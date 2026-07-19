"""
ablation.py — ROAM 模块消融实验
================================
仅用 DeepSeek V4 Flash × 上海，测 4 种配置：

  Full:   搜索 + 聚类 + TSP + RAG  （完整 ROAM）
  -RAG:   搜索 + 聚类 + TSP         （去掉天气/时间上下文）
  -Clust: 搜索 + LLM 分天排序       （去掉聚类和TSP，LLM自己分天）
  Bare:   纯 LLM                    （去掉所有模块）

输出: model/pk/ablation/{variant}/shanghai/q_{query}.json
"""

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.eval.metrics import ItineraryEvaluator
from model.utils.proxy_call import OpenaiCall
from model.agent import TravelAgent
from model.bare import BareAgent

# ==================== 配置 ====================
MODEL_KEY = "deepseek-v4-flash-direct"
MODEL_CFG = {
    "model": "deepseek-v4-flash",
    "api_base": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "label": "DeepSeek-V4-Flash",
}
CITY = "shanghai"
CITY_ZH = "上海"
DAYS = 3
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "pk", "ablation")

# 9 个 query（来自 batch_queries.py）
QUERIES = {
    "A1_dist10km": "我想在上海玩{0}天。每天总移动距离控制在10公里以内，尽量多去景点，按最短路径顺序游览。",
    "A2_walk800m": "在上海玩{0}天，优先步行距离800米以内的相邻景点组成游览群，超过2公里用公共交通衔接，每天不超过3个步行群。",
    "A3_min_transit": "在上海玩{0}天，最小化总交通时间，每天交通控制在1小时内。精确计算移动时间，宁愿少去景点也不超交通预算。",
    "B1_museum_morning": "在上海玩{0}天。博物馆和展览馆只在上午9:00-12:00安排，下午13:00-17:00安排自然风光和公园，晚上17:00-19:00安排美食街区。",
    "B2_stay_duration": "在上海玩{0}天。不同类型景点不同停留时间：博物馆90min、自然风光和公园60min、购物商圈美食45min。每天9:00开始18:00结束。",
    "C1_monday_no_museum": "在上海玩{0}天，出发日期是下周一。完全避开博物馆类景点（周一闭馆），优先公园、自然风光、历史街区和购物商圈。",
    "C2_rain_indoor": "在上海玩{0}天。请查询天气预报，有雨优先室内景点（博物馆、艺术馆、购物中心），晴天安排户外自然风光和公园。",
    "D1_same_district": "在上海玩{0}天。每天景点集中在同一地理区域，同一天内任意两个景点直线距离不超过5公里。不同天有明确地理分隔。",
    "D2_balanced_coverage": "在上海玩{0}天。{0}天行程均衡覆盖不同区域，每天一个主题片区，景点间平均间距控制在3公里以内。",
}


def get_api_key(env_name):
    key = os.getenv(env_name, "")
    if key: return key
    try:
        from model.config import DEEPSEEK_API_KEY
        return DEEPSEEK_API_KEY
    except ImportError:
        return ""


def run_full(query, output_dir):
    """完整 ROAM"""
    proxy = OpenaiCall(api_key=get_api_key("DEEPSEEK_API_KEY"), base_url=MODEL_CFG["api_base"])
    planner = TravelAgent(
        user_reqs=[query], city=CITY, num_days=DAYS, type="zh",
        proxy_call=proxy, use_rag=True,
        travel_date=datetime.date.today().strftime("%Y-%m-%d"),
        citywalk=True, model_override=MODEL_CFG["model"],
        verbose=False, output_dir=output_dir,
    )
    return planner.solve_multiday()


def run_no_rag(query, output_dir):
    """ROAM 去掉 RAG"""
    proxy = OpenaiCall(api_key=get_api_key("DEEPSEEK_API_KEY"), base_url=MODEL_CFG["api_base"])
    planner = TravelAgent(
        user_reqs=[query], city=CITY, num_days=DAYS, type="zh",
        proxy_call=proxy, use_rag=False,
        travel_date=datetime.date.today().strftime("%Y-%m-%d"),
        citywalk=True, model_override=MODEL_CFG["model"],
        verbose=False, output_dir=output_dir,
    )
    return planner.solve_multiday()


def run_no_cluster(query, output_dir):
    """搜索 + LLM 自己分天排序（去掉聚类和TSP）
    
    做法：先用 TravelAgent 的搜索引擎获取 top-K 相关 POI，
    然后把搜索结果（而不是全部500个POI）交给 Bare LLM 规划。
    """
    import pandas as pd
    import numpy as np
    
    proxy = OpenaiCall(api_key=get_api_key("DEEPSEEK_API_KEY"), base_url=MODEL_CFG["api_base"])
    
    # Step 1: 用 ROAM 的搜索获取相关 POI
    planner = TravelAgent(
        user_reqs=[query], city=CITY, num_days=DAYS, type="zh",
        proxy_call=proxy, use_rag=False,
        travel_date=datetime.date.today().strftime("%Y-%m-%d"),
        citywalk=True, model_override=MODEL_CFG["model"],
        verbose=False, output_dir=output_dir,
    )
    # 只做搜索，不做后续步骤
    req_topk_pois, _ = planner._get_reqs_topk()
    search_ids = req_topk_pois[:, 0].astype(int).tolist()
    
    # Step 2: 构建仅含搜索结果的 POI 数据库
    all_dfs = []
    for c in ["fuzhou", "guangzhou", "shenzhen", "shanghai"]:
        df = pd.read_csv(os.path.join("model", "data", f"{c}_zh.csv"))
        df["_city"] = c
        all_dfs.append(df)
    all_data = pd.concat(all_dfs, ignore_index=True)
    
    # 只包含搜索命中的 POI + 上海全部 POI 的基本信息
    city_data = all_data[all_data["_city"] == CITY]
    search_names = set(all_data.loc[search_ids]["name"].tolist()) if search_ids else set()
    
    poi_lines = []
    for _, row in city_data.iterrows():
        name = str(row["name"])
        is_search_hit = "⭐" if name in search_names else " "
        poi_lines.append(
            f"{is_search_hit} [{row['_city']}] {name} | {row.get('category','')} | "
            f"坐标({row.get('lon',''):.4f},{row.get('lat',''):.4f}) | "
            f"{str(row.get('context',''))[:120]}"
        )
    
    # Step 3: 构建 prompt（强调优先选 ⭐ 标记的搜索命中 POI）
    poi_db = "\n".join(poi_lines)
    prompt = f"""你是一位资深旅行规划师。以下是{ CITY_ZH }的全部景点数据库（⭐标记为系统推荐景点，建议优先选择）：

{poi_db}

请为游客规划一个 **{CITY_ZH}** {DAYS} 天的行程。

【游客的具体要求】
{query}

【重要】只选择标记为 [{CITY}] 的景点。优先选择带 ⭐ 标记的推荐景点，但也可以适量搭配其他景点使行程更合理。必须严格遵守游客的所有具体要求。

以严格 JSON 格式返回，不要任何额外文字：
{{"trip_title": "...", "trip_overview": "...", "days": [{{"day":1, "day_theme":"...", "itinerary":"景点A -> 景点B", "day_reason":"...", "pois":{{"1":"简介","2":"简介"}}}}]}}

重要规则：
- itinerary 中的景点名必须从数据库中选择，名称完全一致
- 景点顺序考虑地理位置，避免来回绕路
- 每个景点只能在一天出现，不能跨天重复
- 优先选 ⭐ 推荐景点（至少占60%以上）
- 只返回 JSON，不要任何额外解释或 markdown 代码块标记
"""
    
    # Step 4: 调用 LLM
    print("  ⏳ 搜索->LLM 自主规划（无聚类/TSP）...", end=" ", flush=True)
    t0 = time.time()
    try:
        import re
        response = proxy.chat(model=MODEL_CFG["model"], messages=[{"role":"user","content":prompt}], temperature=0.7)
        raw = response[0] if isinstance(response, tuple) else response
        json_match = re.search(r'\{[\s\S]*\}', str(raw))
        result = json.loads(json_match.group() if json_match else str(raw))
        elapsed = time.time() - t0
        
        # 后处理
        result["city"] = CITY
        result["city_zh"] = CITY_ZH
        result["num_days"] = DAYS
        result["rag_context"] = ""
        result["model"] = "ablation_no_cluster"
        
        for day_idx, day in enumerate(result.get("days", [])):
            day["day"] = day.get("day", day_idx + 1)
            if "weather_note" not in day: day["weather_note"] = "无"
            if "day_theme" not in day: day["day_theme"] = f"第{day_idx+1}天"
            if "day_reason" not in day: day["day_reason"] = ""
            itinerary = day.get("itinerary", "")
            names = [n.strip() for n in itinerary.replace("→","->").split("->") if n.strip()]
            day["itinerary"] = " -> ".join(names)
            day["timeline"] = ""
            day["transit_summary"] = ""
        
        n_pois = sum(len(d.get("itinerary","").split("->")) for d in result.get("days",[]))
        print(f"✅ {elapsed:.0f}s, {n_pois} POIs")
        return result
    except Exception as e:
        print(f"❌ {time.time()-t0:.0f}s: {e}")
        return None


def run_bare(query, output_dir):
    """纯 Bare LLM"""
    proxy = OpenaiCall(api_key=get_api_key("DEEPSEEK_API_KEY"), base_url=MODEL_CFG["api_base"])
    planner = BareAgent(
        user_reqs=[query], city=CITY, num_days=DAYS, type="zh",
        proxy_call=proxy, use_rag=False,
        travel_date=datetime.date.today().strftime("%Y-%m-%d"),
        citywalk=False, model_override=MODEL_CFG["model"],
        verbose=False, output_dir=output_dir,
    )
    return planner.solve()


VARIANTS = {
    "full": {"name": "完整ROAM", "fn": run_full, "desc": "搜索+聚类+TSP+RAG"},
    "no_rag": {"name": "-RAG", "fn": run_no_rag, "desc": "搜索+聚类+TSP（无天气/时间）"},
    "no_cluster": {"name": "-Clust", "fn": run_no_cluster, "desc": "仅搜索+LLM（无聚类/TSP）"},
    "bare": {"name": "Bare", "fn": run_bare, "desc": "纯LLM（无任何模块）"},
}


def run_ablation(variants_to_run, query_keys, output_base):
    """运行消融实验"""
    os.makedirs(output_base, exist_ok=True)
    total = len(variants_to_run) * len(query_keys)
    current = 0
    
    for vkey in variants_to_run:
        vinfo = VARIANTS[vkey]
        var_dir = os.path.join(output_base, vkey, CITY)
        os.makedirs(var_dir, exist_ok=True)
        
        for qkey in query_keys:
            current += 1
            out_path = os.path.join(var_dir, f"q_{qkey}.json")
            
            if os.path.exists(out_path):
                print(f"[{current}/{total}] ⏭️  跳过(已有): {vinfo['name']} {qkey}")
                continue
            
            query = QUERIES[qkey].format(DAYS)
            print(f"\n[{current}/{total}] [{vinfo['name']}] {qkey}", end=" ")
            
            t0 = time.time()
            result = vinfo["fn"](query, var_dir)
            
            if result is not None:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                print(f"  ❌ 失败")


def print_ablation_summary(output_base, query_keys):
    """打印消融对比"""
    from model.eval.metrics import ItineraryEvaluator
    
    keys = ["RC", "SM", "DB", "TT", "CE", "POI"]
    arrows = {"RC": "down", "SM": "up", "DB": "down", "TT": "down", "CE": "up", "POI": "up"}
    query_base_str = "体验当地历史文化与美食"
    evaluator = ItineraryEvaluator(city=CITY)
    
    print("\n" + "=" * 90)
    print("  ROAM 消融实验 — DeepSeek V4 Flash × 上海")
    print("=" * 90)
    
    # Per-query comparison
    vkeys = ["full", "no_rag", "no_cluster", "bare"]
    vnames = [VARIANTS[v]["name"] for v in vkeys]
    vdescs = [VARIANTS[v]["desc"] for v in vkeys]
    
    print(f"\n  {'Query':<22} {'指标':<6}", end="")
    for vn in vnames:
        print(f" {vn:>8}", end="")
    print(f"  {'趋势':<30}")
    print("  " + "-" * 85)
    
    all_wins = {v: {k: 0 for k in keys} for v in vkeys}
    
    for qkey in query_keys:
        results = {}
        for vkey in vkeys:
            path = os.path.join(output_base, vkey, CITY, f"q_{qkey}.json")
            if not os.path.exists(path): continue
            with open(path) as f:
                results[vkey] = json.load(f)
        
        if len(results) < 2: continue
        
        metrics = {}
        for vkey, data in results.items():
            metrics[vkey] = evaluator.compute_all(data, query_base_str)
        
        for k in keys:
            print(f"  {qkey:<22} {k:<6}", end="")
            vals = []
            for vkey in vkeys:
                v = metrics.get(vkey, {}).get(k, 0) if vkey in metrics else None
                if v is not None:
                    vals.append(v)
                    print(f" {v:>8.1f}", end="")
                else:
                    print(f" {'---':>8}", end="")
            
            # Trend: does the metric improve as we add modules?
            if len(vals) >= 3 and all(isinstance(x, (int, float)) for x in vals):
                trend = ""
                # bare -> no_cluster -> no_rag -> full should show improvement for ROAM-favored metrics
                print(f"  ", end="")
            print()
        
        print()
    
    # Summary table
    print("\n" + "=" * 90)
    print("  汇总: 各变体 vs Full ROAM (6指标×9query=54 cells)")
    print("=" * 90)
    print(f"  {'变体':<14} {'描述':<30} {'vs Full 胜':>10} {'vs Full 负':>10}")
    print("  " + "-" * 65)
    
    for vkey in ["no_rag", "no_cluster", "bare"]:
        win = 0; lose = 0
        for qkey in query_keys:
            full_path = os.path.join(output_base, "full", CITY, f"q_{qkey}.json")
            var_path = os.path.join(output_base, vkey, CITY, f"q_{qkey}.json")
            if not os.path.exists(full_path) or not os.path.exists(var_path): continue
            with open(full_path) as f: full = json.load(f)
            with open(var_path) as f: var = json.load(f)
            mF = evaluator.compute_all(full, query_base_str)
            mV = evaluator.compute_all(var, query_base_str)
            for k in keys:
                d = mF.get(k,0) - mV.get(k,0)
                if arrows[k] == "down":
                    if d < -1: win += 1  # full better (lower)
                    elif d > 1: lose += 1
                else:
                    if d > 0.001: win += 1  # full better (higher)
                    elif d < -0.001: lose += 1
        print(f"  {VARIANTS[vkey]['name']:<14} {VARIANTS[vkey]['desc']:<30} {win:>10} {lose:>10}")
    
    print("=" * 90)
    print("  注: '胜'=Full ROAM在该指标上优于该变体, '负'=该变体优于Full")


# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROAM 消融实验")
    parser.add_argument("--variants", type=str, default="full,no_rag,no_cluster,bare",
                        help="逗号分隔的变体: full,no_rag,no_cluster,bare")
    parser.add_argument("--queries", type=str, default="",
                        help="逗号分隔的query key（默认全部9个）")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--summary-only", action="store_true", help="仅打印已有结果的汇总")
    args = parser.parse_args()
    
    output_base = args.output or OUTPUT_BASE
    variants = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS]
    query_keys = [q.strip() for q in args.queries.split(",") if q.strip() in QUERIES] if args.queries else list(QUERIES.keys())
    
    print("=" * 90)
    print("  ROAM 消融实验")
    print("=" * 90)
    print(f"  模型: {MODEL_CFG['label']}  城市: {CITY_ZH}  {DAYS}天")
    print(f"  变体: {', '.join(VARIANTS[v]['name'] + '(' + VARIANTS[v]['desc'] + ')' for v in variants)}")
    print(f"  Query: {len(query_keys)} 个")
    print(f"  总调用: {len(variants) * len(query_keys)} 次")
    print()
    
    if args.summary_only:
        print_ablation_summary(output_base, query_keys)
    else:
        run_ablation(variants, query_keys, output_base)
        print_ablation_summary(output_base, query_keys)
