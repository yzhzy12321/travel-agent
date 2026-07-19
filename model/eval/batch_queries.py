"""
batch_queries.py — 多 Query 分类对比实验
=========================================
设计意图：定义多个 query 类别，每类测试 ROAM 在不同维度上的优势
  - A 类：距离/空间约束 → TSP + 聚类优势
  - B 类：时间窗口约束 → RAG 营业时间优势
  - C 类：天气/日期感知 → RAG 天气优势
  - D 类：行政区/空间均衡 → 聚类优势

输出结构：model/pk/{roam|bare}/{Model}/{city}/q_{query_key}.json

用法：
    python model/eval/batch_queries.py                      # 全部
    python model/eval/batch_queries.py --cat A              # 仅 A 类
    python model/eval/batch_queries.py --models deepseek-v4-pro --cat B
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

# ==================== 路径 ====================
PK_DIR = os.path.join(os.path.dirname(__file__), "..", "pk")
ROAM_DIR = os.path.join(PK_DIR, "roam")
BARE_DIR = os.path.join(PK_DIR, "bare")

# ==================== 模型 ====================
ALL_MODELS = {
    "deepseek-v4-flash-direct": {
        "model": "deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek-V4-Flash",
    },
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek-V4-Pro",
    },
    "qwen3.7-plus": {
        "model": "qwen3.7-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "BAILIAN_API_KEY",
        "label": "Qwen3.7-Plus",
    },
    "qwen3.7-max": {
        "model": "qwen3.7-max",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "BAILIAN_API_KEY",
        "label": "Qwen3.7-Max",
    },
}

CITIES = ["fuzhou", "guangzhou", "shenzhen", "shanghai"]
CITY_ZH = {"fuzhou": "福州", "guangzhou": "广州", "shenzhen": "深圳", "shanghai": "上海"}
DAYS = 3

# ==================== Query 分类定义 ====================
# 每类 query 设计原则：ROAM 的某个模块提供 Bare 无法获取的信息/能力

QUERY_CATEGORIES = {
    "A": {
        "name": "距离硬约束",
        "desc": "测试 TSP 路径优化 + 道路距离精确计算的优势",
        "roam_advantage": "TSP 精确计算累积道路距离，Bare LLM 无法精确估算多段距离之和",
        "queries": {
            "A1_dist10km": (
                "我想在{city}玩{days}天。核心要求：每天的总移动距离必须控制在10公里以内，"
                "在此前提下尽量多去景点。景点之间的游览顺序必须按最短路径排列，避免来回绕路。"
                "请精确计算每段距离，确保总移动量不超标。"
            ),
            "A2_walk800m": (
                "我想在{city}玩{days}天。核心要求：优先选择步行距离在800米以内的相邻景点组成游览群，"
                "景点间超过2公里的必须依赖公共交通。请将步行可达的景点安排在一起，"
                "步行群之间用公共交通衔接。每天不超过3个步行群。"
            ),
            "A3_min_transit": (
                "我想在{city}玩{days}天。核心要求：最小化总交通时间，每天交通时间控制在1小时内。"
                "请精确计算景点间移动时间（步行80m/min、骑行250m/min、驾车500m/min），"
                "选择交通最优的景点组合和顺序。宁愿少去景点也不能超交通预算。"
            ),
        },
    },
    "B": {
        "name": "时间窗口约束",
        "desc": "测试 RAG 营业时间 + 时段感知的优势",
        "roam_advantage": "RAG 提供真实营业时间和时段建议，Bare 统一用 00:00-23:59",
        "queries": {
            "B1_museum_morning": (
                "我想在{city}玩{days}天。核心要求：博物馆和展览馆只在上午9:00-12:00安排参观"
                "（这些场馆通常上午人流较少），下午13:00-17:00安排自然风光和城市公园"
                "（下午光线适合户外），晚上17:00-19:00安排美食街区和夜市。"
                "严格按这个时间窗口分配景点类型。"
            ),
            "B2_stay_duration": (
                "我想在{city}玩{days}天。核心要求：不同类型景点安排不同的合理停留时间——"
                "博物馆和纪念馆 90 分钟、自然风光和公园 60 分钟、购物商圈和美食街区 45 分钟。"
                "每天 9:00 开始、18:00 结束。请合理安排，确保每天时间线可行不超时。"
            ),
        },
    },
    "C": {
        "name": "天气与日期感知",
        "desc": "测试 RAG 天气数据 + 日期感知的优势",
        "roam_advantage": "RAG 提供天气预测和星期几信息，Bare 无此数据",
        "queries": {
            "C1_monday_no_museum": (
                "我想在{city}玩{days}天，出发日期是下周一。核心要求：周一绝大多数博物馆和纪念馆闭馆，"
                "请完全避开博物馆类景点，优先安排公园、自然风光、历史街区和购物商圈。"
                "如果某些场馆周一正常开放可以在行程中注明。"
            ),
            "C2_rain_indoor": (
                "我想在{city}玩{days}天。核心要求：请查询天气预报，如果某天有较大概率下雨，"
                "那天的行程应优先安排室内景点（博物馆、艺术馆、购物中心、美食城），"
                "晴天再安排户外自然风光和公园。每天都要有室内备选方案。"
            ),
        },
    },
    "D": {
        "name": "空间均衡与行政区约束",
        "desc": "测试空间聚类按区域分组 + 跨区协调的优势",
        "roam_advantage": "聚类算法精确按坐标分组，Bare LLM 凭感觉估计区域归属",
        "queries": {
            "D1_same_district": (
                "我想在{city}玩{days}天。核心要求：每天的景点必须集中在同一片地理区域，"
                "同一天内的任意两个景点之间直线距离不超过5公里。不同天的景点群之间要有明确的地理分隔。"
                "避免出现同一天内东南西北到处跑的情况。"
            ),
            "D2_balanced_coverage": (
                "我想在{city}玩{days}天。核心要求：{days}天的行程要均衡覆盖城市的不同区域，"
                "每天一个主题片区。同时每天景点之间的平均间距控制在3公里以内。"
                "既要有区域代表性，又要保证每天的紧凑性。"
            ),
        },
    },
}

# 城市适配的 query 前缀（用于给 Bare 的 prompt 增加城市信息）
# Bare prompt 会让 LLM 从四城 POI 中自选，需要明确城市
CITY_CONTEXT = {
    "fuzhou": "福州（福建省会，三坊七巷、鼓山、西湖等，以闽都文化和温泉闻名）",
    "guangzhou": "广州（广东省会，珠江新城、沙面、长隆、白云山等，以广府美食和商贸闻名）",
    "shenzhen": "深圳（经济特区，南山科技园、华侨城、大小梅沙等，以创新活力和滨海风光闻名）",
    "shanghai": "上海（直辖市，外滩、陆家嘴、新天地、豫园等，以海派文化和摩登都市闻名）",
}


def get_api_key(env_name):
    key = os.getenv(env_name, "")
    if key:
        return key
    try:
        from model.config import DEEPSEEK_API_KEY, BAILIAN_API_KEY
        mapping = {"DEEPSEEK_API_KEY": DEEPSEEK_API_KEY, "BAILIAN_API_KEY": BAILIAN_API_KEY}
        return mapping.get(env_name, "")
    except ImportError:
        return ""


def generate_one(city, days, model_id, api_base, api_key, use_roam, label, output_dir, query):
    """生成单次行程"""
    mode_label = "ROAM" if use_roam else "Bare"
    print(f"  ⏳ [{label}] {CITY_ZH.get(city, city)} {mode_label} ...", end=" ", flush=True)
    t0 = time.time()

    try:
        proxy = OpenaiCall(api_key=api_key, base_url=api_base)
        if use_roam:
            planner = TravelAgent(
                user_reqs=[query],
                city=city,
                num_days=days,
                type="zh",
                proxy_call=proxy,
                use_rag=True,
                travel_date=datetime.date.today().strftime("%Y-%m-%d"),
                citywalk=True,
                model_override=model_id,
                verbose=False,
                output_dir=output_dir,
            )
            result = planner.solve_multiday()
        else:
            planner = BareAgent(
                user_reqs=[query],
                city=city,
                num_days=days,
                type="zh",
                proxy_call=proxy,
                use_rag=False,
                travel_date=datetime.date.today().strftime("%Y-%m-%d"),
                citywalk=False,
                model_override=model_id,
                verbose=False,
                output_dir=output_dir,
            )
            result = planner.solve()
        elapsed = time.time() - t0
        n_pois = sum(len(d.get("itinerary", "").split("->")) for d in result.get("days", []))
        print(f"✅ {elapsed:.0f}s, {n_pois} POIs")
        return result, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ {elapsed:.0f}s: {e}")
        return None, str(e)


def compute_metrics(result, city, query):
    """计算指标"""
    evaluator = ItineraryEvaluator(city=city)
    try:
        return evaluator.compute_all(result, query)
    except Exception as e:
        print(f"    ⚠️ 指标计算失败: {e}")
        return {}


def run_query_experiments(model_keys, cities, days, categories, output_dir, modes=None):
    """主实验函数 — 每个 query 跑 ROAM +/ Bare"""
    if modes is None:
        modes = ["ROAM", "Bare"]
    all_rows = []

    # 展开所有 query
    query_list = []
    for cat_key in categories:
        cat = QUERY_CATEGORIES[cat_key]
        for qkey, qtmpl in cat["queries"].items():
            query_list.append((cat_key, qkey, qtmpl))

    total = len(model_keys) * len(cities) * len(query_list) * len(modes)
    current = 0

    for mkey in model_keys:
        cfg = ALL_MODELS[mkey]
        api_key = get_api_key(cfg["api_key_env"])
        if not api_key:
            print(f"⚠️ [{cfg['label']}] 缺少 API Key，跳过")
            continue

        for city in cities:
            for cat_key, qkey, qtmpl in query_list:
                query = qtmpl.format(city=CITY_ZH.get(city, city), days=days)
                cat = QUERY_CATEGORIES[cat_key]

                # --- ROAM ---
                if "ROAM" in modes:
                    current += 1
                    model_dir = os.path.join(ROAM_DIR, cfg["label"], city)
                    os.makedirs(model_dir, exist_ok=True)
                    roam_path = os.path.join(model_dir, f"q_{qkey}.json")

                    if os.path.exists(roam_path):
                        print(f"\n[{current}/{total}] ⏭️  跳过(已有): {cfg['label']} {CITY_ZH[city]} [{cat_key}] ROAM")
                        with open(roam_path, "r", encoding="utf-8") as f:
                            result_roam = json.load(f)
                    else:
                        print(f"\n[{current}/{total}] [{cat_key}:{cat['name']}]", end=" ")
                        result_roam, err = generate_one(
                            city, days, cfg["model"], cfg["api_base"], api_key,
                            use_roam=True, label=cfg["label"],
                            output_dir=model_dir, query=query,
                        )
                        if result_roam is not None:
                            with open(roam_path, "w", encoding="utf-8") as f:
                                json.dump(result_roam, f, ensure_ascii=False, indent=2)

                # --- Bare ---
                if "Bare" in modes:
                    current += 1
                    bare_model_dir = os.path.join(BARE_DIR, cfg["label"], city)
                    os.makedirs(bare_model_dir, exist_ok=True)
                    bare_path = os.path.join(bare_model_dir, f"q_{qkey}.json")

                    if os.path.exists(bare_path):
                        print(f"\n[{current}/{total}] ⏭️  跳过(已有): {cfg['label']} {CITY_ZH[city]} [{cat_key}] Bare")
                        with open(bare_path, "r", encoding="utf-8") as f:
                            result_bare = json.load(f)
                    else:
                        print(f"[{current}/{total}] [{cat_key}:{cat['name']}]", end=" ")
                        result_bare, err = generate_one(
                            city, days, cfg["model"], cfg["api_base"], api_key,
                            use_roam=False, label=cfg["label"],
                            output_dir=bare_model_dir, query=query,
                        )
                        if result_bare is not None:
                            with open(bare_path, "w", encoding="utf-8") as f:
                                json.dump(result_bare, f, ensure_ascii=False, indent=2)

                # 保存独立指标
                if "ROAM" in modes and result_roam is not None:
                    m_roam = compute_metrics(result_roam, city, query)
                    all_rows.append({
                        "cat": cat_key, "query": qkey, "model": cfg["label"],
                        "city": city, "mode": "ROAM",
                        "n_pois": sum(len(d.get("itinerary","").split("->"))
                                      for d in result_roam.get("days",[])),
                        **(m_roam or {}),
                    })
                if "Bare" in modes and result_bare is not None:
                    m_bare = compute_metrics(result_bare, city, query)
                    all_rows.append({
                        "cat": cat_key, "query": qkey, "model": cfg["label"],
                        "city": city, "mode": "Bare",
                        "n_pois": sum(len(d.get("itinerary","").split("->"))
                                      for d in result_bare.get("days",[])),
                        **(m_bare or {}),
                    })

    return all_rows


def print_summary(all_rows):
    """打印分类汇总"""
    if not all_rows:
        print("无数据")
        return

    import statistics

    keys = ["RC", "SM", "DB", "TT", "CE", "POI"]
    arrows = {"RC": "↓", "SM": "↑", "DB": "↓", "TT": "↓", "CE": "↑", "POI": "↑"}

    # 按类别 + query 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["cat"], r["query"])].append(r)

    print("\n" + "=" * 90)
    print("  🏆 ROAM vs Bare-LLM — 多 Query 分类 PK 汇总")
    print("=" * 90)

    grand_roam = 0
    grand_bare = 0

    for cat_key in ["A", "B", "C", "D"]:
        cat = QUERY_CATEGORIES[cat_key]
        print(f"\n{'─'*90}")
        print(f"  📂 {cat_key}类：{cat['name']}")
        print(f"  🎯 ROAM优势：{cat['roam_advantage']}")
        print(f"{'─'*90}")

        for qkey in cat["queries"]:
            group = groups.get((cat_key, qkey), [])
            if not group:
                continue

            # 按城市汇总
            print(f"\n  📋 {qkey}")
            print(f"  {'城市':<8} {'ROAM':>24} {'Bare':>24} {'Δ':>12}  {'胜者':<6}")
            print(f"  {'':─<8} {'':─<24} {'':─<24} {'':─<12} {'':─<6}")

            cat_roam_w = 0
            cat_bare_w = 0

            for city in CITIES:
                city_group = [r for r in group if r["city"] == city]
                roam_rows = [r for r in city_group if r["mode"] == "ROAM"]
                bare_rows = [r for r in city_group if r["mode"] == "Bare"]

                if not roam_rows or not bare_rows:
                    continue

                rm = roam_rows[0]
                bm = bare_rows[0]

                city_roam_w = 0
                city_bare_w = 0
                parts = []
                for k in keys:
                    rv = rm.get(k, 0)
                    bv = bm.get(k, 0)
                    d = bv - rv
                    w = ""
                    if arrows[k] == "↓":
                        if d > 0.01: w = "ROAM"; city_roam_w += 1; cat_roam_w += 1
                        elif d < -0.01: w = "Bare"; city_bare_w += 1; cat_bare_w += 1
                    else:
                        if d > 0.01: w = "Bare"; city_bare_w += 1; cat_bare_w += 1
                        elif d < -0.01: w = "ROAM"; city_roam_w += 1; cat_roam_w += 1
                    parts.append(f"{k}={rv:.1f}vs{bv:.1f}({w or '平'})")

                line = " | ".join(parts)
                print(f"  {CITY_ZH.get(city, city):<8} {line}")
                print(f"  {'':<8} → ROAM {city_roam_w}-{city_bare_w} Bare")

            grand_roam += cat_roam_w
            grand_bare += cat_bare_w
            print(f"  {'':<8} {'─'*70}")
            print(f"  {'':<8} [{qkey}] ROAM {cat_roam_w}-{cat_bare_w} Bare")

    print(f"\n{'='*90}")
    print(f"  🏁 总计：ROAM {grand_roam} - {grand_bare} Bare")
    if grand_roam > grand_bare:
        print(f"  ✅ ROAM 胜出 +{grand_roam - grand_bare}")
    elif grand_bare > grand_roam:
        print(f"  ❌ Bare 胜出 +{grand_bare - grand_roam}")
    else:
        print(f"  🤝 平局")
    print(f"{'='*90}")


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROAM 多 Query 分类对比实验")
    parser.add_argument("--models", type=str, default="deepseek-v4-flash-direct",
                        help="逗号分隔的模型 key")
    parser.add_argument("--cities", type=str, default="",
                        help="逗号分隔的城市（默认全部）")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--cat", type=str, default="A,B,C,D",
                        help="要跑的类别，逗号分隔（默认全部 A,B,C,D）")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--roam-only", action="store_true", help="仅跑 ROAM")
    parser.add_argument("--bare-only", action="store_true", help="仅跑 Bare")
    args = parser.parse_args()

    output_dir = args.output or PK_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_keys = [k.strip() for k in args.models.split(",") if k.strip() in ALL_MODELS]
    cities = [c.strip() for c in args.cities.split(",") if c.strip() in CITIES] if args.cities else CITIES
    categories = [c.strip() for c in args.cat.split(",") if c.strip() in QUERY_CATEGORIES]

    modes = []
    if args.roam_only: modes = ["ROAM"]
    elif args.bare_only: modes = ["Bare"]
    else: modes = ["ROAM", "Bare"]

    print("=" * 90)
    print("  ROAM vs Bare-LLM 多 Query 分类实验")
    print("=" * 90)
    print(f"  模型: {', '.join(ALL_MODELS[m]['label'] for m in model_keys)}")
    print(f"  城市: {', '.join(CITY_ZH[c] for c in cities)}")
    cat_names = ', '.join(c + '(' + QUERY_CATEGORIES[c]['name'] + ')' for c in categories)
    print(f"  类别: {cat_names}")
    print(f"  模式: {'+'.join(modes)}")
    total_runs = len(model_keys) * len(cities) * sum(len(QUERY_CATEGORIES[c]['queries']) for c in categories) * len(modes)
    print(f"  总 API 调用: {total_runs}")
    print()

    rows = run_query_experiments(model_keys, cities, args.days, categories, output_dir, modes)
    print_summary(rows)
