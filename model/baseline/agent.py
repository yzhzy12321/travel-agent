"""
BaselineAgent v3 — 原始 ITINERA 搜索+聚类 + ROAM Prompt
=========================================================
仅复用原版 ITINERA 的搜索和空间聚类，
LLM 调用使用 ROAM 的多日 prompt 格式。
"""
import json, os, re, sys, time, datetime, numpy as np
from model.agent import TravelAgent

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "ITINERA-main")


class StringProxy:
    def __init__(self, real): self._r = real
    def chat(self, **kw):
        r = self._r.chat(**kw)
        return r[0] if isinstance(r, tuple) else r
    def __getattr__(self, name): return getattr(self._r, name)


class BaselineAgent(TravelAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rag = False; self.rag = None

    def solve(self) -> dict:
        print(f"Baseline: {self.city_zh} (ITINERA搜索聚类 + ROAM Prompt)")
        query = self.user_reqs[0] if self.user_reqs else f"{self.city_zh}旅游"
        t0 = time.time()

        # Step 1: 原版搜索+聚类
        old_path = sys.path.copy()
        sys.path.insert(0, BASELINE_PATH)
        from model import itinera as bi
        bi.ItiNera.MODEL = self.MODEL

        p = bi.ItiNera(
            user_reqs=[query], hours=min(self.num_days*6, 8),
            proxy_call=StringProxy(self.proxy),
            citywalk=True, city=self.city, type=self.type,
        )
        req_topk, pseudo = p.get_reqs_topk()
        _, clusters, _ = p.get_poi_candidates(req_topk, p.must_see_pois, pseudo)
        sys.path = old_path

        # Step 2: 收集聚类 POI
        all_ids = list(set(i for c in clusters for i in c))
        city_data = p.user_favorites
        poi_lines = []
        for pid in all_ids:
            if pid in city_data.index:
                row = city_data.loc[pid]
                poi_lines.append(f"- {row['name']} | {row.get('category','')} | ({row.get('lon',''):.4f},{row.get('lat',''):.4f}) | {str(row.get('context',''))[:120]}")
        poi_db = "\n".join(poi_lines)

        # 聚类提示
        ch = []
        for ci, c in enumerate(clusters):
            names = [city_data.loc[i]['name'] for i in c if i in city_data.index]
            ch.append(f"  第{ci+1}组: {' -> '.join(names[:6])}")
        cluster_hint = "\n".join(ch)

        # Step 3: LLM
        prompt = f"""你是一位资深旅行规划师。以下是{self.city_zh}的相关景点：

{poi_db}

空间聚类建议（同组景点地理位置相近）：
{cluster_hint}

请规划 **{self.city_zh}** {self.num_days} 天行程。
要求：{query}

返回严格 JSON：
{{"trip_title":"...","trip_overview":"...","days":[{{"day":1,"day_theme":"...","itinerary":"景点A -> 景点B","day_reason":"...","pois":{{"1":"简介"}}}}]}}
规则：景点名与数据库完全一致，参考聚类分组，每天5-8个不跨天重复，只返回JSON。"""

        print("  LLM ...", end=" ", flush=True)
        try:
            resp = self.proxy.chat(model=self.MODEL, messages=[{"role":"user","content":prompt}], temperature=0.7)
            raw = resp[0] if isinstance(resp, tuple) else resp
            jm = re.search(r'\{[\s\S]*\}', str(raw))
            result = json.loads(jm.group() if jm else str(raw))
        except Exception as e:
            print(f"FAIL: {e}"); result = None

        elapsed = time.time() - t0
        if result is None:
            result = {"trip_title": f"{self.city_zh}{self.num_days}日游", "days": []}

        result["city"] = self.city; result["city_zh"] = self.city_zh
        result["num_days"] = self.num_days; result["rag_context"] = ""
        result["model"] = "itinera_baseline"

        for di, day in enumerate(result.get("days", [])):
            day["day"] = di+1
            day["weather_note"] = day.get("weather_note","无")
            day["day_theme"] = day.get("day_theme",f"第{di+1}天")
            day["day_reason"] = day.get("day_reason","")
            itin = day.get("itinerary","")
            day["itinerary"] = " -> ".join(n.strip() for n in itin.replace("→","->").split("->") if n.strip())
            day["timeline"] = ""; day["transit_summary"] = ""

        n = sum(len(d.get("itinerary","").split("->")) for d in result.get("days",[]))
        print(f"OK {elapsed:.0f}s, {n} POIs")
        self._save_result(result)
        return result

    def _save_result(self, result):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        with open(os.path.join(self.output_dir, f"{ts}_zh_multiday.json"),"w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
