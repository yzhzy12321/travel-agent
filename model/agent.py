"""
Travel Agent: 多日长行程 + RAG 时空感知旅游规划 Agent
=====================================================
功能：
  - 多日行程支持（1-7天）
  - RAG 时空感知上下文注入（天气、交通、人流）
  - 跨日约束优化（POI 不重复、每日负载均衡）
  - 住宿点周边推荐
  - Folium 交互式地图可视化

用法：
    from model.agent import TravelAgent
    planner = TravelAgent(user_reqs=["..."], city="shanghai", num_days=3, ...)
    result = planner.solve_multiday()
"""

import os
import re
import copy
import json
import time
import datetime
import numpy as np
import folium
import concurrent.futures
from typing import Optional
from xyconvert import gcj2wgs

from model.utils.funcs import (
    RecurringList, compute_consecutive_distances, find_indices,
    sample_items, reorder_list, remove_duplicates, get_user_data_embedding
)
from model.utils.all_prompts import (
    get_start_point_prompt, get_dayplan_prompt, get_system_prompt,
    get_hour_prompt, check_final_reverse_prompt, process_input_prompt,
    get_multiday_hour_prompt, get_multiday_system_prompt,
    get_multiday_dayplan_prompt, get_multiday_overview_prompt
)
from model.search import SearchEngine
from model.spatial import SpatialHandler
from model.rag import SpatiotemporalRAG, create_rag_for_trip
from model.config import CITY_ZH_MAP, MULTIDAY_TIME2NUM, LLM_MODEL
from thefuzz import process as fuzz_process  # 酒店名称模糊匹配


class TravelAgent:
    """
    多日旅游规划 Agent

    能力：
    - num_days: 行程天数（1-7天）
    - use_rag: 是否启用时空 RAG 上下文
    - travel_date: 出发日期
    - hotel: 住宿点（模糊匹配POI库）
    """

    def __init__(self,
                 user_reqs: list,
                 city: str = 'shanghai',
                 num_days: int = 2,
                 type: str = 'zh',
                 min_poi_candidate_num: int = 19,
                 keep_prob: float = 0.8,
                 thresh: int = 1000,
                 hours_per_day: Optional[list] = None,
                 proxy_call=None,
                 citywalk: bool = True,
                 use_rag: bool = True,
                 travel_date: Optional[str] = None,
                 hotel: Optional[str] = None):
        """
        初始化 Travel Agent

        Args:
            user_reqs: 用户需求列表
            city: 城市名称
            num_days: 行程天数
            type: 语言类型
            min_poi_candidate_num: 最小候选 POI 数
            keep_prob: POI 采样保留概率
            thresh: 空间聚类距离阈值
            hours_per_day: 每天小时数（None 则自动推断）
            proxy_call: LLM 调用代理
            citywalk: 是否启用 citywalk 模式
            use_rag: 是否启用 RAG 时空感知
            travel_date: 出发日期 (YYYY-MM-DD)
            hotel: 住宿点名称（模糊匹配 POI 库，None 则不启用住宿模式）
        """
        self.MODEL = LLM_MODEL
        self.type = type
        self.proxy = proxy_call
        self.city = city
        self.city_zh = CITY_ZH_MAP.get(city, city)  # 中文显示名
        self.num_days = num_days
        self.citywalk = citywalk
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh
        self.min_poi_candidate_num = min_poi_candidate_num
        self.use_rag = use_rag
        self.hotel_name = hotel

        # ---- 确定每天参数 ----
        if hours_per_day is not None:
            self.hours_per_day = hours_per_day
        else:
            self.hours_per_day = self._infer_hours_per_day()

        self.daily_params = self._get_daily_params()

        # ---- 解析用户需求 ----
        parsed_request = self._parse_user_request(user_reqs)
        (self.must_see_poi_names, self.itinerary_pos_reqs,
         self.itinerary_neg_reqs, self.user_pos_reqs,
         self.user_neg_reqs, self.start_poi,
         self.end_poi) = self._parse_user_input(parsed_request)

        # ---- 初始化 embedding 和数据 ----
        (self.user_favorites, self.embedding,
         self.r2i, self.i2r,
         self.must_see_pois) = get_user_data_embedding(
            city_name=city,
            must_see_poi_names=self.must_see_poi_names,
            type=type
        )

        # ---- 搜索与空间模块 ----
        self.search_engine = SearchEngine(
            embedding=self.embedding, proxy=self.proxy
        )
        self.spatial_handler = SpatialHandler(
            data=self.user_favorites,
            min_clusters=self.daily_params[0][0],
            min_pois=self.daily_params[0][1],
            citywalk=self.citywalk,
            citywalk_thresh=self.daily_params[0][2]
        )

        # ---- RAG 模块 ----
        if self.use_rag:
            self.rag = create_rag_for_trip(
                city=city,
                num_days=num_days,
                travel_date=travel_date
            )
            # 传入 POI 坐标映射，供 OSRM 路线计算
            poi_coords = {
                row["name"]: (float(row["lon"]), float(row["lat"]))
                for _, row in self.user_favorites.iterrows()
            }
            self.rag.set_poi_coords(poi_coords)
        else:
            self.rag = None

        # ---- 住宿点匹配 ----
        self.hotel_poi_id = None
        self.hotel_info = None
        if self.hotel_name:
            self._match_hotel()

    # ========== 住宿点 ==========

    def _match_hotel(self):
        """模糊匹配住宿点到 POI 库，并计算周边 POI"""
        all_names = self.user_favorites["name"].tolist()
        match, score = fuzz_process.extract(self.hotel_name, all_names, limit=1)[0]

        if score < 60:
            print(f"⚠️ 未在 POI 库中找到匹配的住宿点: '{self.hotel_name}' (最佳匹配: '{match}', 得分: {score})")
            self.hotel_name = None
            return

        idx = all_names.index(match)
        self.hotel_poi_id = self.user_favorites.index[idx]
        row = self.user_favorites.loc[self.hotel_poi_id]
        self.hotel_info = {
            "id": int(self.hotel_poi_id),
            "name": str(row["name"]),
            "address": str(row.get("address", "")),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "category": str(row.get("category", "")),
            "context": str(row.get("context", "")),
        }
        print(f"🏨 住宿点: {self.hotel_info['name']} (匹配得分: {score})")

    def _get_hotel_nearby(self, max_distance: int = 3000, top_k: int = 8) -> list:
        """
        获取住宿点周边的 POI 列表

        Args:
            max_distance: 最大距离（米）
            top_k: 最多返回数量

        Returns:
            list[(poi_id, name, category, distance_m)]: 周边 POI 列表
        """
        if self.hotel_poi_id is None:
            return []

        hotel_loc = self.user_favorites.loc[self.hotel_poi_id][["x", "y"]].astype(float).to_numpy()
        all_locs = self.user_favorites[["x", "y"]].astype(float).to_numpy()
        all_ids = self.user_favorites.index.tolist()

        dists = np.linalg.norm(all_locs - hotel_loc, axis=1)
        # 排除酒店自身
        nearby = []
        for i, d in enumerate(dists):
            if all_ids[i] == self.hotel_poi_id:
                continue
            if d < max_distance:
                nearby.append((
                    all_ids[i],
                    self.user_favorites.loc[all_ids[i], "name"],
                    self.user_favorites.loc[all_ids[i], "category"],
                    int(d)
                ))

        nearby.sort(key=lambda x: x[3])
        return nearby[:top_k]

    def _build_hotel_context(self) -> str:
        """构建住宿点上下文描述，用于注入 LLM prompt"""
        if self.hotel_poi_id is None:
            return ""

        nearby = self._get_hotel_nearby()
        lines = [
            "### 🏨 住宿信息",
            f"- 住宿点: **{self.hotel_info['name']}**",
            f"- 地址: {self.hotel_info['address']}",
            f"- 类别: {self.hotel_info['category']}",
        ]

        if nearby:
            lines.append(f"- 周边推荐 (步行/短途可达):")
            for poi_id, name, cat, dist in nearby:
                mode = "🚶步行" if dist < 1000 else "🚲骑行"
                lines.append(f"    {name}（{cat}）— {dist}m {mode}可达")

        return "\n".join(lines)

    # ========== 参数推断 ==========

    def _infer_hours_per_day(self) -> list:
        """通过 LLM 推断每天的小时数"""
        try:
            msg = [{"role": "user", "content": get_multiday_hour_prompt(
                user_reqs=self.user_reqs, num_days=self.num_days
            )}]
            response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
            hours = json.loads(response)
            hours = [int(h) for h in hours]
            # 确保长度匹配
            while len(hours) < self.num_days:
                hours.append(5)
            return hours[:self.num_days]
        except Exception:
            # 默认每天 5 小时
            return [5] * self.num_days

    def _get_daily_params(self) -> list:
        """获取每天的 (min_clusters, min_pois, citywalk_thresh) 参数"""
        if self.num_days in MULTIDAY_TIME2NUM:
            raw = MULTIDAY_TIME2NUM[self.num_days]
        else:
            # 超过 7 天，用第 7 天的参数扩展
            raw = MULTIDAY_TIME2NUM[7]
            while len(raw) < self.num_days:
                raw.append(raw[-1])

        # 如果 LLM 推断的小时数不同，调整 min_pois
        result = []
        for i, (h_clusters, h_pois, h_thresh) in enumerate(raw):
            actual_hours = (
                self.hours_per_day[i]
                if i < len(self.hours_per_day) else 5
            )
            # 按小时比例调整 POI 数
            adjusted_pois = max(3, int(h_pois * actual_hours / h_clusters))
            result.append((h_clusters, adjusted_pois, h_thresh))
        return result

    # ========== 需求解析 ==========

    def _parse_user_request(self, user_reqs):
        """LLM 解析用户需求为结构化格式"""
        response = self.proxy.chat(
            messages=[{"role": "user", "content": process_input_prompt(user_input=user_reqs)}],
            model=self.MODEL
        ).replace("'", '"')
        try:
            return json.loads(response)
        except Exception:
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            print("无法解析用户需求 JSON")
            return []

    def _parse_user_input(self, structured_input):
        """从结构化输入中提取各类需求"""
        must_see_poi_names = []
        itinerary_pos_reqs, itinerary_neg_reqs = [], []
        user_pos_reqs, user_neg_reqs = [], []
        start_poi, end_poi = None, None

        for req in structured_input:
            if req.get("type") is None:
                req["type"] = "地点"
            if req["type"] == "行程":
                itinerary_pos_reqs.append(req.get("pos", ""))
                if req.get("neg") is not None:
                    itinerary_neg_reqs.append(req["neg"])
            elif req["type"] in ["地点", "起点", "终点"]:
                if req.get("mustsee") is True:
                    must_see_poi_names.append(req.get("pos", ""))
                user_pos_reqs.append(req.get("pos", ""))
                user_neg_reqs.append(req.get("neg"))
                if req["type"] == "起点":
                    start_poi = req.get("pos")
                if req["type"] == "终点":
                    end_poi = req.get("pos")
            else:
                raise ValueError(f"未知需求类型: {req['type']}")

        if len(user_pos_reqs) == 0:
            user_pos_reqs = itinerary_pos_reqs

        return (must_see_poi_names, itinerary_pos_reqs, itinerary_neg_reqs,
                user_pos_reqs, user_neg_reqs, start_poi, end_poi)

    # ========== POI 检索 ==========

    def _get_reqs_topk(self):
        """检索 top-k POI——合并地点需求和行程需求作为搜索条件"""
        def process_request(user_pos_req, user_neg_req):
            top_k = min(self.user_favorites.shape[0], self.min_poi_candidate_num)
            req_pois = self.search_engine.query(
                desc=(user_pos_req, user_neg_req), top_k=top_k
            )
            if req_pois is None or len(req_pois) == 0:
                print(f"⚠️ query 返回空: pos='{user_pos_req}', neg='{user_neg_req}'")
                return np.array([]).reshape(0, 2), []
            pseudo_local = [int(poi) for poi in req_pois[:2, 0]]
            return req_pois, pseudo_local

        # 合并地点需求和行程需求，确保搜索覆盖用户所有意图
        search_reqs = list(self.user_pos_reqs)
        for req in self.itinerary_pos_reqs:
            if req not in search_reqs and len(req) > 0:
                search_reqs.append(req)
        search_negs = list(self.user_neg_reqs)
        # 补齐负向需求长度
        while len(search_negs) < len(search_reqs):
            search_negs.append(None)

        all_reqs_topk = []
        pseudo_must_see_pois = []

        print(f"🔍 搜索需求(合并后): {search_reqs}")

        if len(search_reqs) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_request, pos, neg)
                    for pos, neg in zip(search_reqs, search_negs)
                ]
                for f in futures:
                    req_pois, pseudo_local = f.result()
                    if len(req_pois) > 0:
                        pseudo_must_see_pois.extend(pseudo_local)
                        all_reqs_topk.append(req_pois)
        elif len(search_reqs) == 1:
            neg = search_negs[0] if search_negs else None
            req_pois, pseudo_local = process_request(search_reqs[0], neg)
            if len(req_pois) > 0:
                pseudo_must_see_pois.extend(pseudo_local)
                all_reqs_topk.append(req_pois)
        else:
            raise ValueError(f"无搜索需求: user_pos_reqs={self.user_pos_reqs}, itinerary_pos_reqs={self.itinerary_pos_reqs}")

        if len(all_reqs_topk) == 0:
            raise RuntimeError(
                f"POI 检索返回空结果！\n"
                f"  search_reqs: {search_reqs}\n"
                f"  embedding shape: {self.embedding.shape}"
            )

        all_reqs_topk = np.concatenate(all_reqs_topk, axis=0)
        unique_vals = np.unique(all_reqs_topk[:, 0])
        result = [[v, all_reqs_topk[all_reqs_topk[:, 0] == v][:, 1].sum()]
                   for v in unique_vals]
        result = np.array(result)
        sorted_result = result[result[:, 1].argsort()[::-1]]

        return sorted_result, pseudo_must_see_pois

    # ========== 多日求解 ==========

    def solve_multiday(self) -> dict:
        """
        多日行程求解主入口

        Returns:
            dict: {
                "trip_title": "...",
                "trip_overview": "...",
                "num_days": N,
                "rag_context": "...",       # RAG 时空概览
                "days": [
                    {
                        "day": 1,
                        "day_theme": "...",
                        "itinerary": "...",
                        "day_reason": "...",
                        "weather_note": "...",
                        "pois": {...}
                    },
                    ...
                ]
            }
        """
        print(f"\n{'='*30}")
        print(f"🌍 Travel Agent: {self.num_days}天 {self.city_zh} 行程规划")
        if self.use_rag:
            print(f"📡 RAG 时空感知: 启用")
        if self.hotel_poi_id is not None:
            print(f"🏨 住宿点: {self.hotel_info['name']}")
        print(f"{'='*30}\n")

        # Step 1: 检索候选 POI
        req_topk_pois, pseudo_must_see_pois = self._get_reqs_topk()

        # Step 2: 多日 POI 分配（含住宿点周边增强）
        allpoi_idlist = (
            req_topk_pois[:, 0].astype(int).tolist()
            + self.must_see_pois
        )
        # 住宿点周边 POI 加入候选池
        if self.hotel_poi_id is not None:
            hotel_nearby = self._get_hotel_nearby()
            for poi_id, name, cat, dist in hotel_nearby:
                if poi_id not in allpoi_idlist:
                    allpoi_idlist.append(poi_id)
            # 酒店自身也加入（作为参考点）
            if self.hotel_poi_id not in allpoi_idlist:
                allpoi_idlist.append(self.hotel_poi_id)
        allpoi_idlist = list(set(allpoi_idlist))

        day_poi_candidates, day_poi_scores, day_clusters, _ = \
            self.spatial_handler.get_multiday_poi_candidates(
                allpoi_idlist=allpoi_idlist,
                must_see_poi_idlist=self.must_see_pois,
                req_topk_pois=req_topk_pois,
                num_days=self.num_days,
                min_pois_per_day=self.daily_params[0][1],
                thresh=self.thresh
            )

        # Step 3: 每天 TSP 排序
        day_orders = self.spatial_handler.get_multiday_full_order(
            day_poi_candidates, day_clusters
        )
        # 保存供可视化和调试使用
        self._day_orders = day_orders
        self._day_clusters = day_clusters

        # Step 4: RAG 上下文生成
        if self.use_rag:
            rag_overview = self.rag.generate_multi_day_overview()
        else:
            rag_overview = ""

        # Step 5: 每天生成行程
        day_results = []
        day_themes = []
        global_poi_counter = 1  # 跨天序列号

        for day_idx in range(self.num_days):
            day_order = day_orders[day_idx] if day_idx < len(day_orders) else []

            if len(day_order) == 0:
                day_results.append({
                    "day": day_idx + 1,
                    "day_theme": "自由探索日",
                    "itinerary": "自由安排",
                    "day_reason": "当天无特定行程，建议自由探索城市",
                    "weather_note": "无",
                    "pois": {}
                })
                day_themes.append("自由探索")
                continue

            # 构建当天的 context_string
            context_string, must_see_string = self._build_day_context(
                day_order, day_idx
            )

            # 检查是否需要反转顺序
            day_order = self._check_reverse(day_order)

            # RAG 上下文
            spatiotemporal_ctx = ""
            if self.use_rag:
                ordered_names = self.user_favorites.loc[day_order]["name"].tolist()
                ordered_cats = self.user_favorites.loc[day_order]["category"].tolist()
                # 计算距离
                ordered_with_loop = day_order + [day_order[0]]
                locs = self.user_favorites.loc[ordered_with_loop][["x", "y"]].astype(float).to_numpy()
                dists = np.sqrt(np.sum(np.diff(locs, axis=0) ** 2, axis=1))
                spatiotemporal_ctx = self.rag.generate_full_context(
                    day_index=day_idx,
                    ordered_poi_names=ordered_names,
                    ordered_poi_categories=ordered_cats,
                    distances=dists[:-1]
                )

            # 天气主题建议
            day_theme = ""
            if self.use_rag and hasattr(self.rag, 'daily_weather') and day_idx < len(self.rag.daily_weather):
                w = self.rag.daily_weather[day_idx]
                if "雨" in w["weather"]:
                    day_theme = "室内文化探索日"
                elif "酷热" in w["weather"]:
                    day_theme = "休闲避暑日"
                elif w["day_of_week"] in ["周六", "周日"]:
                    day_theme = "热闹周末游"
                else:
                    day_theme = "深度城市探索日"

            # 住宿点上下文（注入到时空上下文之后）
            hotel_ctx = self._build_hotel_context()
            if hotel_ctx:
                spatiotemporal_ctx = spatiotemporal_ctx + "\n\n" + hotel_ctx if spatiotemporal_ctx else hotel_ctx

            # 调用 LLM 生成当天行程
            hours = (
                self.hours_per_day[day_idx]
                if day_idx < len(self.hours_per_day) else 5
            )
            max_pois = min(
                self.daily_params[day_idx][1] if day_idx < len(self.daily_params) else 7,
                len(day_order)
            )

            day_response = self._generate_day_plan(
                day_index=day_idx,
                context_string=context_string,
                must_see_string=must_see_string,
                spatiotemporal_context=spatiotemporal_ctx,
                hours=hours,
                max_pois=max_pois,
                day_theme=day_theme
            )

            try:
                day_data = json.loads(day_response)
            except Exception:
                try:
                    day_data = json.loads(day_response[8:-4])
                except Exception:
                    day_data = {
                        "day_theme": day_theme or f"第{day_idx+1}天",
                        "itinerary": " -> ".join(
                            self.user_favorites.loc[day_order[:max_pois]]["name"].tolist()
                        ),
                        "day_reason": "自动生成的行程",
                        "weather_note": "无",
                        "pois": {}
                    }

            day_data["day"] = day_idx + 1

            # 交通/天气：只对 LLM 实际选中的 POI 计算
            transit_summary = ""
            weather_ctx = ""
            if self.use_rag:
                itinerary = day_data.get("itinerary", "")
                if itinerary and "->" in itinerary:
                    raw_names = [s.strip() for s in itinerary.split("->")]
                    # 模糊匹配 POI 名到数据库（LLM 可能输出略有不同的名称）
                    from thefuzz import process as fz
                    all_db_names = self.user_favorites["name"].tolist()
                    matched = []
                    for raw in raw_names:
                        if not raw:
                            continue
                        # 先用精确匹配
                        exact = self.user_favorites[self.user_favorites["name"] == raw]
                        if len(exact) > 0:
                            matched.append(exact.index[0])
                        else:
                            # 模糊匹配备选
                            best, score = fz.extract(raw, all_db_names, limit=1)[0]
                            if score > 80:
                                idx = all_db_names.index(best)
                                matched.append(idx)
                    if len(matched) >= 2:
                        selected_names = [self.user_favorites.loc[i, "name"] for i in matched]
                        locs = self.user_favorites.loc[matched][["x", "y"]].astype(float).to_numpy()
                        dists = np.sqrt(np.sum(np.diff(locs, axis=0) ** 2, axis=1))
                        transit_summary = self.rag.get_daily_transit_summary(selected_names, dists)
                weather_ctx = self.rag.get_weather_context(day_idx)

            day_data["transit_summary"] = transit_summary
            day_data["weather_context"] = weather_ctx
            day_results.append(day_data)
            day_themes.append(day_data.get("day_theme", f"Day {day_idx+1}"))

        # Step 6: 生成总览
        overview = self._generate_overview(day_themes)

        # Step 7: 构建结果、保存 JSON + 生成地图
        result = {
            "trip_title": overview.get("trip_title", f"{self.city_zh}{self.num_days}日游"),
            "trip_overview": overview.get("trip_overview", ""),
            "num_days": self.num_days,
            "city": self.city_zh,
            "hotel": self.hotel_info,
            "hotel_nearby": [
                {"name": n, "category": c, "distance_m": d}
                for _, n, c, d in self._get_hotel_nearby()
            ] if self.hotel_poi_id else [],
            "rag_context": rag_overview,
            "daily_summary": overview.get("daily_summary", []),
            "days": day_results
        }

        self._save_result(result)
        self._save_qualitative(result, day_orders, day_clusters)
        return result

    # ========== 辅助方法 ==========

    def _build_day_context(self, day_order, day_idx):
        """为某天构建候选景点上下文"""
        context_parts = []
        must_see_set = set(self.must_see_pois)
        must_see_names = []

        for i, poi_id in enumerate(day_order):
            name = self.user_favorites.loc[poi_id, "name"]
            ctx = self.user_favorites.loc[poi_id, "context"]
            seq = i + 1
            context_parts.append(
                f'序号 {seq}: "{ctx.replace(chr(10), "")[:120]}"'
            )
            if poi_id in must_see_set:
                must_see_names.append(name)

        context_string = "\n".join(context_parts)
        must_see_string = (
            str(must_see_names) if must_see_names
            else "暂无必须选择的 POI"
        )
        return context_string, must_see_string

    def _check_reverse(self, day_order):
        """检查当天的 POI 顺序是否需要反转"""
        if len(day_order) < 2:
            return day_order

        context_parts = []
        for i, poi_id in enumerate(day_order):
            ctx = self.user_favorites.loc[poi_id, "context"]
            context_parts.append(
                f'序号 {i+1}: "{ctx.replace(chr(10), "")[:100]}"'
            )

        msg = [{"role": "user", "content": check_final_reverse_prompt(
            context="\n".join(context_parts),
            user_reqs=self.user_reqs
        )}]
        try:
            response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
            response = json.loads(response)
            if int(response[0]) == 0:
                day_order = list(reversed(day_order))
        except Exception:
            pass

        return day_order

    def _generate_day_plan(self, day_index, context_string, must_see_string,
                           spatiotemporal_context, hours, max_pois, day_theme):
        """调用 LLM 生成某天行程"""
        messages = [
            {"role": "system", "content": get_multiday_system_prompt(
                num_days=self.num_days, max_pois_per_day=max_pois
            )},
            {"role": "user", "content": get_multiday_dayplan_prompt(
                day_index=day_index,
                num_days=self.num_days,
                context_string=context_string,
                must_see_string=must_see_string,
                keyword_reqs=self.user_pos_reqs,
                userReqList=self.user_reqs,
                maxPoiNum=max_pois,
                numCandidates=len(context_string.split("\n")),
                spatiotemporal_context=spatiotemporal_context,
                hours=hours,
                day_theme=day_theme
            )}
        ]
        return self.proxy.chat(messages=messages, model=self.MODEL, temperature=0)

    def _generate_overview(self, day_themes):
        """生成多日行程总览"""
        try:
            msg = [{"role": "user", "content": get_multiday_overview_prompt(
                num_days=self.num_days,
                day_themes=day_themes,
                user_reqs=self.user_reqs
            )}]
            response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
            return json.loads(response)
        except Exception:
            return {
                "trip_title": f"{self.city_zh}{self.num_days}日深度游",
                "trip_overview": f"为期{self.num_days}天的{self.city_zh}行程",
                "daily_summary": [
                    {"day": i+1, "title": t, "one_liner": ""}
                    for i, t in enumerate(day_themes)
                ]
            }

    def _save_result(self, result):
        """保存结果到 JSON 文件"""
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        filename = f"./model/output/{current_time}_{self.type}_multiday.json"
        os.makedirs("./model/output", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 多日行程已保存至: {filename}")

    # ========== 地图可视化 ==========

    def _save_qualitative(self, result, day_orders, day_clusters):
        """
        生成 Folium 交互式地图。

        - 单日模式：生成 fulltsp / response / response_clusters 三张图
        - 多日模式：生成每日路线图 + 一张总览对比图
        """
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        def _create_map(df, polyline_color="green", map_filename="", extra_circles=None):
            """创建单张 Folium 地图"""
            gcj_coords = np.array(df[['lon', 'lat']])
            wgs_coords = gcj2wgs(gcj_coords)
            df = df.copy()
            df['lon'], df['lat'] = wgs_coords[:, 0], wgs_coords[:, 1]

            m = folium.Map(
                location=[df['lat'].mean(), df['lon'].mean()],
                zoom_start=13
            )
            for i, (_, row) in enumerate(df.iterrows()):
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"{i+1} - {row['name']}",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)

            points = list(zip(df['lat'], df['lon']))
            folium.PolyLine(points, color=polyline_color, weight=2.5, opacity=1).add_to(m)

            # 叠加聚类圆圈（可选）
            if extra_circles:
                for circle_info in extra_circles:
                    folium.Circle(
                        location=circle_info["location"],
                        radius=circle_info["radius"],
                        color=circle_info.get("color", "blue"),
                        fill=True,
                        opacity=0.3
                    ).add_to(m)

            m.save(f'./model/output/{map_filename}')
            return m

        # ---- 单日模式 ----
        if self.num_days == 1:
            day_data = result["days"][0] if result.get("days") else {}
            day_order = day_orders[0] if day_orders else []

            if len(day_order) == 0:
                return

            # 全量 TSP 路径图
            df_full = self.user_favorites.loc[day_order]
            _create_map(df_full, map_filename=f"{current_time}_{self.type}_fulltsp.html")

            # 从 LLM 响应中提取实际选中的 POI
            pois_dict = day_data.get("pois", {})
            if pois_dict:
                try:
                    selected_seq = [int(k) for k in pois_dict.keys()]
                    selected_indices = np.array(day_order)[np.array(selected_seq) - 1]
                    df_selected = self.user_favorites.loc[selected_indices]
                except Exception:
                    df_selected = df_full.head(min(10, len(df_full)))
            else:
                df_selected = df_full.head(min(10, len(df_full)))

            _create_map(df_selected, map_filename=f"{current_time}_{self.type}.html")

            # 聚类圆图
            clusters = day_clusters[0] if day_clusters else [list(day_order)]
            centroids = self.spatial_handler.get_cluster_centroids(clusters, lonlat=True)
            radius = self.spatial_handler.citywalk_thresh

            circles = []
            for ci, cluster in enumerate(clusters):
                if ci < len(centroids):
                    clon, clat = centroids[ci]
                    circles.append({
                        "location": [clat, clon],
                        "radius": radius / 2,
                        "color": "blue"
                    })

            _create_map(
                df_selected,
                map_filename=f"{current_time}_{self.type}_response_clusters.html",
                extra_circles=circles
            )
            print(f"🗺️  地图已保存: {current_time}_{self.type}_*.html (3 张)")

        # ---- 多日模式：每日一张路线图 + 总览图 ----
        else:
            day_colors = ["green", "blue", "purple", "orange", "red", "teal", "darkblue"]

            for day_idx in range(min(self.num_days, len(day_orders))):
                day_order = day_orders[day_idx]
                if len(day_order) == 0:
                    continue

                # 获取当天 LLM 选中的 POI
                day_data = result["days"][day_idx] if day_idx < len(result["days"]) else {}
                pois_dict = day_data.get("pois", {})

                if pois_dict:
                    try:
                        selected_seq = [int(k) for k in pois_dict.keys()]
                        selected_indices = np.array(day_order)[np.array(selected_seq) - 1]
                        df_day = self.user_favorites.loc[selected_indices]
                    except Exception:
                        df_day = self.user_favorites.loc[day_order]
                else:
                    df_day = self.user_favorites.loc[day_order]

                color = day_colors[day_idx % len(day_colors)]
                _create_map(
                    df_day,
                    polyline_color=color,
                    map_filename=f"{current_time}_{self.type}_day{day_idx+1}.html"
                )

            # 总览图：所有天的路线叠加，不同颜色
            all_dfs = []
            m = folium.Map(location=[31.23, 121.47], zoom_start=12)
            all_lats, all_lons = [], []

            for day_idx in range(min(self.num_days, len(day_orders))):
                day_order = day_orders[day_idx]
                if len(day_order) == 0:
                    continue
                color = day_colors[day_idx % len(day_colors)]
                df_day = self.user_favorites.loc[day_order].copy()
                gcj = np.array(df_day[['lon', 'lat']])
                wgs = gcj2wgs(gcj)
                df_day['lon'], df_day['lat'] = wgs[:, 0], wgs[:, 1]

                for _, row in df_day.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=5,
                        color=color,
                        fill=True,
                        popup=f"Day{day_idx+1}: {row['name']}"
                    ).add_to(m)
                    all_lats.append(row['lat'])
                    all_lons.append(row['lon'])

                pts = list(zip(df_day['lat'], df_day['lon']))
                folium.PolyLine(pts, color=color, weight=2.5, opacity=1,
                                popup=f"Day {day_idx+1}").add_to(m)

            if all_lats:
                m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
            m.save(f'./model/output/{current_time}_{self.type}_overview.html')

            print(f"🗺️  地图已保存: {current_time}_{self.type}_day*.html + _overview.html ({self.num_days+1} 张)")

    # ========== 兼容接口 ==========

    def solve(self):
        """solve() 接口：多日模式"""
        return self.solve_multiday()
