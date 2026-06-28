"""
Travel Agent: 多日长行程 + RAG 时空感知旅游规划 Agent
=====================================================
功能：
  - 多日行程支持（1-7天）
  - RAG 时空感知上下文注入（天气、交通、人流）
  - 跨日约束优化（POI 不重复、每日负载均衡）
  - 住宿点周边推荐
  - Folium 交互式地图可视化

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
from model.config import CITY_ZH_MAP, MULTIDAY_TIME2NUM, LLM_MODEL, GaodeMap_apikey
from thefuzz import process as fuzz_process  # 酒店名称模糊匹配


class TravelAgent:
    """
    多日旅游规划 Agent

    skills：
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
                 hotel: Optional[str] = None,
                 exclude_pois: Optional[list] = None,
                 start_time: str = "08:00",
                 end_time: str = "20:00",
                 verbose: bool = False,
                 model_override: Optional[str] = None):
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
            exclude_pois: 要排除的景点名称列表（模糊匹配 POI 库）
            start_time: 每天出发时间 "HH:MM"（默认 08:00）
            end_time: 每天结束时间 "HH:MM"（默认 20:00）
            verbose: 是否显示 LLM 思考过程（调试用，配合 --verbose）
            model_override: 覆盖默认 LLM 模型（None 则用 config.LLM_MODEL）
        """
        self.MODEL = model_override or LLM_MODEL
        self.type = type
        self.proxy = proxy_call
        self.city = city
        self.city_zh = CITY_ZH_MAP.get(city, city)
        self.num_days = num_days
        self.verbose = verbose
        self.citywalk = citywalk
        self.user_reqs = user_reqs
        self.start_time = start_time
        self.end_time = end_time
        self.citywalk = citywalk
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh
        self.min_poi_candidate_num = min_poi_candidate_num
        self.use_rag = use_rag
        self.hotel_name = hotel
        self.exclude_poi_names = exclude_pois or []

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
            citywalk_thresh=self.daily_params[0][2],
            start_time=self.start_time
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

        # ---- 排除景点匹配 ----
        self.exclude_poi_ids = set()
        self.exclude_poi_matched = []  # [(输入名, 匹配名, 得分), ...]
        if self.exclude_poi_names:
            self._match_exclude_pois()

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

    # ========== 排除景点 ==========

    def _match_exclude_pois(self):
        """模糊匹配要排除的景点名到 POI 库"""
        all_names = self.user_favorites["name"].tolist()
        for exclude_name in self.exclude_poi_names:
            match, score = fuzz_process.extract(exclude_name, all_names, limit=1)[0]
            if score >= 60:
                idx = all_names.index(match)
                self.exclude_poi_ids.add(self.user_favorites.index[idx])
                self.exclude_poi_matched.append((exclude_name, match, score))
                print(f"🚫 排除景点: {match} (输入: '{exclude_name}', 得分: {score})")
            else:
                print(f"⚠️ 未匹配到排除景点: '{exclude_name}' (最佳: '{match}', 得分: {score})")

    def _build_exclude_context(self) -> str:
        """构建排除景点上下文，注入 LLM prompt"""
        if not self.exclude_poi_matched:
            return ""

        lines = ["### 🚫 用户明确排除的景点（绝对不要加入行程）"]
        for orig, matched, score in self.exclude_poi_matched:
            lines.append(f"- **{matched}**（用户输入: '{orig}'）")
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
            while len(hours) < self.num_days:
                hours.append(5)
            return hours[:self.num_days]
        except Exception:
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
        # print(f"\n{'='*30}")
        print(f"🌍 Travel Agent: {self.num_days}天 {self.city_zh} 行程规划")
        if self.use_rag:
            print(f"📡 RAG 时空感知: 启用")
        if self.hotel_poi_id is not None:
            print(f"🏨 住宿点: {self.hotel_info['name']}")
        print(f"{'='*30}\n")

        # Step 1: 检索候选 POI
        req_topk_pois, pseudo_must_see_pois = self._get_reqs_topk()

        # Step 2: 多日 POI 分配（含住宿点周边增强 + 排除景点过滤）
        allpoi_idlist = (
            req_topk_pois[:, 0].astype(int).tolist()
            + self.must_see_pois
        )
        # 排除用户指定的景点
        if self.exclude_poi_ids:
            before = len(allpoi_idlist)
            allpoi_idlist = [pid for pid in allpoi_idlist if pid not in self.exclude_poi_ids]
            print(f"🚫 已从候选池排除 {before - len(allpoi_idlist)} 个景点")
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

        # Step 3: 每天 TSP 排序（注入品质分数做多目标优化）
        # 构建 POI 品质分数映射（来自搜索相似度，越高越好）
        poi_score_map = {}
        for row in req_topk_pois:
            poi_score_map[int(row[0])] = float(row[1])
        self.spatial_handler.poi_scores = poi_score_map

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

        # Step 5: 每天生成行程（并行化——各天之间无依赖）
        day_jobs = []  # [(day_idx, day_order, context_string, must_see_string, spatiotemporal_ctx, hours, max_pois, day_theme), ...]

        for day_idx in range(self.num_days):
            day_order = day_orders[day_idx] if day_idx < len(day_orders) else []

            if len(day_order) == 0:
                day_jobs.append(None)  # 空占位
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

            # 住宿点上下文
            hotel_ctx = self._build_hotel_context()
            if hotel_ctx:
                spatiotemporal_ctx = spatiotemporal_ctx + "\n\n" + hotel_ctx if spatiotemporal_ctx else hotel_ctx

            # 排除景点上下文
            exclude_ctx = self._build_exclude_context()
            if exclude_ctx:
                spatiotemporal_ctx = spatiotemporal_ctx + "\n\n" + exclude_ctx if spatiotemporal_ctx else exclude_ctx

            hours = (
                self.hours_per_day[day_idx]
                if day_idx < len(self.hours_per_day) else 5
            )
            max_pois = min(
                self.daily_params[day_idx][1] if day_idx < len(self.daily_params) else 7,
                len(day_order)
            )

            day_jobs.append((day_idx, day_order, context_string, must_see_string,
                           spatiotemporal_ctx, hours, max_pois, day_theme))

        # ---- 并行 LLM 调用：所有天同时请求 ----
        # 流式输出加锁，防止多线程 stdout 交错
        import sys
        import threading
        _print_lock = threading.Lock()

        def _gen_one_day(args):
            if args is None:
                return None
            day_idx, day_order, context_string, must_see_string, \
                spatiotemporal_ctx, hours, max_pois, day_theme = args

            # ---- 流式输出思考（仅 verbose 模式） ----
            if self.verbose:
                with _print_lock:
                    print(f"\n{'─'*50}")
                    print(f"💭 Day {day_idx+1} LLM 思考过程（流式）:")
                    print(f"{'─'*50}")

                def on_chunk(chunk_text):
                    with _print_lock:
                        sys.stdout.write(chunk_text)
                        sys.stdout.flush()

                day_response = self._generate_day_plan(
                    day_index=day_idx,
                    context_string=context_string,
                    must_see_string=must_see_string,
                    spatiotemporal_context=spatiotemporal_ctx,
                    hours=hours,
                    max_pois=max_pois,
                    day_theme=day_theme,
                    stream_callback=on_chunk
                )
                with _print_lock:
                    print(f"\n{'─'*50}\n")

                # 提取 JSON（流式输出的内容已实时打印）
                reasoning_text, day_data = self._extract_json_and_reasoning(day_response)
                if reasoning_text and "reasoning" not in (day_data or {}):
                    if day_data:
                        day_data["reasoning"] = reasoning_text
                if day_data is None:
                    try:
                        day_data = json.loads(day_response)
                    except Exception:
                        try:
                            day_data = json.loads(day_response[8:-4])
                        except Exception:
                            day_data = None
            else:
                reasoning_text = ""
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
                        day_data = None

            if day_data is None:
                day_data = {
                    "day_theme": day_theme or f"第{day_idx+1}天",
                    "itinerary": " -> ".join(
                        self.user_favorites.loc[day_order[:max_pois]]["name"].tolist()
                    ),
                    "day_reason": "自动生成的行程",
                    "weather_note": "无",
                    "pois": {}
                }

            # 保留 LLM 返回的 reasoning 字段（如果有），否则用提取的文本
            if reasoning_text and "reasoning" not in day_data:
                day_data["reasoning"] = reasoning_text

            day_data["day"] = day_idx + 1

            # ---- 安全网：检测并修复 LLM 用名字做 pois key 的违规 ----
            pois_dict = day_data.get("pois", {})
            if pois_dict:
                keys_are_names = not all(str(k).isdigit() for k in pois_dict.keys())
                if keys_are_names and day_order:
                    # 构建当天候选的名称→序号 映射
                    from thefuzz import process as fz
                    candidate_names = [self.user_favorites.loc[pid, "name"] for pid in day_order]
                    fixed_pois = {}
                    for name_key, desc in pois_dict.items():
                        best, score = fz.extract(str(name_key), candidate_names, limit=1)[0]
                        if score > 80:
                            seq = candidate_names.index(best) + 1  # 1-based
                            fixed_pois[str(seq)] = desc
                    if fixed_pois:
                        day_data["pois"] = fixed_pois
                        # 同时修复 itinerary：用候选中匹配到的正确名称重建
                        fixed_names = []
                        for seq_str in sorted(fixed_pois.keys(), key=int):
                            pid = day_order[int(seq_str) - 1]
                            fixed_names.append(self.user_favorites.loc[pid, "name"])
                        if fixed_names:
                            day_data["itinerary"] = " -> ".join(fixed_names)

            # 规范化 itinerary：LLM 可能输出序号 → 转为景点名称
            raw_itinerary = day_data.get("itinerary", "")
            if raw_itinerary and day_order:
                tokens = [t.strip() for t in raw_itinerary.split("->")]
                # 检测是否为纯序号（如 "1 -> 7 -> 6"）
                if all(t.isdigit() for t in tokens if t):
                    seqs = [int(t) - 1 for t in tokens if t.isdigit()]
                    names = []
                    for s in seqs:
                        if 0 <= s < len(day_order):
                            names.append(
                                self.user_favorites.loc[day_order[s], "name"]
                            )
                    if names:
                        day_data["itinerary"] = " -> ".join(names)
                        # 同时更新 pois 的 key 为名称（便于后续匹配）
                        if day_data.get("pois"):
                            old_pois = day_data["pois"]
                            new_pois = {}
                            for k, v in old_pois.items():
                                try:
                                    seq = int(k) - 1
                                    if 0 <= seq < len(day_order):
                                        new_key = self.user_favorites.loc[day_order[seq], "name"]
                                        new_pois[new_key] = v
                                    else:
                                        new_pois[k] = v
                                except (ValueError, KeyError):
                                    new_pois[k] = v
                            day_data["pois"] = new_pois

            # LLM 精选后重优化：对选中的 POI 重新 TSP（可能更短）
            itinerary_names = [s.strip() for s in day_data.get("itinerary", "").split("->") if s.strip()]
            if len(itinerary_names) >= 3:
                # 名称→ID 匹配，但限制在当天候选集内（防止匹配到远处同名POI）
                candidate_name_to_id = {
                    self.user_favorites.loc[pid, "name"]: pid
                    for pid in day_order
                }
                selected_ids = []
                for n in itinerary_names:
                    pid = candidate_name_to_id.get(n)
                    if pid is not None:
                        selected_ids.append(pid)
                if len(selected_ids) >= 3:
                    reorder, _, _ = self.spatial_handler.get_tsp_order(selected_ids)
                    new_names = [self.user_favorites.loc[selected_ids[i], "name"] for i in reorder]
                    if new_names != itinerary_names:
                        day_data["itinerary"] = " -> ".join(new_names)

            # 交通/天气：只对 LLM 实际选中的 POI 计算
            transit_summary = ""
            weather_ctx = ""
            if self.use_rag:
                itinerary = day_data.get("itinerary", "")
                if itinerary and "->" in itinerary:
                    raw_names = [s.strip() for s in itinerary.split("->")]
                    # 名称→ID 匹配，限制在当天候选集内
                    candidate_name_to_id = {
                        self.user_favorites.loc[pid, "name"]: pid
                        for pid in day_order
                    }
                    from thefuzz import process as fz
                    candidate_names = list(candidate_name_to_id.keys())
                    matched = []
                    for raw in raw_names:
                        if not raw:
                            continue
                        pid = candidate_name_to_id.get(raw)
                        if pid is not None:
                            matched.append(pid)
                        else:
                            best, score = fz.extract(raw, candidate_names, limit=1)[0]
                            if score > 80:
                                matched.append(candidate_name_to_id[best])
                    if len(matched) >= 2:
                        selected_names = [self.user_favorites.loc[i, "name"] for i in matched]
                        locs = self.user_favorites.loc[matched][["x", "y"]].astype(float).to_numpy()
                        dists = np.sqrt(np.sum(np.diff(locs, axis=0) ** 2, axis=1))
                        transit_summary = self.rag.get_daily_transit_summary(selected_names, dists)
                weather_ctx = self.rag.get_weather_context(day_idx)

            day_data["transit_summary"] = transit_summary
            day_data["weather_context"] = weather_ctx

            # 生成时间线："几点到哪"
            timeline = self._build_timeline(day_data, day_order, day_idx)
            day_data["timeline"] = timeline

            # 最终输出：将 pois 的数字 key 转为景点名称，并确保与 itinerary 一致
            pois_dict = day_data.get("pois", {})
            itinerary_names = [s.strip() for s in day_data.get("itinerary", "").split("->") if s.strip()]

            if pois_dict and day_order:
                # 先尝试数字 key → 名称
                try:
                    new_pois = {}
                    for k, v in pois_dict.items():
                        seq = int(k) - 1
                        if 0 <= seq < len(day_order):
                            name = self.user_favorites.loc[day_order[seq], "name"]
                            new_pois[name] = v
                        else:
                            new_pois[k] = v
                    pois_dict = new_pois
                except (ValueError, KeyError):
                    pass  # 已经是名称 key

                # 校验：pois 的 key 必须和 itinerary 一致
                pois_keys = set(pois_dict.keys())
                it_set = set(itinerary_names)
                if pois_keys != it_set and itinerary_names:
                    # 不一致 → 用 itinerary 重建 pois
                    rebuilt = {}
                    old_descs = list(pois_dict.values())  # 保留 LLM 的描述（顺序可能不同）
                    for i, name in enumerate(itinerary_names):
                        # 尝试从旧 pois 中找匹配的描述（模糊匹配名称）
                        desc = None
                        for old_name, old_desc in pois_dict.items():
                            if name in old_name or old_name in name:
                                desc = old_desc
                                break
                        if desc is None and i < len(old_descs):
                            desc = old_descs[i]  # 按位置 fallback
                        if desc is None:
                            # 最终 fallback：用数据中的 context
                            match = self.user_favorites[self.user_favorites["name"] == name]
                            if len(match) > 0:
                                desc = str(match.iloc[0].get("context", ""))
                        if desc:
                            rebuilt[name] = str(desc)
                    if rebuilt:
                        pois_dict = rebuilt

            day_data["pois"] = pois_dict

            return day_data

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_days) as executor:
            futures = [executor.submit(_gen_one_day, job) for job in day_jobs]
            day_results_raw = [f.result() for f in futures]

        # 组装结果（保持 day 顺序）
        day_results = []
        day_themes = []
        for day_idx in range(self.num_days):
            day_data = day_results_raw[day_idx]
            if day_data is None:
                day_data = {
                    "day": day_idx + 1,
                    "day_theme": "自由探索日",
                    "itinerary": "自由安排",
                    "day_reason": "当天无特定行程，建议自由探索城市",
                    "weather_note": "无",
                    "pois": {}
                }
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
            "exclude": [
                {"input": orig, "matched": matched, "score": score}
                for orig, matched, score in self.exclude_poi_matched
            ] if self.exclude_poi_matched else None,
            "rag_context": rag_overview,
            "daily_summary": overview.get("daily_summary", []),
            "days": day_results
        }

        self._save_result(result)
        self._save_qualitative(result, day_orders, day_clusters)
        return result

    # ========== 辅助方法 ==========

    @staticmethod
    def _extract_json_and_reasoning(response: str):
        """
        从 LLM 响应中分离推理文本和 JSON。

        策略：
        1. 先尝试直接解析整个响应为 JSON
        2. 否则找第一个 { 到最后一个 } 之间的内容作为 JSON
        3. JSON 之前的内容作为推理文本

        Returns:
            tuple[str, dict | None]: (reasoning_text, parsed_json_dict)
        """
        if not response:
            return "", None

        # 方案 1：直接解析
        try:
            return "", json.loads(response)
        except Exception:
            pass

        # 方案 2：括号平衡——逐字符计数 {} 找到真正的 JSON 闭合位置
        first_brace = response.find('{')
        if first_brace != -1:
            reasoning = response[:first_brace].strip()
            depth = 0
            in_string = False
            escape_next = False
            for i in range(first_brace, len(response)):
                ch = response[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = response[first_brace:i + 1]
                        try:
                            return reasoning, json.loads(json_str)
                        except Exception:
                            pass
                        break

        # 方案 3：fallback——全部视为推理
        return response, None

    def _build_timeline(self, day_data: dict, day_order: list, day_idx: int) -> str:
        """
        根据 LLM 精选路线、道路距离和时间窗，生成"几点到哪"时间线。

        Returns:
            str: 格式化的时间线文本
        """
        itinerary = day_data.get("itinerary", "")
        if not itinerary or "->" not in itinerary:
            return ""

        raw_names = [s.strip() for s in itinerary.split("->") if s.strip()]
        if len(raw_names) < 2:
            return ""

        # 模糊匹配 POI 名 → 获取坐标和时间窗
        from thefuzz import process as fz
        all_db_names = self.user_favorites["name"].tolist()
        matched_ids = []
        for raw in raw_names:
            exact = self.user_favorites[self.user_favorites["name"] == raw]
            if len(exact) > 0:
                matched_ids.append(exact.index[0])
            else:
                best, score = fz.extract(raw, all_db_names, limit=1)[0]
                if score > 80:
                    matched_ids.append(all_db_names.index(best))

        if len(matched_ids) < 2:
            return ""

        # 获取坐标
        locs = self.user_favorites.loc[matched_ids][["lon", "lat"]].astype(float).to_numpy()
        # 获取道路距离（用缓存或 OSRM）
        dists_m = []
        for i in range(len(matched_ids) - 1):
            # 尝试从缓存获取
            import numpy as np
            key = (min(matched_ids[i], matched_ids[i+1]),
                   max(matched_ids[i], matched_ids[i+1]))
            if hasattr(self.spatial_handler, '_road_dist_cache') and \
               key in self.spatial_handler._road_dist_cache:
                d = self.spatial_handler._road_dist_cache[key]
            else:
                d = np.sqrt((locs[i][0]-locs[i+1][0])**2 + (locs[i][1]-locs[i+1][1])**2) * 111000
            dists_m.append(d)

        # 获取时间窗
        opens_str = []
        closes_str = []
        for mid in matched_ids:
            try:
                opens_str.append(str(self.user_favorites.loc[mid, "open_time"]))
                closes_str.append(str(self.user_favorites.loc[mid, "close_time"]))
            except KeyError:
                opens_str.append("09:00")
                closes_str.append("18:00")

        def to_min(s):
            parts = s.split(':')
            return int(parts[0]) * 60 + int(parts[1])

        def to_str(m):
            h, mi = divmod(m, 60)
            return f"{h:02d}:{mi:02d}"

        START = int(self.start_time.split(":")[0]) * 60 + int(self.start_time.split(":")[1])
        SPEED_CAR = 500      # m/min 城市车速（30km/h）
        SPEED_WALK = 80      # m/min 步行速度（5km/h）
        SPEED_BIKE = 250     # m/min 骑行速度（15km/h）

        lines = ["### 🕐 预计时间线"]
        prev_depart = START  # 上一站的实际离开时间，用实际停留（可能被关门截断）

        for i, mid in enumerate(matched_ids):
            name = self.user_favorites.loc[mid, "name"]
            open_t = to_min(opens_str[i])
            close_t = to_min(closes_str[i])
            # 从数据读取停留时长，默认 60min
            try:
                visit = int(self.user_favorites.loc[mid, "visittime"])
            except (KeyError, ValueError):
                visit = 60

            if i == 0:
                arrival = max(prev_depart, open_t)
                travel_str = "起点"
            else:
                d = dists_m[i-1]
                # 根据距离自动选择交通方式：<500m步行，500-2000m骑行，>2000m驾车
                if d < 500:
                    travel = max(1, int(d / SPEED_WALK))
                    travel_str = f"🚶 {travel}min"
                elif d < 2000:
                    travel = max(1, int(d / SPEED_BIKE))
                    travel_str = f"🚲 {travel}min"
                else:
                    travel = max(1, int(d / SPEED_CAR))
                    travel_str = f"🚗 {travel}min"
                arrival = max(prev_depart + travel, open_t)

            depart = min(arrival + visit, close_t)
            actual_stay = max(depart - arrival, 0)  # 实际停留（可能被关门时间截断）

            # 到达时已关门则跳过该 POI，但不影响后续时间计算
            if actual_stay <= 0:
                prev_depart = arrival  # 到达即离开，不消耗时间
                continue

            lines.append(
                f"  {to_str(arrival)}-{to_str(depart)}  {name}  "
                f"({travel_str} | 停留{actual_stay}min | 营业 {opens_str[i]}-{closes_str[i]})"
            )
            prev_depart = depart

        lines.append(f"  🏁 {to_str(prev_depart)} 行程结束")
        return "\n".join(lines)

    def _build_day_context(self, day_order, day_idx):
        """为某天构建候选景点上下文"""
        context_parts = []
        must_see_set = set(self.must_see_pois)
        must_see_names = []

        for i, poi_id in enumerate(day_order):
            name = self.user_favorites.loc[poi_id, "name"]
            ctx = self.user_favorites.loc[poi_id, "context"]
            seq = i + 1
            # 同时传递名称和描述，避免 LLM 猜错名字
            context_parts.append(
                f'序号 {seq}: {name} —— "{ctx.replace(chr(10), "")[:120]}"'
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
                           spatiotemporal_context, hours, max_pois, day_theme,
                           stream_callback=None):
        """
        调用 LLM 生成某天行程。

        Args:
            stream_callback: 可选回调函数 fn(chunk_text)，每收到一个 token 调用一次。
                             用于流式输出思考过程到终端/前端。
        """
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
                day_theme=day_theme,
                show_reasoning=self.verbose
            )}
        ]

        if stream_callback is not None:
            # 流式模式：逐 token 回调 + 收集完整响应
            full_response = []
            for chunk in self.proxy.stream_chat(messages, model=self.MODEL, temperature=0):
                if chunk:
                    stream_callback(chunk)
                    full_response.append(chunk)
            return "".join(full_response)
        else:
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
            """创建单张 Folium 地图（高德瓦片 GCJ-02，数据已是 GCJ-02）"""
            df = df.copy()

            m = folium.Map(
                location=[df['lat'].mean(), df['lon'].mean()],
                zoom_start=13,
                tiles=None
            )
            folium.TileLayer(
                tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                attr='高德地图',
                subdomains='1234',
                max_zoom=18
            ).add_to(m)
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

        # ---- 辅助：从 LLM 响应中提取实际选中的 POI ----
        def _get_selected_pois(day_order, day_data):
            """从 LLM 返回的 pois dict 中提取实际选中的 POI 子集（兼容数字key和名称key）"""
            pois_dict = day_data.get("pois", {}) if day_data else {}
            if not pois_dict:
                return self.user_favorites.loc[day_order]

            # 方案1：数字 key → 按序号从 day_order 取
            try:
                selected_seq = [int(k) for k in pois_dict.keys()]
                selected_indices = np.array(day_order)[np.array(selected_seq) - 1]
                return self.user_favorites.loc[selected_indices]
            except (ValueError, IndexError):
                pass

            # 方案2：名称 key → 模糊匹配，但只限于当天候选 POI（防止匹配到远处同名POI）
            from thefuzz import process as fz
            # 当天候选 POI 的名称列表（按 day_order 顺序）
            candidate_names = [self.user_favorites.loc[pid, "name"] for pid in day_order]
            matched = []
            for key in pois_dict.keys():
                best, score = fz.extract(key, candidate_names, limit=1)[0]
                if score > 80:
                    # 找回原始 POI id
                    idx_in_order = candidate_names.index(best)
                    pid = day_order[idx_in_order]
                    if pid not in matched:
                        matched.append(pid)
            if matched:
                return self.user_favorites.loc[matched]

            # 终极 fallback
            return self.user_favorites.loc[day_order]

        # ---- 单日模式 ----
        if self.num_days == 1:
            day_data = result["days"][0] if result.get("days") else {}
            day_order = day_orders[0] if day_orders else []

            if len(day_order) == 0:
                return

            # 全量 TSP 候选路径图（调试用，保留）
            df_full = self.user_favorites.loc[day_order]
            _create_map(df_full, map_filename=f"{current_time}_{self.type}_fulltsp.html")

            # LLM 精选路线图
            df_selected = _get_selected_pois(day_order, day_data)

            _create_map(df_selected, map_filename=f"{current_time}_{self.type}.html")

            # 聚类圆图（使用精选 POI）
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

            # 预先计算每天的精选 POI DataFrame（供每日图和总览图共用）
            day_selected_dfs = []
            for day_idx in range(min(self.num_days, len(day_orders))):
                day_order = day_orders[day_idx]
                if len(day_order) == 0:
                    day_selected_dfs.append(None)
                    continue
                day_data = result["days"][day_idx] if day_idx < len(result["days"]) else {}
                df_day = _get_selected_pois(day_order, day_data)
                day_selected_dfs.append(df_day)

                color = day_colors[day_idx % len(day_colors)]
                _create_map(
                    df_day,
                    polyline_color=color,
                    map_filename=f"{current_time}_{self.type}_day{day_idx+1}.html"
                )

            # 总览图：高德瓦片 + GCJ-02 转换
            m = folium.Map(location=[31.23, 121.47], zoom_start=12, tiles=None)
            folium.TileLayer(
                tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                attr='高德地图',
                subdomains='1234',
                max_zoom=18
            ).add_to(m)
            all_lats, all_lons = [], []

            for day_idx, df_day in enumerate(day_selected_dfs):
                if df_day is None or len(df_day) == 0:
                    continue
                color = day_colors[day_idx % len(day_colors)]
                df_day = df_day.copy()
                # 数据已是 GCJ-02，直接使用

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

    # ========== 对外接口 ==========

    def solve(self):
        """solve() 接口：多日模式"""
        return self.solve_multiday()
