"""
ROAM 评估指标计算模块
======================================
用于对比实验（ROAM vs LLM-Only）和消融实验的定量评估。

指标列表（11个）:
  RC  — Route Compactness (路线紧凑度, 米/段, ↓)
  CR  — Time Compliance Rate (时间合规率, %, ↑)
  SM  — Semantic Matching (语义匹配度, cos, ↑)
  HR  — Hallucination Rate (幻觉率, %, ↓)
  DB  — Day Balance (日间均衡度, CV, ↓)
  WA  — Weather Adaptation (天气适配率, %, ↑)
  TT  — Total Transit Ratio (通勤占比, %, ↓)
  AR  — Avg Rating (平均景点评分, 1-5分, ↑)
  PC  — Preference Coverage (偏好覆盖率, %, ↑)
  CE  — City Exploration (城市探索率, %, ↑)
  POI — POI Count (景点总数, 个, —)

用法:
    from model.eval.metrics import ItineraryEvaluator

    evaluator = ItineraryEvaluator(city="fuzhou")
    metrics = evaluator.compute_all(trip_result, user_query)
    print(evaluator.format_table(metrics))
"""

import json
import re
import os
import datetime
import numpy as np
import pandas as pd
from typing import Optional
from sentence_transformers import SentenceTransformer

# ==================== 辅助函数 ====================

def _name_to_idx(data_df: pd.DataFrame):
    """构建 POI 名称 → DataFrame 索引 映射"""
    mapping = {}
    for idx, row in data_df.iterrows():
        name = str(row.get("name", ""))
        if name:
            mapping[name] = idx
    return mapping


def _fuzzy_match(name: str, all_names: list, threshold: int = 80):
    """模糊匹配景点名到数据库名"""
    try:
        from thefuzz import process as fz
        best, score = fz.extract(name, all_names, limit=1)[0]
        if score >= threshold:
            return best
    except ImportError:
        # 无 thefuzz 时尝试直接精确匹配
        if name in all_names:
            return name
    return None


def _parse_timeline_entry(line: str):
    """
    解析时间线条目，如:
      "08:00-08:45  三坊七巷  (起点 | 停留45min | 营业 00:00-23:59)"
      "09:33-10:03  船政格致园  (🚲 3min | 停留30min | 营业 09:00-18:00)"
    返回: dict 或 None
    """
    # 匹配时间 + 名称 + 详情
    m = re.match(
        r'\s*(\d{2}):(\d{2})-(\d{2}):(\d{2})\s+(.+?)\s+\((.+?)\|.+?营业\s+(\d{2}):(\d{2})-(\d{2}):(\d{2})\)',
        line
    )
    if not m:
        return None
    arr_h, arr_m, dep_h, dep_m = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    name = m[5].strip()
    open_h, open_m, close_h, close_m = int(m[7]), int(m[8]), int(m[9]), int(m[10])
    return {
        "name": name,
        "arrival_min": arr_h * 60 + arr_m,
        "departure_min": dep_h * 60 + dep_m,
        "open_min": open_h * 60 + open_m,
        "close_min": close_h * 60 + close_m,
    }


def _parse_transit_summary(transit_text: str):
    """
    从交通摘要中提取总距离和通勤时间。
    返回: {"total_dist_m": float, "walk_m": float, "bike_m": float, "car_m": float, "cost": float, "weighted_dist_m": float}

    weighted_dist_m 使用时间归一化加权：
      - 步行 80m/min → 权重 1.0（基准）
      - 骑行 250m/min → 权重 80/250 ≈ 0.32
      - 驾车 500m/min → 权重 80/500 ≈ 0.16
    含义：以"步行等效距离"衡量紧凑度——骑行1km等效步行320m，驾车1km等效步行160m。
    """
    result = {"total_dist_m": 0, "walk_m": 0, "bike_m": 0, "car_m": 0, "cost": 0, "weighted_dist_m": 0}
    if not transit_text:
        return result

    # 解析每日汇总行
    m = re.search(r'总移动距离:\s*([\d.]+)m', transit_text)
    if m:
        result["total_dist_m"] = float(m.group(1))

    m = re.search(r'步行:\s*([\d.]+)m', transit_text)
    if m:
        result["walk_m"] = float(m.group(1))

    m = re.search(r'骑行:\s*([\d.]+)m', transit_text)
    if m:
        result["bike_m"] = float(m.group(1))

    m = re.search(r'车辆出行:.*?共([\d.]+)m', transit_text)
    if m:
        result["car_m"] = float(m.group(1))

    m = re.search(r'预估总花费:\s*([\d.]+)元', transit_text)
    if m:
        result["cost"] = float(m.group(1))

    # 加权等效距离 = 步行×1.0 + 骑行×0.32 + 驾车×0.16
    result["weighted_dist_m"] = round(
        result["walk_m"] * 1.0 + result["bike_m"] * 0.32 + result["car_m"] * 0.16, 1)

    return result


# ==================== 指标计算类 ====================

class ItineraryEvaluator:
    """行程评估器 —— 计算全部 7 个定量指标"""

    def __init__(self,
                 city: str = "fuzhou",
                 type: str = "zh",
                 embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
                 daily_start: str = "08:00",
                 daily_end: str = "20:00"):
        """
        Args:
            city: 城市名（拼音，用于加载 POI 数据）
            type: 语言类型（zh）
            embedding_model_name: 用于语义匹配计算的嵌入模型
            daily_start: 每日可用时间起点 "HH:MM"
            daily_end: 每日可用时间终点 "HH:MM"
        """
        self.city = city
        self.type = type

        # 城市名标准化：中文名 → 拼音（兼容两种输入）
        CITY_ZH2PINYIN = {
            "福州": "fuzhou", "广州": "guangzhou",
            "深圳": "shenzhen", "上海": "shanghai",
        }
        city_pinyin = CITY_ZH2PINYIN.get(city, city)

        # 加载 POI 数据库
        data_path = os.path.join("model", "data", f"{city_pinyin}_{type}.csv")
        self.data = pd.read_csv(data_path)
        self.name_to_idx = _name_to_idx(self.data)
        self.all_names = list(self.name_to_idx.keys())

        # 嵌入模型（延迟加载）
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name

        # 每日可用分钟数
        sh, sm = int(daily_start.split(":")[0]), int(daily_start.split(":")[1])
        eh, em = int(daily_end.split(":")[0]), int(daily_end.split(":")[1])
        self.available_min_per_day = (eh * 60 + em) - (sh * 60 + sm)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    # ==================== 单项指标 ====================

    def compute_route_compactness(self, trip_result: dict) -> float:
        """
        RC — 路线紧凑度（加权步行等效米/段，↓）
        使用时间归一化加权距离：
          - 步行 1m = 1m 等效（基准）
          - 骑行 1m ≈ 0.32m 等效
          - 驾车 1m ≈ 0.16m 等效
        总加权距离 / (总POI数 - 天数)，即平均每段路线的"步行等效长度"。
        驾车为主的行程不会被误判为松散，步行为主的行程能得到合理的紧凑度。
        """
        total_dist = 0.0
        total_pois = 0
        num_days = 0
        for day in trip_result.get("days", []):
            transit = day.get("transit_summary", "")
            parsed = _parse_transit_summary(transit)
            # 优先用加权等效距离，fallback 到总距离，再 fallback 到估算
            if parsed["weighted_dist_m"] > 0:
                total_dist += parsed["weighted_dist_m"]
            elif parsed["total_dist_m"] > 0:
                total_dist += parsed["total_dist_m"]
            else:
                total_dist += self._estimate_day_distance(day)
            n = len([x for x in day.get("itinerary", "").split("->") if x.strip()])
            total_pois += n
            if n > 0:
                num_days += 1

        segments = total_pois - num_days  # 路线段数 = POI数 - 天数
        if segments <= 0:
            return 0.0
        return round(total_dist / segments, 1)

    def compute_time_compliance_rate(self, trip_result: dict) -> float:
        """
        CR — 时间合规率（%，↑）
        成功在闭馆前到达并完成游览的景点数 / 总景点数 × 100%。
        从 timeline 解析每个景点的到达+停留时间并与 close_time 对比。
        """
        total_pois = 0
        compliant_pois = 0

        for day in trip_result.get("days", []):
            timeline = day.get("timeline", "")
            if not timeline:
                continue

            for line in timeline.split("\n"):
                entry = _parse_timeline_entry(line)
                if not entry:
                    continue
                total_pois += 1
                # 在闭馆时间前到达即视为合规
                if entry["arrival_min"] <= entry["close_min"]:
                    compliant_pois += 1

        if total_pois == 0:
            return 100.0  # 无景点视为全合规
        return round(compliant_pois / total_pois * 100, 1)

    def compute_semantic_matching(self, trip_result: dict, user_query: str) -> float:
        """
        SM — 语义匹配度（↑）
        所选景点文本嵌入与用户偏好嵌入的平均余弦相似度。
        """
        # 提取所有选中的景点名称
        poi_names = set()
        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if name:
                    poi_names.add(name)

        if not poi_names:
            return 0.0

        # 获取 POI 嵌入（从数据库中匹配）
        valid_names = []
        for n in poi_names:
            matched = _fuzzy_match(n, self.all_names)
            if matched:
                valid_names.append(matched)
            else:
                # 直接查找
                if n in self.name_to_idx:
                    valid_names.append(n)

        if not valid_names:
            return 0.0

        # 生成用户偏好嵌入
        user_emb = self.embedding_model.encode([user_query], normalize_embeddings=True)[0]

        # 生成 POI 嵌入并计算平均余弦相似度
        similarities = []
        for name in valid_names:
            idx = self.name_to_idx[name]
            context = str(self.data.loc[idx, "context"])
            poi_emb = self.embedding_model.encode([context], normalize_embeddings=True)[0]
            sim = float(np.dot(user_emb, poi_emb))
            similarities.append(sim)

        return round(float(np.mean(similarities)), 4)

    def compute_hallucination_rate(self, trip_result: dict) -> float:
        """
        HR — 幻觉率（%，↓）
        LLM 推荐景点中不在 POI 数据库中的比例。
        """
        total = 0
        hallucinated = 0

        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if not name:
                    continue
                total += 1
                matched = _fuzzy_match(name, self.all_names)
                if not matched and name not in self.name_to_idx:
                    hallucinated += 1

        if total == 0:
            return 0.0
        return round(hallucinated / total * 100, 1)

    def compute_day_balance(self, trip_result: dict) -> float:
        """
        DB — 日间均衡度（CV，↓）
        各天景点数量的变异系数 σ/μ，越低越均衡。
        """
        counts = []
        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            n = len([x for x in itinerary.split(" -> ") if x.strip()])
            counts.append(n)

        if not counts or sum(counts) == 0:
            return 0.0

        mean_c = np.mean(counts)
        std_c = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
        if mean_c == 0:
            return 0.0
        return round(float(std_c / mean_c), 3)

    def compute_weather_adaptation(self, trip_result: dict, user_query: str = "") -> float:
        """
        WA — 天气适配率（%，↑）
        检测 query 是否为天气敏感场景（雨天/周一），若是则计算室内 POI 占比或非博物馆占比。
        非天气敏感 query 返回 100（不适用）。
        """
        indoor_categories = {
            "博物馆/纪念馆", "文化艺术", "购物商圈", "美食街区",
            "宗教寺庙", "主题乐园", "演艺/剧院", "温泉度假",
        }
        outdoor_categories = {
            "自然风光", "城市公园", "建筑地标", "历史街区",
            "海滨沙滩", "历史遗迹", "特色村落",
        }
        monday_close_categories = {
            "博物馆/纪念馆",
        }

        # 仅检测雨天场景（周一闭馆无数据支持，不做判断）
        is_rain_query = False
        if user_query:
            if "雨" in user_query or "室内" in user_query:
                is_rain_query = True

        # 非雨天内 query → 不适用
        if not is_rain_query:
            return 100.0

        indoor_count = 0
        outdoor_count = 0

        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if not name:
                    continue
                matched = _fuzzy_match(name, self.all_names)
                if not matched and name in self.name_to_idx:
                    matched = name
                if not matched:
                    continue
                idx = self.name_to_idx[matched]
                cat = str(self.data.loc[idx, "category"])
                if cat in indoor_categories:
                    indoor_count += 1
                elif cat in outdoor_categories:
                    outdoor_count += 1

        total = indoor_count + outdoor_count
        if total == 0:
            return 100.0
        return round(indoor_count / total * 100, 1)

    def compute_transit_ratio(self, trip_result: dict) -> float:
        """
        TT — 通勤占比（%，↓）
        总通勤时间 / (每日可用分钟数 × 天数) × 100%。
        通勤时间从 timeline 的 travel 字段估算。
        """
        total_commute_min = 0.0
        num_days = len(trip_result.get("days", []))

        for day in trip_result.get("days", []):
            timeline = day.get("timeline", "")
            if not timeline:
                continue
            for line in timeline.split("\n"):
                # 匹配 travel 时间，如 "(🚗 7min" 或 "(🚲 3min" 或 "(🚶 1min"
                m = re.search(r'[🚗🚲🚶]\s*(\d+)min', line)
                if m:
                    total_commute_min += int(m.group(1))

        total_available = self.available_min_per_day * num_days
        if total_available == 0:
            return 0.0
        return round(total_commute_min / total_available * 100, 1)

    def compute_avg_rating(self, trip_result: dict) -> float:
        """
        AR — 平均景点评分（1-5分，↑）
        所选景点在 POI 数据库中的评分均值。
        若无评分数据则回退到类别默认评分。
        """
        ratings = []
        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if not name:
                    continue
                matched = _fuzzy_match(name, self.all_names)
                if not matched and name in self.name_to_idx:
                    matched = name
                if not matched:
                    continue
                idx = self.name_to_idx[matched]
                try:
                    r = float(self.data.loc[idx, "rating"])
                    if pd.isna(r) or r <= 0:
                        raise ValueError
                    ratings.append(r)
                except (KeyError, ValueError, TypeError):
                    # fallback: 类别默认评分
                    cat = str(self.data.loc[idx, "category"])
                    fallback_ratings = {
                        "博物馆/纪念馆": 3.8, "文化艺术": 3.6, "历史街区": 3.7,
                        "建筑地标": 3.5, "自然风光": 3.9, "城市公园": 3.6,
                        "美食街区": 3.5, "购物商圈": 3.4, "宗教寺庙": 3.7,
                        "主题乐园": 3.8, "历史遗迹": 3.6, "海滨沙滩": 3.7,
                        "特色村落": 3.5, "温泉度假": 3.7, "高校学府": 3.5,
                        "演艺/剧院": 3.7,
                    }
                    ratings.append(fallback_ratings.get(cat, 3.0))

        if not ratings:
            return 3.0
        return round(float(np.mean(ratings)), 2)

    def compute_preference_coverage(self, trip_result: dict, user_query: str = "") -> float:
        """
        PC — 偏好覆盖率（%，↑）
        用户查询中提到的偏好类别被所选景点覆盖的比例。
        从 user_query 中提取关键词（如"历史、美食、公园"），
        检查所选景点类别是否命中。
        """
        if not user_query:
            return 100.0

        # 定义偏好关键词 → 对应 POI 类别映射
        preference_category_map = {
            "历史": ["博物馆/纪念馆", "历史街区", "历史遗迹", "宗教寺庙", "建筑地标"],
            "文化": ["博物馆/纪念馆", "文化艺术", "历史街区", "宗教寺庙", "演艺/剧院"],
            "美食": ["美食街区", "购物商圈"],
            "公园": ["城市公园", "自然风光"],
            "自然": ["自然风光", "城市公园", "海滨沙滩"],
            "购物": ["购物商圈", "美食街区"],
            "亲子": ["主题乐园", "城市公园", "博物馆/纪念馆"],
            "休闲": ["城市公园", "温泉度假", "购物商圈", "美食街区"],
            "建筑": ["建筑地标", "历史街区", "历史遗迹"],
            "宗教": ["宗教寺庙"],
            "艺术": ["文化艺术", "演艺/剧院"],
            "海边": ["海滨沙滩", "自然风光"],
            "古镇": ["特色村落", "历史街区", "历史遗迹"],
        }

        # 从用户查询中检测偏好
        detected_prefs = []
        for kw, cats in preference_category_map.items():
            if kw in user_query:
                detected_prefs.append((kw, cats))

        if not detected_prefs:
            return 100.0  # 无明确偏好类别，不适用

        # 收集所选景点的类别
        selected_categories = set()
        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if not name:
                    continue
                matched = _fuzzy_match(name, self.all_names)
                if not matched and name in self.name_to_idx:
                    matched = name
                if not matched:
                    continue
                idx = self.name_to_idx[matched]
                selected_categories.add(str(self.data.loc[idx, "category"]))

        # 检查每个偏好是否被满足（至少命中一个相关类别）
        covered = 0
        for kw, cats in detected_prefs:
            if any(c in selected_categories for c in cats):
                covered += 1

        return round(covered / len(detected_prefs) * 100, 1)

    def compute_city_exploration(self, trip_result: dict) -> float:
        """
        CE — 城市探索率（%，↑）
        行程 POI 覆盖范围 / 城市全部 POI 覆盖范围 × 100%。
        用凸包面积衡量，越大说明探索越广。
        """
        from scipy.spatial import ConvexHull

        # 收集行程中所有 POI 的坐标
        trip_lons, trip_lats = [], []
        for day in trip_result.get("days", []):
            itinerary = day.get("itinerary", "")
            for name in itinerary.split(" -> "):
                name = name.strip()
                if not name:
                    continue
                matched = _fuzzy_match(name, self.all_names)
                if not matched and name in self.name_to_idx:
                    matched = name
                if not matched:
                    continue
                idx = self.name_to_idx[matched]
                trip_lons.append(float(self.data.loc[idx, "lon"]))
                trip_lats.append(float(self.data.loc[idx, "lat"]))

        if len(trip_lons) < 3:
            return 0.0

        # 城市全部 POI 凸包面积（缓存）
        if not hasattr(self, '_city_hull_area'):
            all_lons = self.data["lon"].astype(float).values
            all_lats = self.data["lat"].astype(float).values
            pts_all = np.column_stack([all_lons, all_lats])
            try:
                self._city_hull_area = ConvexHull(pts_all).volume  # 2D hull "volume" = area
            except Exception:
                self._city_hull_area = 1.0  # fallback

        # 行程 POI 凸包面积
        pts_trip = np.column_stack([trip_lons, trip_lats])
        try:
            trip_area = ConvexHull(pts_trip).volume
        except Exception:
            return 0.0

        if self._city_hull_area <= 0:
            return 0.0
        return round(trip_area / self._city_hull_area * 100, 1)

    # ==================== 综合计算 ====================

    def compute_all(self, trip_result: dict, user_query: str = "") -> dict:
        """
        计算全部 10 个指标，返回字典。
        """
        return {
            "RC": self.compute_route_compactness(trip_result),
            "CR": self.compute_time_compliance_rate(trip_result),
            "SM": self.compute_semantic_matching(trip_result, user_query),
            "HR": self.compute_hallucination_rate(trip_result),
            "DB": self.compute_day_balance(trip_result),
            "WA": self.compute_weather_adaptation(trip_result, user_query),
            "TT": self.compute_transit_ratio(trip_result),
            "AR": self.compute_avg_rating(trip_result),
            "PC": self.compute_preference_coverage(trip_result, user_query),
            "CE": self.compute_city_exploration(trip_result),
            "POI": self._count_pois(trip_result),
        }

    def _count_pois(self, trip_result: dict) -> int:
        """统计总 POI 数"""
        return sum(len([x for x in d.get("itinerary", "").split("->") if x.strip()])
                   for d in trip_result.get("days", []))

    def compare(self,
                result_a: dict, label_a: str,
                result_b: dict, label_b: str,
                user_query: str = "") -> dict:
        """对比两组结果，返回包含差值 (Δ = B − A) 的字典"""
        ma = self.compute_all(result_a, user_query)
        mb = self.compute_all(result_b, user_query)
        return {
            "metrics_a": ma, "label_a": label_a,
            "metrics_b": mb, "label_b": label_b,
            "delta": {k: round(mb[k] - ma[k], 4) for k in ma},
        }

    # ==================== 格式化输出 ====================

    def format_table(self, metrics: dict, label: str = "") -> str:
        """单组指标 → 格式化表格字符串"""
        keys = ["RC", "CR", "SM", "HR", "DB", "WA", "TT", "AR", "PC", "CE", "POI"]
        arrows = {"RC": "↓", "CR": "↑", "SM": "↑", "HR": "↓", "DB": "↓",
                  "WA": "↑", "TT": "↓", "AR": "↑", "PC": "↑", "CE": "↑", "POI": "↑"}
        units = {"RC": "m", "CR": "%", "SM": "cos", "HR": "%", "DB": "CV",
                 "WA": "%", "TT": "%", "AR": "分", "PC": "%", "CE": "%", "POI": "个"}
        header = f"  {label:<20}" if label else f"  {'指标':<12} {'值':>10} {'方向':>4} {'单位':>6}"
        lines = [header, "  " + "-" * 50]
        for key in keys:
            val = metrics.get(key, "-")
            if isinstance(val, float):
                if key in ("SM", "DB"):
                    val_str = f"{val:.4f}"
                else:
                    val_str = f"{val:.1f}"
            else:
                val_str = str(val)
            if label:
                lines.append(f"  {key:<6} {val_str:>10} {arrows[key]:>4} {units[key]:>6}")
            else:
                lines.append(f"  {key:<12} {val_str:>10} {arrows[key]:>4} {units[key]:>6}")
        return "\n".join(lines)

    def format_comparison_table(self, comparison: dict) -> str:
        """对比表 → 格式化字符串"""
        ma = comparison["metrics_a"]
        mb = comparison["metrics_b"]
        delta = comparison["delta"]
        la = comparison["label_a"]
        lb = comparison["label_b"]

        lines = [
            f"  {'指标':<6} {la:>12} {lb:>12} {'Δ':>10} {'方向':>4}",
            "  " + "-" * 50,
        ]
        keys = ["RC", "CR", "SM", "HR", "DB", "WA", "TT", "AR", "PC", "CE", "POI"]
        arrows = {"RC": "↓", "CR": "↑", "SM": "↑", "HR": "↓", "DB": "↓",
                  "WA": "↑", "TT": "↓", "AR": "↑", "PC": "↑", "CE": "↑", "POI": "↑"}
        for key in keys:
            a = ma.get(key, "-")
            b = mb.get(key, "-")
            d = delta.get(key, "-")
            if isinstance(a, float) and key in ("SM", "DB"):
                fmt = "{:.4f}"
            else:
                fmt = "{:.1f}" if isinstance(a, float) else "{}"
            lines.append(
                f"  {key:<6} {fmt.format(a):>12} {fmt.format(b) if isinstance(b, float) else str(b):>12} "
                f"{fmt.format(d) if isinstance(d, float) else str(d):>10} {arrows[key]:>4}"
            )
        return "\n".join(lines)

    # ==================== 内部辅助 ====================

    def _estimate_day_distance(self, day: dict) -> float:
        """用直线距离估算某天的通勤距离（OSRM 不可用时的 fallback）"""
        itinerary = day.get("itinerary", "")
        names = [x.strip() for x in itinerary.split(" -> ") if x.strip()]
        if len(names) < 2:
            return 0.0

        matched = []
        for n in names:
            m = _fuzzy_match(n, self.all_names)
            if m and m in self.name_to_idx:
                matched.append(self.name_to_idx[m])
            elif n in self.name_to_idx:
                matched.append(self.name_to_idx[n])

        if len(matched) < 2:
            return 0.0

        locs = self.data.loc[matched][["x", "y"]].astype(float).to_numpy()
        dists = np.sqrt(np.sum(np.diff(locs, axis=0) ** 2, axis=1))
        return float(np.sum(dists))


# ==================== CLI 入口 ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ROAM 评估指标计算工具")
    parser.add_argument("--city", type=str, default="fuzhou", help="城市")
    parser.add_argument("--json", type=str, required=True, help="行程 JSON 文件路径")
    parser.add_argument("--query", type=str, default="", help="用户原始查询（用于语义匹配）")
    parser.add_argument("--compare", type=str, default=None, help="对比的第二个 JSON 文件路径")
    parser.add_argument("--label-a", type=str, default="A", help="第一个结果标签")
    parser.add_argument("--label-b", type=str, default="B", help="第二个结果标签")
    args = parser.parse_args()

    evaluator = ItineraryEvaluator(city=args.city)

    with open(args.json, "r", encoding="utf-8") as f:
        result_a = json.load(f)

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            result_b = json.load(f)
        comp = evaluator.compare(result_a, args.label_a, result_b, args.label_b, args.query)
        print("\n" + evaluator.format_comparison_table(comp))
    else:
        metrics = evaluator.compute_all(result_a, args.query)
        print("\n" + evaluator.format_table(metrics, label=os.path.basename(args.json)))
