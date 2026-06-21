"""
Travel Agent: 时空感知 RAG 模块
================================
为旅游规划 Agent 提供增强的时空上下文：
  - 天气感知（影响户外/室内 POI 推荐）
  - 时段感知（POI 开放时间、人流高峰）
  - 交通感知（POI 间通勤时间预估）
  - 事件感知（节假日、临时活动、闭馆信息）
"""

import os
import datetime
import numpy as np
import requests
from typing import Optional
from model.config import (
    WEATHER_TYPES, WEATHER_ACTIVITY_MAP, TIME_SLOT_CROWD,
    CITY_EN_MAP, CITY_ZH_MAP, WTTR_WEATHER_MAP, CATEGORY_TIPS
)


class SpatiotemporalRAG:
    """
    时空检索增强生成模块
    在 embedding 检索之上，注入动态时空上下文，
    让 LLM 规划时能感知真实世界约束。

    所有静态映射表统一在 model/config.py 中管理。
    """

    def __init__(self,
                 city: str,
                 num_days: int = 1,
                 travel_date: Optional[str] = None,
                 use_simulated: bool = True):
        """
        初始化时空 RAG 模块

        Args:
            city: 城市名称
            num_days: 行程天数
            travel_date: 出发日期 (YYYY-MM-DD)，None 则使用当前日期
            use_simulated: 是否使用模拟数据（无 API key 时）
        """
        self.city = city
        self.city_zh = CITY_ZH_MAP.get(city, city)  # 中文显示
        self.num_days = num_days
        self.use_simulated = use_simulated

        if travel_date:
            self.start_date = datetime.date.fromisoformat(travel_date)
        else:
            self.start_date = datetime.date.today()

        # 获取天气：wttr.in API
        if not self.use_simulated:
            self.daily_weather = self._fetch_wttr_weather()
        else:
            self.daily_weather = []
        if not self.daily_weather:
            self.daily_weather = []

    # ==================== 天气上下文 ====================

    def _fetch_wttr_weather(self) -> list[dict]:
        """
        调用 wttr.in 免费 API 获取 3 天天气预报。

        wttr.in 完全免费，无需注册，无需 API Key。

        Returns:
            list[dict]: 天气预报列表；失败返回空列表
        """
        city_en = CITY_EN_MAP.get(self.city, self.city)

        try:
            url = f"https://wttr.in/{city_en}?format=j1"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            weather_list = data.get("weather", [])
            if not weather_list:
                print(f"⚠️ wttr.in 返回空数据，天气不可用")
                return []

            print(f"🌤️ 天气 (wttr.in): {city_en}")

            forecasts = []
            for i, day in enumerate(weather_list):
                if i >= self.num_days:
                    break

                date = datetime.date.fromisoformat(day["date"])
                hourly = day.get("hourly", [{}])
                mid = hourly[min(4, len(hourly)-1)]
                desc_en = mid.get("weatherDesc", [{}])[0].get("value", "Sunny")
                weather = WTTR_WEATHER_MAP.get(desc_en.lower(), "多云⛅")

                forecasts.append({
                    "date": day["date"],
                    "day_of_week": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.weekday()],
                    "weather": weather,
                    "temp_high": int(day.get("maxtempC", 25)),
                    "temp_low": int(day.get("mintempC", 18)),
                    "activity_advice": WEATHER_ACTIVITY_MAP.get(weather, "天气适宜出行"),
                    "humidity": int(mid.get("humidity", 60)),
                    "wind_level": int(float(mid.get("windspeedKmph", 10)) / 10),
                    "source": "wttr.in",
                })

            print(f"✅ 天气已获取: {len(forecasts)} 天预报")
            return forecasts

        except requests.exceptions.Timeout:
            print("⚠️ wttr.in 超时，天气不可用")
        except requests.exceptions.ConnectionError:
            print("⚠️ wttr.in 网络不通，天气不可用")
        except Exception as e:
            print(f"⚠️ wttr.in 异常: {e}，天气不可用")
        return []

    def get_weather_context(self, day_index: int = 0) -> str:
        """
        获取指定天的天气上下文，用于注入 LLM prompt。

        无天气数据时返回空字符串，不影响规划。
        """
        if day_index >= len(self.daily_weather) or not self.daily_weather:
            return "⚠️ 天气信息暂缺，请根据季节常识自行安排"

        w = self.daily_weather[day_index]
        return (
            f"📅 日期：{w['date']}（{w['day_of_week']}）\n"
            f"🌤️ 天气：{w['weather']}\n"
            f"🌡️ 温度：{w['temp_low']}°C ~ {w['temp_high']}°C\n"
            f"💧 湿度：{w['humidity']}%\n"
            f"💡 活动建议：{w['activity_advice']}"
        )

    def get_all_weather_summary(self) -> str:
        """获取全部天数的天气摘要"""
        if not self.daily_weather:
            return "⚠️ 未能获取天气数据，行程规划将不考虑天气因素"
        lines = ["### 🌤️ 旅行期间天气预报"]
        for i, w in enumerate(self.daily_weather):
            lines.append(
                f"  Day {i+1} ({w['date']} {w['day_of_week']}): "
                f"{w['weather']} {w['temp_low']}~{w['temp_high']}°C"
            )
        return "\n".join(lines)

    # ==================== POI 动态上下文 ====================

    def get_poi_dynamic_context(self,
                                 poi_name: str,
                                 poi_category: str,
                                 day_index: int = 0) -> str:
        """
        获取单个 POI 的动态信息（开放时间、人流预估、注意事项）。
        """
        weather = self.daily_weather[day_index]["weather"] if day_index < len(self.daily_weather) else ""
        day_of_week = self.daily_weather[day_index]["day_of_week"] if day_index < len(self.daily_weather) else "未知"
        is_weekend = day_of_week in ["周六", "周日"]

        # 根据类别和天气给出建议
        is_outdoor = poi_category in ["自然风光", "城市公园", "建筑地标", "历史街区", "主题乐园"]
        is_indoor = poi_category in ["博物馆/纪念馆", "文化艺术", "购物商圈", "美食街区", "宗教寺庙"]

        notes = []

        # 天气影响
        if is_outdoor and "雨" in weather:
            notes.append("⚠️ 雨天户外体验较差，建议备选室内景点")
        elif is_outdoor and "酷热" in weather:
            notes.append("⚠️ 高温天气，建议避开 12:00-15:00 时段")
        elif is_indoor and "晴天" in weather:
            notes.append("💡 晴天也可作为避暑/避雨备选")

        # 周末人流
        if is_weekend:
            notes.append("📊 周末客流量较大，建议上午早些出发")
        else:
            notes.append("📊 工作日客流量适中")

        tip = CATEGORY_TIPS.get(poi_category)
        if tip:
            notes.append(tip)

        return f"📍 {poi_name}（{poi_category}）\n" + "\n".join(f"  {n}" for n in notes)

    def get_all_poi_dynamic_context(self,
                                     poi_names: list[str],
                                     poi_categories: list[str],
                                     day_index: int = 0) -> str:
        """
        批量获取多个 POI 的动态上下文

        Args:
            poi_names: POI 名称列表
            poi_categories: POI 类别列表
            day_index: 第几天

        Returns:
            汇总的动态上下文字符串
        """
        lines = [f"### 📍 Day {day_index+1} POI 动态信息"]
        for name, cat in zip(poi_names, poi_categories):
            lines.append(self.get_poi_dynamic_context(name, cat, day_index))
            lines.append("")
        return "\n".join(lines)

    # ==================== 交通上下文 ====================

    def get_transit_context(self,
                            from_poi: str,
                            to_poi: str,
                            distance_meters: float) -> str:
        """
        获取两点间的交通建议。

        策略：
        1. 短距离 (<2000m) → 步行/骑行（基于距离的启发式估算）
        2. 长距离 → OSRM 驾车路线（真实道路数据，无需注册）
        3. OSRM 不可用时 → 启发式估算

        Args:
            from_poi: 起点 POI 名称
            to_poi: 终点 POI 名称
            distance_meters: 直线距离（米）

        Returns:
            交通建议字符串
        """
        # 短距离步行/骑行
        if distance_meters < 2000:
            walk_min = int(distance_meters / 80)
            bike_min = int(distance_meters / 250)
            advice = f"🚶 步行约{walk_min}分，或 🚲 骑行约{bike_min}分"
            return f"  {from_poi} → {to_poi}: 约{int(distance_meters)}m，{advice}"

        # 长距离尝试 OSRM 真实路线
        from_lon, from_lat = self._get_poi_coords(from_poi)
        to_lon, to_lat = self._get_poi_coords(to_poi)

        if from_lon and from_lat and to_lon and to_lat:
            duration_min, road_dist = self._fetch_osrm_route(from_lon, from_lat, to_lon, to_lat)
            if duration_min is not None:
                taxi_fee = int(road_dist / 1000 * 2.5 + 12)  # 起步价+里程费
                return (
                    f"  {from_poi} → {to_poi}: 直线{int(distance_meters)}m，"
                    f"道路{int(road_dist)}m，🚗 驾车约{duration_min}分 "
                    f"（约{taxi_fee}元）"
                )

        # 回退：基于直线距离的启发式估算
        transit_min = int(distance_meters / 500 + 5)
        taxi_fee = int(distance_meters / 1000 * 3)
        return (
            f"  {from_poi} → {to_poi}: 约{int(distance_meters)}m，"
            f"🚇 地铁约{transit_min}分，或 🚗 打车约{taxi_fee}元"
        )

    def _get_poi_coords(self, name: str) -> tuple:
        """根据 POI 名称查经纬度（来自预先传入的坐标映射）"""
        if not hasattr(self, '_poi_coords') or not self._poi_coords:
            return None, None
        return self._poi_coords.get(name, (None, None))

    def set_poi_coords(self, poi_coords: dict):
        """
        设置 POI 名称→经纬度映射，供 OSRM 路线计算使用。

        Args:
            poi_coords: {POI名称: (lon, lat)}
        """
        self._poi_coords = poi_coords
        print(f"🗺️ 已加载 {len(poi_coords)} 个 POI 坐标，OSRM 路线计算已就绪")

    def _fetch_osrm_route(self, lon1: float, lat1: float,
                          lon2: float, lat2: float) -> tuple:
        """
        调用 OSRM 免费 API 获取驾车路线。

        完全免费，无需注册，无需 API Key。
        基于 OpenStreetMap 数据。

        Returns:
            tuple: (duration_min, distance_m) 或 (None, None) 失败
        """
        try:
            url = (
                f"https://router.project-osrm.org/route/v1/driving/"
                f"{lon1},{lat1};{lon2},{lat2}"
                f"?overview=false"
            )
            resp = requests.get(url, timeout=8)
            data = resp.json()

            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                duration_min = int(route["duration"] / 60)
                distance_m = int(route["distance"])
                return duration_min, distance_m
        except Exception:
            pass
        return None, None

    def get_daily_transit_summary(self,
                                   ordered_poi_names: list[str],
                                   distances: np.ndarray) -> str:
        """
        生成一天内连续 POI 之间的交通摘要，包含详细费用分解。
        """
        segments = []
        total_walk = 0
        total_bike = 0
        total_transit = 0
        transit_count = 0
        total_cost = 0
        total_dist = 0

        for i in range(len(distances)):
            d = distances[i]
            frm = ordered_poi_names[i]
            to = ordered_poi_names[i + 1]
            total_dist += d

            if d < 500:
                # 纯步行
                walk_min = int(d / 80)
                seg = f"  {frm} → {to}: 🚶 步行{int(d)}m（约{walk_min}分）"
                total_walk += d
                segments.append((seg, d, "walk", 0))

            elif d < 2000:
                # 步行或骑行
                walk_min = int(d / 80)
                bike_min = int(d / 250)
                total_bike += d
                seg = f"  {frm} → {to}: 🚲 骑行{int(d)}m（约{bike_min}分，或步行约{walk_min}分）"
                segments.append((seg, d, "bike", 0))

            else:
                # 尝试 OSRM 驾车路线
                from_lon, from_lat = self._get_poi_coords(frm)
                to_lon, to_lat = self._get_poi_coords(to)
                duration_min, road_dist = None, None

                if from_lon and to_lon:
                    duration_min, road_dist = self._fetch_osrm_route(
                        from_lon, from_lat, to_lon, to_lat
                    )

                if duration_min is not None:
                    # OSRM 真实路线
                    taxi_fee = int(road_dist / 1000 * 2.5 + 12)
                    total_cost += taxi_fee
                    total_transit += road_dist  # 用道路距离
                    transit_count += 1
                    seg = (
                        f"  {frm} → {to}: 🚗 驾车{road_dist}m（约{duration_min}分，"
                        f"打车约{taxi_fee}元）"
                    )
                    segments.append((seg, road_dist, "car", taxi_fee))
                else:
                    # 回退：地铁估算
                    transit_min = int(d / 500 + 5)
                    fee = int(d / 1000 * 3)
                    total_cost += fee
                    total_transit += d
                    transit_count += 1
                    seg = (
                        f"  {frm} → {to}: 🚇 轨道{int(d)}m（约{transit_min}分，"
                        f"约{fee}元）"
                    )
                    segments.append((seg, d, "transit", fee))

        # 组装输出（总距离用分段实际距离，非直线距离）
        lines = ["### 🚗 交通建议与费用明细"]
        for seg_str, _, _, _ in segments:
            lines.append(seg_str)

        # 汇总（用 seg 中记录的实际距离）
        total_walk = sum(d for _, d, m, _ in segments if m == "walk")
        total_bike = sum(d for _, d, m, _ in segments if m == "bike")
        total_transit = sum(d for _, d, m, _ in segments if m in ("car", "transit"))
        transit_count = sum(1 for _, _, m, _ in segments if m in ("car", "transit"))
        total_cost = sum(c for _, _, _, c in segments)
        total_dist = total_walk + total_bike + total_transit
        lines.append("")
        lines.append("  ───────────────────────────────────")
        lines.append(f"  📊 当日交通汇总")
        if total_walk > 0:
            lines.append(f"    🚶 步行: {int(total_walk)}m")
        if total_bike > 0:
            lines.append(f"    🚲 骑行: {int(total_bike)}m")
        if total_transit > 0:
            lines.append(f"    🚗 车辆出行: {transit_count}次，共{int(total_transit)}m")
        lines.append(f"    📏 总移动距离: {int(total_dist)}m")
        if total_cost > 0:
            lines.append(f"    💰 预估总花费: {total_cost}元（打车和地铁）")
        else:
            lines.append(f"    💰 预估总花费: 0元（全程步行/骑行，无交通费用）")

        return "\n".join(lines)

    # ==================== 综合上下文生成 ====================

    def generate_full_context(self,
                               day_index: int,
                               ordered_poi_names: list[str],
                               ordered_poi_categories: list[str],
                               distances: np.ndarray) -> str:
        """
        生成注入 LLM prompt 的完整时空上下文

        这是 RAG 模块的主入口——将天气、POI 动态信息、交通建议
        整合为一段结构化文本，注入到规划 prompt 中。

        Args:
            day_index: 第几天
            ordered_poi_names: 排序后的 POI 名称
            ordered_poi_categories: 对应的 POI 类别
            distances: 相邻 POI 距离

        Returns:
            完整的时空上下文文本
        """
        parts = []

        # 1. 天气
        parts.append(self.get_weather_context(day_index))
        parts.append("")

        # 2. POI 动态信息
        parts.append(self.get_all_poi_dynamic_context(
            ordered_poi_names, ordered_poi_categories, day_index
        ))

        # 3. 交通
        parts.append(self.get_daily_transit_summary(ordered_poi_names, distances))

        return "\n".join(parts)

    def generate_multi_day_overview(self) -> str:
        """
        生成多日行程的整体概览（天气趋势 + 每日主题建议）
        """
        weather_source = self.daily_weather[0].get("source", "wttr.in") if self.daily_weather else "无（天气不可用）"
        lines = [
            "=" * 50,
            "🌍 时空感知上下文 (Spatiotemporal RAG Context)",
            f"📍 目的地: {self.city_zh}",
            f"📅 行程: {self.num_days} 天 ({self.start_date} ~ "
            f"{self.start_date + datetime.timedelta(days=self.num_days-1)})",
            f"📡 天气来源: {weather_source}",
            "=" * 50,
            "",
            self.get_all_weather_summary(),
            "",
        ]

        # 每日主题建议
        lines.append("### 🎯 每日主题建议")
        for i, w in enumerate(self.daily_weather):
            is_rain = "雨" in w["weather"]
            is_hot = "酷热" in w["weather"]
            is_weekend = w["day_of_week"] in ["周六", "周日"]

            if is_rain:
                theme = "室内文化日（博物馆、美术馆、购物）"
            elif is_hot:
                theme = "休闲避暑日（上午户外、下午室内）"
            elif is_weekend:
                theme = "热门景点日（周末人多但气氛好）"
            else:
                theme = "深度探索日（工作日人少，适合热门户外景点）"

            lines.append(f"  Day {i+1}: {theme} — {w['weather']} {w['temp_low']}~{w['temp_high']}°C")

        return "\n".join(lines)


# ==================== 便捷工厂函数 ====================

def create_rag_for_trip(city: str,
                        num_days: int,
                        travel_date: Optional[str] = None) -> SpatiotemporalRAG:
    """
    为一次旅行创建 RAG 模块实例。

    天气数据：自动使用 wttr.in 免费 API，网络不通时跳过天气。

    Args:
        city: 城市（中文名，如"上海"）
        num_days: 天数（wttr.in 提供 3 天，超出自动模拟补充）
        travel_date: 出发日期

    Returns:
        SpatiotemporalRAG 实例
    """
    return SpatiotemporalRAG(
        city=city,
        num_days=num_days,
        travel_date=travel_date,
        use_simulated=False,
    )
