"""
BareAgent — 裸 LLM 行程规划（无 ROAM 模块）
=============================================
用于对比实验：ROAM 完整体 vs LLM-Only 基线。

BareAgent 只保留：
  - POI 搜索（和 ROAM 相同的搜索引擎）
  - POI 数据库格式化
  - 单一 LLM 调用：LLM 自己分天、排序、写文案

移除的 ROAM 模块：
  - 空间聚类 (SpatialHandler)
  - TSP 路线排序
  - RAG 时空上下文（天气/交通/人流）
  - 多轮 LLM 调用（每日独立规划）

输出格式与 ROAM 完全一致，可直接用 metrics.py 对比。
"""

import json
import os
import re
import time
import datetime
import numpy as np
import pandas as pd
from typing import Optional

from model.agent import TravelAgent
from model.utils.funcs import get_user_data_embedding
from model.utils.all_prompts import process_input_prompt
from model.spatial import SpatialHandler


class BareAgent(TravelAgent):
    """
    裸 LLM 行程规划器 — 只给 LLM 提供 POI 列表，不做任何优化。

    继承 TravelAgent 以复用：
      - POI 搜索 (search_engine)
      - 住宿点匹配 (_match_hotel)
      - 排除景点 (_match_exclude_pois)
      - 地图渲染 (refresh_trip_rendering)
      - JSON 保存 (_save_result)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 裸模式固定关闭 RAG
        self.use_rag = False
        self.rag = None

    # ========== 裸 LLM 求解 ==========

    def solve(self) -> dict:
        """
        裸 LLM 模式主入口。
        与 solve_multiday() 输出格式完全一致。
        """
        print(f"🌍 Bare LLM: {self.num_days}天 {self.city_zh} 行程规划 (无ROAM)")
        print(f"{'='*30}\n")

        # Step 1: 四个城市的全部 POI（无搜索，裸 LLM 自己识别目标城市）
        import pandas as pd
        all_dfs = []
        for c in ["fuzhou", "guangzhou", "shenzhen", "shanghai"]:
            df = pd.read_csv(os.path.join("model", "data", f"{c}_zh.csv"))
            df["_city"] = c
            all_dfs.append(df)
        self._all_city_data = pd.concat(all_dfs, ignore_index=True)
        allpoi_ids = list(self._all_city_data.index)

        # Step 2: 构建纯文本 POI 数据库（四城全部）
        poi_lines = []
        for pid in allpoi_ids:
            row = self._all_city_data.loc[pid]
            name = str(row["name"])
            poi_lines.append(
                f"- [{row['_city']}] {name} | {row.get('category','')} | "
                f"坐标({row.get('lon',''):.4f},{row.get('lat',''):.4f}) | "
                f"{str(row.get('context',''))[:120]}"
            )
        poi_db = "\n".join(poi_lines)

        # Step 3: 构建 prompt — LLM 自己做全部规划（需要输出完整格式）
        bare_prompt = self._build_bare_prompt(poi_db)

        # Step 4: 调用 LLM
        print("  ⏳ 调用 LLM 自主规划（无 ROAM）...", end=" ", flush=True)
        t0 = time.time()

        try:
            response = self.proxy.chat(
                model=self.MODEL,
                messages=[{"role": "user", "content": bare_prompt}],
                temperature=0.7,
            )
            if isinstance(response, tuple):
                raw = response[0]
            else:
                raw = response

            elapsed = time.time() - t0

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', str(raw))
            result = json.loads(json_match.group() if json_match else str(raw))

        except Exception as e:
            print(f"❌ 解析失败: {e}")
            elapsed = time.time() - t0
            result = None

        if result is None:
            print(f"  ❌ LLM 未返回有效 JSON，回退到简单模板")
            result = self._build_fallback_result(allpoi_ids)

        # Step 5: 后处理 — 补齐 ROAM 兼容字段
        result["city"] = self.city
        result["city_zh"] = self.city_zh
        result["num_days"] = self.num_days
        result["rag_context"] = ""
        result["model"] = "bare_llm"
        result["hotel"] = self.hotel_info
        result["hotel_nearby"] = [
            {"name": n, "category": c, "distance_m": d}
            for _, n, c, d in self._get_hotel_nearby()
        ] if self.hotel_poi_id else []

        # 补齐每天字段 + 生成 timeline + transit_summary（使用道路距离）
        for day_idx, day in enumerate(result.get("days", [])):
            day["day"] = day.get("day", day_idx + 1)
            if "weather_note" not in day:
                day["weather_note"] = "无"
            if "day_theme" not in day:
                day["day_theme"] = f"第{day_idx+1}天"
            if "day_reason" not in day:
                day["day_reason"] = ""

            itinerary = day.get("itinerary", "")
            names = [n.strip() for n in itinerary.replace("→", "->").split("->") if n.strip()]
            day["itinerary"] = " -> ".join(names)

            # 匹配 POI ID
            matched_ids = self._match_poi_names(names)

            # Bare 的时间线：不用营业时间，只按距离排列，每个 POI 默认 45min
            if len(matched_ids) >= 2:
                day["timeline"] = self._build_simple_timeline(matched_ids, names)

                # 计算道路距离
                dists_m = []
                locs = self._all_city_data.loc[matched_ids][["lon", "lat"]].astype(float).to_numpy()
                for i in range(len(matched_ids) - 1):
                    key = (min(matched_ids[i], matched_ids[i+1]),
                           max(matched_ids[i], matched_ids[i+1]))
                    if hasattr(self.spatial_handler, '_road_dist_cache') and \
                       key in self.spatial_handler._road_dist_cache:
                        d = self.spatial_handler._road_dist_cache[key]
                    else:
                        d = np.sqrt((locs[i][0]-locs[i+1][0])**2 +
                                    (locs[i][1]-locs[i+1][1])**2) * 111000
                    dists_m.append(d)

                total_m = float(np.sum(dists_m))
                walk_m = sum(d for d in dists_m if d < 500)
                bike_m = sum(d for d in dists_m if 500 <= d < 2000)
                car_m = sum(d for d in dists_m if d >= 2000)
                day["transit_summary"] = (
                    f"交通汇总\n"
                    f"{'─'*30}\n"
                    f"总移动距离: {total_m:.0f}m\n"
                    f"步行: {walk_m:.0f}m\n"
                    f"骑行: {bike_m:.0f}m\n"
                    f"车辆出行: 共{car_m:.0f}m\n"
                    f"预估总花费: 0元"
                )
            else:
                day["timeline"] = ""
                day["transit_summary"] = ""

            # 规范化 pois
            pois = day.get("pois", {})
            if pois:
                # 将名称 key 转为数字 key
                fixed_pois = {}
                for seq, name in enumerate(names, 1):
                    # 尝试匹配 pois 中的描述
                    desc = pois.get(str(seq), "") or pois.get(name, "")
                    if not desc:
                        # 从其他 key 模糊匹配
                        for k, v in pois.items():
                            if name in str(k) or str(k) in name:
                                desc = v
                                break
                    fixed_pois[str(seq)] = desc or f"{name}，值得一游"
                day["pois"] = fixed_pois

        n_pois = sum(len(d.get("itinerary", "").split("->")) for d in result.get("days", []))
        print(f"✅ {elapsed:.0f}s, {n_pois} POIs")

        self._save_result(result)
        self._save_maps(result)
        return result

    # ========== 地图生成 ==========

    def _save_maps(self, result: dict):
        """为 Bare 结果生成 Folium 地图（和 ROAM 格式一致）"""
        import folium
        import datetime as dt
        import pandas as pd

        ts = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
        day_dfs = []

        for day_idx, day in enumerate(result.get("days", [])):
            itinerary = day.get("itinerary", "")
            names = [n.strip() for n in itinerary.replace("→", "->").split("->") if n.strip()]
            matched_ids = self._match_poi_names(names)
            if len(matched_ids) < 2:
                continue

            df_day = self._all_city_data.loc[matched_ids].copy()
            day_dfs.append(df_day)

            try:
                m = folium.Map(
                    location=[df_day["lat"].mean(), df_day["lon"].mean()],
                    zoom_start=13, tiles=None
                )
                folium.TileLayer(
                    tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                    attr='高德地图', subdomains='1234', max_zoom=18
                ).add_to(m)
                for _, row in df_day.iterrows():
                    folium.Marker(
                        location=[row["lat"], row["lon"]],
                        popup=row["name"],
                        icon=folium.Icon(color="blue", icon="info-sign")
                    ).add_to(m)
                points = list(zip(df_day["lat"], df_day["lon"]))
                folium.PolyLine(points, color="green", weight=2.5, opacity=1).add_to(m)
                day_path = os.path.join(self.output_dir, f"{ts}_zh_day{day_idx+1}.html")
                m.save(day_path)
            except Exception as e:
                print(f"[bare] Day{day_idx+1} 地图失败: {e}")

        # 总览地图
        if day_dfs:
            try:
                all_lats, all_lons = [], []
                m = folium.Map(location=[31.23, 121.47], zoom_start=12, tiles=None)
                folium.TileLayer(
                    tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                    attr='高德地图', subdomains='1234', max_zoom=18
                ).add_to(m)
                colors = ["green", "blue", "purple", "orange", "red", "teal", "darkblue"]
                for di, df_day in enumerate(day_dfs):
                    color = colors[di % len(colors)]
                    for _, row in df_day.iterrows():
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]], radius=5,
                            color=color, fill=True, popup=f"Day{di+1}: {row['name']}"
                        ).add_to(m)
                        all_lats.append(row["lat"]); all_lons.append(row["lon"])
                    pts = list(zip(df_day["lat"], df_day["lon"]))
                    folium.PolyLine(pts, color=color, weight=2.5, opacity=1,
                                    popup=f"Day {di+1}").add_to(m)
                if all_lats:
                    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
                ov_path = os.path.join(self.output_dir, f"{ts}_zh_overview.html")
                m.save(ov_path)
                print(f"🗺️  地图已保存: {ts}_zh_day*.html + _overview.html ({len(day_dfs)+1} 张)")
            except Exception as e:
                print(f"[bare] 总览地图失败: {e}")

    def _build_simple_timeline(self, matched_ids: list, names: list) -> str:
        """Bare 专用：不含营业时间的简化时间线，纯按距离排列每个 POI 45min"""
        locs = self._all_city_data.loc[matched_ids][["lon", "lat"]].astype(float).to_numpy()
        dists_m = []
        for i in range(len(matched_ids) - 1):
            key = (min(matched_ids[i], matched_ids[i+1]), max(matched_ids[i], matched_ids[i+1]))
            if hasattr(self.spatial_handler, '_road_dist_cache') and key in self.spatial_handler._road_dist_cache:
                d = self.spatial_handler._road_dist_cache[key]
            else:
                d = np.sqrt((locs[i][0]-locs[i+1][0])**2 + (locs[i][1]-locs[i+1][1])**2) * 111000
            dists_m.append(d)

        START = int(self.start_time.split(":")[0]) * 60 + int(self.start_time.split(":")[1])
        SPEED_WALK = 80; SPEED_BIKE = 250; SPEED_CAR = 500
        lines = ["### 🕐 预计时间线（无营业时间约束）"]
        t = START
        for i, (mid, name) in enumerate(zip(matched_ids, names)):
            if i == 0:
                travel_str = "起点"
            else:
                d = dists_m[i-1]
                if d < 500:
                    travel = max(1, int(d / SPEED_WALK)); travel_str = f"🚶 {travel}min"
                elif d < 2000:
                    travel = max(1, int(d / SPEED_BIKE)); travel_str = f"🚲 {travel}min"
                else:
                    travel = max(1, int(d / SPEED_CAR)); travel_str = f"🚗 {travel}min"
                t += travel
            h_arr, m_arr = divmod(t, 60); h_dep, m_dep = divmod(t + 45, 60)
            lines.append(f"  {h_arr:02d}:{m_arr:02d}-{h_dep:02d}:{m_dep:02d}  {name}  ({travel_str} | 停留45min | 营业 00:00-23:59)")
            t += 45
        lines.append(f"  🏁 {t//60:02d}:{t%60:02d} 行程结束")
        return "\n".join(lines)

    # ========== 辅助方法 ==========

    def _build_bare_prompt(self, poi_db: str) -> str:
        """构建裸 LLM 的 prompt（包含用户具体需求）"""
        # 获取用户需求
        user_req = self.user_reqs[0] if self.user_reqs else f"体验{self.city_zh}的历史文化与美食"
        return f"""你是一位资深旅行规划师。以下是四个城市（福州、广州、深圳、上海）的全部景点数据库：

{poi_db}

请为游客规划一个 **{self.city_zh}** {self.num_days} 天的行程。

【游客的具体要求】
{user_req}

【重要】只选择标记为 [{self.city}] 的景点，不要选其他城市的景点。必须严格遵守游客的所有具体要求。

你需要自己决定：
1. 每天去哪些景点（每天 5-8 个，景点不能跨天重复）
2. 每天的游览顺序（考虑地理位置邻近性，避免来回绕路）
3. 每天的主题和推荐理由

以严格 JSON 格式返回，不要任何额外文字：

{{
  "trip_title": "行程标题",
  "trip_overview": "行程总体介绍（50字内）",
  "days": [
    {{
      "day": 1,
      "day_theme": "当日主题",
      "itinerary": "景点A -> 景点B -> 景点C",
      "day_reason": "当日行程推荐理由（30字内）",
      "pois": {{"1": "景点A的20字简介", "2": "景点B的20字简介"}}
    }},
    ...
  ]
}}

重要规则：
- itinerary 中的景点名必须从数据库中选择，名称完全一致
- 景点顺序考虑地理位置，避免来回绕路
- pois 的 key 用数字序号 1,2,3...，value 写 20 字以内的简介
- 每个景点只能在一天出现，不能跨天重复
- **只选 [{self.city}] 的景点**
- 只返回 JSON，不要任何额外解释或 markdown 代码块标记
"""

    def _match_poi_names(self, names: list) -> list:
        """将景点名称列表匹配为 POI ID 列表（仅限目标城市）"""
        from thefuzz import process as fz
        # 仅搜索目标城市的 POI，避免跨城同名匹配
        city_mask = self._all_city_data["_city"] == self.city
        city_data = self._all_city_data[city_mask]
        all_names = city_data["name"].tolist()
        matched = []
        for name in names:
            exact = city_data[city_data["name"] == name]
            if len(exact) > 0:
                matched.append(exact.index[0])
            else:
                matches = fz.extract(name, all_names, limit=1)
                if matches and matches[0][1] > 80:
                    best = matches[0][0]
                    idx = all_names.index(best)
                    # 映射回全局 index
                    matched.append(city_data.index[idx])
        return matched

    def _build_fallback_result(self, allpoi_ids: list) -> dict:
        """LLM 失败时的回退结果（简单均分 POI 到每天）"""
        days = []
        n_per_day = max(5, len(allpoi_ids) // self.num_days)
        for d in range(self.num_days):
            start = d * n_per_day
            end = start + n_per_day if d < self.num_days - 1 else len(allpoi_ids)
            day_ids = allpoi_ids[start:end]
            names = [str(self._all_city_data.loc[pid, "name"]) for pid in day_ids if pid in self._all_city_data.index]
            days.append({
                "day": d + 1,
                "day_theme": f"第{d+1}天",
                "itinerary": " -> ".join(names),
                "day_reason": "自动生成（LLM 未返回有效结果）",
                "weather_note": "无",
                "pois": {str(i+1): name for i, name in enumerate(names)},
            })
        return {
            "trip_title": f"{self.city_zh}{self.num_days}日游",
            "trip_overview": "",
            "days": days,
        }
