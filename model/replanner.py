"""
Replanner Agent: 基于 function calling 的行程对话与修改
=========================================================
用户在生成初始行程后，可用自然语言查看、修改、优化行程。
LLM 通过 function calling 自主调用工具完成查询/替换/删除/添加/重排/展示。

用法:
    from model.replanner import Replanner
    rp = Replanner(agent, result_json_path)
    resp, steps = rp.replan("把第一天下午的公园换成博物馆")
"""
import json
import copy
import datetime
import numpy as np
import requests
from typing import Optional
from thefuzz import process as fuzz_process
from model.config import CITY_EN_MAP, WTTR_WEATHER_MAP, WEATHER_ACTIVITY_MAP


# ==================== Function Calling 工具定义 ====================

TOOLS = [
    # ── 查询类工具 ──
    {
        "type": "function",
        "function": {
            "name": "search_pois",
            "description": "根据关键词在城市POI库中搜索景点，返回匹配的POI列表（名称+类别+评分+地址+营业时间）",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词，如'博物馆'、'公园'、'美食'、'海边'"},
                    "category": {"type": "string", "description": "限定类别，如'博物馆/纪念馆'、'自然风光'、'美食街区'等，可为空"},
                    "top_k": {"type": "integer", "description": "返回前K个结果，默认5", "default": 5}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi_detail",
            "description": "获取某个POI的详细信息（营业时间、地址、评分、类别、建议游览时长）",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {"type": "string", "description": "POI名称"}
                },
                "required": ["poi_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_distance",
            "description": "获取两个POI之间的驾车距离和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_poi": {"type": "string", "description": "起点POI名称"},
                    "to_poi": {"type": "string", "description": "终点POI名称"}
                },
                "required": ["from_poi", "to_poi"]
            }
        }
    },
    # ── 修改类工具 ──
    {
        "type": "function",
        "function": {
            "name": "replace_poi",
            "description": "替换行程中某天的某个景点为另一个景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based）"},
                    "old_poi": {"type": "string", "description": "要替换掉的景点名称"},
                    "new_poi": {"type": "string", "description": "新加入的景点名称"}
                },
                "required": ["day", "old_poi", "new_poi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_poi",
            "description": "从行程中删除某个景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based）"},
                    "poi": {"type": "string", "description": "要删除的景点名称"}
                },
                "required": ["day", "poi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_poi",
            "description": "在行程中插入一个新景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based）"},
                    "poi": {"type": "string", "description": "要添加的景点名称"},
                    "position": {"type": "integer", "description": "插入位置（1=第一个之前，-1=末尾），默认-1"}
                },
                "required": ["day", "poi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reroute_day",
            "description": "对某天的行程重新做TSP路径优化排序",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based）"}
                },
                "required": ["day"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_time_feasibility",
            "description": "检查某天行程的时间窗是否合理（是否会关门/太赶/时间不够），返回预计时间线",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based）"}
                },
                "required": ["day"]
            }
        }
    },
    # ── 展示类工具 ──
    {
        "type": "function",
        "function": {
            "name": "show_timeline",
            "description": "展示某天的详细时间线（几点到几点在哪、交通方式、停留时长）",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based），不传则展示所有天"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_transit",
            "description": "展示某天的交通汇总（步行/骑行/驾车距离、预估费用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based），不传则展示所有天"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_weather",
            "description": "实时查询并展示行程期间的天气预报（调用 wttr.in 在线 API，非缓存数据）。用户问'天气怎么样'、'会下雨吗'时优先调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_itinerary",
            "description": "展示某天的行程路线（景点列表 + 主题 + 推荐理由）",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based），不传则展示所有天"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_trip_overview",
            "description": "展示整个行程的概览（标题、总览、每日主题摘要、住宿信息）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_map",
            "description": "生成并展示行程的交互式地图（每日路线图+总览图）。调用后地图会自动渲染到页面上。",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "第几天（1-based），不传则生成所有天的地图+总览图"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_replan",
            "description": "完成修改，保存结果到文件。做了任何景点增删改后必须调用此函数来持久化。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "修改摘要，描述做了哪些改动"}
                },
                "required": ["summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_day",
            "description": "在行程末尾新增一天（空行程），用于扩容。新增后可用 add_poi 往里面加景点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "description": "新一天的推荐主题，如'自然风光日'、'美食探索日'"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_day",
            "description": "删除某一天（缩容）。该天的所有景点会被移除，后续天数自动前移。",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "description": "要删除的第几天（1-based）"}
                },
                "required": ["day"]
            }
        }
    },
    # ── 住宿管理 ──
    {
        "type": "function",
        "function": {
            "name": "show_hotel",
            "description": "查看当前住宿点信息（名称、地址、类别）以及周边步行/短途可达的推荐景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_distance": {"type": "integer", "description": "周边搜索最大距离（米），默认3000", "default": 3000}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_hotel",
            "description": "修改住宿点（模糊匹配POI库），更改后自动更新周边推荐",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel_name": {"type": "string", "description": "新的住宿点名称，如'三坊七巷附近'、'天河城'"}
                },
                "required": ["hotel_name"]
            }
        }
    },
    # ── 排除景点管理 ──
    {
        "type": "function",
        "function": {
            "name": "show_exclude_pois",
            "description": "查看当前已排除的景点列表（这些景点在搜索和添加时会被过滤掉）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_exclude_poi",
            "description": "添加一个景点到排除列表（模糊匹配POI库），后续搜索将自动过滤此景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {"type": "string", "description": "要排除的景点名称"}
                },
                "required": ["poi_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_exclude_poi",
            "description": "从排除列表中移除一个景点，使其重新可被搜索和添加",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {"type": "string", "description": "要取消排除的景点名称"}
                },
                "required": ["poi_name"]
            }
        }
    },
    # ── 搜索增强 ──
    {
        "type": "function",
        "function": {
            "name": "amap_search",
            "description": "通过高德地图API在线搜索POI（当本地POI库搜索无结果时使用）。返回名称+坐标+地址。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"},
                    "city": {"type": "string", "description": "城市名（中文），默认使用当前行程城市"}
                },
                "required": ["keyword"]
            }
        }
    },
    # ── 行程重生成 ──
    {
        "type": "function",
        "function": {
            "name": "regenerate_trip_overview",
            "description": "重新生成行程概览（标题、总览描述、每日主题、每日推荐理由）。在增删改景点后调用，使描述与最新行程一致。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # ── 时间设置 ──
    {
        "type": "function",
        "function": {
            "name": "set_trip_time",
            "description": "修改每天的出发时间和结束时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "每天出发时间，如'08:00'、'09:30'"},
                    "end_time": {"type": "string", "description": "每天结束时间，如'20:00'、'21:00'"}
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
]

SYSTEM_PROMPT = """你是一个专业的旅游行程助手，拥有完整的工具链来帮用户查看和修改行程。默认使用简体中文进行回复，除非用户指定使用的语言

你有以下工具：

📖 查询工具：
- show_trip_overview: 查看行程概览
- show_itinerary: 查看某天行程路线
- show_timeline: 查看某天详细时间线
- show_transit: 查看某天交通汇总
- show_weather: 实时查询天气预报（在线 API，非缓存）
- show_hotel: 查看住宿点+周边推荐
- show_exclude_pois: 查看已排除的景点
- show_map: 生成交互式地图（每日路线+总览）
- search_pois: 搜索本地景点库
- amap_search: 高德地图在线搜索（本地库无结果时用）
- get_poi_detail: 查看景点详情
- get_route_distance: 查两点间距离

✏️ 修改工具：
- replace_poi: 替换景点
- remove_poi: 删除景点
- add_poi: 添加景点（只能加在已有天数内）
- add_day: 新增一天（扩容）
- remove_day: 删除某一天（缩容）
- change_hotel: 修改住宿点
- add_exclude_poi: 排除某个景点（不再推荐）
- remove_exclude_poi: 取消排除
- set_trip_time: 修改每天出发/结束时间
- reroute_day: 重新优化路径顺序
- check_time_feasibility: 检查时间是否合理
- regenerate_trip_overview: 重新生成行程概览/主题/理由
- finish_replan: 保存修改

⚠️ 关键行为准则（严格遵循）：

1. 【工具调用时】调用工具时 content 必须为空字符串 "" 或仅写一句极简短的话（≤10字）。绝对禁止在 content 里写"好的！"、"搞定！"、"完美！"、"来看看效果"、"现在开始..."等流水账旁白。

2. 【最终回复】当所有工具调用完成后，输出一个干净、结构化的 Markdown 回复。最终回复里绝对禁止出现：
   - 操作过程的流水账（"先删掉再重建"、"调了几次顺序"）
   - 对工具返回结果的逐条点评（"reroute把顺序打乱了"、"时间检查通过了"）
   - 任何"让我来..."、"我来手动调整"之类的自言自语
   最终回复只包含：用户要的信息 + 最终结果。

3. 【修改景点后】必须：reroute_day → check_time_feasibility → regenerate_trip_overview → finish_replan

4. 【回答风格】简洁、结构化，用 Markdown 格式。直接给出结果，不废话。"""


# ==================== Replanner 类 ====================

class Replanner:
    """行程重规划 Agent"""

    def __init__(self, agent, result_json_path: str):
        """
        Args:
            agent: TravelAgent 实例（需要其 proxy、spatial_handler、user_favorites）
            result_json_path: 已保存的行程 JSON 文件路径
        """
        self.agent = agent
        self.proxy = agent.proxy
        self.data = agent.user_favorites  # POI DataFrame
        self.spatial = agent.spatial_handler
        self._SYSTEM_PROMPT = SYSTEM_PROMPT  # 暴露给外部状态机使用

        # 加载行程
        with open(result_json_path, "r", encoding="utf-8") as f:
            self.result = json.load(f)
        self.json_path = result_json_path

        # 构建 POI 名称→DataFrame索引 映射
        self.name_to_idx = {}
        for idx, row in self.data.iterrows():
            self.name_to_idx[row["name"]] = idx

    # ========== 工具实现 ==========

    def _search_pois(self, keyword: str, category: str = None, top_k: int = 5) -> str:
        """搜索POI（使用模糊匹配，支持多词搜索）"""
        from thefuzz import process as fz
        all_names = list(self.name_to_idx.keys())
        # 直接用 thefuzz 模糊匹配，不要求 keyword 是 name 的子串
        raw_matches = fz.extract(keyword, all_names, limit=min(top_k * 3, len(all_names)))
        results = []
        for name, score in raw_matches:
            if score < 50:  # 过滤低分匹配
                continue
            idx = self.name_to_idx[name]
            row = self.data.loc[idx]
            if category and category not in str(row.get("category", "")):
                continue
            results.append({
                "name": name,
                "category": str(row.get("category", "")),
                "rating": float(row.get("rating", 0)) if row.get("rating") not in [None, "", float("nan")] else 0,
                "address": str(row.get("address", "")),
                "open_time": str(row.get("open_time", "")),
                "close_time": str(row.get("close_time", "")),
                "_score": score,
            })
        # 按评分排序
        results.sort(key=lambda x: x["rating"], reverse=True)
        # 移除内部字段
        for r in results:
            r.pop("_score", None)
        return json.dumps(results[:top_k], ensure_ascii=False, indent=2)

    def _get_poi_detail(self, poi_name: str) -> str:
        """获取POI详情"""
        idx = self._fuzzy_match(poi_name)
        if idx is None:
            return json.dumps({"error": f"未找到景点: {poi_name}"})
        row = self.data.loc[idx]
        return json.dumps({
            "name": str(row["name"]),
            "category": str(row.get("category", "")),
            "address": str(row.get("address", "")),
            "rating": float(row.get("rating", 0)) if row.get("rating") not in [None, "", float("nan")] else 0,
            "open_time": str(row.get("open_time", "")),
            "close_time": str(row.get("close_time", "")),
            "visittime_min": int(row.get("visittime", 60)),
        }, ensure_ascii=False)

    def _get_route_distance(self, from_poi: str, to_poi: str) -> str:
        """获取两点间距离"""
        idx1 = self._fuzzy_match(from_poi)
        idx2 = self._fuzzy_match(to_poi)
        if idx1 is None or idx2 is None:
            return json.dumps({"error": "POI未找到"})

        # 尝试 OSRM
        lon1, lat1 = float(self.data.loc[idx1, "lon"]), float(self.data.loc[idx1, "lat"])
        lon2, lat2 = float(self.data.loc[idx2, "lon"]), float(self.data.loc[idx2, "lat"])

        import numpy as np
        key = (min(idx1, idx2), max(idx1, idx2))
        if hasattr(self.spatial, '_road_dist_cache') and key in self.spatial._road_dist_cache:
            dist_m = self.spatial._road_dist_cache[key]
        else:
            dist_m = np.sqrt((lon1-lon2)**2 + (lat1-lat2)**2) * 111000

        travel_min = int(dist_m / 500)  # ~30km/h
        return json.dumps({
            "from": from_poi, "to": to_poi,
            "distance_m": int(dist_m),
            "travel_min": travel_min,
        }, ensure_ascii=False)

    def _replace_poi(self, day: int, old_poi: str, new_poi: str) -> str:
        """替换景点"""
        days = self.result.get("days", [])
        if day < 1 or day > len(days):
            return json.dumps({"error": f"无效天数: {day}"})

        day_data = days[day - 1]
        itinerary = day_data.get("itinerary", "")
        pois = [s.strip() for s in itinerary.split("->")]

        old_match = self._fuzzy_match_in_list(old_poi, pois)
        if old_match is None:
            return json.dumps({"error": f"行程中未找到: {old_poi}"})

        new_idx = self._fuzzy_match(new_poi)
        if new_idx is None:
            return json.dumps({"error": f"POI库中未找到: {new_poi}"})

        new_name = self.data.loc[new_idx, "name"]
        pois[pois.index(old_match)] = new_name
        day_data["itinerary"] = " -> ".join(pois)

        # 更新 pois dict
        if day_data.get("pois"):
            old_pois = day_data["pois"]
            new_pois = {}
            for k, v in old_pois.items():
                if k == old_match:
                    new_pois[new_name] = f"{new_name}的推荐理由。"
                else:
                    new_pois[k] = v
            day_data["pois"] = new_pois

        return json.dumps({"success": True, "day": day,
                           "old": old_match, "new": new_name,
                           "itinerary": day_data["itinerary"]}, ensure_ascii=False)

    def _remove_poi(self, day: int, poi: str) -> str:
        """删除景点"""
        days = self.result.get("days", [])
        day_data = days[day - 1]
        itinerary = day_data.get("itinerary", "")
        pois = [s.strip() for s in itinerary.split("->")]

        match = self._fuzzy_match_in_list(poi, pois)
        if match is None:
            return json.dumps({"error": f"行程中未找到: {poi}"})

        pois.remove(match)
        day_data["itinerary"] = " -> ".join(pois)

        if day_data.get("pois") and match in day_data["pois"]:
            del day_data["pois"][match]

        return json.dumps({"success": True, "day": day, "removed": match,
                           "itinerary": day_data["itinerary"]}, ensure_ascii=False)

    def _add_poi(self, day: int, poi: str, position: int = -1) -> str:
        """添加景点"""
        days = self.result.get("days", [])
        if day < 1 or day > len(days):
            return json.dumps({"error": f"无效天数: {day}（当前共{len(days)}天）"})
        day_data = days[day - 1]
        itinerary = day_data.get("itinerary", "")
        pois = [s.strip() for s in itinerary.split("->") if s.strip()]

        new_idx = self._fuzzy_match(poi)
        if new_idx is None:
            return json.dumps({"error": f"POI库中未找到: {poi}"})

        new_name = self.data.loc[new_idx, "name"]
        if position < 0 or position > len(pois):
            pois.append(new_name)
        else:
            pois.insert(position - 1, new_name)

        day_data["itinerary"] = " -> ".join(pois)
        if day_data.get("pois") is not None:
            day_data["pois"][new_name] = f"{new_name}的推荐理由。"
        return json.dumps({"success": True, "day": day, "added": new_name,
                           "itinerary": day_data["itinerary"]}, ensure_ascii=False)

    def _reroute_day(self, day: int) -> str:
        """重排某天路径"""
        days = self.result.get("days", [])
        if day < 1 or day > len(days):
            return json.dumps({"error": f"无效天数: {day}（当前共{len(days)}天）"})
        day_data = days[day - 1]
        itinerary = day_data.get("itinerary", "")
        pois = [s.strip() for s in itinerary.split("->") if s.strip()]

        ids = []
        for p in pois:
            idx = self._fuzzy_match(p)
            if idx is not None:
                ids.append(idx)

        if len(ids) >= 2:
            order, _, _ = self.spatial.get_tsp_order(ids)
            new_pois = [self.data.loc[ids[o], "name"] for o in order]
            day_data["itinerary"] = " -> ".join(new_pois)
            return json.dumps({"success": True, "day": day,
                               "new_order": day_data["itinerary"]}, ensure_ascii=False)

        return json.dumps({"error": "POI数量不足，无法重排"})

    def _add_day(self, theme: str = "") -> str:
        """新增一天"""
        days = self.result.get("days", [])
        new_day_num = len(days) + 1
        new_day = {
            "day": new_day_num,
            "day_theme": theme or f"第{new_day_num}天",
            "itinerary": "",
            "day_reason": "",
            "weather_note": "",
            "pois": {},
        }
        days.append(new_day)
        self.result["num_days"] = len(days)
        return json.dumps({
            "success": True,
            "new_day": new_day_num,
            "total_days": len(days),
            "message": f"已添加第{new_day_num}天，共{len(days)}天。请用 add_poi 往里加景点。"
        }, ensure_ascii=False)

    def _remove_day(self, day: int) -> str:
        """删除某一天"""
        days = self.result.get("days", [])
        if day < 1 or day > len(days):
            return json.dumps({"error": f"无效天数: {day}（当前共{len(days)}天）"})
        removed = days.pop(day - 1)
        removed_pois = list(removed.get("pois", {}).keys())
        # 重新编号后续天数
        for i, d in enumerate(days):
            d["day"] = i + 1
        self.result["num_days"] = len(days)
        return json.dumps({
            "success": True,
            "removed_day": day,
            "removed_pois": removed_pois,
            "total_days": len(days),
            "message": f"已删除第{day}天（含{len(removed_pois)}个景点），现共{len(days)}天"
        }, ensure_ascii=False)

    def _check_time_feasibility(self, day: int) -> str:
        """检查时间可行性"""
        days = self.result.get("days", [])
        if day < 1 or day > len(days):
            return json.dumps({"error": f"无效天数: {day}（当前共{len(days)}天）"})
        day_data = days[day - 1]
        # 复用 agent 的 timeline 方法
        timeline = self.agent._build_timeline(day_data, [], day - 1)
        return json.dumps({"day": day, "timeline": timeline}, ensure_ascii=False)

    def _finish_replan(self, summary: str) -> str:
        """完成修改，保存JSON"""
        self.result["replan_summary"] = summary
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.result, f, ensure_ascii=False, indent=2)
        return json.dumps({"success": True, "message": f"已保存: {self.json_path}",
                           "summary": summary}, ensure_ascii=False)

    # ========== 展示类工具 ==========

    def _show_timeline(self, day: int = None) -> str:
        """展示时间线"""
        days = self.result.get("days", [])
        if day is not None:
            if day < 1 or day > len(days):
                return json.dumps({"error": f"无效天数: {day}"})
            return json.dumps({
                "day": day,
                "day_theme": days[day - 1].get("day_theme", ""),
                "timeline": days[day - 1].get("timeline", "暂无时间线")
            }, ensure_ascii=False, indent=2)
        else:
            all_timelines = []
            for d in days:
                all_timelines.append({
                    "day": d["day"],
                    "day_theme": d.get("day_theme", ""),
                    "timeline": d.get("timeline", "暂无")
                })
            return json.dumps(all_timelines, ensure_ascii=False, indent=2)

    def _show_transit(self, day: int = None) -> str:
        """展示交通汇总"""
        days = self.result.get("days", [])
        if day is not None:
            if day < 1 or day > len(days):
                return json.dumps({"error": f"无效天数: {day}"})
            return json.dumps({
                "day": day,
                "day_theme": days[day - 1].get("day_theme", ""),
                "transit_summary": days[day - 1].get("transit_summary", "暂无交通信息")
            }, ensure_ascii=False, indent=2)
        else:
            all_transit = []
            for d in days:
                all_transit.append({
                    "day": d["day"],
                    "day_theme": d.get("day_theme", ""),
                    "transit_summary": d.get("transit_summary", "暂无")
                })
            return json.dumps(all_transit, ensure_ascii=False, indent=2)

    def _show_weather(self) -> str:
        """实时查询行程期间天气预报（wttr.in），失败时回退到缓存数据"""
        city_zh = self.result.get("city_zh", "") or self.result.get("city", "")
        city_en = CITY_EN_MAP.get(city_zh, city_zh)
        num_days = self.result.get("num_days", len(self.result.get("days", [])))

        # ── 尝试实时拉取 wttr.in ──
        live_forecasts = []
        try:
            url = f"https://wttr.in/{city_en}?format=j1"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            weather_list = data.get("weather", [])
            if weather_list:
                for i, day in enumerate(weather_list):
                    if i >= num_days:
                        break
                    date = datetime.date.fromisoformat(day["date"])
                    hourly = day.get("hourly", [{}])
                    mid = hourly[min(4, len(hourly) - 1)]
                    desc_en = mid.get("weatherDesc", [{}])[0].get("value", "Sunny")
                    weather = WTTR_WEATHER_MAP.get(desc_en.lower(), "多云⛅")
                    live_forecasts.append({
                        "day": i + 1,
                        "date": day["date"],
                        "day_of_week": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.weekday()],
                        "weather": weather,
                        "temp_high": int(day.get("maxtempC", 25)),
                        "temp_low": int(day.get("mintempC", 18)),
                        "humidity": int(mid.get("humidity", 60)),
                        "activity_advice": WEATHER_ACTIVITY_MAP.get(weather, "天气适宜出行"),
                        "source": "wttr.in 实时",
                    })
        except Exception:
            pass  # 失败则回退到缓存

        # ── 回退：读缓存数据 ──
        if not live_forecasts:
            days = self.result.get("days", [])
            for d in days:
                wc = d.get("weather_context", "")
                if wc:
                    live_forecasts.append({
                        "day": d["day"],
                        "weather_context": wc,
                        "source": "历史缓存",
                    })
            if not live_forecasts:
                rag = self.result.get("rag_context", "")
                return json.dumps({
                    "source": "无实时数据，缓存也为空",
                    "rag_summary": rag[:500] if rag else "无天气数据",
                    "hint": "请检查网络连接后重试"
                }, ensure_ascii=False, indent=2)

        return json.dumps({
            "city": city_zh,
            "num_days": num_days,
            "source": live_forecasts[0].get("source", "未知"),
            "forecasts": live_forecasts,
        }, ensure_ascii=False, indent=2)

    def _show_itinerary(self, day: int = None) -> str:
        """展示行程路线"""
        days = self.result.get("days", [])
        if day is not None:
            if day < 1 or day > len(days):
                return json.dumps({"error": f"无效天数: {day}"})
            d = days[day - 1]
            return json.dumps({
                "day": d["day"],
                "day_theme": d.get("day_theme", ""),
                "itinerary": d.get("itinerary", ""),
                "day_reason": d.get("day_reason", ""),
                "weather_note": d.get("weather_note", "无"),
                "pois": d.get("pois", {})
            }, ensure_ascii=False, indent=2)
        else:
            all_info = []
            for d in days:
                all_info.append({
                    "day": d["day"],
                    "day_theme": d.get("day_theme", ""),
                    "itinerary": d.get("itinerary", ""),
                    "day_reason": d.get("day_reason", ""),
                    "poi_count": len(d.get("pois", {}))
                })
            return json.dumps(all_info, ensure_ascii=False, indent=2)

    def _show_trip_overview(self) -> str:
        """展示行程概览"""
        return json.dumps({
            "trip_title": self.result.get("trip_title", ""),
            "trip_overview": self.result.get("trip_overview", ""),
            "num_days": self.result.get("num_days", 0),
            "city": self.result.get("city_zh", self.result.get("city", "")),
            "hotel": self.result.get("hotel"),
            "hotel_nearby": self.result.get("hotel_nearby", []),
            "daily_summary": self.result.get("daily_summary", []),
            "exclude": self.result.get("exclude"),
        }, ensure_ascii=False, indent=2)

    def _show_map(self, day: int = None) -> str:
        """生成并展示交互式地图"""
        # 检查 agent 的 user_favorites 与行程中的 POI 是否匹配
        trip_city = self.result.get("city", "") or self.result.get("city_zh", "")
        agent_city = getattr(self.agent, 'city', '?')
        if trip_city and agent_city != '?' and trip_city != agent_city:
            return json.dumps({
                "error": f"Agent 数据城市({agent_city})与行程城市({trip_city})不匹配，请刷新页面重试"
            }, ensure_ascii=False)

        # 诊断：打印 agent 数据信息
        uf = getattr(self.agent, 'user_favorites', None)
        if uf is not None:
            print(f"[_show_map] agent.user_favorites: {uf.shape[0]} rows, city={agent_city}")
            print(f"[_show_map] names sample: {uf['name'].tolist()[:5]}")
        else:
            print("[_show_map] ⚠️ agent.user_favorites is None!")

        map_paths = self.agent.refresh_trip_rendering(self.result)
        print(f"[_show_map] got map_paths: overview={map_paths.get('map_overview')}, days={len(map_paths.get('map_days',[]))}")

        if day is not None:
            day_idx = day - 1
            if 0 <= day_idx < len(map_paths.get("map_days", [])):
                return json.dumps({
                    "success": True,
                    "day": day,
                    "map_generated": True,
                    "message": f"第{day}天地图已生成，请查看页面顶部的地图面板"
                }, ensure_ascii=False)
            else:
                total_days = len(map_paths.get("map_days", []))
                if total_days == 0:
                    days_data = self.result.get('days', [])
                    day_info = ""
                    if days_data:
                        d0 = days_data[0]
                        day_info = f" Day1 itinerary: {d0.get('itinerary','?')[:80]}"
                    return json.dumps({
                        "error": f"第{day}天没有地图数据（共{len(days_data)}天行程，但地图生成为0天）。{day_info}"
                                 f" 请用 search_pois 确认景点名称是否正确。"
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        "error": f"第{day}天没有地图数据（仅有第1-{total_days}天的地图）"
                    }, ensure_ascii=False)
        return json.dumps({
            "success": True,
            "map_generated": True,
            "total_days": len(map_paths.get("map_days", [])),
            "message": "全天地图+总览图已生成，请查看页面顶部的地图面板"
        }, ensure_ascii=False)

    # ========== 辅助方法 ==========

    def _fuzzy_match(self, name: str) -> Optional[int]:
        """模糊匹配POI名→DataFrame索引"""
        if name in self.name_to_idx:
            return self.name_to_idx[name]
        all_names = list(self.name_to_idx.keys())
        match, score = fuzz_process.extract(name, all_names, limit=1)[0]
        if score > 70:
            return self.name_to_idx[match]
        return None

    def _fuzzy_match_in_list(self, name: str, candidates: list) -> Optional[str]:
        """在候选列表里模糊匹配"""
        if name in candidates:
            return name
        match, score = fuzz_process.extract(name, candidates, limit=1)[0]
        return match if score > 70 else None

    # ========== 主循环 ==========

    def replan(self, user_input: str, max_turns: int = 10, callback=None):
        """处理一次对话（查询或修改行程）
        Args:
            user_input: 用户输入
            max_turns: 最大对话轮数
            callback: 可选回调函数 callback(step_dict)，每步即时调用（用于流式展示）
        返回: (final_text, steps_list)
        """
        steps = []
        modified = False
        city_zh = self.result.get("city_zh", "")
        city = self.result.get("city", "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + f"\n\n📍 当前城市: {city_zh} ({city})。所有搜索和添加景点操作仅限于此城市。"},
            {"role": "user", "content": f"当前行程 ({city_zh}):\n{json.dumps(self.result.get('days', []), ensure_ascii=False, indent=2)}\n\n用户请求: {user_input}"}
        ]

        def _emit(step: dict):
            """记录步骤并回调前端"""
            steps.append(step)
            if callback:
                callback(step)

        for turn in range(max_turns):
            resp, reasoning = self.proxy.chat(messages, model=self.agent.MODEL, tools=TOOLS)

            if reasoning:
                _emit({"type": "reasoning", "content": reasoning})

            if isinstance(resp, str):
                if modified:
                    self._auto_save()
                return resp, steps

            msg = resp
            messages.append(msg)

            if hasattr(msg, 'content') and msg.content:
                _emit({"type": "text", "content": msg.content})

            for tc in msg.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)
                _emit({"type": "tool_call", "name": func_name, "args": args})

                if func_name in ("replace_poi", "remove_poi", "add_poi", "reroute_day"):
                    modified = True

                tool_result = self._call_tool(func_name, args)
                result_preview = tool_result[:2000] if len(tool_result) > 2000 else tool_result
                _emit({"type": "tool_result", "name": func_name, "content": result_preview})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

        if modified:
            self._auto_save()
        return "达到最大对话轮数，请简化你的请求。", steps

    def _auto_save(self):
        """自动保存当前结果到 JSON 文件"""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========== 住宿管理 ==========

    def _show_hotel(self, max_distance: int = 3000) -> str:
        """查看住宿点+周边推荐"""
        hotel = self.result.get("hotel")
        hotel_nearby = self.result.get("hotel_nearby", [])
        if not hotel:
            return json.dumps({
                "has_hotel": False,
                "message": "当前行程未设置住宿点。可以用 change_hotel 添加。"
            }, ensure_ascii=False, indent=2)

        # 如果有 agent，尝试实时计算周边
        nearby = []
        if hasattr(self.agent, '_get_hotel_nearby') and self.agent.hotel_poi_id is not None:
            raw = self.agent._get_hotel_nearby(max_distance=max_distance)
            for poi_id, name, cat, dist in raw:
                mode = "🚶步行" if dist < 1000 else "🚲骑行"
                nearby.append({"name": name, "category": cat, "distance_m": dist, "mode": mode})

        return json.dumps({
            "has_hotel": True,
            "hotel": hotel,
            "nearby_count": len(nearby) or len(hotel_nearby),
            "nearby": nearby if nearby else hotel_nearby,
        }, ensure_ascii=False, indent=2)

    def _change_hotel(self, hotel_name: str) -> str:
        """修改住宿点"""
        if not hasattr(self.agent, 'user_favorites') or self.agent.user_favorites is None:
            return json.dumps({"error": "Agent 数据不可用，无法匹配住宿点"})

        from thefuzz import process as fz
        all_names = self.agent.user_favorites["name"].tolist()
        match, score = fz.extract(hotel_name, all_names, limit=1)[0]

        if score < 60:
            return json.dumps({
                "error": f"未在POI库中找到匹配的住宿点: '{hotel_name}'",
                "best_match": match,
                "score": score,
                "hint": "请换一个更具体的地名或商圈名"
            }, ensure_ascii=False)

        # 更新 agent 的住宿信息
        idx = all_names.index(match)
        row = self.agent.user_favorites.loc[idx]
        self.agent.hotel_name = hotel_name
        self.agent.hotel_poi_id = idx
        self.agent.hotel_info = {
            "id": int(idx),
            "name": str(row["name"]),
            "address": str(row.get("address", "")),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "category": str(row.get("category", "")),
        }

        # 同步到 trip_result
        self.result["hotel"] = str(row["name"])
        nearby_raw = self.agent._get_hotel_nearby()
        self.result["hotel_nearby"] = [
            {"name": n, "category": c, "distance_m": d}
            for _, n, c, d in nearby_raw[:8]
        ]

        return json.dumps({
            "success": True,
            "hotel": str(row["name"]),
            "address": str(row.get("address", "")),
            "match_score": score,
            "nearby_count": len(self.result["hotel_nearby"]),
        }, ensure_ascii=False, indent=2)

    # ========== 排除景点管理 ==========

    def _show_exclude_pois(self) -> str:
        """查看已排除景点"""
        exclude = self.result.get("exclude")
        exclude_matched = getattr(self.agent, 'exclude_poi_matched', [])
        exclude_names = []
        for item in exclude_matched:
            if isinstance(item, tuple):
                exclude_names.append({"input": item[0], "matched": item[1], "score": item[2]})
        if not exclude_names and exclude:
            exclude_names = [{"input": exclude, "matched": exclude, "score": 100}]

        return json.dumps({
            "exclude_raw": exclude,
            "exclude_count": len(exclude_names),
            "exclude_list": exclude_names,
        }, ensure_ascii=False, indent=2)

    def _add_exclude_poi(self, poi_name: str) -> str:
        """添加排除景点"""
        if not hasattr(self.agent, 'user_favorites') or self.agent.user_favorites is None:
            return json.dumps({"error": "Agent 数据不可用"})

        from thefuzz import process as fz
        all_names = self.agent.user_favorites["name"].tolist()
        match, score = fz.extract(poi_name, all_names, limit=1)[0]

        if score < 60:
            return json.dumps({
                "error": f"未匹配到景点: '{poi_name}'",
                "best_match": match,
                "score": score,
            }, ensure_ascii=False)

        idx = all_names.index(match)
        poi_id = self.agent.user_favorites.index[idx]
        if not hasattr(self.agent, 'exclude_poi_ids'):
            self.agent.exclude_poi_ids = set()
        if not hasattr(self.agent, 'exclude_poi_matched'):
            self.agent.exclude_poi_matched = []

        self.agent.exclude_poi_ids.add(poi_id)
        self.agent.exclude_poi_matched.append((poi_name, match, score))

        # 同步到 result
        existing = self.result.get("exclude") or ""
        if existing:
            self.result["exclude"] = existing + ", " + match
        else:
            self.result["exclude"] = match

        return json.dumps({
            "success": True,
            "excluded": match,
            "score": score,
            "hint": "此景点将在后续搜索中被自动过滤",
        }, ensure_ascii=False, indent=2)

    def _remove_exclude_poi(self, poi_name: str) -> str:
        """移除排除景点"""
        exclude_matched = getattr(self.agent, 'exclude_poi_matched', [])
        from thefuzz import process as fz

        # 在已排除列表中模糊匹配
        matched_names = [item[1] if isinstance(item, tuple) else item for item in exclude_matched]
        if not matched_names:
            return json.dumps({"error": "当前没有已排除的景点"})

        match, score = fz.extract(poi_name, matched_names, limit=1)[0]
        if score < 60:
            return json.dumps({
                "error": f"在排除列表中未找到: '{poi_name}'",
                "current_excludes": matched_names,
            }, ensure_ascii=False)

        # 从 agent 中移除
        if hasattr(self.agent, 'exclude_poi_ids'):
            all_names = self.agent.user_favorites["name"].tolist()
            if match in all_names:
                idx = all_names.index(match)
                poi_id = self.agent.user_favorites.index[idx]
                self.agent.exclude_poi_ids.discard(poi_id)

        self.agent.exclude_poi_matched = [
            item for item in exclude_matched
            if (isinstance(item, tuple) and item[1] != match) or (item != match)
        ]

        # 同步到 result
        remaining = [item[1] if isinstance(item, tuple) else item for item in self.agent.exclude_poi_matched]
        self.result["exclude"] = ", ".join(remaining) if remaining else None

        return json.dumps({
            "success": True,
            "removed": match,
            "remaining": remaining,
        }, ensure_ascii=False, indent=2)

    # ========== 高德搜索 ==========

    def _amap_search(self, keyword: str, city: str = None) -> str:
        """高德地图POI搜索（fallback）"""
        from model.config import GaodeMap_apikey
        try:
            params = {
                "key": GaodeMap_apikey,
                "keywords": keyword,
                "city": city or self.result.get("city_zh", ""),
                "extensions": "all",
                "offset": 5,
            }
            resp = requests.get("https://restapi.amap.com/v3/place/text", params=params, timeout=8)
            data = resp.json()
            if data.get("status") != "1" or int(data.get("count", 0)) == 0:
                return json.dumps({
                    "error": "高德搜索无结果",
                    "keyword": keyword,
                    "hint": "请换个关键词试试",
                }, ensure_ascii=False)

            results = []
            for poi in data.get("pois", [])[:5]:
                location = poi.get("location", "")
                lon, lat = (None, None)
                if location and "," in location:
                    lon_str, lat_str = location.split(",")
                    lon, lat = float(lon_str), float(lat_str)
                results.append({
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "category": poi.get("type", ""),
                    "rating": poi.get("biz_ext", {}).get("rating", ""),
                    "lon": lon, "lat": lat,
                })
            return json.dumps({
                "source": "高德地图在线",
                "count": len(results),
                "results": results,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"高德API异常: {e}"}, ensure_ascii=False)

    # ========== 重生成概览 ==========

    def _regenerate_trip_overview(self) -> str:
        """重新生成行程概览/主题/理由"""
        if not hasattr(self.agent, '_generate_overview') or not hasattr(self.agent, 'proxy'):
            return json.dumps({"error": "Agent 不支持在线重生成，请刷新页面后重新规划"})

        # 收集当前每日主题
        days = self.result.get("days", [])
        day_themes = [d.get("day_theme", f"第{i+1}天") for i, d in enumerate(days)]

        try:
            overview = self.agent._generate_overview(day_themes)
            self.result["trip_title"] = overview.get("trip_title", self.result.get("trip_title", ""))
            self.result["trip_overview"] = overview.get("trip_overview", self.result.get("trip_overview", ""))
            daily_summary = overview.get("daily_summary", [])
            for i, ds in enumerate(daily_summary):
                if i < len(days):
                    days[i]["day_theme"] = ds.get("title", days[i].get("day_theme", ""))
                    days[i]["one_liner"] = ds.get("one_liner", "")
                    # 重写 day_reason（简单拼接，LLM重生成太贵）
                    itinerary = days[i].get("itinerary", "")
                    if itinerary:
                        days[i]["day_reason"] = f"{ds.get('title','')}行程：途经{itinerary}"
            self.result["daily_summary"] = daily_summary
            return json.dumps({
                "success": True,
                "trip_title": self.result["trip_title"],
                "trip_overview": self.result["trip_overview"][:200],
                "daily_summary": daily_summary,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"重生成失败: {e}"}, ensure_ascii=False)

    # ========== 时间设置 ==========

    def _set_trip_time(self, start_time: str, end_time: str) -> str:
        """修改每天出发/结束时间"""
        import re
        if not re.match(r'^\d{1,2}:\d{2}$', start_time) or not re.match(r'^\d{1,2}:\d{2}$', end_time):
            return json.dumps({"error": "时间格式错误，请使用 HH:MM 格式，如'08:00'"})

        self.result["start_time"] = start_time
        self.result["end_time"] = end_time

        # 同步到 agent
        if hasattr(self.agent, 'start_time'):
            self.agent.start_time = start_time
        if hasattr(self.agent, 'end_time'):
            self.agent.end_time = end_time

        return json.dumps({
            "success": True,
            "start_time": start_time,
            "end_time": end_time,
            "hint": "建议修改后调用 check_time_feasibility 检查每天时间是否够用",
        }, ensure_ascii=False, indent=2)

    # ========== 工具分发 ==========

    def _call_tool(self, name: str, args: dict) -> str:
        """分发工具调用"""
        tool_map = {
            "search_pois": self._search_pois,
            "get_poi_detail": self._get_poi_detail,
            "get_route_distance": self._get_route_distance,
            "replace_poi": self._replace_poi,
            "remove_poi": self._remove_poi,
            "add_poi": self._add_poi,
            "reroute_day": self._reroute_day,
            "check_time_feasibility": self._check_time_feasibility,
            "finish_replan": self._finish_replan,
            "show_timeline": self._show_timeline,
            "show_transit": self._show_transit,
            "show_weather": self._show_weather,
            "show_itinerary": self._show_itinerary,
            "show_trip_overview": self._show_trip_overview,
            "show_map": self._show_map,
            "add_day": self._add_day,
            "remove_day": self._remove_day,
            "show_hotel": self._show_hotel,
            "change_hotel": self._change_hotel,
            "show_exclude_pois": self._show_exclude_pois,
            "add_exclude_poi": self._add_exclude_poi,
            "remove_exclude_poi": self._remove_exclude_poi,
            "amap_search": self._amap_search,
            "regenerate_trip_overview": self._regenerate_trip_overview,
            "set_trip_time": self._set_trip_time,
        }
        func = tool_map.get(name)
        if func is None:
            return json.dumps({"error": f"未知工具: {name}"})
        try:
            return func(**args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"error": f"工具 {name} 执行失败: {e}"}, ensure_ascii=False)
