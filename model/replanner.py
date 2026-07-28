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
import numpy as np
from typing import Optional
from thefuzz import process as fuzz_process


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
            "description": "展示行程期间的天气预报",
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
]

SYSTEM_PROMPT = """你是一个专业的旅游行程助手，拥有完整的工具链来帮用户查看和修改行程。

你有以下工具：

📖 查询工具：
- show_trip_overview: 查看行程概览
- show_itinerary: 查看某天行程路线
- show_timeline: 查看某天详细时间线
- show_transit: 查看某天交通汇总
- show_weather: 查看天气预报
- search_pois: 搜索景点
- get_poi_detail: 查看景点详情
- get_route_distance: 查两点间距离

✏️ 修改工具：
- replace_poi: 替换景点
- remove_poi: 删除景点
- add_poi: 添加景点
- reroute_day: 重新优化路径顺序
- check_time_feasibility: 检查时间是否合理
- finish_replan: 保存修改

工作原则：
1. 用户问行程相关的问题时，优先调用展示工具获取准确数据再回答，不要凭记忆瞎编
2. 用户要求修改行程时，先搜索确认新景点可行，再执行修改
3. 修改景点后必须重排路径（reroute_day）和检查时间（check_time_feasibility）
4. 做了任何增删改操作后，必须调用 finish_replan 保存
5. 用户只是聊天/问问题时，不需要调用 finish_replan
6. 回答要简洁、结构化，用 Markdown 格式"""


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
        """搜索POI"""
        results = []
        for idx, row in self.data.iterrows():
            name = str(row["name"])
            if keyword.lower() in name.lower() or keyword in str(row.get("context", "")):
                if category and category not in str(row.get("category", "")):
                    continue
                results.append({
                    "name": name,
                    "category": str(row.get("category", "")),
                    "rating": float(row.get("rating", 0)) if row.get("rating") not in [None, "", float("nan")] else 0,
                    "address": str(row.get("address", "")),
                    "open_time": str(row.get("open_time", "")),
                    "close_time": str(row.get("close_time", "")),
                })
            if len(results) >= top_k * 3:
                break

        # 按评分排序
        results.sort(key=lambda x: x["rating"], reverse=True)
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

    def _check_time_feasibility(self, day: int) -> str:
        """检查时间可行性"""
        days = self.result.get("days", [])
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
        """展示天气"""
        rag = self.result.get("rag_context", "")
        days = self.result.get("days", [])
        weather_info = []
        for d in days:
            wc = d.get("weather_context", "")
            if wc:
                weather_info.append({"day": d["day"], "weather": wc})
        return json.dumps({
            "rag_summary": rag[:500] if rag else "无天气数据",
            "daily_weather": weather_info
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"当前行程:\n{json.dumps(self.result.get('days', []), ensure_ascii=False, indent=2)}\n\n用户请求: {user_input}"}
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
        }
        func = tool_map.get(name)
        if func is None:
            return json.dumps({"error": f"未知工具: {name}"})
        return func(**args)
