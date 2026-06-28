"""
Travel Agent — Streamlit 前端
=========================================
用法: streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import datetime
import io
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.utils.proxy_call import OpenaiCall
from model.agent import TravelAgent


def _stream_chat(proxy, messages: list, model: str):
    for chunk in proxy.stream_chat(messages=messages, model=model):
        if chunk:
            yield chunk


st.set_page_config(
    page_title="Travel Agent",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== Session State ====================

for key, default in [
    ("chat_history", []),
    ("trip_result", None),
    ("trip_in_chat_idx", None),
    ("console_logs", []),
    ("llm_model", "deepseek-v4-flash"),
    ("api_base", "https://api.deepseek.com"),
    ("api_key_input", ""),
    ("provider", "deepseek"),
    ("preset_idx", 0),
    ("show_planner", False),
    ("view_mode", "chat"),
    ("confirm_clear", False),
    ("trip_source", "generated"),
    ("current_chat_id", None),  # 当前对话的保存文件名（不含扩展名）
]:
    if key not in st.session_state:
        st.session_state[key] = default

CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

# ==================== 辅助函数 ====================

def _save_chat():
    """保存当前对话到文件"""
    if not st.session_state.chat_history:
        return
    # 用已有 id 或新建
    chat_id = st.session_state.current_chat_id
    if not chat_id:
        first_msg = st.session_state.chat_history[0].get("content", "chat")[:30]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_id = f"{ts}_{first_msg}"
        st.session_state.current_chat_id = chat_id
    # 清洗非法文件名字符
    safe_id = "".join(c for c in chat_id if c.isalnum() or c in "._-()[]  ") .strip()[:80]
    st.session_state.current_chat_id = safe_id

    data = {
        "chat_history": st.session_state.chat_history,
        "trip_result": st.session_state.trip_result,
        "trip_source": st.session_state.trip_source,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    with open(CHAT_DIR / f"{safe_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _list_saved_chats():
    """列出所有已保存的对话"""
    chats = []
    for f in sorted(CHAT_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            title = ""
            for msg in data.get("chat_history", []):
                if msg.get("role") == "trip":
                    title = f"🗺️ {msg.get('title','行程')}"
                    break
                elif msg.get("role") == "user":
                    title = msg["content"][:40]
                    break
            chats.append({
                "id": f.stem,
                "title": title or "(空对话)",
                "saved_at": data.get("saved_at", ""),
                "msg_count": len(data.get("chat_history", [])),
            })
        except Exception:
            pass
    return chats

# ==================== 侧边栏 ====================

st.sidebar.title("🗺️ Travel Agent")

# ── API 配置 ──
with st.sidebar.expander("🔑 API 配置", expanded=True):
    MODEL_PRESETS = {
        "DeepSeek V4 Flash（推荐）": {"model":"deepseek-v4-flash","base_url":"https://api.deepseek.com","provider":"deepseek"},
        "DeepSeek V4 Pro": {"model":"deepseek-v4","base_url":"https://api.deepseek.com","provider":"deepseek"},
        "GPT-4o": {"model":"gpt-4o","base_url":"https://api.openai.com/v1","provider":"openai"},
        "GPT-4o-mini": {"model":"gpt-4o-mini","base_url":"https://api.openai.com/v1","provider":"openai"},
        "Claude 3.5 Sonnet": {"model":"claude-3-5-sonnet-20241022","base_url":"https://api.anthropic.com/v1","provider":"anthropic"},
        "Claude 3 Haiku": {"model":"claude-3-haiku-20240307","base_url":"https://api.anthropic.com/v1","provider":"anthropic"},
        "MiniMax-abab7": {"model":"abab7-chat","base_url":"https://api.minimax.chat/v1","provider":"minimax"},
        "MiniMax-abab6.5s": {"model":"abab6.5s-chat","base_url":"https://api.minimax.chat/v1","provider":"minimax"},
        "阶跃星辰 Step-2": {"model":"step-2-16k","base_url":"https://api.stepfun.com/v1","provider":"stepfun"},
        "Gemini 2.0 Flash": {"model":"gemini-2.0-flash","base_url":"https://generativelanguage.googleapis.com/v1beta/openai","provider":"gemini"},
        "🛠️ 自定义…": {"model":"","base_url":"","provider":"custom"},
    }
    names = list(MODEL_PRESETS.keys())
    idx = st.selectbox("模型", range(len(names)), format_func=lambda i: names[i], index=st.session_state.preset_idx)
    st.session_state.preset_idx = idx
    p = MODEL_PRESETS[names[idx]]
    if p["model"]:
        st.session_state.llm_model = p["model"]; st.session_state.api_base = p["base_url"]; st.session_state.provider = p["provider"]
        st.caption(f"🔗 `{p['base_url']}`")
    else:
        st.session_state.llm_model = st.text_input("模型名", st.session_state.llm_model)
        st.session_state.api_base = st.text_input("Base URL", st.session_state.api_base)
    st.session_state.api_key_input = st.text_input("API Key", st.session_state.api_key_input, type="password", placeholder="sk-...")

# ── 行程规划入口 ──
st.sidebar.divider()

# � 对话历史
with st.sidebar.expander("💾 对话记录", expanded=False):
    saved = _list_saved_chats()
    if saved:
        for chat in saved[:20]:
            c_col1, c_col2 = st.columns([5, 1])
            with c_col1:
                label = f"{chat['title']} ({chat['msg_count']}条)"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True,
                             help=f"保存于 {chat['saved_at'][:16]}"):
                    with open(CHAT_DIR / f"{chat['id']}.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    st.session_state.chat_history = data.get("chat_history", [])
                    st.session_state.trip_result = data.get("trip_result")
                    st.session_state.trip_source = data.get("trip_source", "generated")
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.console_logs = [f"✅ 已恢复对话: {chat['title']}"]
                    st.session_state.view_mode = "chat"
                    st.rerun()
            with c_col2:
                if st.button("🗑️", key=f"del_{chat['id']}", help="删除此对话"):
                    (CHAT_DIR / f"{chat['id']}.json").unlink(missing_ok=True)
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                    st.rerun()
    else:
        st.caption("暂无保存的对话")
    if st.button("💾 保存当前对话", use_container_width=True):
        _save_chat()
        st.rerun()

# �🔧 调试按钮：快速加载已生成的行程
with st.sidebar.expander("🔧 调试工具", expanded=False):
    if st.button("⚡ 加载最近一次行程", use_container_width=True):
        json_files = sorted(Path("model/output").glob("*_zh_multiday.json"), key=os.path.getmtime, reverse=True)
        if json_files:
            jf = json_files[0]
            with open(jf, "r", encoding="utf-8") as f:
                trip = json.load(f)
            st.session_state.trip_result = trip
            st.session_state.trip_source = "debug"
            st.session_state.console_logs = [f"✅ 已加载(调试): {jf.name}"]

            # 匹配同名地图文件
            ts_prefix = jf.name.replace("_zh_multiday.json", "")
            output_dir = Path("model/output")
            all_html = sorted(output_dir.glob(f"{ts_prefix}*.html"), key=os.path.getmtime, reverse=True)
            st.session_state.map_overview = None
            st.session_state.map_days = []
            for hf in all_html:
                if "_overview" in hf.name:
                    st.session_state.map_overview = str(hf)
                elif "_day" in hf.name:
                    st.session_state.map_days.append(str(hf))
                elif "_fulltsp" not in hf.name and "_response_clusters" not in hf.name:
                    st.session_state.map_days.append(str(hf))
            st.session_state.map_days.sort()
            trip_card = {
                "role": "trip",
                "title": trip.get("trip_title", "行程"),
                "overview": trip.get("trip_overview", ""),
                "data": trip,
            }
            st.session_state.chat_history.append(trip_card)
            st.session_state.show_planner = False
            st.session_state.view_mode = "chat"
            st.rerun()
        else:
            st.warning("无可用行程文件")

# 有行程结果 → 显示操作按钮
if st.session_state.trip_result and not st.session_state.show_planner:
    trip_title = st.session_state.trip_result.get("trip_title", "行程")
    debug_tag = " (调试)" if st.session_state.get("trip_source") == "debug" else ""

    # 行程按钮 + 右侧菜单
    btn_col, menu_col = st.sidebar.columns([5, 1])
    with btn_col:
        if st.button(f"📋 {debug_tag}{trip_title}", use_container_width=True, key="view_trip_btn"):
            st.session_state.view_mode = "trip_detail"
            st.rerun()
    with menu_col:
        with st.popover("·", help="更多操作"):
            # 1. 重命名
            new_name = st.text_input("重命名", value=trip_title, key="rename_trip_input")
            if st.button("✅ 确认重命名", key="rename_confirm", use_container_width=True):
                st.session_state.trip_result["trip_title"] = new_name
                # 同步更新 chat 中的卡片
                for msg in st.session_state.chat_history:
                    if msg.get("role") == "trip":
                        msg["title"] = new_name
                st.rerun()

            st.divider()

            # 2. 置顶
            if st.button("📌 置顶", key="pin_trip", use_container_width=True):
                # 把 trip 卡片移到 chat_history 最前面
                trip_msgs = [m for m in st.session_state.chat_history if m.get("role") == "trip"]
                other_msgs = [m for m in st.session_state.chat_history if m.get("role") != "trip"]
                st.session_state.chat_history = trip_msgs + other_msgs
                st.rerun()

            st.divider()

            # 3. 删除（仅删行程，保留聊天）
            if st.button("🗑️ 删除行程", key="delete_trip", use_container_width=True):
                st.session_state.trip_result = None
                st.session_state.trip_source = "generated"
                st.session_state.console_logs = []
                st.session_state.current_chat_id = None
                st.session_state.chat_history = [m for m in st.session_state.chat_history if m.get("role") != "trip"]
                st.session_state.view_mode = "chat"
                st.rerun()

    # 清空 + 新建 并排
    c_clear, c_new = st.sidebar.columns([1, 1])
    with c_clear:
        if st.button("🗑️ 清空", use_container_width=True, key="clear_trip_btn"):
            st.session_state.confirm_clear = True
            st.rerun()
    with c_new:
        if st.button("🗺️ 新建", use_container_width=True, key="new_trip_btn"):
            st.session_state.chat_history = []
            st.session_state.trip_result = None
            st.session_state.console_logs = []
            st.session_state.current_chat_id = None
            st.session_state.trip_source = "generated"
            st.session_state.show_planner = True
            st.session_state.view_mode = "chat"
            st.rerun()

    # 二次确认清空
    if st.session_state.get("confirm_clear"):
        st.sidebar.warning("⚠️ 确定要清空当前行程吗？聊天记录也会一并清除。")
        cc1, cc2 = st.sidebar.columns([1, 1])
        with cc1:
            if st.button("✅ 确认清空", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.trip_result = None
                st.session_state.console_logs = []
                st.session_state.current_chat_id = None
                st.session_state.confirm_clear = False
                st.session_state.view_mode = "chat"
                st.rerun()
        with cc2:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

# 无行程且未展开设置 → 显示"新建行程规划"
if not st.session_state.trip_result and not st.session_state.show_planner:
    if st.sidebar.button("🗺️ 新建行程规划", type="primary", use_container_width=True):
        st.session_state.show_planner = True
        st.rerun()

# 规划设置面板
if st.session_state.show_planner:
    with st.sidebar.expander("⚙️ 行程设置", expanded=True):
        city = st.selectbox("目的地", ["fuzhou","guangzhou","shenzhen","shanghai"],
            format_func=lambda x: {"fuzhou":"福州","guangzhou":"广州","shenzhen":"深圳","shanghai":"上海"}[x])
        num_days = st.slider("天数", 1, 7, 3)
        travel_date = st.date_input("出发日期", datetime.date.today() + datetime.timedelta(days=1))
        start_time = st.text_input("出发时间", "08:00")
        end_time = st.text_input("结束时间", "20:00")
        hotel = st.text_input("住宿点", placeholder="如：外滩")
        exclude = st.text_input("排除景点", placeholder="鼓山,西湖公园")

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("✖ 取消", use_container_width=True, key="cancel_planner"):
                st.session_state.show_planner = False
                st.rerun()
        with c2:
            start_btn = st.button("🚀 开始规划", type="primary", use_container_width=True)

        if start_btn:
            st.session_state.console_logs = []
            city_zh = {"fuzhou":"福州","guangzhou":"广州","shenzhen":"深圳","shanghai":"上海"}[city]
            query = f"我想在{city_zh}玩{num_days}天，体验当地的历史文化与现代风貌，品尝地道美食，逛逛特色街区和公园。"
            el = [x.strip() for x in exclude.split(",") if x.strip()] if exclude else None

            import os as _os
            from model.config import DEEPSEEK_API_KEY,OPENAI_API_KEY,ANTHROPIC_API_KEY,MINIMAX_API_KEY,STEPFUN_API_KEY,GEMINI_API_KEY
            KEYS = {"deepseek":(DEEPSEEK_API_KEY,"DEEPSEEK_API_KEY"),"openai":(OPENAI_API_KEY,"OPENAI_API_KEY"),
                    "anthropic":(ANTHROPIC_API_KEY,"ANTHROPIC_API_KEY"),"minimax":(MINIMAX_API_KEY,"MINIMAX_API_KEY"),
                    "stepfun":(STEPFUN_API_KEY,"STEPFUN_API_KEY"),"gemini":(GEMINI_API_KEY,"GEMINI_API_KEY")}
            fk = st.session_state.api_key_input
            if not fk and st.session_state.provider in KEYS:
                cfg_key, env_name = KEYS[st.session_state.provider]
                fk = _os.getenv(env_name) or cfg_key
            if not fk:
                st.error(f"❌ 未配置 API Key"); st.stop()

            with st.spinner(f"⏳ 正在规划 {city_zh} {num_days} 日游，请稍候…"):
                try:
                    ts_before = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
                    log_cap = io.StringIO(); _out = sys.stdout; sys.stdout = log_cap
                    proxy = OpenaiCall(api_key=fk, base_url=st.session_state.api_base)
                    planner = TravelAgent(user_reqs=[query], city=city, num_days=num_days, type="zh",
                        proxy_call=proxy, use_rag=True, travel_date=travel_date.strftime("%Y-%m-%d"),
                        citywalk=True, hotel=hotel or None, exclude_pois=el,
                        start_time=start_time, end_time=end_time, model_override=st.session_state.llm_model)
                    trip = planner.solve_multiday()
                    sys.stdout = _out
                    st.session_state.console_logs = log_cap.getvalue().strip().split("\n")
                    st.session_state.trip_result = trip
                    st.session_state.trip_source = "generated"

                    # 匹配地图文件
                    output_dir = Path("model/output")
                    all_html = sorted(output_dir.glob("*.html"), key=os.path.getmtime, reverse=True)
                    st.session_state.map_overview = None
                    st.session_state.map_days = []
                    for hf in all_html:
                        if hf.name.startswith(ts_before[:13]):
                            if "_overview" in hf.name:
                                st.session_state.map_overview = str(hf)
                            elif "_day" in hf.name:
                                st.session_state.map_days.append(str(hf))
                            elif "_fulltsp" not in hf.name and "_response_clusters" not in hf.name:
                                # 单日地图（如 2026_06_28_14_22_zh.html）
                                st.session_state.map_days.append(str(hf))
                    # 单日地图排序
                    st.session_state.map_days.sort()

                    # 将规划结果卡片插入聊天历史
                    trip_card = {
                        "role": "trip",
                        "title": trip.get("trip_title", f"{city_zh}{num_days}日游"),
                        "overview": trip.get("trip_overview", ""),
                        "data": trip,
                    }
                    st.session_state.chat_history.append(trip_card)
                    st.session_state.trip_in_chat_idx = len(st.session_state.chat_history) - 1
                    st.session_state.show_planner = False
                    _save_chat()
                    st.rerun()
                except Exception as e:
                    sys.stdout = _out; st.error(f"规划失败: {e}"); st.stop()

    # 已有行程时快速查看各天

# ==================== 主区域 ====================

# ── 行程详情模式（从侧边栏按钮进入）──
if st.session_state.get("view_mode") == "trip_detail" and st.session_state.trip_result:
    trip = st.session_state.trip_result

    # 返回按钮
    if st.button("← 返回对话", key="back_to_chat"):
        st.session_state.view_mode = "chat"
        st.rerun()

    st.title(f"🎯 {trip.get('trip_title', '')}")
    st.markdown(f"*{trip.get('trip_overview', '')}*")

    days = trip.get("days", [])
    tabs = st.tabs([f"Day {d['day']}: {d.get('day_theme','')}" for d in days])
    for tab, day in zip(tabs, days):
        with tab:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("📍 路线")
                st.markdown(f"**{day.get('itinerary','—')}**")
                st.caption(f"_{day.get('day_reason','')}_")
                if day.get("weather_note","无") != "无":
                    st.info(f"🌤️ {day['weather_note']}")
                ts = day.get("transit_summary","")
                if ts:
                    # 去掉"当日交通汇总"前的横线，换成分隔线
                    ts_clean = ts.replace("───────────────────────────────────", "")
                    with st.expander("🚗 交通", expanded=False): st.markdown(ts_clean)
            with c2:
                tl = day.get("timeline","")
                if tl:
                    tl_lines = tl.strip().split("\n")
                    tl_clean = "\n".join(line.strip() for line in tl_lines if line.strip())
                    with st.expander("🕐 预计时间线", expanded=True):
                        st.markdown(tl)
            st.divider()
            pois = day.get("pois",{})
            if pois:
                cols = st.columns(min(len(pois), 3))
                for j, (name, desc) in enumerate(pois.items()):
                    with cols[j % 3]:
                        with st.container(border=True):
                            st.markdown(f"**{name}**")
                            st.caption(str(desc)[:200])

    # 地图
    ov = st.session_state.get("map_overview")
    if ov and os.path.exists(ov):
        with st.expander("🗺️ 总览地图", expanded=True):
            with open(ov, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500)
    day_maps = st.session_state.get("map_days", [])
    if day_maps:
        mtabs = st.tabs([f"Day {k+1} 地图" for k in range(len(day_maps))])
        for k, (mt, mp) in enumerate(zip(mtabs, day_maps)):
            with mt:
                if os.path.exists(mp):
                    with open(mp, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=400)

    st.stop()

# ── 正常对话模式 ──
st.title("🗺️ Travel Agent")
st.caption("多日旅游行程规划 + AI 对话")

# ── 运行日志 ──
if st.session_state.console_logs:
    with st.expander("📋 运行日志", expanded=False):
        for line in st.session_state.console_logs:
            if line.strip(): st.text(line)

# ── 聊天区域 ──
for i, msg in enumerate(st.session_state.chat_history):
    role = msg.get("role", "trip")

    if role == "trip":
        # 行程卡片
        with st.chat_message("assistant"):
            trip = msg["data"]
            with st.expander(f"🗺️ {msg['title']} — {msg['overview'][:50]}…", expanded=False):
                days = trip.get("days", [])
                tabs = st.tabs([f"Day {d['day']}: {d.get('day_theme','')}" for d in days])
                for tab, day in zip(tabs, days):
                    with tab:
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.subheader("📍 路线")
                            st.markdown(f"**{day.get('itinerary','—')}**")
                            st.caption(f"_{day.get('day_reason','')}_")
                            if day.get("weather_note","无") != "无":
                                st.info(f"🌤️ {day['weather_note']}")
                            ts = day.get("transit_summary","")
                            if ts:
                                ts_clean = ts.replace("───────────────────────────────────", "")
                                with st.expander("🚗 交通", expanded=False): st.markdown(ts_clean)
                        with c2:
                            tl = day.get("timeline","")
                            if tl:
                                tl_lines = tl.strip().split("\n")
                                tl_clean = "\n".join(line.strip() for line in tl_lines if line.strip())
                                with st.expander("🕐 预计时间线", expanded=True):
                                    st.markdown(tl_clean)
                        st.divider()
                        pois = day.get("pois",{})
                        if pois:
                            cols = st.columns(min(len(pois), 3))
                            for j, (name, desc) in enumerate(pois.items()):
                                with cols[j % 3]:
                                    with st.container(border=True):
                                        st.markdown(f"**{name}**")
                                        st.caption(str(desc)[:200])

                # 地图
                ov = st.session_state.get("map_overview")
                if ov and os.path.exists(ov):
                    with st.expander("🗺️ 总览地图", expanded=False):
                        with open(ov, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=400)
                day_maps = st.session_state.get("map_days", [])
                if day_maps:
                    mtabs = st.tabs([f"Day {k+1} 地图" for k in range(len(day_maps))])
                    for k, (mt, mp) in enumerate(zip(mtabs, day_maps)):
                        with mt:
                            if os.path.exists(mp):
                                with open(mp, "r", encoding="utf-8") as f:
                                    st.components.v1.html(f.read(), height=400)

    elif role in ("user", "assistant"):
        with st.chat_message(role):
            st.markdown(msg["content"])

# ── 聊天输入 ──
if question := st.chat_input("输入消息…"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    from model.config import (
        DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
        MINIMAX_API_KEY, STEPFUN_API_KEY, GEMINI_API_KEY,
    )
    KC = {"deepseek": DEEPSEEK_API_KEY, "openai": OPENAI_API_KEY,
          "anthropic": ANTHROPIC_API_KEY, "minimax": MINIMAX_API_KEY,
          "stepfun": STEPFUN_API_KEY, "gemini": GEMINI_API_KEY}
    ck = st.session_state.api_key_input or KC.get(st.session_state.provider, "")

    if not ck:
        answer = "⚠️ 请先在左侧「🔑 API 配置」中设置 API Key。"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        messages = []
        if st.session_state.trip_result:
            trip_json = json.dumps(st.session_state.trip_result, ensure_ascii=False)
            messages.append({
                "role": "system",
                "content": f"以下是用户的旅行行程（JSON）。用户可能就此行程提问，请结合行程内容回答：\n{trip_json[:4000]}"
            })
        messages += [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state.chat_history if m.get("role") in ("user", "assistant")]

        try:
            proxy = OpenaiCall(api_key=ck, base_url=st.session_state.api_base)
            with st.chat_message("assistant"):
                response = st.write_stream(_stream_chat(proxy, messages, st.session_state.llm_model))
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "(空回复)"})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ 调用失败: {e}"})

    _save_chat()
    st.rerun()

# ── 清空 ──
col1, col2 = st.columns([1, 10])
# with col1:
#     if st.button("🗑️ 清空对话", use_container_width=True):
#         st.session_state.chat_history = []
#         st.session_state.trip_result = None
#         st.session_state.trip_in_chat_idx = None
#         st.rerun()
