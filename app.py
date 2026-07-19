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
from model.config import CITY_ZH_MAP

# 中文城市名 → 英文城市名 反向映射
_CITY_ZH2EN = {v: k for k, v in CITY_ZH_MAP.items()}


def _norm_city(city_val: str) -> str:
    """将中文城市名转回英文名（兼容旧数据），如 '福州' → 'fuzhou'"""
    return _CITY_ZH2EN.get(city_val, city_val)


def _stream_assistant(proxy, messages: list, model: str):
    """流式输出，返回 (reply_text, thinking_text)"""
    think_expander = st.expander("🧠 深度思考", expanded=True)
    think_placeholder = think_expander.empty()
    reply_placeholder = st.empty()

    thinking_parts = []
    reply_parts = []

    for item in proxy.stream_chat(messages=messages, model=model):
        if isinstance(item, tuple):
            reasoning, content = item
        else:
            reasoning, content = "", item or ""

        if reasoning:
            thinking_parts.append(reasoning)
            think_placeholder.markdown("".join(thinking_parts))

        if content:
            reply_parts.append(content)
            reply_placeholder.markdown("".join(reply_parts))

    return "".join(reply_parts), "".join(thinking_parts)


st.set_page_config(
    page_title="Travel Agent",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 加宽侧边栏
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 420px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State ====================

for key, default in [
    ("chat_history", []),
    ("trip_result", None),    ("trip_planner", None),    ("trip_in_chat_idx", None),
    ("console_logs", []),
    ("llm_model", "deepseek-v4-flash"),
    ("api_base", "https://api.deepseek.com"),
    # replan 状态机
    ("_rp_active", False),
    ("_rp_steps", []),
    ("_rp_msgs", None),
    ("_rp_replanner", None),
    ("_rp_modified", False),
    ("_rp_question", ""),
    ("_rp_tools", None),
    ("_rp_conversation", []),    # 上下文记忆：累积对话历史
    ("_rp_progress", None),      # 当前进度数据（{steps, count}），用于在聊天区底部渲染
    ("map_overview", None),      # Folium 总览地图路径
    ("map_days", []),            # Folium 每日地图路径列表
    ("_show_map_inline", False), # 标记：在对话中内联展示地图
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
    """保存当前对话到文件（含地图在同一文件夹）"""
    if not st.session_state.chat_history:
        return
    chat_id = st.session_state.current_chat_id
    if not chat_id:
        first_msg = st.session_state.chat_history[0].get("content", "chat")[:30]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_id = f"{ts}_{first_msg}"
        st.session_state.current_chat_id = chat_id
    safe_id = "".join(c for c in chat_id if c.isalnum() or c in "._-()[]  ") .strip()[:80]
    st.session_state.current_chat_id = safe_id

    chat_folder = CHAT_DIR / safe_id
    chat_folder.mkdir(parents=True, exist_ok=True)

    data = {
        "chat_history": st.session_state.chat_history,
        "trip_result": st.session_state.trip_result,
        "trip_source": st.session_state.trip_source,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    with open(chat_folder / f"{safe_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _list_saved_chats():
    """列出所有已保存的对话"""
    chats = []
    for f in sorted(CHAT_DIR.glob("*/*.json"), key=os.path.getmtime, reverse=True):
        # 只读取与目录同名的 JSON（如 chat_history/<id>/<id>.json），跳过 trip.json / multiday.json 等
        if f.stem != f.parent.name:
            continue
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


def _match_map_files(folder: Path, prefix: str = ""):
    """扫描文件夹中的 HTML 地图文件，返回 (map_overview, map_days)

    Args:
        folder: 包含 HTML 地图文件的文件夹
        prefix: 文件名前缀过滤（如时间戳前缀），为空则匹配所有
    """
    if not folder.exists():
        return None, []
    all_html = sorted(folder.glob("*.html"), key=os.path.getmtime, reverse=True)
    map_overview = None
    map_days = []
    for hf in all_html:
        if prefix and not hf.name.startswith(prefix):
            continue
        if "_overview" in hf.name:
            map_overview = str(hf)
        elif "_day" in hf.name:
            map_days.append(str(hf))
        elif "_fulltsp" not in hf.name and "_response_clusters" not in hf.name:
            map_days.append(str(hf))
    map_days.sort()
    return map_overview, map_days


def _render_map_section(map_overview, map_days, height=500, expanded=False, use_chat_wrapper=False):
    """统一的地图渲染组件：单日/无 overview 时用 day_maps[0] 兜底，多日用分日标签页"""
    # 确定实际可用的地图文件
    ov_ok = map_overview and os.path.exists(map_overview)
    day_maps_ok = [mp for mp in (map_days or []) if os.path.exists(mp)]

    if len(day_maps_ok) <= 1:
        # 单日或无分日地图：优先 overview，回退到 day_maps[0]
        map_file = map_overview if ov_ok else (day_maps_ok[0] if day_maps_ok else None)
        if map_file and os.path.exists(map_file):
            if use_chat_wrapper:
                with st.chat_message("assistant"):
                    with st.expander("🗺️ 查看地图", expanded=expanded):
                        with open(map_file, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=height, scrolling=True)
            else:
                with st.expander("🗺️ 总览地图", expanded=expanded):
                    with open(map_file, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=height, scrolling=True)
    else:
        # 多日：分日标签页
        if use_chat_wrapper:
            with st.chat_message("assistant"):
                mtabs = st.tabs([f"Day {k+1}" for k in range(len(day_maps_ok))])
                for k, (mt, mp) in enumerate(zip(mtabs, day_maps_ok)):
                    with mt:
                        with open(mp, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=height, scrolling=True)
        else:
            mtabs = st.tabs([f"Day {k+1} 地图" for k in range(len(day_maps_ok))])
            for k, (mt, mp) in enumerate(zip(mtabs, day_maps_ok)):
                with mt:
                    with open(mp, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=height, scrolling=True)

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
        "Qwen3.7-Plus（百炼）": {"model":"qwen3.7-plus","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "Qwen3.7-Max（百炼）": {"model":"qwen3.7-max","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "Qwen-Flash（百炼）": {"model":"qwen-flash","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "DeepSeek V4 Flash（百炼）": {"model":"deepseek-v4-flash","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "Kimi K2.7（百炼）": {"model":"kimi-k2.7-code","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "GLM-5.2（百炼）": {"model":"glm-5.2","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
        "MiniMax M3（百炼）": {"model":"MiniMax-M3","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","provider":"bailian"},
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
                    with open(CHAT_DIR / chat['id'] / f"{chat['id']}.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    st.session_state.chat_history = data.get("chat_history", [])
                    st.session_state.trip_result = data.get("trip_result")
                    st.session_state.trip_source = data.get("trip_source", "generated")
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.console_logs = [f"✅ 已恢复对话: {chat['title']}"]
                    st.session_state.view_mode = "chat"
                    # 恢复地图文件路径
                    ov, dm = _match_map_files(CHAT_DIR / chat["id"])
                    st.session_state.map_overview = ov
                    st.session_state.map_days = dm
                    # 清除旧 replan 上下文，避免串到其他行程
                    st.session_state._rp_conversation = []
                    st.session_state._rp_active = False
                    st.rerun()
            with c_col2:
                if st.button("🗑️", key=f"del_{chat['id']}", help="删除此对话"):
                    import shutil; shutil.rmtree(CHAT_DIR / chat['id'], ignore_errors=True)
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

            # 为 replan 创建轻量 agent（兼容新旧数据：city 可能是中文或英文）
            city = _norm_city(trip.get("city", "fuzhou"))
            num_days = len(trip.get("days", []))
            proxy = OpenaiCall(api_key=st.session_state.api_key_input, base_url=st.session_state.api_base)
            st.session_state.trip_planner = TravelAgent(
                user_reqs=["debug"], city=city, num_days=num_days,
                proxy_call=proxy, use_rag=False, model_override=st.session_state.llm_model
            )

            # 匹配同名地图文件
            ts_prefix = jf.name.replace("_zh_multiday.json", "")
            ov, dm = _match_map_files(Path("model/output"), prefix=ts_prefix)
            st.session_state.map_overview = ov
            st.session_state.map_days = dm
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

    st.sidebar.info("💬 在下方聊天框自由对话：\n"
                    "• \"把公园换成博物馆\" — 修改行程\n"
                    "• \"给我看看时间线\" — 展示详情\n"
                    "• \"第一天天气怎么样\" — 查询信息\n"
                    "• 直接聊天也可以，我都会调用合适的工具")

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
            from model.config import DEEPSEEK_API_KEY,OPENAI_API_KEY,ANTHROPIC_API_KEY,MINIMAX_API_KEY,STEPFUN_API_KEY,GEMINI_API_KEY,BAILIAN_API_KEY
            KEYS = {"deepseek":(DEEPSEEK_API_KEY,"DEEPSEEK_API_KEY"),"openai":(OPENAI_API_KEY,"OPENAI_API_KEY"),
                    "anthropic":(ANTHROPIC_API_KEY,"ANTHROPIC_API_KEY"),"minimax":(MINIMAX_API_KEY,"MINIMAX_API_KEY"),
                    "stepfun":(STEPFUN_API_KEY,"STEPFUN_API_KEY"),"gemini":(GEMINI_API_KEY,"GEMINI_API_KEY"),
                    "bailian":(BAILIAN_API_KEY,"BAILIAN_API_KEY")}
            fk = st.session_state.api_key_input
            if not fk and st.session_state.provider in KEYS:
                cfg_key, env_name = KEYS[st.session_state.provider]
                fk = _os.getenv(env_name) or cfg_key
            if not fk:
                st.error(f"❌ 未配置 API Key"); st.stop()

            with st.spinner(f"⏳ 正在规划 {city_zh} {num_days} 日游，请稍候…"):
                try:
                    # 预建 chat folder
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    chat_id = f"{ts}_{city_zh}{num_days}日游"
                    chat_folder = CHAT_DIR / chat_id
                    chat_folder.mkdir(parents=True, exist_ok=True)

                    ts_before = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
                    log_cap = io.StringIO(); _out = sys.stdout; sys.stdout = log_cap
                    proxy = OpenaiCall(api_key=fk, base_url=st.session_state.api_base)
                    planner = TravelAgent(user_reqs=[query], city=city, num_days=num_days, type="zh",
                        proxy_call=proxy, use_rag=True, travel_date=travel_date.strftime("%Y-%m-%d"),
                        citywalk=True, hotel=hotel or None, exclude_pois=el,
                        start_time=start_time, end_time=end_time, model_override=st.session_state.llm_model,
                        output_dir=str(chat_folder))
                    trip = planner.solve_multiday()
                    sys.stdout = _out
                    st.session_state.console_logs = log_cap.getvalue().strip().split("\n")
                    st.session_state.trip_result = trip
                    st.session_state.trip_source = "generated"
                    st.session_state.trip_planner = planner
                    st.session_state.current_chat_id = chat_id
                    # 新行程清除旧 replan 上下文
                    st.session_state._rp_conversation = []
                    st.session_state._rp_active = False

                    # 匹配地图文件（现在在 chat_folder 里）
                    ov, dm = _match_map_files(chat_folder, prefix=ts_before[:13])
                    st.session_state.map_overview = ov
                    st.session_state.map_days = dm

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
    _render_map_section(
        st.session_state.get("map_overview"),
        st.session_state.get("map_days", []),
        height=650, expanded=True,
    )

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

                # 地图：单日或无 overview 时回退到 day_maps[0]；多日分日标签页
                _render_map_section(
                    st.session_state.get("map_overview"),
                    st.session_state.get("map_days", []),
                    height=500, expanded=False,
                )

    elif role in ("user", "assistant"):
        with st.chat_message(role):
            if role == "assistant" and msg.get("thinking"):
                with st.expander("🧠 深度思考", expanded=False):
                    st.markdown(msg["thinking"])
            st.markdown(msg["content"])

# ── 内联地图：在所有对话之后、紧跟最后一条消息展示 ──
if st.session_state.get("_show_map_inline"):
    _render_map_section(
        st.session_state.get("map_overview"),
        st.session_state.get("map_days", []),
        height=500, expanded=True, use_chat_wrapper=True,
    )
    st.session_state._show_map_inline = False

# ── 聊天输入 ──
if question := st.chat_input("输入消息…"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    from model.config import (
        DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
        MINIMAX_API_KEY, STEPFUN_API_KEY, GEMINI_API_KEY,
        BAILIAN_API_KEY,
    )
    KC = {"deepseek": DEEPSEEK_API_KEY, "openai": OPENAI_API_KEY,
          "anthropic": ANTHROPIC_API_KEY, "minimax": MINIMAX_API_KEY,
          "stepfun": STEPFUN_API_KEY, "gemini": GEMINI_API_KEY,
          "bailian": BAILIAN_API_KEY}
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
            # ── 有行程时：内联流式 function calling ──
            if st.session_state.trip_result:
                from model.replanner import Replanner, TOOLS as RP_TOOLS
                chat_folder = CHAT_DIR / (st.session_state.current_chat_id or "chat")
                chat_folder.mkdir(parents=True, exist_ok=True)
                trip_json_path = chat_folder / "trip.json"
                with open(trip_json_path, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.trip_result, f, ensure_ascii=False, indent=2)

                agent = st.session_state.get("trip_planner")
                if agent is not None:
                    trip_city = _norm_city(st.session_state.trip_result.get("city", "fuzhou"))
                    if agent.city != trip_city:
                        agent = None
                        st.session_state.trip_planner = None
                if agent is None:
                    trip = st.session_state.trip_result
                    city = _norm_city(trip.get("city", "fuzhou"))
                    num_days = trip.get("num_days", len(trip.get("days", [])))
                    proxy = OpenaiCall(api_key=ck, base_url=st.session_state.api_base)
                    agent = TravelAgent(
                        user_reqs=["chat"], city=city, num_days=num_days,
                        proxy_call=proxy, use_rag=False, model_override=st.session_state.llm_model,
                        output_dir=str(chat_folder)
                    )
                    st.session_state.trip_planner = agent

                rp = Replanner(agent, str(trip_json_path))
                proxy = agent.proxy  # 从 agent 获取，避免第二次对话时未定义

                # ── 上下文记忆 ──
                sys_msg = {"role": "system", "content": rp._SYSTEM_PROMPT}
                user_msg = {"role": "user",
                            "content": f"当前行程:\n{json.dumps(rp.result.get('days', []), ensure_ascii=False, indent=2)}\n\n用户请求: {question}"}
                prev_conv = st.session_state.get("_rp_conversation", [])
                if prev_conv:
                    msgs = [sys_msg] + prev_conv + [user_msg]
                else:
                    msgs = [sys_msg, user_msg]

                # ── 流式容器 ──
                with st.chat_message("assistant"):
                    think_exp = st.expander("🧠 深度思考", expanded=True)
                    think_ph = think_exp.empty()
                    reply_ph = st.empty()
                    status_ph = st.empty()

                    all_reasoning_parts = []
                    all_reply_parts = []
                    modified = False
                    max_turns = 12
                    final_text = None

                    for turn in range(max_turns):
                        # 流式调用 LLM
                        turn_reasoning = []
                        turn_content = []
                        tool_calls_from_stream = None

                        stream_events = proxy.stream_chat_with_tools(
                            messages=msgs,
                            model=st.session_state.llm_model,
                            tools=RP_TOOLS,
                        )
                        for event in stream_events:
                            if event["type"] == "reasoning":
                                turn_reasoning.append(event["content"])
                                all_reasoning_parts.append(event["content"])
                                think_ph.markdown("".join(all_reasoning_parts))
                            elif event["type"] == "content":
                                turn_content.append(event["content"])
                                all_reply_parts.append(event["content"])
                                reply_ph.markdown("".join(all_reply_parts))
                            elif event["type"] == "done":
                                if event.get("tool_calls"):
                                    tool_calls_from_stream = event["tool_calls"]
                                    if event.get("reasoning"):
                                        all_reasoning_parts.append(event["reasoning"])
                                elif event.get("final_text"):
                                    final_text = event["final_text"]
                                    if event.get("reasoning"):
                                        all_reasoning_parts.append(event["reasoning"])
                                break

                        # ── 无 tool_calls → 最终回复 ──
                        if tool_calls_from_stream is None:
                            final_text = final_text or "".join(all_reply_parts)
                            break

                        # ── 有 tool_calls → 执行并继续 ──
                        assistant_msg = {
                            "role": "assistant",
                            "content": "".join(turn_content) if turn_content else None,
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    }
                                }
                                for tc in tool_calls_from_stream
                            ]
                        }
                        msgs.append(assistant_msg)

                        for tc in tool_calls_from_stream:
                            func_name = tc["function"]["name"]
                            try:
                                args = json.loads(tc["function"]["arguments"])
                            except Exception:
                                args = {}

                            # ── 记录工具调用到思考面板 ──
                            args_str = json.dumps(args, ensure_ascii=False)
                            all_reasoning_parts.append(f"\n\n🔧 **`{func_name}`**\n```json\n{args_str}\n```\n")
                            think_ph.markdown("".join(all_reasoning_parts))

                            # 更新状态
                            status_ph.caption(f"🔧 `{func_name}`")

                            if func_name in ("replace_poi", "remove_poi", "add_poi",
                                             "reroute_day", "add_day", "remove_day"):
                                modified = True

                            tool_result = rp._call_tool(func_name, args)
                            preview = tool_result[:2000] if len(tool_result) > 2000 else tool_result

                            msgs.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result,
                            })

                            # show_map 即时更新地图
                            if func_name == "show_map":
                                map_paths = agent.refresh_trip_rendering(rp.result)
                                if map_paths.get("map_overview"):
                                    st.session_state.map_overview = map_paths["map_overview"]
                                if map_paths.get("map_days"):
                                    st.session_state.map_days = map_paths["map_days"]
                                    st.session_state._show_map_inline = True

                            status_ph.caption(f"📋 `{func_name}` 完成 ({preview[:80]}...)")

                            # ── 记录工具返回到思考面板 ──
                            all_reasoning_parts.append(f"\n📋 **`{func_name}` 返回**\n```json\n{preview[:500]}\n```\n")
                            think_ph.markdown("".join(all_reasoning_parts))

                            # 流式输出工具返回摘要
                            all_reply_parts.append(f"\n\n✅ `{func_name}` 执行完毕\n")
                            reply_ph.markdown("".join(all_reply_parts))

                    # ── 循环结束：构建最终回复 ──
                    if final_text is None:
                        final_text = "".join(all_reply_parts) or "操作已完成"

                    # 显示最终回复
                    reply_ph.markdown(final_text)

                    thinking_md = "".join(all_reasoning_parts)

                    # 追加到 chat_history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_text,
                        "thinking": thinking_md,
                    })

                    # 同步结果
                    st.session_state.trip_result = rp.result

                    # 重刷地图
                    map_paths = agent.refresh_trip_rendering(rp.result)
                    if map_paths.get("map_overview"):
                        st.session_state.map_overview = map_paths["map_overview"]
                    if map_paths.get("map_days"):
                        st.session_state.map_days = map_paths["map_days"]
                        st.session_state._show_map_inline = True

                    # 保存对话上下文
                    conv = []
                    for m in msgs:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if role in ("tool", "system"):
                            continue
                        if role == "assistant" and m.get("tool_calls"):
                            if content:
                                conv.append({"role": "assistant", "content": content})
                            continue
                        if role in ("user", "assistant"):
                            conv.append({"role": role, "content": content})
                    conv.append({"role": "assistant", "content": final_text})
                    st.session_state._rp_conversation = conv[-20:]

                    if modified:
                        rp._auto_save()

                    _save_chat()
                    st.rerun()

            else:
                # 无行程时：普通流式对话
                proxy = OpenaiCall(api_key=ck, base_url=st.session_state.api_base)
                with st.chat_message("assistant"):
                    response, thinking = _stream_assistant(proxy, messages, st.session_state.llm_model)
                msg = {"role": "assistant", "content": response or "(空回复)"}
                if thinking:
                    msg["thinking"] = thinking
                st.session_state.chat_history.append(msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
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
