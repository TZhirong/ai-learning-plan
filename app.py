"""Streamlit 智能学习计划定制系统 —— 主入口。

运行方式：
    streamlit run app.py

包含 4 个标签页：
    1. 计划定制 —— 采集学习信息，借助 DeepSeek 生成结构化学习计划
    2. 我的计划 —— 查看 / 管理已保存的学习计划与任务
    3. 学习打卡 —— 对任务进行每日打卡
    4. 数据看板 —— 学习数据统计概览

页面层只负责展示与交互，数据访问统一走 database.py，
AI 生成统一走 llm_client.py，配置统一走 config.py。
"""
import calendar
import datetime
import json
import time

import pandas as pd
import streamlit as st

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from database import (
    add_daily_checkin,
    add_quiz_result,
    add_task,
    add_wrong_answer,
    create_plan,
    delete_plan,
    get_checkin_dates,
    get_dashboard_stats,
    get_kp_explanations,
    get_quiz_stats,
    init_db,
    list_daily_checkins,
    list_plans,
    list_tasks,
    list_wrong_answers,
    mark_wrong_mastered,
    save_kp_explanations,
    update_task_status,
)
from llm_client import generate_kp_explanations, generate_learning_plan, generate_weekly_quiz
from knowledge_base import (
    add_document,
    delete_document,
    init_kb,
    kb_chunk_count,
    list_documents,
    retrieve,
)
from knowledge_graph import WEEK_COLORS, build_graph
from streamlit_echarts import JsCode, st_echarts
from browser_storage import get_saved_key, remove_saved_key, set_saved_key

# 表单选项常量
LEVEL_OPTIONS = ["零基础", "了解入门概念", "有一定实践经验", "有系统学习经历"]
STYLE_OPTIONS = ["偏实操", "偏理论", "均衡"]


# ==================== 工具函数 ====================
def _fmt(value):
    """格式化数字：去除多余的 .0，例如 2.0 -> 2。"""
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(value)


def get_effective_api_key() -> str:
    """返回当前生效的 API Key：侧边栏手动输入 > 环境变量。"""
    manual = st.session_state.get("manual_api_key", "").strip()
    return manual or DEEPSEEK_API_KEY


def plan_to_markdown(plan: dict) -> str:
    """将计划 dict 转为简洁的 Markdown 文本，用于存储到数据库。"""
    lines = []
    ov = plan.get("plan_overview", {})
    if ov:
        lines.append("### 计划总览")
        lines.append(f"- 总时长：{ov.get('total_duration', '—')}")
        lines.append(f"- 学习目标：{ov.get('learning_goal', '—')}")
        lines.append(f"- 每日时长：{_fmt(ov.get('daily_hours'))} 小时")
        lines.append(f"- 难度等级：{ov.get('difficulty_level', '—')}")
        if ov.get("core_knowledge_points"):
            lines.append(f"- 核心知识点：{'、'.join(ov['core_knowledge_points'])}")
        lines.append("")
    for week in plan.get("weekly_plan", []):
        lines.append(f"### 第 {week.get('week', '?')} 周：{week.get('week_theme', '')}")
        for task in week.get("daily_tasks", []):
            lines.append(
                f"- 第 {task.get('day', '?')} 天 {task.get('task_title', '')}"
                f"（{_fmt(task.get('duration'))} 小时）"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_plan_json(plan_json):
    """解析存储的结构化计划 JSON，失败或为空时返回 None。"""
    if not plan_json:
        return None
    try:
        data = json.loads(plan_json)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _inject_css():
    """注入科技蓝主题样式。"""
    st.markdown(
        """
        <style>
        :root { --primary: #165DFF; }
        .stApp { background: #f5f7fb; }
        .stApp h1, .stApp h2, .stApp h3 { color: #1d2129; font-weight: 600; }
        /* 主按钮 */
        .stButton > button[kind="primary"] {
            background-color: #165DFF;
            border: none;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(22, 93, 255, 0.3);
        }
        .stButton > button[kind="primary"]:hover { background-color: #0e42d2; }
        /* 指标卡片 */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6ebf5;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(22, 93, 255, 0.06);
        }
        [data-testid="stMetricValue"] { color: #165DFF; font-weight: 700; }
        /* 折叠面板卡片化 */
        [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e6ebf5;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }
        /* 带边框容器卡片化 */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(22, 93, 255, 0.06);
        }
        /* 标签页高亮 */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [aria-selected="true"] { color: #165DFF; }
        /* 侧边栏 */
        [data-testid="stSidebar"] {
            background: #f0f5ff;
            border-right: 1px solid #dce5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sync_browser_key():
    """首次成功读取浏览器中保存的 Key 后，回填到输入框并记录状态。"""
    if st.session_state.get("ls_initialized"):
        return
    res = get_saved_key()
    if res.get("loaded"):
        saved = res.get("value")
        st.session_state["browser_saved_key"] = saved
        # 仅当用户尚未手动输入时才回填，避免覆盖当前输入
        if saved and not st.session_state.get("manual_api_key"):
            st.session_state["manual_api_key"] = saved
        st.session_state["ls_initialized"] = True
    else:
        st.session_state.setdefault("browser_saved_key", None)


def _render_browser_persistence_controls():
    """渲染 API Key 的浏览器持久化控制：保存到浏览器 / 删除。"""
    current = st.session_state.get("manual_api_key", "").strip()
    saved = st.session_state.get("browser_saved_key")

    if saved:
        st.caption(f"💾 浏览器已保存 Key：`****{saved[-4:]}`")
    else:
        st.caption("💾 浏览器中暂无已保存的 Key")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 保存到浏览器", disabled=not current, width="stretch"):
            set_saved_key(current)
            st.session_state["browser_saved_key"] = current
            st.success("✅ 已保存到浏览器本地")
    with c2:
        if st.button("🗑️ 删除已保存", disabled=not saved, width="stretch"):
            remove_saved_key()
            st.session_state["browser_saved_key"] = None
            st.success("✅ 已删除浏览器中保存的 Key")


def render_sidebar():
    """渲染侧边栏：API 配置区、使用说明与状态提示。"""
    with st.sidebar:
        st.markdown("## ⚙️ 配置中心")

        # 从浏览器本地读取已保存的 Key，并回填到输入框
        _sync_browser_key()

        # 状态提示
        if get_effective_api_key():
            st.success("✅ 密钥已就绪，AI 功能可用")
        else:
            st.warning("⚠️ 未配置密钥，AI 生成不可用")

        # API 配置区
        with st.expander("🔑 API 配置", expanded=not get_effective_api_key()):
            st.text_input(
                "DeepSeek API Key",
                type="password",
                key="manual_api_key",
                placeholder="sk-...",
                help="优先级高于 .env 中的 DEEPSEEK_API_KEY；点击下方「保存到浏览器」可持久化到本地。",
            )
            st.caption(f"接口地址：{DEEPSEEK_BASE_URL}")
            st.caption(f"模型：{DEEPSEEK_MODEL}")

            # 浏览器持久化：保存 / 删除
            _render_browser_persistence_controls()

        # 使用说明
        with st.expander("📖 使用说明", expanded=False):
            st.markdown(
                """
                1. **计划定制**：填写学习目标等信息，AI 生成结构化学习计划并保存。
                2. **我的计划**：查看、管理已保存的学习计划与任务。
                3. **学习打卡**：按周 / 天记录每日学习进度。
                4. **数据看板**：查看学习统计与知识点掌握情况。
                """
            )

        st.divider()
        st.caption("💡 密钥默认仅本次会话有效；点击「保存到浏览器」后才会持久化到本地。")


# ==================== 标签页 1：计划定制 ====================
def render_plan_creator():
    """采集学习信息 -> 调用 AI 生成结构化计划 -> 展示与保存。"""
    st.subheader("📝 计划定制")
    st.caption("填写以下信息，AI 将为你生成一份结构化的个性化学习计划")

    # ---- 采集表单（两列布局，共 5 项信息） ----
    with st.form("plan_form"):
        col1, col2 = st.columns(2)
        with col1:
            learning_goal = st.text_input(
                "学习目标", placeholder="例如：3个月掌握Python数据分析"
            )
            current_level = st.selectbox("当前基础", LEVEL_OPTIONS)
            learning_style = st.selectbox("学习偏好", STYLE_OPTIONS)
        with col2:
            daily_hours = st.slider(
                "每日可用时长（小时）", 0.5, 8.0, 2.0, 0.5
            )
            weeks = st.slider("学习周期（周）", 1, 24, 8, 1)

        submitted = st.form_submit_button("✨ 生成学习计划", type="primary")

    # ---- 仅在点击「生成」后执行生成逻辑 ----
    if submitted:
        if not learning_goal.strip():
            st.warning("请先填写学习目标。")
        else:
            api_key = get_effective_api_key()
            if not api_key:
                st.error(
                    "⚠️ 未配置 DeepSeek API Key。请在左侧边栏填写，"
                    "或在 .env 中配置 DEEPSEEK_API_KEY。"
                )
            else:
                # 检索知识库（空知识库自动返回空列表，降级为通用模式）
                context = retrieve(learning_goal.strip(), top_k=5)

                # 流式生成：逐段展示内容，减少等待焦虑
                st.caption("⏳ 正在生成学习计划（流式输出）……")
                if context:
                    st.info(f"📚 已从知识库检索到 {len(context)} 段相关内容，知识点将标注资料来源。")
                streamed = []
                placeholder = st.empty()
                last_flush = [0.0]  # 节流：限制前端重绘频率

                def on_chunk(delta):
                    streamed.append(delta)
                    now = time.time()
                    # 最多约每 0.15 秒重绘一次，避免逐字符刷屏导致卡顿
                    if now - last_flush[0] >= 0.15:
                        placeholder.code("".join(streamed), language="json")
                        last_flush[0] = now

                try:
                    plan = generate_learning_plan(
                        learning_goal=learning_goal.strip(),
                        current_level=current_level,
                        learning_style=learning_style,
                        daily_hours=float(daily_hours),
                        weeks=int(weeks),
                        api_key=api_key,
                        on_chunk=on_chunk,
                        context=context,
                    )
                except ValueError as e:
                    st.error(f"❌ {e}")
                    return
                except Exception:
                    st.error("❌ 生成失败，请稍后重试或检查网络连接。")
                    return
                finally:
                    placeholder.empty()

                st.success("✅ 生成完成")

                # 暂存结果与元信息，供下方渲染与保存使用
                st.session_state["generated_plan"] = plan
                st.session_state["plan_meta"] = {
                    "goal": learning_goal.strip(),
                    "daily_hours": float(daily_hours),
                    "weeks": int(weeks),
                }
                st.session_state["plan_saved"] = False

    # ---- 渲染已有结果（刚生成或之前生成但尚未刷新） ----
    plan = st.session_state.get("generated_plan")
    if plan:
        render_plan_result(plan)


def render_plan_result(plan: dict):
    """渲染结构化学习计划，并提供保存入口。"""
    st.markdown("## 📋 生成结果")

    # 计划总览
    overview = plan.get("plan_overview", {})
    if overview:
        st.markdown("### 📌 计划总览")
        c1, c2, c3 = st.columns(3)
        c1.metric("学习周期", overview.get("total_duration", "—"))
        c2.metric("每日时长", f"{_fmt(overview.get('daily_hours'))} 小时")
        c3.metric("难度等级", overview.get("difficulty_level", "—"))
        if overview.get("learning_goal"):
            st.markdown(f"**学习目标**：{overview['learning_goal']}")
        if overview.get("core_knowledge_points"):
            st.markdown(
                "**核心知识点**：" + "、".join(overview["core_knowledge_points"])
            )

    # 周计划（每周一个折叠面板）
    weekly = plan.get("weekly_plan", [])
    if weekly:
        st.markdown("### 🗓️ 周计划")
        for week in weekly:
            with st.expander(
                f"第 {week.get('week', '?')} 周 · {week.get('week_theme', '')}"
            ):
                for task in week.get("daily_tasks", []):
                    st.markdown(
                        f"**第 {task.get('day', '?')} 天 · {task.get('task_title', '')}"
                        f"（{_fmt(task.get('duration'))} 小时）**"
                    )
                    if task.get("knowledge_points"):
                        st.markdown(
                            "知识点：" + "、".join(task["knowledge_points"])
                        )
                    if task.get("practice_task"):
                        st.markdown(f"🛠️ 实操：{task['practice_task']}")
                    if task.get("review_note"):
                        st.markdown(f"🔁 复习：{task['review_note']}")
                    st.divider()

    # 复习安排
    review = plan.get("review_schedule", {})
    if review and (review.get("key_nodes") or review.get("review_method")):
        st.markdown("### 🔁 复习安排")
        if review.get("key_nodes"):
            st.markdown("关键节点：" + "、".join(review["key_nodes"]))
        if review.get("review_method"):
            st.markdown(f"复习方法：{review['review_method']}")

    # 个性化建议
    suggestions = plan.get("learning_suggestions", [])
    if suggestions:
        st.markdown("### 💡 个性化建议")
        for s in suggestions:
            st.markdown(f"- {s}")

    # 保存入口（避免重复保存）
    if not st.session_state.get("plan_saved", False):
        if st.button("💾 保存到「我的计划」", type="primary"):
            meta = st.session_state.get("plan_meta", {})
            title = (meta.get("goal", "") or "未命名计划")[:30]
            plan_id = create_plan(
                title=title,
                goal=meta.get("goal", ""),
                description=plan_to_markdown(plan),
                plan_json=json.dumps(plan, ensure_ascii=False),
            )
            st.session_state["plan_saved"] = True
            st.success(f"已保存学习计划（ID: {plan_id}），可在「我的计划」中查看。")
    else:
        st.caption("✅ 该计划已保存到「我的计划」。")


# ==================== 标签页 2：我的计划 ====================
def render_my_plans():
    """以卡片形式结构化展示已保存的学习计划。"""
    st.subheader("📁 我的计划")

    plans = list_plans()
    if not plans:
        # 空状态引导
        st.info("📭 暂无学习计划")
        st.markdown(
            "前往 **📝 计划定制** 标签页，填写学习信息并生成你的第一份个性化学习计划。"
        )
        return

    for plan in plans:
        render_plan_card(plan)


def render_plan_card(plan: dict):
    """渲染单个学习计划卡片：结构化 JSON 优先，回退旧版描述展示。"""
    plan_json = _parse_plan_json(plan.get("plan_json"))

    with st.container(border=True):
        # 卡片头部
        st.markdown(f"### 📌 {plan['title']}")
        st.caption(f"创建于 {plan['created_at']}")

        if plan_json:
            _render_structured_plan(plan_json, plan["id"])
        else:
            # 旧版 / 无结构化数据：回退到描述展示
            if plan.get("goal"):
                st.markdown(f"**目标**：{plan['goal']}")
            if plan.get("description"):
                st.markdown("**计划详情**")
                st.markdown(plan["description"])

        st.divider()

        # 任务清单管理
        _render_task_management(plan)


def _render_structured_plan(plan: dict, plan_id: int):
    """渲染结构化计划的四部分：总览卡片、目标与知识点、周计划（含周测）、复习与建议。"""
    overview = plan.get("plan_overview", {})
    knowledge_points = overview.get("core_knowledge_points", []) or []

    # 1. 顶部 4 个数据卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("学习周期", overview.get("total_duration", "—"))
    c2.metric("每日时长", f"{_fmt(overview.get('daily_hours'))} 小时")
    c3.metric("难度等级", overview.get("difficulty_level", "—"))
    c4.metric("核心知识点", f"{len(knowledge_points)} 个")

    # 2. 学习目标 + 核心知识点（折叠面板）
    with st.expander("🎯 学习目标与核心知识点", expanded=True):
        if overview.get("learning_goal"):
            st.markdown(f"**学习目标**：{overview['learning_goal']}")
        if knowledge_points:
            st.markdown("**核心知识点**：")
            for kp in knowledge_points:
                st.markdown(f"- {kp}")

    # 3. 周计划（按周折叠，周内按天列出任务）
    weekly = plan.get("weekly_plan", [])
    if weekly:
        st.markdown("#### 🗓️ 周计划")
        for week in weekly:
            theme = week.get("week_theme", "")
            with st.expander(f"第 {week.get('week', '?')} 周 · {theme}"):
                for task in week.get("daily_tasks", []):
                    with st.container(border=True):
                        st.markdown(
                            f"**第 {task.get('day', '?')} 天 · {task.get('task_title', '')}"
                            f"（{_fmt(task.get('duration'))} 小时）**"
                        )
                        if task.get("knowledge_points"):
                            st.markdown(
                                "**知识点**：" + "、".join(task["knowledge_points"])
                            )
                        if task.get("practice_task"):
                            st.markdown(f"🛠️ **实操**：{task['practice_task']}")
                        if task.get("review_note"):
                            st.markdown(f"🔁 **复习**：{task['review_note']}")

                # 周度自测
                _render_quiz_section(plan_id, week)

    # 4. 整体复习安排 + 个性化建议
    review = plan.get("review_schedule", {})
    suggestions = plan.get("learning_suggestions", []) or []
    if review or suggestions:
        col_review, col_suggest = st.columns(2)
        with col_review:
            st.markdown("#### 🔁 整体复习安排")
            if review.get("key_nodes"):
                st.markdown("**关键节点**：")
                for n in review["key_nodes"]:
                    st.markdown(f"- {n}")
            if review.get("review_method"):
                st.markdown(f"**复习方法**：{review['review_method']}")
        with col_suggest:
            st.markdown("#### 💡 个性化建议")
            for s in suggestions:
                st.markdown(f"- {s}")


def _render_task_management(plan: dict):
    """渲染计划下的任务清单、新增任务与删除入口。"""
    st.markdown("#### ✅ 任务清单")

    tasks = list_tasks(plan_id=plan["id"])
    if tasks:
        for task in tasks:
            done = task["status"] == "done"
            label = f"✅ {task['title']}" if done else f"⏳ {task['title']}"
            new_done = st.checkbox(label, value=done, key=f"task_{task['id']}")
            if new_done != done:
                update_task_status(task["id"], "done" if new_done else "pending")
                st.rerun()
    else:
        st.caption("（暂无手动添加的任务，可在下方添加）")

    # 手动新增任务
    with st.form(f"add_task_{plan['id']}"):
        c1, c2 = st.columns([3, 1])
        with c1:
            task_title = st.text_input("任务名称", key=f"task_title_{plan['id']}")
        with c2:
            minutes = st.number_input(
                "预计时长(分钟)",
                min_value=5,
                value=30,
                step=5,
                key=f"task_min_{plan['id']}",
            )
        if st.form_submit_button("➕ 添加任务"):
            if task_title.strip():
                add_task(plan["id"], task_title.strip(), int(minutes))
                st.success("任务已添加。")
                st.rerun()
            else:
                st.warning("任务名称不能为空。")

    # 删除入口
    if st.button("🗑️ 删除该计划", key=f"del_{plan['id']}"):
        delete_plan(plan["id"])
        st.success("计划已删除。")
        st.rerun()


def _render_quiz_section(plan_id, week):
    """在每周任务下方渲染「开始周测」按钮与自测界面。"""
    week_no = week.get("week")
    tasks = week.get("daily_tasks", []) or []
    kps = []
    for t in tasks:
        for kp in (t.get("knowledge_points", []) or []):
            if kp and kp not in kps:
                kps.append(kp)
    if not kps:
        return

    quiz_key = f"quiz_{plan_id}_{week_no}"

    if st.button("🧪 开始周测", key=f"start_quiz_{plan_id}_{week_no}"):
        api_key = get_effective_api_key()
        if not api_key:
            st.error("⚠️ 未配置 DeepSeek API Key，请在左侧边栏填写。")
        else:
            with st.spinner("正在生成周测题目……"):
                try:
                    quiz = generate_weekly_quiz(kps, week.get("week_theme", ""), api_key=api_key)
                    st.session_state[quiz_key] = {
                        "questions": quiz.get("questions", []),
                        "submitted": False,
                    }
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception:
                    st.error("❌ 周测生成失败，请稍后重试。")

    quiz_state = st.session_state.get(quiz_key)
    if quiz_state:
        _render_quiz(quiz_state, plan_id, week_no, quiz_key)


def _render_quiz(quiz_state, plan_id, week_no, quiz_key):
    """渲染周测作答界面与判分结果。"""
    questions = quiz_state.get("questions", [])
    if not questions:
        return

    st.markdown("#### 🧪 周度自测")

    if not quiz_state.get("submitted"):
        # 作答界面
        answers = {}
        for i, q in enumerate(questions):
            qtype = q.get("type")
            if qtype == "choice":
                answers[i] = st.radio(
                    f"**{i + 1}. {q.get('question', '')}**",
                    q.get("options", []),
                    key=f"quiz_ans_{plan_id}_{week_no}_{i}",
                    index=None,
                )
            else:
                st.markdown(f"**{i + 1}. {q.get('question', '')}**（编程题）")
                answers[i] = st.checkbox(
                    "✅ 我已完成并做对",
                    key=f"quiz_prog_{plan_id}_{week_no}",
                    value=False,
                )

        if st.button("📝 提交自测", key=f"submit_quiz_{plan_id}_{week_no}", type="primary"):
            total = len(questions)
            correct = 0
            details = []
            for i, q in enumerate(questions):
                qtype = q.get("type")
                if qtype == "choice":
                    user = answers[i]
                    ok = user == q.get("answer")
                    if ok:
                        correct += 1
                    else:
                        add_wrong_answer(
                            plan_id, week_no,
                            question=q.get("question", ""),
                            options=json.dumps(q.get("options", []), ensure_ascii=False),
                            correct_answer=q.get("answer", ""),
                            user_answer=user or "",
                            explanation=q.get("explanation", ""),
                            knowledge_point=q.get("knowledge_point", ""),
                            question_type="choice",
                        )
                    details.append({
                        "question": q.get("question", ""),
                        "user": user,
                        "correct_answer": q.get("answer", ""),
                        "explanation": q.get("explanation", ""),
                        "is_correct": ok,
                        "qtype": "choice",
                    })
                else:
                    ok = bool(answers[i])
                    if ok:
                        correct += 1
                    else:
                        add_wrong_answer(
                            plan_id, week_no,
                            question=q.get("question", ""),
                            options="[]",
                            correct_answer=q.get("reference_answer", ""),
                            user_answer="",
                            explanation=q.get("explanation", ""),
                            knowledge_point=q.get("knowledge_point", ""),
                            question_type="programming",
                        )
                    details.append({
                        "question": q.get("question", ""),
                        "user": "已完成" if ok else "未完成",
                        "correct_answer": q.get("reference_answer", ""),
                        "explanation": q.get("explanation", ""),
                        "is_correct": ok,
                        "qtype": "programming",
                    })

            score = round(correct / total * 100, 1) if total else 0.0
            add_quiz_result(plan_id, week_no, total, correct, score)
            quiz_state["submitted"] = True
            quiz_state["result"] = {
                "score": score, "correct": correct, "total": total, "details": details,
            }
            st.rerun()

    else:
        # 结果展示
        result = quiz_state.get("result", {})
        st.markdown(
            f"### 🎯 得分：**{result.get('score', 0)} 分**（答对 {result.get('correct')}/{result.get('total')}）"
        )
        for i, d in enumerate(result.get("details", [])):
            icon = "✅" if d.get("is_correct") else "❌"
            st.markdown(f"**{i + 1}. {icon} {d.get('question', '')}**")
            if d.get("qtype") == "choice":
                st.markdown(
                    f"- 你的答案：{d.get('user') or '未作答'} ｜ 正确答案：{d.get('correct_answer')}"
                )
            else:
                st.markdown(f"- 状态：{d.get('user')} ｜ 参考答案：")
                st.code(d.get("correct_answer", "") or "", language="python")
            st.markdown(f"- 解析：{d.get('explanation', '')}")
            st.divider()
        if st.button("🔁 重新自测", key=f"retake_quiz_{plan_id}_{week_no}"):
            st.session_state.pop(quiz_key, None)
            st.rerun()


# ==================== 标签页 3：学习打卡 ====================
def _get_active_plan():
    """返回当前学习计划。

    优先选择最新的、含结构化数据（plan_json）的计划，用于打卡 / 图谱等功能；
    若全部为旧版无结构化计划，则回退任意计划；无计划返回 None。
    """
    plans = list_plans()
    if not plans:
        return None
    for p in plans:
        if p.get("plan_json"):
            return p
    return plans[0]


def _render_calendar(checkin_dates):
    """渲染当月日历，已完成打卡的日期标记为绿色。"""
    today = datetime.date.today()
    year, month = today.year, today.month
    weeks = calendar.monthcalendar(year, month)
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]

    html = [
        '<table style="width:100%;border-collapse:separate;'
        'border-spacing:4px;text-align:center;">'
    ]
    html.append(
        "<tr>"
        + "".join(
            f"<th style='padding:6px;color:#666;font-weight:600;'>{w}</th>"
            for w in weekdays
        )
        + "</tr>"
    )
    for week in weeks:
        html.append("<tr>")
        for d in week:
            if d == 0:
                html.append("<td></td>")
                continue
            ds = datetime.date(year, month, d).isoformat()
            if ds in checkin_dates:
                cell = (
                    f"<span style='display:inline-block;width:32px;height:32px;"
                    f"line-height:32px;background:#28a745;color:#fff;"
                    f"border-radius:6px;font-weight:bold;'>{d}</span>"
                )
            else:
                cell = (
                    f"<span style='display:inline-block;width:32px;height:32px;"
                    f"line-height:32px;'>{d}</span>"
                )
            html.append(f"<td style='padding:4px;'>{cell}</td>")
        html.append("</tr>")
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_checkin():
    """选择周数 / 天数打卡，并展示打卡统计与当月日历。"""
    st.subheader("✅ 学习打卡")

    plan = _get_active_plan()
    if not plan:
        st.info("暂无学习计划。请先在「计划定制」生成并保存计划。")
        return

    plan_json = _parse_plan_json(plan.get("plan_json"))
    weekly = (plan_json or {}).get("weekly_plan", []) or []
    if not weekly:
        st.info("当前计划暂无结构化周任务，无法打卡。")
        return

    # 周数 / 天数选择
    week_options = [w.get("week") for w in weekly if w.get("week") is not None]
    selected_week = st.selectbox("选择周数", week_options, key="checkin_week")

    week_days = []
    for w in weekly:
        if w.get("week") == selected_week:
            week_days = [
                t.get("day")
                for t in (w.get("daily_tasks", []) or [])
                if t.get("day") is not None
            ]
            break
    if not week_days:
        st.info("该周暂无任务安排。")
        return

    selected_day = st.selectbox("选择天数", week_days, key="checkin_day")

    if st.button("📅 打卡", type="primary"):
        ok = add_daily_checkin(plan["id"], int(selected_week), int(selected_day))
        if ok:
            st.success(f"✅ 已打卡：第 {selected_week} 周 · 第 {selected_day} 天")
        else:
            st.info("该天已完成打卡，无需重复打卡。")

    # 统计指标
    stats = get_dashboard_stats()
    st.markdown("### 📊 打卡统计")
    c1, c2, c3 = st.columns(3)
    c1.metric("累计打卡天数", stats["total_study_days"])
    c2.metric("总完成率", f"{stats['total_completion']:.1f}%")
    c3.metric("连续打卡天数", stats["longest_streak"])
    st.markdown("**整体进度**")
    st.progress(min(stats["total_completion"] / 100, 1.0))
    st.caption(f"已完成 {stats['total_checkins']} / {stats['total_planned_days']} 天")

    # 当月日历视图
    st.markdown("### 🗓️ 本月打卡日历")
    _render_calendar(get_checkin_dates())


# ==================== 标签页 4：数据看板 ====================
def render_dashboard():
    """展示学习数据统计：核心指标、趋势折线图、知识点掌握进度。"""
    st.subheader("📊 数据看板")

    stats = get_dashboard_stats()

    # 无计划（无任何学习数据）时显示空状态引导
    if stats["total_planned_days"] == 0:
        st.info("📭 暂无学习数据")
        st.markdown(
            "前往 **📝 计划定制** 生成学习计划，再到 **✅ 学习打卡** 记录每日进度，"
            "即可在此查看学习统计。"
        )
        return

    # 1. 顶部 4 个核心指标卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计学习天数", f"{stats['total_study_days']} 天")
    c2.metric("总完成率", f"{stats['total_completion']:.1f}%")
    c3.metric("平均周完成率", f"{stats['avg_week_completion']:.1f}%")
    c4.metric("最长连续打卡", f"{stats['longest_streak']} 天")

    st.divider()

    # 2. 近7天完成率趋势折线图（Streamlit 原生图表）
    st.markdown("#### 📈 近7天完成率趋势")
    trend = stats["recent_7"]
    if trend:
        df = pd.DataFrame(
            {"完成率(%)": [r["rate"] for r in trend]},
            index=[r["date"] for r in trend],
        )
        st.line_chart(df)
    else:
        st.caption("暂无近7天数据。")

    st.divider()

    # 3. 知识点掌握进度概览
    st.markdown("#### 🧠 知识点掌握进度")
    knowledge = stats["knowledge"]
    if knowledge["total"] > 0:
        st.markdown(
            f"**已掌握 {knowledge['mastered']} / {knowledge['total']} 个知识点**"
        )
        st.progress(knowledge["mastered"] / knowledge["total"])
        for p in knowledge["points"]:
            icon = "✅" if p["mastered"] else "⬜"
            st.markdown(f"{icon} {p['name']}")
    else:
        st.caption("暂无知识点数据。")

    st.divider()

    # 4. 周度自测成绩
    st.markdown("#### 📝 周度自测成绩")
    qs = get_quiz_stats()
    if qs["total_tests"] > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("自测次数", qs["total_tests"])
        c2.metric("平均成绩", f"{qs['avg_score']} 分")
        c3.metric("最佳成绩", f"{qs['best_score']} 分")
        recent = qs["recent"]
        if recent:
            df = pd.DataFrame(
                {"成绩": [r["score"] for r in reversed(recent)]},
                index=[f"第{r['week']}周" for r in reversed(recent)],
            )
            st.line_chart(df)
    else:
        st.caption("暂无自测成绩，前往「我的计划」开始周测吧。")

    # 有计划但尚未打卡时的补充提示
    if stats["total_checkins"] == 0:
        st.info("还没有打卡记录，前往「✅ 学习打卡」开始记录吧。")


# ==================== 标签页 5：知识库 ====================
def render_knowledge_base():
    """上传教材/大纲文档，管理知识库，为 RAG 生成提供检索语料。"""
    st.subheader("📚 知识库")
    st.caption("上传 PDF / Word / Markdown 教材或课程大纲，AI 生成计划时将自动检索并标注资料来源")

    # 上传（用文件签名去重，避免每次重跑重复入库）
    uploaded = st.file_uploader(
        "上传文档（PDF / Word / Markdown）",
        type=["pdf", "docx", "md", "markdown"],
        accept_multiple_files=False,
        key="kb_uploader",
    )
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get("kb_last_upload") != sig:
            with st.spinner("正在解析并向量化文档……"):
                try:
                    doc_id, chunk_count = add_document(uploaded.name, uploaded.getvalue())
                    st.session_state["kb_last_upload"] = sig
                    st.success(f"✅ 已入库「{uploaded.name}」（{chunk_count} 个文本块）")
                except Exception:
                    st.error("❌ 文档解析失败，请确认文件格式正确且未损坏。")

    # 统计与空状态
    docs = list_documents()
    if docs:
        st.markdown(f"**已收录 {len(docs)} 个文档 · {kb_chunk_count()} 个文本块**")
    else:
        st.info("📭 知识库为空。上传教材或课程大纲后，生成学习计划时将自动检索相关内容。")

    # 文档列表与删除
    for d in docs:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"📄 **{d['name']}**")
                st.caption(
                    f"{d['file_type'].upper()} · {d['chunk_count']} 个文本块 · 创建于 {d['created_at']}"
                )
            with c2:
                if st.button("🗑️ 删除", key=f"del_doc_{d['id']}"):
                    delete_document(d["id"])
                    st.success("已删除文档。")
                    st.rerun()


# ==================== 标签页 6：错题本 ====================
def render_wrong_answers():
    """展示错题本，支持重新练习。"""
    st.subheader("📕 错题本")
    wrong = list_wrong_answers(status="wrong")
    if not wrong:
        st.info("🎉 暂无错题，继续保持！")
        return

    st.markdown(f"共 **{len(wrong)}** 道错题待复习")

    for wa in wrong:
        wa_id = wa["id"]
        title = wa["knowledge_point"] or "未分类"
        with st.expander(f"❌ {title} · {wa['question'][:40]}..."):
            st.markdown(f"**题目**：{wa['question']}")
            if wa["question_type"] == "choice":
                options = json.loads(wa["options"] or "[]")
                st.markdown("**选项**：")
                for op in options:
                    st.markdown(f"- {op}")
                st.markdown(
                    f"**你的答案**：{wa['user_answer'] or '未作答'} ｜ **正确答案**：{wa['correct_answer']}"
                )
                retry = st.radio("重新作答", options, key=f"retry_{wa_id}", index=None)
                if st.button("提交重练", key=f"retry_btn_{wa_id}"):
                    if retry == wa["correct_answer"]:
                        mark_wrong_mastered(wa_id)
                        st.success("✅ 回答正确，已移出错题本！")
                        st.rerun()
                    else:
                        st.error("❌ 仍不正确，查看解析后再试试。")
            else:
                st.markdown("**参考答案**：")
                st.code(wa["correct_answer"] or "", language="python")
                if st.button("✅ 已掌握", key=f"master_{wa_id}"):
                    mark_wrong_mastered(wa_id)
                    st.success("已标记为掌握。")
                    st.rerun()
            st.markdown(f"**解析**：{wa['explanation']}")


# ==================== 标签页 7：知识图谱 ====================
def render_knowledge_graph():
    """知识图谱可视化：解析知识点前置依赖、周分色、打卡高亮。"""
    st.subheader("🕸️ 知识图谱")
    st.caption("知识点前置依赖关系 · 节点按周分色 · 绿框为已打卡 · 支持拖拽缩放")

    plan = _get_active_plan()
    if not plan:
        st.info("暂无学习计划，请先在「计划定制」生成计划。")
        return
    plan_json = _parse_plan_json(plan.get("plan_json"))
    if not plan_json:
        st.info("当前计划无结构化数据，无法生成图谱。")
        return

    # 已打卡的 (week, day) 集合
    checked_pairs = set()
    for c in list_daily_checkins(plan_id=plan["id"]):
        checked_pairs.add((c["week"], c["day"]))

    g = build_graph(plan_json, checked_pairs)
    if not g["nodes"]:
        st.info("计划中暂无知识点数据。")
        return

    # 获取知识点解释（缓存优先，缺失批量生成并入库）
    names = g["seq"]
    explanations = get_kp_explanations(names)
    missing = [n for n in names if n not in explanations]
    if missing:
        api_key = get_effective_api_key()
        if api_key:
            with st.spinner(f"正在为 {len(missing)} 个知识点生成解释……"):
                try:
                    generated = generate_kp_explanations(missing, api_key=api_key)
                    save_kp_explanations(generated)
                    explanations.update(generated)
                except Exception:
                    pass
    for node in g["nodes"]:
        node["explanation"] = explanations.get(node["name"], "")

    # ECharts 有向关系图谱
    tooltip_formatter = JsCode(
        """
        function(params) {
            if (params.dataType === 'node') {
                var d = params.data;
                var html = '<div style="max-width:360px;line-height:1.7;">';
                html += '<b style="font-size:14px;color:#165DFF;">' + d.name + '</b><br/>';
                if (d.explanation) {
                    html += '<div style="margin:4px 0;color:#333;white-space:normal;word-break:break-word;overflow-wrap:break-word;">' + d.explanation + '</div>';
                }
                html += '<div style="color:#86909c;font-size:12px;">第 ' + d.week + ' 周 · ' + (d.checked ? '已打卡' : '未打卡') + '</div>';
                html += '</div>';
                return html;
            }
            return '';
        }
        """
    )
    options = {
        "color": WEEK_COLORS,
        "tooltip": {
            "trigger": "item",
            "formatter": tooltip_formatter,
            "backgroundColor": "rgba(255,255,255,0.96)",
            "borderColor": "#165DFF",
            "borderWidth": 1,
            "textStyle": {"color": "#333", "fontSize": 12},
            "confine": True,
            "extraCssText": "box-shadow: 0 2px 8px rgba(22,93,255,0.15); border-radius: 8px; max-width: 400px; white-space: normal; word-break: break-word; overflow-wrap: break-word;",
        },
        "legend": [{"data": [c["name"] for c in g["categories"]], "top": 10}],
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,           # 缩放 + 平移
            "draggable": True,      # 拖拽节点
            "data": g["nodes"],
            "links": g["links"],
            "categories": g["categories"],
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": [0, 8],
            "label": {"show": True, "position": "bottom", "fontSize": 11, "color": "#333"},
            "force": {"repulsion": 300, "edgeLength": 130, "gravity": 0.08},
            "emphasis": {"focus": "adjacency"},
            "lineStyle": {"color": "source", "curveness": 0.15, "width": 1.5},
        }],
    }

    events = {
        "click": "function(params) { if (params.dataType === 'node') { return params.data; } return null; }"
    }
    result = st_echarts(options=options, events=events, height="600px", key="knowledge_graph")

    # 点击节点查看详情
    clicked = None
    if isinstance(result, dict):
        clicked = result.get("chart_event")
    if clicked and isinstance(clicked, dict) and clicked.get("name"):
        name = clicked["name"]
        week = clicked.get("week")
        checked = clicked.get("checked", False)
        pre = [l["source"] for l in g["links"] if l["target"] == name]
        succ = [l["target"] for l in g["links"] if l["source"] == name]
        with st.container(border=True):
            st.markdown(f"### 📌 {name}")
            st.markdown(f"- **所属周数**：第 {week} 周")
            st.markdown(f"- **掌握状态**：{'✅ 已打卡' if checked else '⏳ 未打卡'}")
            if pre:
                st.markdown(f"- **前置依赖**：{pre[0]}")
            if succ:
                st.markdown(f"- **后续知识点**：{succ[0]}")
    else:
        st.caption("💡 点击图中节点查看详情")


# ==================== 主函数 ====================
def main():
    """应用入口：配置页面、渲染侧边栏、初始化数据库、渲染 4 个标签页。"""
    # 页面基础配置（必须放在所有 st 组件之前）
    st.set_page_config(
        page_title="智能学习计划定制系统",
        page_icon="📚",
        layout="wide",
    )

    # 注入科技蓝主题样式
    _inject_css()

    # 初始化数据库表结构
    init_db()

    # 初始化知识库表结构
    init_kb()

    # 侧边栏（API Key 设置）
    render_sidebar()

    st.title("📚 智能学习计划定制系统")
    st.caption("基于 AI 的个性化学习计划规划与打卡跟踪工具")

    # 未配置 API Key 时给出引导
    if not get_effective_api_key():
        st.warning(
            "⚠️ 未检测到 DeepSeek API Key，AI 计划生成功能不可用。"
            "请在左侧边栏填写，或复制 `.env.example` 为 `.env` 并填入密钥后重启应用。"
        )

    # 四个标签页
    tab_plan, tab_my_plans, tab_checkin, tab_dashboard, tab_kb, tab_wrong, tab_graph = st.tabs(
        ["📝 计划定制", "📁 我的计划", "✅ 学习打卡", "📊 数据看板", "📚 知识库", "📕 错题本", "🕸️ 知识图谱"]
    )

    with tab_plan:
        render_plan_creator()
    with tab_my_plans:
        render_my_plans()
    with tab_checkin:
        render_checkin()
    with tab_dashboard:
        render_dashboard()
    with tab_kb:
        render_knowledge_base()
    with tab_wrong:
        render_wrong_answers()
    with tab_graph:
        render_knowledge_graph()


if __name__ == "__main__":
    main()
