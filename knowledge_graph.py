"""知识图谱模块：从学习计划解析知识点，构建有向图数据。

- 节点：计划中的知识点（去重，按首次出现顺序）
- 有向边：按学习顺序构建「前置依赖」（前一个知识点 -> 后一个知识点）
- 分色：节点按所属周数着色
- 高亮：已打卡的知识点用绿色边框 + 发光标记
"""
from collections import OrderedDict

# 周节点配色（循环使用，科技蓝系 + 互补色）
WEEK_COLORS = [
    "#165DFF", "#14C9C9", "#F7BA1E", "#F53F3F", "#722ED1",
    "#0FC6C2", "#3491FA", "#FADC19", "#F77234", "#9FDB1D",
    "#F5319D", "#86909C",
]


def build_graph(plan_json: dict, checked_pairs: set) -> dict:
    """构建知识图谱数据。

    参数:
        plan_json:     计划的结构化 JSON（含 weekly_plan）
        checked_pairs: 已打卡的 (week, day) 集合

    返回:
        {"nodes", "links", "categories", "seq", "node_map"}
    """
    weekly = plan_json.get("weekly_plan", []) or []

    # 1. 收集知识点：首次出现周数、所有出现位置、简要介绍
    kp_first_week = OrderedDict()  # name -> week（保持顺序）
    kp_occurrences = {}            # name -> [(week, day), ...]
    kp_intro = {}                  # name -> {"task", "practice", "review"}
    for w in weekly:
        week = w.get("week")
        for t in (w.get("daily_tasks", []) or []):
            day = t.get("day")
            for kp in (t.get("knowledge_points", []) or []):
                kp = str(kp).strip()
                if not kp:
                    continue
                if kp not in kp_first_week:
                    kp_first_week[kp] = week
                    kp_intro[kp] = {
                        "task": t.get("task_title", "") or "",
                        "practice": t.get("practice_task", "") or "",
                        "review": t.get("review_note", "") or "",
                    }
                kp_occurrences.setdefault(kp, []).append((week, day))

    seq = list(kp_first_week.keys())
    if not seq:
        return {"nodes": [], "links": [], "categories": [], "seq": [], "node_map": {}}

    # 2. 每周一个分类（用于分色 + 图例）
    weeks = sorted({w for w in kp_first_week.values() if w is not None})
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    categories = [{"name": f"第{w}周"} for w in weeks]

    # 3. 节点
    node_map = {}
    nodes = []
    for kp in seq:
        week = kp_first_week[kp]
        checked = any((w, d) in checked_pairs for (w, d) in kp_occurrences[kp])
        cat = week_to_idx.get(week, 0)
        intro = kp_intro.get(kp, {})
        nodes.append({
            "name": kp,
            "week": week,
            "checked": checked,
            "category": cat,
            "symbolSize": 55 if checked else 45,
            "task": intro.get("task", ""),
            "practice": intro.get("practice", ""),
            "review": intro.get("review", ""),
            "itemStyle": {
                "borderColor": "#28a745" if checked else "rgba(0,0,0,0.15)",
                "borderWidth": 3 if checked else 1,
                "shadowBlur": 12 if checked else 0,
                "shadowColor": "rgba(40,167,69,0.5)" if checked else "transparent",
            },
        })
        node_map[kp] = {"week": week, "checked": checked}

    # 4. 有向边：按学习顺序（前置依赖）
    links = [{"source": seq[i], "target": seq[i + 1]} for i in range(len(seq) - 1)]

    return {
        "nodes": nodes,
        "links": links,
        "categories": categories,
        "seq": seq,
        "node_map": node_map,
    }
