"""数据库层模块。

使用 Python 标准库 sqlite3（无需额外安装），负责：
- 初始化数据库与表结构
- 提供学习计划 / 学习任务 / 打卡记录的增删改查接口

业务逻辑统一通过本模块访问数据库，页面层不直接拼 SQL。
"""
import datetime
import json
import sqlite3
from contextlib import contextmanager

from config import DB_PATH


@contextmanager
def get_conn():
    """获取数据库连接（上下文管理器，自动提交 / 回滚 / 关闭）。

    用法：
        with get_conn() as conn:
            conn.execute(...)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果支持按列名访问，如 row["title"]
    conn.execute("PRAGMA foreign_keys = ON")  # 启用外键约束，使 ON DELETE CASCADE 生效
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """初始化数据库表结构（幂等：表已存在则跳过，可重复调用）。"""
    with get_conn() as conn:
        cur = conn.cursor()

        # 学习计划表：一条计划对应一个学习目标
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,                -- 计划标题
                goal        TEXT,                         -- 学习目标
                description TEXT,                         -- 计划描述
                plan_json   TEXT,                         -- 结构化计划 JSON（用于结构化展示）
                created_at  TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )

        # 学习任务表：隶属于某个计划，可单独跟踪完成状态
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id           INTEGER NOT NULL,      -- 所属计划 id
                title             TEXT NOT NULL,          -- 任务名称
                estimated_minutes INTEGER DEFAULT 30,     -- 预计时长（分钟）
                status            TEXT DEFAULT 'pending',  -- pending=未完成 / done=已完成
                FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )

        # 每日打卡表：记录学习计划中某周某天的完成状态
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_checkins (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id      INTEGER NOT NULL,            -- 所属计划 id
                week         INTEGER NOT NULL,            -- 周数
                day          INTEGER NOT NULL,            -- 天数（第几天）
                status       TEXT NOT NULL DEFAULT 'done',-- 完成状态
                checkin_time TEXT NOT NULL DEFAULT (datetime('now', 'localtime')), -- 打卡时间
                UNIQUE(plan_id, week, day),               -- 同一天只打卡一次
                FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )

        # 周测成绩表
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id         INTEGER NOT NULL,
                week            INTEGER NOT NULL,
                total_questions INTEGER DEFAULT 6,
                correct_count   INTEGER DEFAULT 0,
                score           REAL DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )

        # 错题本
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wrong_answers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id         INTEGER NOT NULL,
                week            INTEGER NOT NULL,
                question        TEXT NOT NULL,
                options         TEXT,
                correct_answer  TEXT,
                user_answer     TEXT,
                explanation     TEXT,
                knowledge_point TEXT,
                question_type   TEXT DEFAULT 'choice',
                status          TEXT DEFAULT 'wrong',
                created_at      TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )

        # 知识点解释表（用于知识图谱悬浮提示）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kp_explanations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT UNIQUE NOT NULL,
                explanation TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )

        # 迁移：为已存在的旧表补充 plan_json 列（幂等，可重复执行）
        cols = [r[1] for r in cur.execute("PRAGMA table_info(plans)").fetchall()]
        if "plan_json" not in cols:
            cur.execute("ALTER TABLE plans ADD COLUMN plan_json TEXT")


# ---------- 学习计划：增 / 查 / 删 ----------
def create_plan(
    title: str, goal: str = "", description: str = "", plan_json: str = None
) -> int:
    """新建学习计划，返回新计划的 id。

    plan_json: 结构化计划 JSON 字符串（可选，用于「我的计划」结构化展示）。
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO plans (title, goal, description, plan_json) VALUES (?, ?, ?, ?)",
            (title, goal, description, plan_json),
        )
        return cur.lastrowid


def list_plans():
    """返回全部学习计划（按创建时间倒序），列表元素为 dict。"""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM plans ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_plan(plan_id: int):
    """按 id 查询单个学习计划，不存在时返回 None。"""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        return dict(row) if row else None


def delete_plan(plan_id: int):
    """删除指定计划（级联删除其下任务与打卡记录）。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM plans WHERE id = ?", (plan_id,))


# ---------- 学习任务：增 / 查 / 更新状态 ----------
def add_task(plan_id: int, title: str, estimated_minutes: int = 30) -> int:
    """为指定计划新增任务，返回任务 id。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (plan_id, title, estimated_minutes) VALUES (?, ?, ?)",
            (plan_id, title, estimated_minutes),
        )
        return cur.lastrowid


def list_tasks(plan_id: int = None):
    """返回任务列表；传入 plan_id 则只查该计划下的任务。"""
    with get_conn() as conn:
        if plan_id is None:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE plan_id = ? ORDER BY id DESC", (plan_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def update_task_status(task_id: int, status: str):
    """更新任务状态（pending / done）。"""
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


# ---------- 每日打卡：增 / 查 ----------
def add_daily_checkin(plan_id: int, week: int, day: int) -> bool:
    """记录某计划第 week 周第 day 天的打卡。重复打卡返回 False。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO daily_checkins (plan_id, week, day) VALUES (?, ?, ?)",
            (plan_id, week, day),
        )
        return cur.rowcount > 0


def list_daily_checkins(plan_id: int = None):
    """返回每日打卡记录；传入 plan_id 则只查该计划。"""
    with get_conn() as conn:
        if plan_id is None:
            rows = conn.execute(
                "SELECT * FROM daily_checkins ORDER BY id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM daily_checkins WHERE plan_id = ? ORDER BY id DESC",
                (plan_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_checkin_dates():
    """返回所有打卡的日历日期集合（YYYY-MM-DD），用于日历视图。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date(checkin_time) AS d FROM daily_checkins"
        ).fetchall()
        return {r["d"] for r in rows}


# ---------- 数据看板：统计 ----------
def _parse_json(s):
    """解析 JSON 字符串为 dict，失败返回 None。"""
    if not s:
        return None
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _longest_consecutive(dates):
    """计算最长连续日历天数。dates 为升序去重的 YYYY-MM-DD 列表。"""
    if not dates:
        return 0
    longest = cur = 1
    prev = datetime.date.fromisoformat(dates[0])
    for d in dates[1:]:
        cur_d = datetime.date.fromisoformat(d)
        diff = (cur_d - prev).days
        if diff == 1:
            cur += 1
        elif diff > 1:
            cur = 1
        prev = cur_d
        longest = max(longest, cur)
    return longest


def _recent_7day_trend(total_planned_days, checkins):
    """近7天累计完成率趋势（含窗口前的历史）。返回 [{'date','rate'}] 升序。"""
    today = datetime.date.today()
    daily = {}
    for c in checkins:
        d = c["checkin_time"][:10]
        daily[d] = daily.get(d, 0) + 1

    cutoff = (today - datetime.timedelta(days=6)).isoformat()
    cum = sum(n for d, n in daily.items() if d < cutoff)
    result = []
    for i in range(6, -1, -1):
        d = (today - datetime.timedelta(days=i)).isoformat()
        cum += daily.get(d, 0)
        rate = cum / total_planned_days * 100 if total_planned_days else 0.0
        result.append({"date": d, "rate": round(rate, 1)})
    return result


def _knowledge_mastery(kp_task_map, checked, all_kp):
    """计算知识点掌握进度。返回 {'mastered','total','points':[{'name','mastered'}]}。"""
    point_status = {}
    for kp in all_kp:
        point_status.setdefault(kp, False)
    for (pid, week, day, kps) in kp_task_map:
        if (pid, week, day) in checked:
            for kp in kps:
                point_status[kp] = True
    points = [{"name": k, "mastered": point_status[k]} for k in point_status]
    mastered = sum(1 for v in point_status.values() if v)
    return {"mastered": mastered, "total": len(point_status), "points": points}


def get_dashboard_stats():
    """汇总数据看板所需指标：累计天数、完成率、周完成率、连续打卡、趋势、知识点。"""
    with get_conn() as conn:
        plans = conn.execute("SELECT id, plan_json FROM plans").fetchall()
        checkins = conn.execute(
            "SELECT plan_id, week, day, checkin_time FROM daily_checkins ORDER BY checkin_time"
        ).fetchall()

    # ---- 解析计划结构：总计划天数 / 每周计划天数 / 知识点 ----
    total_planned_days = 0
    week_planned = {}   # (plan_id, week) -> 计划天数
    kp_task_map = []    # (plan_id, week, day, [知识点])
    all_kp = []         # 全部知识点（含重复）
    for plan in plans:
        pj = _parse_json(plan["plan_json"])
        if not pj:
            continue
        for wk in pj.get("weekly_plan", []) or []:
            week_no = wk.get("week")
            days = wk.get("daily_tasks", []) or []
            week_planned[(plan["id"], week_no)] = len(days)
            total_planned_days += len(days)
            for task in days:
                kps = task.get("knowledge_points", []) or []
                kp_task_map.append((plan["id"], week_no, task.get("day"), kps))
                all_kp.extend(kps)

    # ---- 打卡集合与日期 ----
    checked = set()   # (plan_id, week, day)
    dates = []        # 打卡日历日期
    for c in checkins:
        checked.add((c["plan_id"], c["week"], c["day"]))
        dates.append(c["checkin_time"][:10])

    total_checkins = len(checked)

    # 累计学习天数（去重日历日）
    total_study_days = len(set(dates))

    # 总完成率
    total_completion = (
        total_checkins / total_planned_days * 100 if total_planned_days else 0.0
    )

    # 平均周完成率
    week_rates = []
    for (pid, week), planned in week_planned.items():
        if planned:
            done = sum(
                1 for c in checkins if c["plan_id"] == pid and c["week"] == week
            )
            week_rates.append(done / planned * 100)
    avg_week_completion = sum(week_rates) / len(week_rates) if week_rates else 0.0

    # 最长连续打卡
    longest_streak = _longest_consecutive(sorted(set(dates)))

    # 近7天完成率趋势
    recent_7 = _recent_7day_trend(total_planned_days, checkins)

    # 知识点掌握进度
    knowledge = _knowledge_mastery(kp_task_map, checked, all_kp)

    return {
        "total_study_days": total_study_days,
        "total_completion": round(total_completion, 1),
        "avg_week_completion": round(avg_week_completion, 1),
        "longest_streak": longest_streak,
        "total_checkins": total_checkins,
        "total_planned_days": total_planned_days,
        "recent_7": recent_7,
        "knowledge": knowledge,
    }


# ---------- 周测：成绩与错题 ----------
def add_quiz_result(plan_id: int, week: int, total_questions: int, correct_count: int, score: float):
    """保存一次周测成绩。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO quiz_results (plan_id, week, total_questions, correct_count, score) "
            "VALUES (?, ?, ?, ?, ?)",
            (plan_id, week, total_questions, correct_count, score),
        )


def add_wrong_answer(
    plan_id: int,
    week: int,
    question: str,
    options: str,
    correct_answer: str,
    user_answer: str,
    explanation: str,
    knowledge_point: str,
    question_type: str = "choice",
):
    """将错题存入错题本。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO wrong_answers "
            "(plan_id, week, question, options, correct_answer, user_answer, explanation, knowledge_point, question_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (plan_id, week, question, options, correct_answer, user_answer, explanation, knowledge_point, question_type),
        )


def list_wrong_answers(status: str = "wrong"):
    """返回错题本中的错题（默认只查待复习的）。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM wrong_answers WHERE status = ? ORDER BY id DESC", (status,)
        ).fetchall()
        return [dict(r) for r in rows]


def mark_wrong_mastered(wrong_id: int):
    """将错题标记为已掌握。"""
    with get_conn() as conn:
        conn.execute("UPDATE wrong_answers SET status = 'mastered' WHERE id = ?", (wrong_id,))


def get_quiz_stats():
    """返回周测统计：次数、平均分、最佳分、最近成绩。"""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM quiz_results").fetchone()[0]
        if total == 0:
            return {"total_tests": 0, "avg_score": 0.0, "best_score": 0.0, "recent": []}
        avg = conn.execute("SELECT AVG(score) FROM quiz_results").fetchone()[0]
        best = conn.execute("SELECT MAX(score) FROM quiz_results").fetchone()[0]
        recent = conn.execute(
            "SELECT week, score, created_at FROM quiz_results ORDER BY id DESC LIMIT 10"
        ).fetchall()
    return {
        "total_tests": total,
        "avg_score": round(avg or 0, 1),
        "best_score": round(best or 0, 1),
        "recent": [dict(r) for r in recent],
    }


# ---------- 知识点解释 ----------
def get_kp_explanations(names):
    """返回 names 中已有解释的 dict {name: explanation}。"""
    if not names:
        return {}
    placeholders = ",".join("?" * len(names))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT name, explanation FROM kp_explanations WHERE name IN ({placeholders})",
            list(names),
        ).fetchall()
    return {r["name"]: r["explanation"] for r in rows}


def save_kp_explanations(expl_map):
    """批量保存知识点解释（按 name 覆盖）。"""
    if not expl_map:
        return
    with get_conn() as conn:
        for name, expl in expl_map.items():
            conn.execute(
                "INSERT OR REPLACE INTO kp_explanations (name, explanation) VALUES (?, ?)",
                (name, expl),
            )
