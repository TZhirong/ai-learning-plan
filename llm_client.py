"""LLM 客户端模块。

封装 DeepSeek（OpenAI 兼容接口）的调用，供「计划定制」标签页使用。
提供统一的生成函数，具备以下能力：
- 密钥支持「环境变量」与「手动传入（侧边栏输入）」，手动传入优先；
- 流式输出（on_chunk 回调逐段返回内容），减少等待焦虑；
- 输入参数合法性校验；
- 相同参数短时间重复生成直接返回缓存；
- API 调用失败自动重试一次；
- JSON 解析失败自动追加格式提示并二次生成；
- 对外只抛出可友好提示的异常，不暴露技术细节。
"""
import hashlib
import json
import re
import time

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# ---------------------------------------------------------------------------
# 系统提示词：角色设定 + 强制规则 + 输出 schema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一位拥有10年经验的计算机专业教学规划师，擅长为不同基础的学习者制定科学、可执行、个性化的学习计划。

【必须遵守的强制规则】
1. 知识点前置依赖校验：安排内容时必须确保前置知识已在此前的任务中覆盖，不得出现知识跳跃。
2. 难度平缓递进：每天的学习难度应逐步、平缓地提升，避免陡增。
3. 每日时长误差不超10%：每个 week 的 daily_tasks 中所有任务的 duration 之和，与用户设定的每日可用时长误差不得超过 ±10%。
4. 嵌入艾宾浩斯复习节点：在 weekly_plan 中安排基于艾宾浩斯遗忘曲线（学习后第1、2、4、7、15天）的复习节点。
5. 理论实操比例不低于1:1：practice_task（实操）的占比不得低于理论学习。

【输出要求】
- 必须只输出一个合法的 JSON 对象，不包含任何额外文本、解释或 Markdown 代码块标记（不要用 ```json 包裹）。
- 字段名、类型、层级必须严格遵循以下 schema：
{
  "plan_overview": {
    "total_duration": "字符串，例如：8周",
    "learning_goal": "字符串，学习目标",
    "daily_hours": 数字，每日时长（小时）,
    "core_knowledge_points": ["字符串数组，核心知识点"],
    "difficulty_level": "字符串，整体难度等级说明"
  },
  "weekly_plan": [
    {
      "week": 数字，第几周（从1开始）,
      "week_theme": "字符串，本周主题",
      "daily_tasks": [
        {
          "day": 数字，第几天（从1开始）,
          "task_title": "字符串，任务标题",
          "duration": 数字，该任务时长（小时）,
          "knowledge_points": ["字符串数组，本任务涉及的知识点"],
          "practice_task": "字符串，实操任务",
          "review_note": "字符串，复习提示（含艾宾浩斯节点）"
        }
      ]
    }
  ],
  "review_schedule": {
    "key_nodes": ["字符串数组，关键复习节点"],
    "review_method": "字符串，复习方法说明"
  },
  "learning_suggestions": ["字符串数组，个性化学习建议"]
}
"""

# JSON 解析失败时追加的格式修正提示
FORMAT_REMINDER = (
    "你上次的输出不是合法的 JSON，无法被程序解析。"
    "请重新输出，且只输出一个合法的 JSON 对象，"
    "不要包含任何额外文本、解释或 Markdown 代码块标记。"
)

# 周测出题提示词
QUIZ_SYSTEM_PROMPT = """你是一位教学出题专家。请根据给定的知识点，为学习自测出题。

【出题要求】
1. 共出 5 道单项选择题 + 1 道编程题。
2. 选择题每题 4 个选项，答案唯一（用 A/B/C/D 表示）。
3. 编程题给出题目描述与参考答案。
4. 每题都附详细解析，解析要能帮助理解知识点。

【输出要求】
- 只输出一个合法的 JSON 对象，不包含任何额外文本、解释或 Markdown 代码块标记。
- 结构如下：
{
  "questions": [
    {"type": "choice", "question": "题干", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A", "explanation": "解析", "knowledge_point": "知识点"},
    {"type": "programming", "question": "编程题描述", "reference_answer": "参考答案", "explanation": "解析", "knowledge_point": "知识点"}
  ]
}
"""

# 知识点解释提示词
KP_EXPLANATION_PROMPT = """你是一位知识库助手，擅长用通俗易懂的语言解释概念。

【任务】为给定的每个知识点写一句简洁、准确、通俗的解释，帮助学习者理解其含义。

【要求】
- 只输出一个合法 JSON 对象，键为知识点名称，值为其解释。
- 每个解释 20~60 字，可包含公式或类比，例如「回归图 = 散点图 + 拟合的回归线，用于直观展示自变量与因变量的数量关系」。
- 不包含任何额外文本、解释或 Markdown 代码块标记。
"""

# 生成参数
TEMPERATURE = 0.3
MAX_RETRIES = 1    # API 调用失败自动重试次数
CACHE_TTL = 300    # 结果缓存有效期（秒）

# 生成结果缓存：key -> (时间戳, 结果)。进程内缓存，短时间重复生成直接复用。
_cache: dict = {}


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """创建 DeepSeek 客户端。密钥优先使用手动传入值，其次回退到环境变量。"""
    key = api_key or DEEPSEEK_API_KEY
    url = base_url or DEEPSEEK_BASE_URL
    return OpenAI(api_key=key, base_url=url)


def _build_user_message(
    learning_goal: str,
    current_level: str,
    learning_style: str,
    daily_hours: float,
    weeks: int,
    context: list | None = None,
) -> str:
    """根据表单采集的 5 项信息构造用户消息；有知识库上下文时附带参考资料。"""
    msg = (
        "请根据以下信息，为我生成一份个性化学习计划：\n"
        f"- 学习目标：{learning_goal}\n"
        f"- 当前基础：{current_level}\n"
        f"- 学习偏好：{learning_style}\n"
        f"- 每日可用时长：{daily_hours} 小时\n"
        f"- 学习周期：{weeks} 周\n"
    )
    if context:
        refs = "\n".join(
            f"- 来源【{c.get('source') or '未知'}】：{c.get('text', '')[:200]}"
            for c in context
        )
        msg += (
            "\n【参考资料（来自知识库）】\n"
            f"{refs}\n"
            "请优先基于上述参考资料规划知识点，"
            "并在每个来自资料的知识点名称后标注来源，"
            "格式：知识点名（来源：章节/页码）。"
        )
    msg += "请严格以 JSON 格式输出。"
    return msg


def _parse_json_response(text: str) -> dict:
    """稳健地解析模型返回的 JSON（容忍可能包裹的 Markdown 代码块）。"""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("模型未返回预期的 JSON 对象")
    return data


def validate_inputs(
    learning_goal: str,
    current_level: str,
    learning_style: str,
    daily_hours: float,
    weeks: int,
):
    """校验输入参数合法性。返回 (是否合法, 错误信息)。"""
    if not learning_goal or not str(learning_goal).strip():
        return False, "请填写学习目标"
    if not current_level:
        return False, "请选择当前基础"
    if not learning_style:
        return False, "请选择学习偏好"
    if not isinstance(daily_hours, (int, float)) or not (0.5 <= float(daily_hours) <= 8):
        return False, "每日可用时长需在 0.5~8 小时之间"
    if not isinstance(weeks, (int, float)) or not (1 <= float(weeks) <= 24):
        return False, "学习周期需在 1~24 周之间"
    return True, ""


def _cache_key(learning_goal, current_level, learning_style, daily_hours, weeks, context=None) -> str:
    """根据生成参数构造缓存键（含知识库上下文指纹）。"""
    ctx_fp = (
        hashlib.md5(json.dumps(context, ensure_ascii=False).encode("utf-8")).hexdigest()
        if context else ""
    )
    return json.dumps(
        {
            "goal": str(learning_goal).strip(),
            "level": current_level,
            "style": learning_style,
            "hours": float(daily_hours),
            "weeks": int(weeks),
            "context": ctx_fp,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _call_api(client: OpenAI, messages: list, on_chunk=None, max_retries: int = MAX_RETRIES, response_format=None) -> str:
    """调用 API 返回完整文本。支持流式输出，失败自动重试。"""
    stream = on_chunk is not None
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(
                model=DEEPSEEK_MODEL,
                temperature=TEMPERATURE,
                stream=stream,
                messages=messages,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            if stream:
                parts = []
                for chunk in resp:
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except (IndexError, AttributeError):
                        continue
                    if delta:
                        parts.append(delta)
                        on_chunk(delta)
                return "".join(parts)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
    raise RuntimeError("AI 服务调用失败") from last_error


def generate_learning_plan(
    learning_goal: str,
    current_level: str,
    learning_style: str,
    daily_hours: float,
    weeks: int,
    api_key: str | None = None,
    on_chunk=None,
    use_cache: bool = True,
    context: list | None = None,
) -> dict:
    """生成结构化学习计划。

    参数:
        learning_goal / current_level / learning_style / daily_hours / weeks: 表单采集信息
        api_key:    手动传入的密钥（可选，优先级高于环境变量）
        on_chunk:   流式回调（可选，接收逐段文本）
        use_cache:  是否启用结果缓存
        context:    知识库检索到的参考资料（可选，[{text, source}]）

    返回:
        解析后的计划 dict。

    异常:
        参数不合法 / 未配置密钥时抛出 ValueError（信息友好）；
        API 全部重试失败时抛出 RuntimeError。
    """
    # 1. 输入合法性校验
    ok, msg = validate_inputs(learning_goal, current_level, learning_style, daily_hours, weeks)
    if not ok:
        raise ValueError(msg)

    # 2. 密钥检查
    if not (api_key or DEEPSEEK_API_KEY):
        raise ValueError("未配置 DeepSeek API Key")

    # 3. 缓存命中检查
    key = _cache_key(learning_goal, current_level, learning_style, daily_hours, weeks, context)
    if use_cache:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        if hit:
            _cache.pop(key, None)  # 过期清理

    # 4. 首次生成
    client = get_client(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(
                learning_goal, current_level, learning_style, daily_hours, weeks, context
            ),
        },
    ]
    content = _call_api(client, messages, on_chunk=on_chunk)

    # 5. JSON 解析失败 -> 追加格式提示二次生成
    try:
        result = _parse_json_response(content)
    except Exception:
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": FORMAT_REMINDER},
        ]
        content2 = _call_api(client, messages, on_chunk=on_chunk)
        result = _parse_json_response(content2)

    # 6. 写缓存
    _cache[key] = (time.time(), result)
    return result


def generate_weekly_quiz(knowledge_points, week_theme: str = "", api_key: str | None = None) -> dict:
    """根据当周知识点生成周测（5 选择 + 1 编程，强制 JSON 输出）。

    参数:
        knowledge_points: 知识点字符串列表
        week_theme:       本周主题
        api_key:          手动密钥（可选）

    返回:
        {"questions": [...]}；异常时抛出 ValueError / RuntimeError。
    """
    if not knowledge_points:
        raise ValueError("本周暂无知识点，无法生成周测")
    if not (api_key or DEEPSEEK_API_KEY):
        raise ValueError("未配置 DeepSeek API Key")

    client = get_client(api_key=api_key)
    kps = "、".join(str(k) for k in knowledge_points)
    user = f"本周主题：{week_theme or '未指定'}\n知识点：{kps}\n请根据以上知识点出题。"
    messages = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    # 非流式 + 强制 JSON 模式
    content = _call_api(client, messages, response_format={"type": "json_object"})
    try:
        result = _parse_json_response(content)
    except Exception:
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": FORMAT_REMINDER},
        ]
        content2 = _call_api(client, messages, response_format={"type": "json_object"})
        result = _parse_json_response(content2)
    return result


def generate_kp_explanations(names, api_key: str | None = None) -> dict:
    """批量生成知识点解释。返回 {name: explanation}。

    参数:
        names:   知识点名称列表
        api_key: 手动密钥（可选）

    返回:
        只包含成功生成解释的知识点；异常时抛出 ValueError / RuntimeError。
    """
    if not names:
        return {}
    if not (api_key or DEEPSEEK_API_KEY):
        raise ValueError("未配置 DeepSeek API Key")

    client = get_client(api_key=api_key)
    user = "知识点列表：\n" + "\n".join(f"- {n}" for n in names)
    messages = [
        {"role": "system", "content": KP_EXPLANATION_PROMPT},
        {"role": "user", "content": user},
    ]

    content = _call_api(client, messages, response_format={"type": "json_object"})
    try:
        result = _parse_json_response(content)
    except Exception:
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": FORMAT_REMINDER},
        ]
        content2 = _call_api(client, messages, response_format={"type": "json_object"})
        result = _parse_json_response(content2)

    # 只保留 names 中且解释非空的结果
    return {n: str(result[n]) for n in names if result.get(n)}
