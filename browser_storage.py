"""浏览器本地存储（localStorage）桥接模块。

为「配置中心」提供把 DeepSeek API Key 持久化到浏览器本地的能力：

- get_saved_key()    读取（JS -> Python），通过极简自定义组件 localstorage_component 实现
- set_saved_key()    写入（Python -> JS），通过注入脚本实现
- remove_saved_key() 删除（Python -> JS），通过注入脚本实现

说明：Streamlit 组件的 iframe 携带 allow-same-origin，可正常访问 localStorage。
"""
import json
import os

from streamlit.components.v1 import declare_component
from streamlit.components.v1 import html as _render_html

# 浏览器 localStorage 中的存储键
STORAGE_KEY = "deepseek_api_key"

# 极简自定义组件：读取 localStorage 并回传 Python
_read_component = declare_component(
    "localstorage",
    path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "localstorage_component"
    ),
)


def get_saved_key():
    """读取浏览器中保存的 API Key。

    返回 {"loaded": bool, "value": str|None}。首次运行时 loaded=False（前端尚未回传），
    前端回传触发重跑后 loaded=True。调用方需结合 loaded 判断是否已拿到真实结果。
    """
    res = _read_component(
        local_key=STORAGE_KEY,
        default={"loaded": False, "value": None},
        key="ls_read",
    )
    if isinstance(res, dict):
        return res
    return {"loaded": False, "value": None}


def set_saved_key(value):
    """把 API Key 写入浏览器 localStorage。"""
    _inject(
        "localStorage.setItem(" + json.dumps(STORAGE_KEY) + ", " + json.dumps(value or "") + ");"
    )


def remove_saved_key():
    """删除浏览器 localStorage 中保存的 API Key。"""
    _inject("localStorage.removeItem(" + json.dumps(STORAGE_KEY) + ");")


def _inject(statement):
    _render_html(
        "<script>try { " + statement + " } catch(e) { console.error(e); }</script>",
        height=0,
        width=0,
    )
