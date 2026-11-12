"""配置管理模块。

负责加载 .env 环境变量，集中管理 DeepSeek API 相关配置与本地数据库路径。
所有需要用到密钥 / 模型名的地方，统一从这里导入，避免散落各处、便于维护。
"""
import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（若不存在则静默跳过，不报错）
load_dotenv()

# ---------- DeepSeek API 配置（DeepSeek 提供 OpenAI 兼容接口） ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- 本地 SQLite 数据库文件路径（与项目同目录） ----------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning_plan.db")
