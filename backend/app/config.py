"""\n配置管理\n统一从项目根目录的 .env 文件加载配置\n"""

import os
import secrets
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


CHATGPT_PLAN_MODE = os.environ.get('CHATGPT_PLAN_MODE', '0') == '1'
_LOCAL_PROXY_TOKEN = (os.environ.get('MIROFISH_PLAN_PROXY_TOKEN') or secrets.token_urlsafe(32)) if CHATGPT_PLAN_MODE else None
if CHATGPT_PLAN_MODE:
    # Existing MiroFish/CAMEL clients still use the OpenAI-shaped local endpoint.
    # This is an ephemeral localhost credential, never a ChatGPT OAuth token.
    os.environ['LLM_API_KEY'] = _LOCAL_PROXY_TOKEN
    os.environ['MIROFISH_PLAN_PROXY_TOKEN'] = _LOCAL_PROXY_TOKEN
    os.environ['LLM_BASE_URL'] = 'http://127.0.0.1:5001/v1'
    # The UI sets this only after checking the account's visible model list.
    # Preserve it when a simulation worker imports this module in a child process.
    os.environ['LLM_MODEL_NAME'] = os.environ.get('MIROFISH_PLAN_SELECTED_MODEL', '')
    for name in ('LLM_BOOST_API_KEY', 'LLM_BOOST_BASE_URL', 'LLM_BOOST_MODEL_NAME'):
        os.environ.pop(name, None)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or (secrets.token_urlsafe(32) if CHATGPT_PLAN_MODE else 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    CHATGPT_PLAN_MODE = CHATGPT_PLAN_MODE
    LOCAL_PROXY_TOKEN = _LOCAL_PROXY_TOKEN
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def zep_is_configured(cls) -> bool:
        """A configured key is present; this does not verify the Zep account."""
        return bool(cls.ZEP_API_KEY and cls.ZEP_API_KEY != 'your_zep_api_key_here')
    
    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if not cls.zep_is_configured() and not cls.CHATGPT_PLAN_MODE:
            errors.append("ZEP_API_KEY 未配置")
        if cls.CHATGPT_PLAN_MODE:
            if os.environ.get('FLASK_HOST', '127.0.0.1') != '127.0.0.1':
                errors.append('ChatGPT plan mode requires FLASK_HOST=127.0.0.1')
            if os.environ.get('FLASK_PORT', '5001') != '5001':
                errors.append('ChatGPT plan mode requires FLASK_PORT=5001')
            if cls.DEBUG:
                errors.append('ChatGPT plan mode requires FLASK_DEBUG=False')
        if os.environ.get("ZEP_API_URL"):
            errors.append("ZEP_API_URL 不受支持；MiroFish 仅连接 Zep Cloud")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
