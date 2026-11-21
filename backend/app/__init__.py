"""
MiroFish Backend - Flask应用工厂
"""

import os
import ipaddress
import warnings

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, abort, g, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 设置JSON编码：确保中文直接显示（而不是 \uXXXX 格式）
    # Flask >= 2.3 使用 app.json.ensure_ascii，旧版本使用 JSON_AS_ASCII 配置
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # 设置日志
    logger = setup_logger('mirofish')
    
    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend 启动中...")
        logger.info("=" * 50)
    
    # 启用CORS
    CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:3000"}})

    @app.before_request
    def restrict_browser_origin():
        if Config.CHATGPT_PLAN_MODE:
            try:
                remote_is_local = ipaddress.ip_address(request.remote_addr or '').is_loopback
            except ValueError:
                remote_is_local = False
            if request.host != '127.0.0.1:5001' or not remote_is_local:
                abort(403)
        if request.path.startswith(('/api/', '/v1/')):
            origin = request.headers.get('Origin')
            fetch_site = request.headers.get('Sec-Fetch-Site')
            # OpenAI returns to this exact URL in a top-level cross-site navigation.
            is_oauth_callback = request.method == 'GET' and request.path == '/api/chatgpt/callback'
            if origin not in (None, 'http://127.0.0.1:3000') or (fetch_site == 'cross-site' and not is_oauth_callback):
                abort(403)

    if Config.CHATGPT_PLAN_MODE:
        from .services.plan_work_guard import PlanWorkBusyError, plan_work_guard

        @app.before_request
        def require_plan_connection_for_work():
            if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
                return
            is_plan_api_work = request.path.startswith(('/api/graph/', '/api/simulation/', '/api/report/'))
            is_plan_proxy_work = request.path == '/v1/chat/completions'
            if not (is_plan_api_work or is_plan_proxy_work):
                return
            try:
                g.plan_work_lease = plan_work_guard.claim()
            except PlanWorkBusyError as exc:
                abort(409, str(exc))
            if is_plan_proxy_work:
                return
            from .services.chatgpt_plan_auth import auth
            if not (auth.status()['plan_usage_enabled'] and Config.memory_backend_configured() and Config.LLM_MODEL_NAME):
                abort(503, 'Connect ChatGPT, select a model, and configure graph memory first.')

        @app.teardown_request
        def release_request_plan_work(_error):
            lease = getattr(g, 'plan_work_lease', None)
            if lease is not None:
                lease.release()
                del g.plan_work_lease
    
    # 注册模拟进程清理函数（确保服务器关闭时终止所有模拟进程）
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已注册模拟进程清理函数")
    
    # 请求日志中间件
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"请求: {request.method} {request.path}")
        # The plan adapter can receive prompts and tool results. Never write
        # JSON request bodies to disk in this local mode, even at DEBUG level.
        if not Config.CHATGPT_PLAN_MODE and request.content_type and 'json' in request.content_type:
            logger.debug(f"请求体: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"响应: {response.status_code}")
        return response
    
    # 注册蓝图
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    if Config.CHATGPT_PLAN_MODE:
        from .api.chatgpt_plan import chatgpt_plan_bp
        from .api.chatgpt_proxy import chatgpt_proxy_bp
        app.register_blueprint(chatgpt_plan_bp, url_prefix='/api/chatgpt')
        app.register_blueprint(chatgpt_proxy_bp)
    else:
        @app.get('/api/chatgpt/status')
        def chatgpt_plan_disabled_status():
            return {'planMode': False}
    
    # 健康检查
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend 启动完成")
    
    return app

