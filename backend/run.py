"""
MiroFish Backend 启动入口
"""

import os
import sys

# 解决 Windows 控制台中文乱码问题：在所有导入之前设置 UTF-8 编码
if sys.platform == 'win32':
    # 设置环境变量确保 Python 使用 UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # 重新配置标准输出流为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config
from werkzeug.serving import WSGIRequestHandler


class _PrivateOAuthRequestHandler(WSGIRequestHandler):
    """Avoid printing one-use OAuth code and state in Werkzeug access logs."""

    def log_request(self, code='-', size='-'):
        if getattr(self, 'path', '').split('?', 1)[0] == '/api/chatgpt/callback':
            original_path = self.path
            try:
                self.path = '/api/chatgpt/callback'
                return super().log_request(code, size)
            finally:
                self.path = original_path
        return super().log_request(code, size)


def main():
    """主函数"""
    # 验证配置
    errors = Config.validate()
    if errors:
        print("配置错误:")
        for err in errors:
            print(f"  - {err}")
        print("\n请检查 .env 文件中的配置")
        sys.exit(1)
    
    # 创建应用
    app = create_app()
    
    # 获取运行配置
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # 启动服务
    run_options = {'request_handler': _PrivateOAuthRequestHandler} if Config.CHATGPT_PLAN_MODE else {}
    app.run(host=host, port=port, debug=debug, threaded=True, **run_options)


if __name__ == '__main__':
    main()

