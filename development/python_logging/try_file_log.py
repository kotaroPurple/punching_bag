
import logging
from logging.handlers import RotatingFileHandler
import datetime

# ログファイルの名前設定
log_filename = f"app_log_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# コンソールハンドラの設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# ファイルハンドラの設定
file_handler = RotatingFileHandler(log_filename, maxBytes=10**6, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# ロガーにハンドラを追加
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ログメッセージの出力
logger.debug('これはデバッグメッセージです')
logger.info('これは情報メッセージです')
logger.warning('これは警告メッセージです')
logger.error('これはエラーメッセージです')
logger.critical('これは致命的なエラーメッセージです')
