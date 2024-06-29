
import logging

logger = logging.getLogger(__name__)

def function_b():
    logger.warning('moduleBのfunction_bが警告を発しました')
    # 何らかの処理
    logger.error('moduleBのエラーが発生しました')
