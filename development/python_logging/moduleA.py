
import logging

logger = logging.getLogger(__name__)

def function_a():
    logger.debug('moduleAのfunction_aが呼び出されました')
    # 何らかの処理
    logger.info('moduleAの処理が完了しました')
