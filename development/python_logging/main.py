
import logging
import moduleA
import moduleB

# メインロガーの設定
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')

logger = logging.getLogger(__name__)

def main():
    logger.info('メインモジュールが開始されました')
    moduleA.function_a()
    moduleB.function_b()

if __name__ == '__main__':
    main()
