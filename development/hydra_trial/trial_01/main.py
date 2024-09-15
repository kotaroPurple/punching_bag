import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="my_config")  # main.yamlを使用
def main(cfg: DictConfig):
    print(cfg)
    # for directory in cfg.dirs.dirs:  # 外部ファイルdirs.yamlから読み込んだリストを使用
    #     print(f"Processing directory: {directory}")
    #     if not os.path.exists(directory):
    #         print(f"Directory {directory} does not exist.")
    #         continue

    #     process_directory(directory, cfg.processing.method)

def process_directory(directory, method):
    print(f"Applying {method} on {directory}")

if __name__ == "__main__":
    main()
