
import tempfile
import numpy as np
import pandas as pd
import os


def create_temp_csv(num_files=3, num_rows=10):
    temp_files = []
    for i in range(num_files):
        # 一時ファイルを作成（削除しない設定）
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='', encoding='utf-8')
        temp_files.append(temp.name)

        # サンプルデータを作成
        data = {
            'column1': np.arange(num_rows),
            'column2': np.random.randint(0, 100, num_rows),
        }
        df = pd.DataFrame(data)
        # CSVに書き込み
        df.to_csv(temp.name, index=False)
        # ファイルを閉じる
        temp.close()

    return temp_files


class MultiCSVReader:
    def __init__(self, file_paths: list[str], chunksize: int = 100000, **read_csv_kwargs):
        """
        初期化メソッド。

        :param file_paths: 読み込むCSVファイルのパスのリスト。
        :param chunksize: 読み込む行数のチャンクサイズ。
        :param read_csv_kwargs: pandas.read_csv に渡す追加のキーワード引数。
        """
        self.file_paths = file_paths
        self.chunksize = chunksize
        self.read_csv_kwargs = read_csv_kwargs
        self.readers = []
        self._initialize_readers()

    def _initialize_readers(self):
        """各ファイルのイテレータを初期化します。"""
        self.readers = [
            pd.read_csv(file, chunksize=self.chunksize, **self.read_csv_kwargs) \
                for file in self.file_paths]

    def __iter__(self):
        """イテレータとして自身を返します。"""
        return self

    def __next__(self):
        """
        次のチャンクを取得します。

        :return: 各ファイルからのDataFrameのリスト。
        """
        chunks = []
        for reader in self.readers:
            try:
                chunk = next(reader)
                chunks.append(chunk)
            except StopIteration:
                # すべてのファイルで読み込みが終了した場合
                if len(chunks) == 0:
                    raise StopIteration
                else:
                    # 他のファイルはまだ読み込める場合、Noneを追加
                    chunks.append(None)
        return chunks

    def close(self):
        """全てのイテレータを閉じます。"""
        for reader in self.readers:
            if hasattr(reader, 'close'):
                reader.close()


def main():
    # 一時CSVファイルを作成
    n_files = 3
    n_rows = 50

    temp_csv_files = create_temp_csv(num_files=n_files, num_rows=n_rows)

    print("作成された一時CSVファイル:")
    for file in temp_csv_files:
        print(file)

    # 作成した一時CSVファイルのリスト
    file_list = temp_csv_files

    # MultiCSVReaderのインスタンスを作成
    reader = MultiCSVReader(
        file_list,
        chunksize=15,  # 各チャンクの行数
        usecols=['column1', 'column2'],
        dtype={'column1': 'int32', 'column2': 'int32'}
    )

    # イテレーションを実行
    try:
        for chunk_list in reader:
            print("\n新しいチャンクセット:")
            for idx, chunk in enumerate(chunk_list):
                if chunk is not None:
                    print(f"\nファイル {file_list[idx]} のチャンク:")
                    print(chunk)
                else:
                    print(f"\nファイル {file_list[idx]} の読み込みが完了しました。")
            # 必要な処理をここに追加
    except StopIteration:
        print("全てのファイルの読み込みが完了しました。")
    finally:
        # リソースの解放
        reader.close()

        # 一時ファイルを削除（必要に応じて）
        for file in temp_csv_files:
            os.unlink(file)
            print(f"一時ファイル {file} を削除しました。")


if __name__ == '__main__':
    main()
