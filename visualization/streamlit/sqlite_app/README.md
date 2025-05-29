# SQLite + Streamlit サンプルアプリ

このアプリケーションは、Streamlitを使用してSQLiteデータベースにアクセスし、データの登録・表示を行う簡単な例です。

## 機能

- ユーザー情報（名前、メールアドレス）の登録
- 登録されたユーザー一覧の表示
- ユーザーデータのCSVダウンロード

## 実行方法

1. 必要なパッケージをインストール:
```
pip install streamlit pandas
```

2. アプリケーションを実行:
```
streamlit run app.py
```

## アプリケーションの構成

- `app.py`: メインのStreamlitアプリケーションコード
- `user_data.db`: SQLiteデータベースファイル（自動生成）