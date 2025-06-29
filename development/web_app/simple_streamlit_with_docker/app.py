import streamlit as st
import sqlite3
import pandas as pd


def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def insert_item(name, description):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO items (name, description) VALUES (?, ?)', (name, description))
    conn.commit()
    conn.close()


def get_items():
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query('SELECT * FROM items ORDER BY created_at DESC', conn)
    conn.close()
    return df


def main():
    st.title("データ管理アプリ")
    init_db()
    page = st.sidebar.selectbox("ページを選択", ["データ登録", "データ一覧"])
    if page == "データ登録":
        st.header("データ登録")
        with st.form("item_form"):
            name = st.text_input("名前")
            description = st.text_area("説明")
            submitted = st.form_submit_button("登録")
            if submitted and name:
                insert_item(name, description)
                st.success("データを登録しました！")
                st.rerun()
    elif page == "データ一覧":
        st.header("データ一覧")
        df = get_items()
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("データがありません")

if __name__ == "__main__":
    main()