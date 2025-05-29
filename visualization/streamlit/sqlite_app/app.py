import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# データベースの初期化
def init_db():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ユーザーの追加
def add_user(name, email):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)',
              (name, email, created_at))
    conn.commit()
    conn.close()
    
    # データが追加されたらsession_stateのデータを更新
    st.session_state.users_df = get_users()
    st.session_state.data_loaded = True

# ユーザー一覧の取得
def get_users():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

# Streamlitアプリケーション
def main():
    st.title('SQLiteデータベース操作アプリ')
    
    # データベースの初期化
    init_db()
    
    # session_stateの初期化
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'users_df' not in st.session_state or not st.session_state.data_loaded:
        st.session_state.users_df = get_users()
        st.session_state.data_loaded = True
    
    # データ再読み込みボタン
    if st.button('データを再読み込み'):
        st.session_state.users_df = get_users()
        st.success('データを再読み込みしました')
    
    # サイドバーにフォームを配置
    st.sidebar.header('ユーザー登録')
    with st.sidebar.form(key='user_form'):
        name = st.text_input('名前')
        email = st.text_input('メールアドレス')
        submit_button = st.form_submit_button(label='登録')
        
        if submit_button:
            if name and email:
                add_user(name, email)
                st.sidebar.success(f'{name}さんを登録しました！')
            else:
                st.sidebar.error('名前とメールアドレスを入力してください')
    
    # メイン画面にユーザー一覧を表示
    st.header('登録ユーザー一覧')
    users_df = st.session_state.users_df
    
    if not users_df.empty:
        # ユーザー選択用のセレクトボックスを表示
        user_ids = users_df['id'].tolist()
        user_names = users_df['name'].tolist()
        user_options = [f"{id} - {name}" for id, name in zip(user_ids, user_names)]
        
        selected_user = st.selectbox('ユーザーを選択して詳細を表示', options=user_options)
        
        # 選択されたユーザーの詳細を表示
        if selected_user:
            user_id = int(selected_user.split(' - ')[0])
            # DBから読み込まずにsession_stateのデータから取得
            user_details = users_df[users_df['id'] == user_id].iloc[0]
            
            st.subheader('ユーザー詳細')
            col1, col2 = st.columns(2)
            with col1:
                st.write('**ID:**', user_details['id'])
                st.write('**名前:**', user_details['name'])
            with col2:
                st.write('**メールアドレス:**', user_details['email'])
                st.write('**登録日時:**', user_details['created_at'])
        
        # 全ユーザーデータを表示
        st.subheader('全ユーザーデータ')
        st.dataframe(users_df)
        
        # CSVダウンロードボタン
        csv = users_df.to_csv(index=False)
        st.download_button(
            label="CSVダウンロード",
            data=csv,
            file_name="users.csv",
            mime="text/csv"
        )
    else:
        st.info('登録されたユーザーはまだいません')

if __name__ == '__main__':
    main()