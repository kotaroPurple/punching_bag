
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


# matplotlib のカラーマップを使用して役職ごとに色を割り当てる
def colorize_role(val, names: list[str], colormap='tab10'):
    # カラーマップを取得
    cmap = cm.get_cmap(colormap)
    role_colors = {name: cmap(i%10) for i, name in enumerate(names)}
    # role_colors = {
    #     'マネージャー': cmap(0),  # C0
    #     'エンジニア': cmap(1),  # C1
    #     'デザイナー': cmap(2)  # C2
    # }
    color = role_colors.get(val, 'white')
    return f'background-color: {matplotlib.colors.to_hex(color)}'


def colorize_value(val, min_val, max_val, colors=['red', 'green', 'blue']):
    """
    val: 対象の値
    min_val: 最小値
    max_val: 最大値
    colors: 色のリスト、[赤, 緑, 青] の順番で指定。デフォルトは ['red', 'green', 'blue']
    """
    # min_val と max_val の範囲を0〜1に正規化
    norm_val = (val - min_val) / (max_val - min_val)
    norm_val = max(0, min(1, norm_val))  # 値が範囲外にならないようにクリップ

    # RGBの各成分に対して色の強弱を計算
    color_values = []
    for color in colors:
        if color == 'red':
            color_value = int(norm_val * 255)  # norm_valに応じて赤を強くする
        elif color == 'green':
            color_value = int((1 - norm_val) * 255)  # norm_valに応じて緑を弱くする
        elif color == 'blue':
            color_value = int((1 - norm_val) * 255)  # norm_valに応じて青を弱くする
        else:
            color_value = 0  # 未定義の色は0
        color_values.append(color_value)

    return f'background-color: rgb({color_values[0]}, {color_values[1]}, {color_values[2]})'


# カテゴリカル変数の選択肢
C1_choices = ['A', 'B', 'C']
C2_choices = ['X', 'Y']
C3_choices = ['P', 'Q', 'R']

# データ数を指定
num_records = 300

df = pd.DataFrame({
    'C1': np.random.choice(C1_choices, size=num_records),  # C1はA, B, Cのいずれか
    'C2': np.random.choice(C2_choices, size=num_records),  # C2はX, Yのいずれか
    'C3': np.random.choice(C3_choices, size=num_records),  # C3はP, Q, Rのいずれか
    'true_value': np.random.randint(10, 20, size=num_records),  # true_valueは10から20の範囲でランダム
    'predicted_value': np.random.randint(10, 20, size=num_records)  # predicted_valueも同様にランダム
})

# 推定値と真値の差を計算
df['error'] = df['predicted_value'] - df['true_value']

# 3つの条件（C1, C2, C3）の組み合わせを新しいカラムとして定義
df['condition_combination'] = df['C1'] + '-' + df['C2'] + '-' + df['C3']

# 最小と最大を指定
min_error = df['error'].min()
max_error = df['error'].max()
min_predicted = df['predicted_value'].min()
max_predicted = df['predicted_value'].max()

# スタイルを適用
styled_df = df.style.applymap(lambda val: colorize_role(val, names=C1_choices, colormap='tab10'), subset=['C1']) \
                    .applymap(lambda val: colorize_value(val, min_error, max_error, colors=['red', 'red', 'red']), subset=['error']) \
                    .applymap(lambda val: colorize_value(val, min_predicted, max_predicted, colors=['green', 'green', 'green']), subset=['predicted_value'])

# Streamlitで表示
st.dataframe(styled_df)

# groupbyを使ってC1, C2, trialごとの差の中央値を計算
median_errors = df.groupby(['C1', 'C2', 'C3'])['error'].median().reset_index()
# print(median_errors.describe())

# 誤差の中央値が小さい順に並べ替え
sorted_median_errors = median_errors.sort_values(by='error', ascending=True).reset_index(drop=True)

# 結果の表示
st.dataframe(sorted_median_errors)

# Seabornを使って条件の組み合わせごとのエラーの散布図を描画
fig = plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='condition_combination', y='error', hue='C1', style='C2', palette='Set2')

# グラフのタイトルとラベル
plt.title('Error Scatter by Condition Combination (C1, C2, C3)', fontsize=14)
plt.xlabel('Condition Combination (C1-C2-C3)', fontsize=12)
plt.ylabel('Error (Predicted - True)', fontsize=12)
plt.xticks(rotation=90)  # x軸のラベルを90度回転させて表示
plt.legend(title='C1 and C2')

st.pyplot(fig)
