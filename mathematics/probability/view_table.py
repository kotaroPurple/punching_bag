
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# 仮のデータ例
df = pd.DataFrame({
    'subject': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'location': ['X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'X', 'Y'],
    'time': ['T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1'],
    'trial_count': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'estimated_value': [10, 12, 11, 14, 15, 16, 18, 19, 20]
})

# 各カテゴリごとの推定値の平均を計算
summary_stats = df.groupby(['subject', 'location', 'time']).agg({
    'estimated_value': ['mean', 'std', 'min', 'max']
}).reset_index()

print(summary_stats)

# %%
# df = pd.DataFrame({
#     'C1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
#     'C2': ['X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'X', 'Y'],
#     'trial': [1, 2, 1, 1, 2, 1, 1, 2, 1],
#     'true_value': [10, 12, 11, 14, 15, 16, 18, 19, 20],
#     'predicted_value': [11, 11, 12, 13, 14, 16, 17, 20, 22]
# })
# データ数を指定
num_records = 1000  # 例として1000行のデータを生成

# カテゴリカル変数の選択肢
C1_choices = ['A', 'B', 'C']
C2_choices = ['X', 'Y']
C3_choices = ['P', 'Q', 'R']

# ランダムにカテゴリカル変数と数値を生成
# np.random.seed(42)  # 再現性を確保するために乱数の種を固定

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

# Seabornを使って条件の組み合わせごとのエラーの散布図を描画
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='condition_combination', y='error', hue='C1', style='C2', palette='Set2')

# グラフのタイトルとラベル
plt.title('Error Scatter by Condition Combination (C1, C2, C3)', fontsize=14)
plt.xlabel('Condition Combination (C1-C2-C3)', fontsize=12)
plt.ylabel('Error (Predicted - True)', fontsize=12)
plt.xticks(rotation=90)  # x軸のラベルを90度回転させて表示
plt.legend(title='C1 and C2')

plt.show()

# groupbyを使ってC1, C2, trialごとの差の中央値を計算
median_errors = df.groupby(['C1', 'C2', 'C3'])['error'].median().reset_index()
print(median_errors.describe())

# 誤差の中央値が小さい順に並べ替え
sorted_median_errors = median_errors.sort_values(by='error', ascending=True).reset_index(drop=True)

# 結果の表示
print(sorted_median_errors.head())

# %%
