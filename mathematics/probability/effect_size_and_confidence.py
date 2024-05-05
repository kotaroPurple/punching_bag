
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# データ生成
# np.random.seed(0)  # 乱数の再現性を確保するためにシードを設定
group_size = 10  # グループごとのサンプルサイズ
num_groups = 3  # グループ数
true_mean_diff = 0  # 真の平均値との差

# グループごとにサンプリング
group_data = []
for _ in range(num_groups):
    group_data.append(np.random.normal(loc=true_mean_diff, scale=0.5, size=group_size))  # 平均=0, 標準偏差=1の正規分布からサンプリング

# 平均の信頼区間を計算
mean_diffs = [np.mean(data) for data in group_data]  # 各グループの平均値
std_errs = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in group_data]  # 各グループの標準誤差
t_critical = stats.t.ppf(0.975, df=group_size - 1)  # 自由度=サンプルサイズ-1のt分布の上側2.5%点
ci_lower = [mean_diff - t_critical * std_err for mean_diff, std_err in zip(mean_diffs, std_errs)]  # 下側信頼限界
ci_upper = [mean_diff + t_critical * std_err for mean_diff, std_err in zip(mean_diffs, std_errs)]  # 上側信頼限界

# 効果量の計算
# pooled_std = np.sqrt(np.mean([np.var(data, ddof=1) for data in group_data]))  # グループ間の標準偏差の平均値
# hedges_g = [(mean_diff - true_mean_diff) / pooled_std for mean_diff in mean_diffs]  # 各グループの効果量
hedges_g = [(mean_diff / np.std(data, ddof=1)) * (1 - (3 / (4 * len(data) - 9))) for mean_diff, data in zip(mean_diffs, group_data)]
print(f'{hedges_g=}')

var_list = np.array([np.var(data, ddof=1) for data in group_data])
mu_minus_3s = np.array([mean_diff - 3*np.sqrt(s) for mean_diff, s in zip(mean_diffs, var_list)])
mu_plus_3s = np.array([mean_diff + 3*np.sqrt(s) for mean_diff, s in zip(mean_diffs, var_list)])

for i, (minus, plus) in enumerate(zip(mu_minus_3s, mu_plus_3s)):
    print(f'Gropu {i+1} : {minus} to {plus}')

# 各グループの信頼区間が -1 から 1 の範囲に含まれるかを確認する
for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
    if -1 <= lower <= 1 and -1 <= upper <= 1:
        print(f'Group {i+1} has a valuable effect (95% CI: [{lower}, {upper}]).')
    else:
        print(f'Group {i+1} does not have a valuable effect (95% CI: [{lower}, {upper}]).')


# 箱ひげ図
plt.figure(figsize=(10, 6))
plt.boxplot(group_data, patch_artist=True)
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Box plot with confidence interval')

# 信頼区間を描画
for i in range(num_groups):
    plt.plot([i + 1, i + 1], [ci_lower[i], ci_upper[i]], color='black', linewidth=2)

# 効果量の棒グラフ
plt.bar(np.arange(num_groups) + 1.2, hedges_g, width=0.4, label="Hedges's g")
plt.xticks(np.arange(num_groups) + 1.2, [f'Group {i+1}' for i in range(num_groups)])
plt.ylabel("Effect size (Hedge's g)")

# 凡例の作成
plt.legend(['95% CI'])

plt.show()
