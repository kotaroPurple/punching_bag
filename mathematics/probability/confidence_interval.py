
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# パラメーターの設定
mu = 0.5  # 平均
sigma = 1  # 標準偏差
sample_size = 50  # サンプルサイズ
confidence_level = 0.95  # 信頼レベル

# 信頼区間を計算する関数
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_error = stats.sem(data)  # 標準誤差
    # ppf : inverse of cdf
    z_score = stats.norm.ppf((1 + confidence_level) / 2)  # 信頼レベルに基づくZスコア
    margin_of_error = z_score * std_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

# 繰り返し試行して、95%信頼区間が真の値を含む確率を計算
num_trials = 100
contained_count = 0
lowers = []
uppers = []

for _ in range(num_trials):
    # 正規分布からサンプルを生成
    data = np.random.normal(mu, sigma, sample_size)
    lower_bound, upper_bound = calculate_confidence_interval(data)
    lowers.append(lower_bound)
    uppers.append(upper_bound)

    # 真の値が信頼区間に含まれるか確認
    if lower_bound <= mu <= upper_bound:
        contained_count += 1

# 真の値が95%信頼区間に含まれる確率を計算
contained_probability = contained_count / num_trials
print("真の値が95%信頼区間に含まれる確率:", contained_probability)

for i, (lower, upper) in enumerate(zip(lowers, uppers)):
    plt.plot([lower, upper], [i, i], c='C0')

print(f'信頼区間の平均 = {(np.array(uppers) - np.array(lowers)).mean()}')

plt.vlines(mu, 0, num_trials, colors='black')
plt.show()
