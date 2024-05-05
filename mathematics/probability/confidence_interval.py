
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import chi2


# パラメーターの設定
mu = 0.5  # 平均
sigma = 1.5  # 標準偏差
sample_size = 500  # サンプルサイズ
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


def confidence_interval_variance(data, confidence_level=0.95):
    n = len(data)
    sample_variance = np.var(data, ddof=1)  # 不偏分散を計算

    # カイ二乗分布のパーセントポイントを計算
    chi2_lower = chi2.ppf((1 - confidence_level) / 2, df=n - 1)
    chi2_upper = chi2.ppf((1 + confidence_level) / 2, df=n - 1)

    # 信頼区間の計算
    lower_bound = (n - 1) * sample_variance / chi2_upper
    upper_bound = (n - 1) * sample_variance / chi2_lower

    return lower_bound, upper_bound



# 繰り返し試行して、95%信頼区間が真の値を含む確率を計算
num_trials = 100
contained_count = 0
contained_count_s = 0
lowers = []
uppers = []
lowers_s = []
uppers_s = []

for _ in range(num_trials):
    # 正規分布からサンプルを生成
    data = np.random.normal(mu, sigma, sample_size)
    lower_bound, upper_bound = calculate_confidence_interval(data)
    lower_s, upper_s = confidence_interval_variance(data, confidence_level)
    lowers.append(lower_bound)
    uppers.append(upper_bound)
    lowers_s.append(np.sqrt(lower_s))
    uppers_s.append(np.sqrt(upper_s))

    # 真の値が信頼区間に含まれるか確認
    if lower_bound <= mu <= upper_bound:
        contained_count += 1

    if lowers_s[-1] <= sigma <= uppers_s[-1]:
        contained_count_s += 1


# 真の値が95%信頼区間に含まれる確率を計算
contained_probability = contained_count / num_trials
print("真の値が95%信頼区間に含まれる確率:", contained_probability)
print(f'信頼区間の平均 = {(np.array(uppers) - np.array(lowers)).mean()}')
print()
contained_probability = contained_count_s / num_trials
print("真の標準偏差が95%信頼区間に含まれる確率:", contained_probability)
print(f'信頼区間の平均 = {(np.array(uppers_s) - np.array(lowers_s)).mean()}')

plt.subplot(211)

for i, (lower, upper) in enumerate(zip(lowers, uppers)):
    plt.plot([lower, upper], [i, i], c='C0')


plt.vlines(mu, 0, num_trials, colors='black')

plt.subplot(212)

for i, (lower, upper) in enumerate(zip(lowers_s, uppers_s)):
    plt.plot([lower, upper], [i, i], c='C1')

plt.vlines(sigma, 0, num_trials, colors='black')

plt.show()
