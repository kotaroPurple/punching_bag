
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# データの準備
data = np.array([2, 3, 5, 7, 8, 12, 15, 20, 100])  # 少ないデータの例

# 外れ値の定義を調整
def adjusted_outliers(data, factor=2.0):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data < lower_bound) | (data > upper_bound)]

# 外れ値の検出
outliers = adjusted_outliers(data, factor=2.0)
print("外れ値:", outliers)

# ブートストラップ法による信頼区間の計算
def bootstrap(data, n=1000):
    n_size = len(data)
    samples = np.random.choice(data, (n, n_size), replace=True)
    return samples

def bootstrap_ci(data, stat_func, alpha=0.05, n_bootstrap=1000):
    samples = bootstrap(data, n=n_bootstrap)
    stats = np.array([stat_func(sample) for sample in samples])
    lower_bound = np.percentile(stats, 100 * (alpha / 2))
    upper_bound = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

# 中央値、Q1、Q3の信頼区間を計算
median_ci = bootstrap_ci(data, np.median, n_bootstrap=100)
q1_ci = bootstrap_ci(data, lambda x: np.percentile(x, 25))
q3_ci = bootstrap_ci(data, lambda x: np.percentile(x, 75))

print("中央値の95%信頼区間:", median_ci)
print("第一四分位数の95%信頼区間:", q1_ci)
print("第三四分位数の95%信頼区間:", q3_ci)

# 箱ひげ図の描画
sns.boxplot(data=data)
plt.show()
