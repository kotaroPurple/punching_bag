
# 参考: https://datachemeng.com/wp-content/uploads/outlierdetection.pdf

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# サンプルデータ
data = np.array([2, 3, 5, 7, 8, 12, 15, 20, 100, 30, 50])

# 箱ひげ図による外れ値検出
def boxplot_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data < lower_bound) | (data > upper_bound)]

boxplot_outliers = boxplot_outliers(data)
print("箱ひげ図の外れ値:", boxplot_outliers)

# Hampel Identifierによる外れ値検出
def hampel_outliers(data, threshold=3):
    median = np.median(data)
    deviation = np.abs(data - median)
    mad = np.median(deviation)
    mad_scaled = mad * 1.4826
    lower_bound = median - threshold * mad_scaled
    upper_bound = median + threshold * mad_scaled
    return data[(data < lower_bound) | (data > upper_bound)]

hampel_outliers = hampel_outliers(data)
print("Hampel Identifierの外れ値:", hampel_outliers)

# 箱ひげ図の描画
sns.boxplot(data=data)
plt.show()
