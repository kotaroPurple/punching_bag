
import numpy as np

def _phi(time_series, m, r):
    N = len(time_series)
    patterns = [time_series[i:i + m] for i in range(N - m + 1)]
    count = 0

    for i in range(len(patterns)):
        for j in range(len(patterns)):
            if i != j:
                distance = np.max(np.abs(patterns[i] - patterns[j]))
                if distance <= r:
                    count += 1

    return count / (N - m + 1)

def sample_entropy(time_series, m, r):
    r *= np.std(time_series)
    phi_m = _phi(time_series, m, r)
    phi_m_plus_1 = _phi(time_series, m + 1, r)
    return -np.log(phi_m_plus_1 / phi_m)

# サンプル信号の生成
np.random.seed(0)
signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.5 * np.random.randn(1000)

# サンプルエントロピーの計算
m = 2
r = 0.2
sampen = sample_entropy(signal, m, r)
print(f'Sample Entropy: {sampen}')


# def compute_differences(vector):
#     # ベクトルの長さを取得
#     n = len(vector)

#     # すべての i < j を満たすインデックスペアを取得
#     i_indices, j_indices = np.triu_indices(n, k=1)

#     # i < j を満たす差分を計算
#     differences = vector[j_indices] - vector[i_indices]

#     return differences
