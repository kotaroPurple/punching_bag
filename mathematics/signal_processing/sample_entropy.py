
import numpy as np
from numpy.typing import NDArray


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


def estimate_n0_n1(data_length: int) -> tuple[int, int]:
    # n0 = max(1024, int(np.sqrt(data_length)))
    n0 = int(np.sqrt(data_length))
    n1 = min(5 + int(np.log2(data_length)), int(data_length / n0))
    return n0, n1


def monte_carlo_sample_entropy(data: NDArray, m: int, r: float, n0: int, n1: int) -> float:
    biased_r = r * np.std(data)
    length = len(data)
    n_sequence = length - m - 1
    count_A = 0
    count_B = 0
    for _ in range(n1):
        s_indices = np.random.choice(n_sequence, n0, replace=False)
        a_indices = np.concatenate([np.repeat(s, n0 - 1 - i) for i, s in enumerate(s_indices[:-1])])
        b_indices = np.concatenate([s_indices[i+1:] for i in range(n0 - 1)])
        #
        count_A += count_similarity(data, a_indices, b_indices, m, biased_r)
        count_B += count_similarity(data, a_indices, b_indices, m + 1, biased_r)
    avg_A = count_A / n1
    avg_B = count_B / n1
    if avg_A > 0 and avg_B > 0:
        return -np.log(avg_B / avg_A)
    else:
        print('A and B are 0')
        return -np.log(2/(n_sequence * (n_sequence + 1)))


def count_similarity(data: NDArray, a_indices_: NDArray, b_indices_: NDArray, m_: int, r_: float) -> int:
    _a_indices = np.array([np.arange(_a, _a+m_) for _a in a_indices_])
    _b_indices = np.array([np.arange(_b, _b+m_) for _b in b_indices_])
    vec_a = data[_a_indices].reshape(-1, m_)
    vec_b = data[_b_indices].reshape(-1, m_)
    return _count_lower_thresohld(vec_a, vec_b, r_)


def _count_lower_thresohld(vec_a: NDArray, vec_b: NDArray, r: float) -> int:
    abs_diff = np.abs(vec_a - vec_b)
    lower_than_threshold = np.max(abs_diff, axis=1) < r
    return np.sum(lower_than_threshold)


# サンプル信号の生成
# np.random.seed(0)
length = 512
signal = np.sin(np.linspace(0, 10 * np.pi, length)) + 0.1 * np.random.randn(length)

# サンプルエントロピーの計算
m = 2
r = 0.2
sampen = sample_entropy(signal, m, r)
print(f'Sample Entropy: {sampen}')

n0, n1 = estimate_n0_n1(len(signal))
# n0 = 512
# n1 = 2
print(f'{n0=}, {n1=}')
mc_se = monte_carlo_sample_entropy(signal, m, r, n0, n1)
print(f'Monte-Carlo Sample Entropy: {mc_se}')
