
import numpy as np
from numpy.typing import NDArray


def generate_data(number: int, p: float) -> NDArray:
    return np.random.binomial(1, p, number)


def count_valid_windows(data: NDArray, threshold_count: int, window: int) -> int:
    window_sums = np.convolve(data, np.ones(window, dtype=int), 'valid')
    success_count = (window_sums >= threshold_count).sum()
    return success_count


def _monte_carlo_onetime(
        number: int, p: float, threshold_count: int, window: int, target_ratio: float) -> bool:
    # data
    n_windows = number - (window - 1)
    data = generate_data(number, p)
    # count: greater or equal threshold
    success_count = count_valid_windows(data, threshold_count, window)
    # check over target ratio
    return (int(success_count) >= n_windows * target_ratio)


def monte_carlo_probability(
        number: int, p: float, threshold_count: int, window: int, target_ratio: float,
        trials: int) -> float:
    # trial
    result_bools = [_monte_carlo_onetime(number, p, threshold_count, window, target_ratio) for _ in range(trials)]  # NOQA
    return sum(result_bools) / trials


# パラメータの設定例
p = 0.8  # 1が生成される確率
number = 500  # データ列の長さ
m = 9  # 部分列の長さ
s = 5  # 部分列に含まれる1の最小数
alpha = 0.5  # 目標割合
num_sim = 1000  # シミュレーションの回数

# # 確率の推定
estimated_prob = monte_carlo_probability(number, p, s, m, alpha, num_sim)
print(f"推定された確率: {estimated_prob * 100:.2f}%")
