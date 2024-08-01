
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.typing import NDArray


@dataclass
class CountInfo:
    continuous: NDArray
    count: NDArray


@dataclass
class CountEvaluation:
    value: int = 0
    equal: bool = False  # value を含む
    more: bool = False  # value より大きいものが 1つ以上ある
    equal_and_more: bool = False  # value 以上のものが 1つ以上ある
    less: bool = False  # 全てが value より小さい
    equal_and_less: bool = False  # 全てが value 以下
    maximum: bool = False  # 最大が value
    minimum: bool = False  # 最小が value


def generate_samples(number: int, one_prob: float) -> NDArray:
    if not (0. <= one_prob <= 1.):
        raise ValueError()
    data = np.random.choice([0, 1], number, p=[1. - one_prob, one_prob])
    return data


def count_continuous_zero_and_one(data: NDArray) -> tuple[CountInfo, CountInfo]:
    # to bool
    _data = data != 0
    # not(data[0]), data[:], not(data[-1])
    data_with_side = np.r_[~_data[0], data, ~_data[-1]]
    # find not-continuous indices
    xor_value = np.logical_xor(data_with_side[1:], data_with_side[:-1])
    true_indices, = np.where(xor_value)
    # continuous count
    counts = true_indices[1:] - true_indices[:-1]
    # count 0, 00, 000,..., 1, 11, 111, ...
    if _data[0]:
        count_zero = counts[1::2]
        count_one = counts[::2]
    else:
        count_zero = counts[::2]
        count_one = counts[1::2]
    # get unique from counts
    zero_continuous, zero_continuous_count = np.unique(count_zero, return_counts=True)
    one_continuous, one_continuous_count = np.unique(count_one, return_counts=True)
    # result
    zero_info = CountInfo(zero_continuous, zero_continuous_count)
    one_info = CountInfo(one_continuous, one_continuous_count)
    return zero_info, one_info


def evaluate_continuous(info: CountInfo, continuous_count: int) -> CountEvaluation:
    evaluation = CountEvaluation(value=continuous_count)
    if len(info.continuous) == 0:
        return evaluation
    evaluation.equal = continuous_count in info.continuous
    evaluation.equal_and_more = bool(np.any(info.continuous >= continuous_count))  # 1つでも大きい
    evaluation.more = evaluation.equal_and_more and (evaluation.equal is False)
    evaluation.equal_and_less = bool(np.all(info.continuous <= continuous_count))  # 全てが小さい
    evaluation.less = evaluation.equal_and_less and (evaluation.equal is False)
    evaluation.maximum = info.continuous[-1] == continuous_count
    evaluation.minimum = info.continuous[0] == continuous_count
    return evaluation


def make_prob_matrix(p_list: list[float]) -> NDArray:
    length = len(p_list)
    mat = np.zeros((length + 1, length + 1))
    for i, p_i_to_next in enumerate(p_list):
        p_back = 1. - p_i_to_next
        mat[i, 0] = p_back
        mat[i, i+1] = p_i_to_next
    mat[-1, 0] = 1.0
    return mat.T


def calcualte_eigens(mat: NDArray) -> None:
    pass


def main2():
    p_list = [0.05, 0.05, 0.05]
    p_mat = make_prob_matrix(p_list)
    print(p_mat)
    eigen_w, eigen_v = np.linalg.eig(p_mat)
    inv_v = np.linalg.inv(eigen_v)
    tmp_mat = np.zeros_like(p_mat)
    tmp_mat[0, 0] = 1.0
    result = eigen_v @ tmp_mat @ inv_v
    start_vec = np.zeros(len(p_list) + 1)
    start_vec[0] = 1.
    convergence = result @ start_vec
    real_convergence = np.real(convergence)
    print(real_convergence)
    print(np.sum(real_convergence))
    plt.scatter(np.arange(len(real_convergence)), real_convergence)
    plt.show()


def main():
    # np.random.seed(1)
    n_trial = 10000
    one_prob = 0.05
    number = 400
    one_hist = np.zeros(number, dtype=np.int64)
    maximum_info = {key: 0 for key in range(1, 6)}
    for _ in tqdm(range(n_trial), desc='Trial'):
        data = generate_samples(number, one_prob)
        _, one_info = count_continuous_zero_and_one(data)
        one_hist[one_info.continuous] += one_info.count
        for count in range(1, 6):
            evaluation = evaluate_continuous(one_info, count)
            maximum_info[count] += int(evaluation.equal_and_more)

    # show
    print(maximum_info)
    for key, value in maximum_info.items():
        print(key, value / n_trial)

    max_prob = maximum_info[4] / n_trial

    one_year = 365
    print(f'{((1. - max_prob)**one_year)}')
    # plt.plot(one_hist)
    # plt.show()


if __name__ == '__main__':
    main()
    main2()
