
import numpy as np
import matplotlib.pyplot as plt
from typing import Final
from numpy.typing import NDArray


# x1
X1_FROM: Final[float] = 4.0
X1_TO: Final[float] = 10.0

# x2
X2_FROM: Final[float] = 2.0
X2_TO: Final[float] = 7.0


def generate_sample(is_x1: bool, number: int) -> NDArray:
    if is_x1:
        start = X1_FROM
        end = X1_TO
    else:
        start = X2_FROM
        end = X2_TO
    samples = np.random.uniform(start, end, number)
    return samples


def generate_probability(x_array: NDArray, is_x1: bool) -> NDArray:
    if is_x1:
        start = X1_FROM
        end = X1_TO
    else:
        start = X2_FROM
        end = X2_TO
    probs = np.zeros_like(x_array)
    probs[x_array < start] = 0.
    probs[x_array > end] = 0.
    valid_position = (x_array >= start) * (x_array <= end)
    probs[valid_position] = 1. / (end - start)
    return probs


def cumulative_distribution(x_array: NDArray, is_x1: bool) -> NDArray:
    if is_x1:
        start = X1_FROM
        end = X1_TO
    else:
        start = X2_FROM
        end = X2_TO
    y_array = np.zeros_like(x_array)
    y_array[x_array < start] = 0.
    y_array[x_array > end] = 1.
    valid_position = (x_array >= start) * (x_array <= end)
    valid_x = x_array[valid_position]
    y_array[valid_position] = (valid_x - start) / (end - start)
    return y_array


def montecarlo_way(number: int) -> float:
    x1_samples = generate_sample(True, number)
    x2_samples = generate_sample(False, number)
    x1_over_x2 = x1_samples > x2_samples
    return float(np.sum(x1_over_x2)) / number


def analytical_way(x_array: NDArray) -> tuple[NDArray, float]:
    cumulative_x2 = cumulative_distribution(x_array, is_x1=False)
    prob_x1 = generate_probability(x_array, is_x1=True)
    prob_x1_over_x2 = prob_x1 * cumulative_x2
    diff_x = x_array[1:] - x_array[:-1]
    cumulative = np.cumsum(prob_x1_over_x2[1:] * diff_x)
    return cumulative, cumulative[-1]


def main():
    # montecarlo
    number = 5000
    p_montecarlo = montecarlo_way(number)
    print(f'{p_montecarlo=}')
    # analytical
    x_array = np.linspace(X2_FROM - 1., X1_TO + 1., number)
    p_analytical_distribution, p_analytical = analytical_way(x_array)
    print(f'{p_analytical=}')
    plt.plot(x_array[1:], p_analytical_distribution)
    plt.show()


if __name__ == '__main__':
    main()
