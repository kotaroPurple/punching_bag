
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def calculate_cos_cos(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = 0.5 * np.sin((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = angles - start_rad
    else:
        term2 = 0.5 * np.sin((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def calculate_sin_sin(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = -0.5 * np.sin((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = angles - start_rad
    else:
        term2 = 0.5 * np.sin((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def calculate_sin_cos(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = -0.5 * np.cos((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = np.zeros_like(angles)
    else:
        term2 = -0.5 * np.cos((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def main():
    start_rad = 0.
    end_rad = 2. * np.pi
    ms = [i for i in range(1, 6)]
    ns = [i for i in range(1, 6)]

    fig, axes = plt.subplots(nrows=len(ns), ncols=len(ms), figsize=(8, 8))
    for j, n_ in enumerate(ns):
        for i, m_ in enumerate(ms):
            angles, values = calculate_cos_cos(start_rad, end_rad, m_, n_, number=100)
            axes[j, i].plot(angles, values)
            axes[j, i].set_ylim(-end_rad, end_rad)
    plt.show()


if __name__ == '__main__':
    main()
