
# ref: https://qiita.com/tanaka_soichiro/items/56c02ad53bc0babe62ab

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from core import make_hankel_matrix, apply_svd, apply_standard_dmd, reconstruct_by_dmd


def make_data(seed: int = 123) -> list[NDArray]:
    np.random.seed(seed)
    number = 400
    process_time = 1.0
    t_raw = np.linspace(0, process_time, number)
    trend = 10 * (t_raw - process_time) ** 2
    periodic1 = np.sin(10 * 2 * np.pi * t_raw) / np.exp(-2 * t_raw)
    periodic2 = np.sin(20 * 2 * np.pi * t_raw)
    noise = 1.5 * (np.random.rand(number) - 0.5)
    data = trend + periodic1 + periodic2 + noise
    return [data, trend, periodic1, periodic2, noise, t_raw]


def main():
    data, trend, periodic1, periodic2, noise, times = make_data()

    window_size = 70
    hankel_mat = make_hankel_matrix(data, window_size)
    mat_x = hankel_mat[:, :-1]
    mat_y = hankel_mat[:, 1:]

    # SVD
    _U, _S, _Vh = apply_svd(mat_x)
    # sigma_sumsq = (_S ** 2).sum()
    # sigma_array = (_S ** 2) / sigma_sumsq
    # cumlative_contrib = (_S ** 2).cumsum() / sigma_sumsq * 100.


    # DMD
    low_rank = 5
    eigens, mat_phi, amps = apply_standard_dmd(mat_x, mat_y, _U, _S, _Vh, low_rank)

    # reconstruction
    reconstructed, wave_list = reconstruct_by_dmd(eigens, mat_phi, amps, times)

    # frequency
    dt = times[1] - times[0]
    freqs = np.angle(eigens) / (2 * np.pi * dt)
    print()
    print('---- frequency [Hz] ----')
    print(f'{freqs=}')

    _, ax = plt.subplots(figsize=(8, 4))
    for i, one_phi in enumerate(mat_phi.T):
        ax.plot(one_phi.real, alpha=0.5, label=f'freq: {freqs[i]:.2f} [Hz]')
    ax.legend()

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, data)
    ax.plot(times, reconstructed)

    for sub_wave in wave_list:
        ax.plot(times, sub_wave, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
