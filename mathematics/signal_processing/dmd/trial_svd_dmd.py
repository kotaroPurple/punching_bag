
import numpy as np
import matplotlib.pyplot as plt

from core import (
    apply_svd, apply_standard_dmd, make_hankel_matrix, reconstruct_by_dmd)
from util import generate_data


def main():
    # data
    data_list = generate_data(mode=0)
    data = data_list[0]
    times = data_list[-1]

    # DMD
    n_rows = 100
    low_rank = 5
    hankel_mat = make_hankel_matrix(data, n_rows)
    mat_x, mat_y = hankel_mat[:, :-1], hankel_mat[:, 1:]
    svd_u, svd_s, svd_vh = apply_svd(mat_x)
    eigens, mat_phi, amps = apply_standard_dmd(mat_x, mat_y, svd_u, svd_s, svd_vh, low_rank)

    # reconstruction
    reconstructed, sub_waves = reconstruct_by_dmd(eigens, mat_phi, amps, times)
    wave_without_trend = np.sum(np.array(sub_waves[1:]), axis=0)

    plt.plot(times, data, alpha=0.5, label='original')
    plt.plot(times, reconstructed, alpha=0.5, label='DMD')
    plt.plot(times, sub_waves[0], alpha=0.5, label='trend', c='black')
    plt.plot(times, wave_without_trend, alpha=0.5, label='w/o')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
