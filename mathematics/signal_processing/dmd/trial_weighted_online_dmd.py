
import numpy as np
import matplotlib.pyplot as plt

from weighted_online_dmd import WeightedOnlineDmd
from util import generate_data


def main() -> None:
    # mode option
    mode = 1
    # data
    data_list = generate_data(mode)
    if mode == 0:
        data = data_list[0]
        times = data_list[-1]
        window_size = 70
        rho = 0.98
    elif mode == 1:
        data = data_list[0][:]
        times = data_list[-1][:]
        window_size = 100
        rho = 0.98

    # Online DMD
    remain_size = len(data) // 2
    first_size = len(data) - remain_size
    first_data = data[:first_size]
    remain_data = data[first_size:]

    dmd = WeightedOnlineDmd(window_size, rho=rho)
    dmd.set_initial_data(first_data, low_rank_threshold=0.99)

    for one_data in remain_data:
        dmd.update(one_data)

    wave_list = dmd.reconstruct(valid_number=0, time_index=len(data) - window_size + 1, threshold=0.98)
    reconstructed = np.sum(np.array(wave_list), axis=0)

    for i, one_wave in enumerate(wave_list):
        plt.plot(times, one_wave[:len(data)], alpha=0.5, label=f'sub {i}')
    plt.plot(times, reconstructed[:len(data)], alpha=0.5, c='red', label='reconstruted')
    plt.plot(times, data, alpha=0.5, c='black', label='original')
    plt.legend()
    plt.show()

    wave_list = dmd.reconstruct_from_start(
        start_vec=data[:window_size], valid_number=0, time_index=len(data) - window_size + 1, threshold=0.98)
    reconstructed = np.sum(np.array(wave_list), axis=0)

    for i, one_wave in enumerate(wave_list):
        plt.plot(times, one_wave[:len(data)], alpha=0.5, label=f'sub {i}')
    plt.plot(times, reconstructed[:len(data)], alpha=0.5, c='red', label='reconstruted')
    plt.plot(times, data, alpha=0.5, c='black', label='original')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
