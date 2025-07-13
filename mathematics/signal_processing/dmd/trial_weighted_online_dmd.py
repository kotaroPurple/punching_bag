
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from weighted_online_dmd import WeightedOnlineDmd
from util import generate_data


def reconstruct(
        dmd: WeightedOnlineDmd, length: int, threshold: float = 0.98) \
            -> tuple[list[NDArray], NDArray]:
    wave_list = dmd.reconstruct(valid_number=0, time_index=length, threshold=threshold)
    reconstructed = np.sum(np.array(wave_list), axis=0)
    return wave_list, reconstructed


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
        n_blocks = 5
    elif mode == 1:
        data = data_list[0][:]
        times = data_list[-1][:]
        window_size = 40
        rho = 0.98
        n_blocks = 6

    # Online DMD
    remain_size = int(0.8 * len(data))
    first_size = len(data) - remain_size
    first_data = data[:first_size]
    remain_data = data[first_size:]

    dmd = WeightedOnlineDmd(window_size, rho=rho)

    # 1st data
    wave_keeper = []
    reconstructed_list = []
    dmd.set_initial_data(first_data, low_rank_threshold=0.99)
    _wave_list, _reconstructed = reconstruct(dmd, len(first_data))
    wave_keeper.append(_wave_list)
    reconstructed_list.append(_reconstructed)

    # for remain data
    one_block = remain_size // n_blocks
    for i in range(n_blocks):
        input_data = remain_data[i * one_block:(i + 1) * one_block]
        for one_data in input_data:
            dmd.update(float(one_data))
        _wave_list, _reconstructed = reconstruct(dmd, len(input_data))
        wave_keeper.append(_wave_list)
        reconstructed_list.append(_reconstructed)

    # plot reconstruction
    count = 0
    last_time = times[0]
    last_value = reconstructed_list[0][0]

    plt.figure(figsize=(14, 6))
    for i, one_block in enumerate(reconstructed_list):
        sub_times = times[count:count + len(one_block)]
        count += len(one_block)
        plt.plot(
            np.r_[last_time, sub_times], np.r_[last_value, one_block[:len(sub_times)]],
            alpha=0.7, c=f'C{i}', label=f'Block {i}')
        # for next
        last_time = sub_times[-1]
        last_value = one_block[len(sub_times)-1]
    plt.plot(times[:count], data[:count], alpha=0.5, c='black', label='original')
    plt.legend()
    plt.show()

    # plot modes
    count = 0

    plt.figure(figsize=(14, 6))
    for i, wave_list in enumerate(wave_keeper):
        size = len(wave_list[0])
        sub_times = times[count:count + size]
        count += size
        for one_wave in wave_list:
            plt.plot(sub_times, one_wave[:len(sub_times)], alpha=0.7, c=f'C{i}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
