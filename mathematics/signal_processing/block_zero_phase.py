
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def filter_zero_phase(
        data: NDArray, filter_a: NDArray, filter_b: NDArray,
        block: int, converge: int, use_init_zi: bool) -> NDArray:
    # data is already one side filtered
    length = len(data)
    n_block = (length - converge) // block
    result = []
    # init zi
    if use_init_zi:
        zi = signal.lfilter_zi(filter_b, filter_a)
    else:
        zi = None
    # main
    for i in range(n_block):
        block_data = data[i*block:(i+1)*block + converge]
        inv_block_data = block_data.copy()[::-1]
        use_zi = zi if zi is None else zi * 1  # inv_block_data[0]
        zero_phased = signal.lfilter(filter_b, filter_a, inv_block_data, zi=use_zi)
        zero_phased = zero_phased[::-1] if use_zi is None else zero_phased[0][::-1]
        result.append(zero_phased[:block])
    result_array = np.concatenate(result)
    if (result_size := len(result_array)) < length:
        result_array = np.r_[result_array, [0.] * (length - result_size)]
    elif result_size > length:
        result_array = result_array[:length]
    return result_array


def main():
    np.random.seed(1)
    raw_data = 100 * (2 * np.random.random(10_000) - 1.)

    fs = 500.
    low_f = 3
    high_f = 10
    filter_b, filter_a = butter_bandpass(low_f, high_f, fs, order=3)

    block = 500
    converge = 250

    # filter
    one_side_filtered = signal.lfilter(filter_b, filter_a, raw_data)
    zero_phase_filtered = signal.filtfilt(filter_b, filter_a, raw_data)
    block_zero_phase = filter_zero_phase(
        one_side_filtered, filter_a, filter_b, block, converge, use_init_zi=False)
    block_zero_phase_with_zi = filter_zero_phase(
        one_side_filtered, filter_a, filter_b, block, converge, use_init_zi=True)
    diff = zero_phase_filtered - block_zero_phase
    diff_with_zi = zero_phase_filtered - block_zero_phase_with_zi

    plt.subplot(211)
    # plt.plot(one_side_filtered, label='one side', alpha=0.5, c='C3')
    plt.plot(zero_phase_filtered, label='zero-phase', alpha=0.5, c='C0')
    plt.plot(block_zero_phase, label='block', alpha=0.5, c='C1')
    plt.plot(block_zero_phase_with_zi, label='block zi', alpha=0.5, c='C2')
    plt.legend()

    plt.subplot(212)
    plt.plot(diff, label='simple diff', alpha=0.5, c='C0')
    plt.plot(diff_with_zi, label='zi', alpha=0.5, c='C1')
    plt.legend()
    plt.ylim((-3, 3))
    plt.show()


if __name__ == '__main__':
    main()
