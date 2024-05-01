
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray


def calculate_frequency_response(
        filter_a: NDArray, filter_b: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    frequency, response = signal.freqz(filter_b, filter_a, fs=fs)
    return frequency, response


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print(low, high)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def precise_butter_bandpass(
        samplerate, fp: list[float], fs: list[float], gpass: float, gstop: float, order: int) \
        -> tuple[NDArray, NDArray]:
    fn = samplerate / 2
    wp = np.array(fp) / fn
    ws = np.array(fs) / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    print(N, min(order, N))
    print(Wn)
    b, a = signal.butter(min(order, N), Wn, "band")
    return b, a


def main():
    # np.random.seed(1)
    raw_data = 50 * (2 * np.random.random(10_000) - 1.)

    fs = 500.
    low_f = 3
    high_f = 10
    filter_b, filter_a = butter_bandpass(low_f, high_f, fs, order=3)
    filter_b2, filter_a2 = precise_butter_bandpass(fs, [low_f, high_f], [1, 20], 10, 50, order=4)
    name_list = ['Simple', 'Precise']

    res_freq_list = []
    response_list = []
    for a, b in zip((filter_a, filter_a2), (filter_b, filter_b2)):
        res_freq, response = calculate_frequency_response(a, b, fs)
        res_freq_list.append(res_freq)
        response_list.append(response)

    print(f'{filter_a=}')
    # print(f'{filter_b=}')
    print(f'{filter_a2=}')
    # print(f'{filter_b2=}')

    # filter
    one_side_filtered = signal.lfilter(filter_b2, filter_a2, raw_data)
    zero_phase_filtered = signal.filtfilt(filter_b2, filter_a2, raw_data)
    pseudo_zero_phase_filtered = signal.lfilter(filter_b2, filter_a2, one_side_filtered[::-1])[::-1]
    diff = zero_phase_filtered - pseudo_zero_phase_filtered

    # fft
    start = len(raw_data) // 2
    length = 512
    fft1 = np.fft.rfft(one_side_filtered[start:start+length], length)
    fft2 = np.fft.rfft(zero_phase_filtered[start:start+length], length)
    abs_fft1 = np.abs(fft1)
    abs_fft2 = np.abs(fft2)

    # 振幅応答のプロット
    plt.figure()
    plt.subplot(5, 1, 1)
    for name, freq, response in zip(name_list, res_freq_list, response_list):
        plt.plot(freq, 20 * np.log10(abs(response)), label=name, alpha=0.5)
    # plt.xlim(0, 50)
    plt.xscale('log')
    plt.ylim(-100, 2)
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response (Magnitude)')
    plt.legend()

    # 位相応答のプロット
    plt.subplot(5, 1, 2)
    for name, freq, response in zip(name_list, res_freq_list, response_list):
        plt.plot(freq, np.unwrap(np.angle(response)), label=name, alpha=0.5)
    # plt.xlim(0, 50)
    plt.xscale('log')
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Phase (radians)')
    plt.title('Frequency Response (Phase)')
    plt.legend()

    # filter
    plt.subplot(5, 1, 3)
    plt.plot(one_side_filtered, label='lfilter', alpha=0.5)
    plt.plot(zero_phase_filtered, label='zero-phase', alpha=0.5)
    plt.plot(pseudo_zero_phase_filtered, label='pseudo', alpha=0.5)
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(diff, label='diff zero-phase')
    plt.legend()

    # fft
    plt.subplot(5, 1, 5)
    plt.plot(abs_fft1, label='fft lfilter')
    plt.plot(abs_fft2, label='fft zero-phase')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
