
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.typing import NDArray


def generate_data(
        total_time: float, fs: int, first_time: float,
        sigma: float, n_gauss: int, step: float, gauss_coeff: float, fluctuation: float) \
            -> tuple[NDArray, NDArray]:
    # calculate length in data
    def calc_length(_time: float) -> int:
        return int(fs * _time)

    length = calc_length(total_time)
    gauss_step = calc_length(step)
    first_position = calc_length(first_time)
    gauss_sigma = calc_length(sigma)
    _fluctuation = calc_length(fluctuation)

    data = np.zeros(length)
    times = np.arange(length) / length * total_time

    for i in range(n_gauss):
        delta = np.random.uniform(-_fluctuation, _fluctuation + 1)
        center = delta + first_position + i * gauss_step
        left_side = max(0, int(center - 3 * gauss_sigma))
        right_side = min(length - 1, int(center + 3 * gauss_sigma))
        if left_side > length - 1:
            break
        x = np.arange(left_side, right_side + 1)
        normal_pdf = norm.pdf(x, loc=center, scale=float(gauss_sigma))
        if np.random.random() > 0.:
            data[x] = gauss_coeff * normal_pdf * 1  # np.random.uniform(0.5, 3.5, 1)
        # if np.max(x + gauss_step // 2) <= length - 1:
        #     data[x + gauss_step // 2] = -data[x]
    return data, times


def generate_wave(
        total_time: float, fs: int, first_time: float, n_wave: int, step: float, coeff: float) \
            -> tuple[NDArray, NDArray]:
    # calculate length in data
    def calc_length(_time: float) -> int:
        return int(fs * _time)

    length = calc_length(total_time)
    _step = calc_length(step)
    first_position = calc_length(first_time)

    data = np.zeros(length)
    times = np.arange(length) / length * total_time

    for i in range(n_wave):
        center = first_position + i * _step
        left_side = max(0, int(center - _step // 2))
        right_side = min(length - 1, int(center + _step // 2))
        if left_side > length - 1:
            break
        x = np.arange(left_side, right_side + 1)
        y = coeff * np.sin(2 * np.pi * x / _step)
        y += 2 * coeff * np.sin(4 * np.pi * x / _step)
        data[x] = y
    return data, times


def run_stft(data: NDArray, step_time: float, fs: int, ft_range: float) -> tuple[NDArray, NDArray]:
    def calc_length(_time: float) -> int:
        return int(fs * _time)

    step = calc_length(step_time)
    _ft_range = calc_length(ft_range)
    n_fft = int((len(data) - _ft_range) / step) + 1

    frequency = np.fft.fftfreq(_ft_range, 1 / fs)

    result = []
    window = np.hanning(_ft_range)
    # window = np.ones(_ft_range)
    for i in range(n_fft):
        start = i * step
        end = start + _ft_range
        tmp_result = np.fft.fft(data[start:end] * window)
        result.append(tmp_result)
        # if i == 0:
        #     print(start, end)
        #     plt.plot(data[start:end])
        #     plt.show()
    return np.array(result), frequency


def run_cepstrum(stft_data: NDArray) -> NDArray:
    amp_stft = np.abs(stft_data)
    log_spectrum = np.log(amp_stft)

    cepstrum = np.fft.ifft(log_spectrum, axis=1).real
    return cepstrum


def main():
    total_time = 30.
    fs = 500
    # gauss setting
    first_position = 0.1
    sigma = 0.05
    n_gauss = 30
    gauss_step = 1 / 1.3
    gauss_coeff = 1.
    fluctuation = 0.
    # stft setting
    step_time = 0.5
    short_to_normal = 5
    short_range = gauss_step * 2
    ft_range = short_range * short_to_normal

    data, times = generate_data(
        total_time, fs, first_position, sigma, n_gauss, gauss_step, gauss_coeff, fluctuation)
    # data, times = generate_wave(total_time, fs, first_position, n_gauss, gauss_step, gauss_coeff)
    fft_array, freq = run_stft(data, step_time, fs, ft_range)
    fft_array_short, freq_short = run_stft(data, step_time, fs, short_range)

    fft_amp = np.abs(fft_array) / short_to_normal
    fft_amp_short = np.abs(fft_array_short) / 1
    cepstrum = run_cepstrum(fft_array_short)
    quefrency = np.arange(cepstrum.shape[1]) / fs
    fft_amp[:, 0] = 0.
    fft_amp_short[:, 0] = 0.
    half = len(freq) // 2
    half_short = len(freq_short) // 2

    print(f'True Frequency = {1/gauss_step:.3f}')

    _, ax = plt.subplots(3)
    ax[0].plot(times, data)
    index = 3
    ymax1 = np.max(fft_amp[index, :])
    ymax2 = np.max(fft_amp_short[index, :])
    ymax1, ymax2 = 1, 1
    for i in range(index, index + 1):
        ax[1].bar(freq[:half], fft_amp[i, :half] / ymax1, width=0.1, alpha=0.5)
        ax[1].bar(freq_short[:half_short//2], fft_amp_short[i, :half_short//2] / ymax2, color='C1', width=0.1, alpha=0.5)
        # ax[1].plot(freq[:half], fft_amp[i, :half])
        # ax[1].plot(freq_short[:half_short//2], fft_amp_short[i, :half_short//2])
        ax[2].plot(1 / quefrency[:], cepstrum[i, :])
        # ax[1].plot(freq_short, fft_amp_short[i, :], color='black')
    ax[1].vlines(1 / gauss_step, ymin=0, ymax=np.max(fft_amp[i, :half]), color='black', alpha=0.5)
    ax[2].vlines(1 / gauss_step, ymin=0, ymax=np.max(cepstrum[i, :]), color='black', alpha=0.5)
    ax[1].set_xscale('log')
    # ax[2].set_xscale('log')
    ax[2].set_xlim(0, 20)
    plt.show()


if __name__ == '__main__':
    main()
