
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import jn
from numpy.typing import NDArray


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int=5) -> NDArray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def apply_bandpass_filter(
        data: NDArray, lowcut: float, highcut: float, fs: int, order: int=5) -> NDArray:
    sos_values = butter_bandpass(lowcut, highcut, fs, order)
    filtered = signal.sosfiltfilt(sos_values, data, axis=-1)
    return filtered


def calculate_bessel_coeff(n_degree: int, amplitude: float) -> float:
    return jn(n_degree, ALPHA * amplitude)


def calculate_bessel_coeff_list(n_degree: list[int], amplitude: list[float]) -> list[float]:
    result = [calculate_bessel_coeff(n, amp) for n, amp in zip(n_degree, amplitude)]
    return result


def generate_waves(
        fs: int, time_range: float, d_list: list[float], freq_list: list[float],
        delta_list: list[float]) -> tuple[NDArray, NDArray, NDArray]:
    # make waves
    length = int(fs * time_range)
    times = np.arange(length) / fs
    # phase
    phase = np.zeros_like(times)
    for d, f, delta in zip(d_list, freq_list, delta_list):
        phase += ALPHA * d * np.sin(2 * np.pi * f * times - delta)
    # waves
    i_wave = np.cos(phase)
    q_wave = np.sin(phase)
    iq_wave = i_wave + 1j * q_wave
    return times, iq_wave, phase


def calculate_fft(wave: NDArray, fs: int) -> tuple[NDArray, NDArray]:
    # multiply window function
    fft_window = 2**(int(np.log2(len(wave))))
    wave_with_window = wave[:fft_window] * np.hanning(fft_window)
    # fft
    fft_result = np.fft.fft(wave_with_window, fft_window)
    fft_freq = np.fft.fftfreq(fft_window, 1/fs)
    return fft_result, fft_freq


def get_abs_without_minus_frequency(fft_result: NDArray, fft_freq: NDArray) -> tuple[NDArray, NDArray]:
    # amp
    _fft_amp = np.abs(fft_result) / len(fft_result)
    fft_amp = _fft_amp[:len(_fft_amp)//2]
    fft_amp[1:] += _fft_amp[-1:-len(_fft_amp)//2:-1]
    # fft_amp[0] = 0.
    fft_freq = fft_freq[:len(fft_freq)//2]
    return fft_amp, fft_freq


def calculate_theoretical_iq_coeff(
        d_list: list[float], f_list: list[float], max_n_degree: int) \
            -> tuple[list[float], list[float]]:
    degree_list = [n for n in range(-abs(max_n_degree), abs(max_n_degree) + 1)]
    bessel_list = []
    frequency_list = []
    for d, f in zip(d_list, f_list):
        bessel = calculate_bessel_coeff_list(degree_list, [d] * len(degree_list))
        freq = [f * n for n in degree_list]
        bessel_list.append(bessel)
        frequency_list.append(freq)
    # main
    combination_indices = itertools.product(*[range(len(lst)) for lst in bessel_list])
    result_coeff_list = []
    result_freq_list = []
    for indices in combination_indices:
        coeff = 1.
        sum_freq = 0.
        for i, bessel, freq in zip(indices, bessel_list, frequency_list):
            coeff *= bessel[i]
            sum_freq += freq[i]
        result_coeff_list.append(coeff)
        result_freq_list.append(sum_freq)
    return result_coeff_list, result_freq_list


def calculate_theoretical_stft(
        amplitude: float, max_bessel_degree: int, base_angular: float, target_freq_coeff: float,
        start_time: float, end_time: float, delta_time: float, number: int = 100) \
            -> tuple[NDArray, NDArray, list[int], NDArray, NDArray, NDArray]:
    # degrees
    degree_list = [i for i in range(-abs(max_bessel_degree), abs(max_bessel_degree) + 1)]
    # times
    starts = np.linspace(start_time, end_time, number)
    ends = starts + delta_time
    # theoretical stft
    stft_w = base_angular * target_freq_coeff
    real_part = np.zeros(len(starts))
    imag_part = np.zeros_like(real_part)
    bessel_list = []
    real_list = []
    imag_list = []
    for n in degree_list:
        bessel = calculate_bessel_coeff(n, amplitude)
        n_w = n * base_angular
        if n == target_freq_coeff:
            _real = ends - starts
            _imag = np.zeros_like(ends)
        else:
            _real = (np.sin((n_w - stft_w) * ends) - np.sin((n_w - stft_w) * starts)) \
                / (n_w - stft_w)
            _imag = -1. * (np.cos((n_w - stft_w) * ends) - np.cos((n_w - stft_w) * starts)) \
                / (n_w - stft_w)
        # addition
        real_part += bessel * _real
        imag_part += bessel * _imag
        # for output
        bessel_list.append(bessel)
        real_list.append(_real)
        imag_list.append(_imag)
    # output
    stft_value = real_part + 1.j * imag_part
    return starts, stft_value, degree_list, np.array(bessel_list), np.array(real_list), np.array(imag_list)


def main():
    # prepare
    fs = 500
    time_range = 70.
    # amplitude, frequency
    base_d = 200.e-6  # heart
    base_f = 1.  # heart
    base_w = 2 * np.pi * base_f
    # waves
    times, iq_wave, phase = generate_waves(fs, time_range, [base_d], [base_f], [0.])

    # bandpass filter
    lowcut = 3.
    highcut = 10.
    iq_filtered = apply_bandpass_filter(iq_wave, lowcut, highcut, fs, order=5)

    # theoretical stft
    max_bessel_degree = 10
    # stft_w_coeff = [1., 2., 3., 4., 5.]  # base_w * w_coeff
    # stft_w_coeff = [1. + 0.5 * i for i in range(10)]  # base_w * w_coeff
    stft_w_coeff = [3. + 0.1 * i for i in range(40)]
    start_time = 0.
    end_time = 3 * 1/base_f
    stft_time = 0.5 * (1/base_f)

    values = None
    for coeff in stft_w_coeff:
        times, stft_value, degree_list, bessel_values, real_part, imag_part = calculate_theoretical_stft(
            base_d, max_bessel_degree, base_w, coeff, start_time, end_time, stft_time)
        if values is None:
            values = np.abs(stft_value)
        else:
            values += np.abs(stft_value)
    plt.plot(times, np.abs(values))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
