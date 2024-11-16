
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


def calculate_signal_power(data: NDArray, times: NDArray, step: int, window: int) -> tuple[NDArray, NDArray]:
    power_array = []
    time_array = []
    for i in range(0, len(data)-window, step):
        segment = data[i:i+window]
        power = np.sqrt(np.mean(segment * np.conj(segment)))
        power_array.append(power)
        time_array.append(times[i])
    return np.array(power_array), np.array(time_array)


def calculate_bessel_coeff(n_degree: int, amplitude: float) -> float:
    return jn(n_degree, ALPHA * amplitude)


def calculate_bessel_coeff_list(n_degree: list[int], amplitude: list[float]) -> list[float]:
    result = [calculate_bessel_coeff(n, amp) for n, amp in zip(n_degree, amplitude)]
    return result


def generate_waves(
        fs: int, time_range: float, d_list: list[float], freq_list: list[float],
        initial_position: list[float]) -> tuple[NDArray, NDArray, NDArray]:
    # make waves
    length = int(fs * time_range)
    times = np.arange(length) / fs
    # phase
    phase = np.zeros_like(times)
    for d, f, init_p in zip(d_list, freq_list, initial_position):
        phase += ALPHA * d * np.sin(2 * np.pi * f * times) + ALPHA * init_p
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


def allan_variance(x, max_tau_exp=10):
    N = len(x)
    tau_values = []
    allan_variances = []
    print(x.shape)

    # 2の累乗でスケールを取る
    tau_values = [2**i for i in range(1, max_tau_exp+1)]
    taus = []

    for tau in tau_values:
        m = N // tau  # スロット数
        if m <= 1:
            continue

        # 各スロットの平均を計算
        averages = [np.mean(x[i * tau: (i + 1) * tau]) for i in range(m)]

        # 隣接スロット間の変動を計算
        delta_x = np.diff(averages)

        # アラン分散を計算
        allan_var = 0.5 * np.mean(delta_x * delta_x.conj())

        taus.append(tau)
        allan_variances.append(allan_var)

    return np.array(taus), np.array(allan_variances)



def main():
    # prepare
    fs = 500
    time_range = 70.
    # amplitudes, frequency list, delta phase list
    # mix
    d_list = [200.e-6, 4.e-3]
    f_list = [1., 0.3]
    init_theta_list = [0., 0.]
    # # mix
    # d_list = [1.e-3, 5.e-3]
    # f_list = [0.5, 0.3]
    # init_theta_list = [0., 0.]
    # # heart
    # d_list = [200.e-6]
    # f_list = [1.]
    # init_theta_list = [np.pi/2, np.pi/2]
    # # respiration
    # d_list = [5.e-3]
    # f_list = [0.3]
    # init_theta_list = [np.pi/2, np.pi/2]
    # waves
    init_position_list = [theta / ALPHA for theta in init_theta_list]
    times, iq_wave, phase = generate_waves(fs, time_range, d_list, f_list, init_position_list)

    # bandpass filter
    lowcut = 3.
    highcut = 10.
    iq_filtered = apply_bandpass_filter(iq_wave, lowcut, highcut, fs, order=5)

    # power
    iq_power, power_times = calculate_signal_power(iq_filtered, times, 5, fs//1)

    # fft
    # fft_result, fft_freq = calculate_fft(iq_wave, fs)
    fft_result, fft_freq = calculate_fft(iq_filtered, fs)
    fft_all_amp = np.abs(fft_result) / len(fft_result)
    fft_amp_without_minus, fft_freq_without_minus = get_abs_without_minus_frequency(
        fft_result, fft_freq)
    # shift: minus to plus
    fft_all_amp = np.fft.fftshift(fft_all_amp)
    fft_freq = np.fft.fftshift(fft_freq)

    # predict theoretical coefficients
    theoretical_coeffs, theoretical_freq = calculate_theoretical_iq_coeff(d_list, f_list, 10)
    abs_theoretical_coeffs = np.abs(theoretical_coeffs)
    _fft_amp_max = np.max(fft_all_amp)
    _abs_thoretical_max = np.max(abs_theoretical_coeffs)
    abs_theoretical_coeffs = abs_theoretical_coeffs * (_fft_amp_max / _abs_thoretical_max)

    # allan variance
    taus, allan_vars = allan_variance(iq_wave, max_tau_exp=12)
    taus_filtered, allan_vars_filterd = allan_variance(iq_filtered, max_tau_exp=12)

    # plot
    n_plot = 4
    plt.subplot(n_plot, 1, 1)
    plt.plot(times, np.real(iq_wave), alpha=0.5)
    plt.plot(times, np.imag(iq_wave), alpha=0.5)
    # plt.plot(times, np.real(iq_filtered), alpha=0.5)
    # plt.plot(times, np.imag(iq_filtered), alpha=0.5)
    # plt.plot(times, np.abs(iq_filtered), c='black', alpha=0.5)
    # plt.plot(power_times, iq_power, c='black', alpha=0.5)

    plt.subplot(n_plot, 1, 2)
    plt.plot(fft_freq_without_minus, fft_amp_without_minus)
    plt.xlim(-5., 5.)

    plt.subplot(n_plot, 1, 3)
    plt.plot(fft_freq, fft_all_amp)
    plt.vlines(theoretical_freq, 0., abs_theoretical_coeffs, color='red', alpha=0.5)
    plt.xlim(-5., 5.)

    plt.subplot(n_plot, 1, 4)
    plt.plot(taus / fs, allan_vars / np.max(allan_vars), label='original')
    plt.plot(taus_filtered / fs, allan_vars_filterd / np.max(allan_vars_filterd), label='filtered')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
