

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.special import jn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


LIGHT_SPEED = 3.e8
SENSOR_FREQ = 24.e9
WAVE_NUMBER = 2 * np.pi / (LIGHT_SPEED / SENSOR_FREQ)


def calculate_one_term(
        alpha: float, order_n: int, in_omega: float, omega_map: NDArray, tau_map: NDArray, \
        time_range: float) -> NDArray:
    # Jn(a).sinc((nw-omega).time_range/2).exp(inw.tau)
    values = jn(order_n, alpha) * np.sinc(0.5 * (order_n * in_omega - omega_map) / np.pi * time_range) * np.exp(1.j * order_n * in_omega * tau_map)
    return values


def calculate_stft(
        alpha: float, order_n_list: list[int], in_omega: float, omega_map: NDArray, tau_map: NDArray, \
        time_range: float, initial_phase: float = 0.) -> NDArray:
    # sum term
    values = np.zeros(omega_map.shape, dtype=np.complex128)
    for order_n in order_n_list:
        values += calculate_one_term(alpha, order_n, in_omega, omega_map, tau_map, time_range)
    # phase term
    phases = time_range * np.exp(1.j * (initial_phase - omega_map * tau_map))
    return phases * values


def generate_omega_and_time_map(
        start_time: float, end_time: float, step_time: float, start_omega: float, \
        end_omega: float, step_omega: float, minus_omega: bool) -> tuple[NDArray, NDArray]:
    # prepare
    times = np.arange(start_time, end_time, step_time)
    omegas = np.arange(start_omega, end_omega, step_omega)
    if minus_omega:
        if omegas[0] == 0.:
            appending = -1. * omegas[1:][::-1]
        else:
            appending = -1. * omegas[::-1]
        omegas = np.r_[appending, omegas]
    # meshgrid
    omegas2, times2 = np.meshgrid(omegas, times)
    return omegas2, times2


def generate_bessel_wave(
        start_time: float, end_time: float, step_time: float,
        alpha: float, order_n_list: list[int], in_omega: float) -> tuple[NDArray, NDArray]:
    # prepare
    times = np.arange(start_time, end_time, step_time)
    # main
    values = np.zeros(times.shape, dtype=np.complex128)
    for order_n in order_n_list:
        values += jn(order_n, alpha) * np.exp(1.j * order_n * in_omega * times)
    return values, times


def calculate_iq_power(
        times: NDArray, iq_wave: NDArray, step: int, length: int) -> tuple[NDArray, NDArray]:
    powers = iq_wave * iq_wave.conj()
    power_mat = sliding_window_view(powers.real, length)[::step]
    result_power = power_mat.mean(axis=1)
    indices = np.arange(0, len(times) - length + 1, step=step)
    return result_power, times[indices]


def remove_negative_frequency(data: NDArray) -> NDArray:
    fft_value = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(fft_value))
    negative_indices = fft_freq < 0.
    fft_value[negative_indices] = 0.
    reconstructed = np.fft.ifft(fft_value)
    return reconstructed


def extract_frequency_info(data: NDArray, fs: int) -> tuple[NDArray, NDArray]:
    fft_value = np.fft.fftshift(np.fft.fft(data))
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(fft_value), d=1/fs))
    fft_abs = np.abs(fft_value) / len(data)
    return fft_abs, fft_freq


def mix_wave(data: NDArray, omega: float, fs: int) -> NDArray:
    times = np.arange(len(data)) / fs
    mixed_wave = data * np.exp(1.j * omega * times)
    return mixed_wave


def apply_lowpass_filter(data: NDArray, cutoff: float, fs: int) -> NDArray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(4, normal_cutoff, 'lowpass', output='sos', analog=False)
    return signal.sosfiltfilt(sos, data)


def remove_negative_frequency_trial():
    object_freq = 1.
    object_omega = 2. * np.pi * object_freq
    object_displacement = 0.005  # [m]
    start_time = 0.
    n_wave = 10
    end_time = n_wave / object_freq
    fs = 100
    alpha = 2 * WAVE_NUMBER * object_displacement
    # wave freq
    start_n = 2
    end_n = 6
    order_n_list = np.arange(-end_n, end_n+1)
    order_n_list = [int(value) for value in order_n_list if abs(value) >= start_n]
    positive_order_n_list = [value for value in order_n_list if value >= 0]

    # original wave
    iq_wave, wave_times = generate_bessel_wave(start_time, end_time, 1/fs, alpha, order_n_list, object_omega)
    # mixing, lowpass, inverse mixing
    mix_freq_value = -(max(positive_order_n_list) + min(positive_order_n_list)) / 2
    mix_omega_value = 2 * np.pi * mix_freq_value
    mixed_wave = mix_wave(iq_wave, mix_omega_value, fs)
    lowpassed = apply_lowpass_filter(mixed_wave, abs(mix_freq_value), fs)
    positive_iq_wave = mix_wave(lowpassed, -mix_omega_value, fs)

    # ffts
    first_abs_fft, fft_freq = extract_frequency_info(iq_wave, fs)
    mixed_abs_fft, _ = extract_frequency_info(mixed_wave, fs)
    lowpass_abs_fft, _ = extract_frequency_info(lowpassed, fs)
    positive_abs_fft, _ = extract_frequency_info(positive_iq_wave, fs)

    # plot
    max_value = max(np.max(iq_wave.real), np.max(iq_wave.imag))
    min_value = min(np.min(iq_wave.real), np.min(iq_wave.imag))
    freq_max = max(order_n_list) + abs(mix_freq_value)
    fig = plt.figure(figsize=(12, 6))
    gs= GridSpec(5, 2, figure=fig)

    # #
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wave_times, iq_wave.real, alpha=0.5)
    ax1.plot(wave_times, iq_wave.imag, alpha=0.5)
    ax1.set_ylim(min_value, max_value)

    ax1_fft = fig.add_subplot(gs[0, 1])
    ax1_fft.plot(fft_freq, first_abs_fft)
    ax1_fft.set_xlim(-freq_max, freq_max)

    # #
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(wave_times, mixed_wave.real, alpha=0.5)
    ax2.plot(wave_times, mixed_wave.imag, alpha=0.5)
    ax2.set_ylim(min_value, max_value)

    ax2_fft = fig.add_subplot(gs[1, 1])
    ax2_fft.plot(fft_freq, mixed_abs_fft)
    ax2_fft.set_xlim(-freq_max, freq_max)

    # #
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(wave_times, lowpassed.real, alpha=0.5)
    ax3.plot(wave_times, lowpassed.imag, alpha=0.5)
    ax3.set_ylim(min_value, max_value)

    ax3_fft = fig.add_subplot(gs[2, 1])
    ax3_fft.plot(fft_freq, lowpass_abs_fft)
    ax3_fft.set_xlim(-freq_max, freq_max)

    # #
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(wave_times, positive_iq_wave.real, alpha=0.5)
    ax4.plot(wave_times, positive_iq_wave.imag, alpha=0.5)
    ax4.set_ylim(min_value, max_value)

    ax4_fft = fig.add_subplot(gs[3, 1])
    ax4_fft.plot(fft_freq, positive_abs_fft)
    ax4_fft.set_xlim(-freq_max, freq_max)

    # # compare to ideal
    # natural_positive_iq_wave, _ = generate_bessel_wave(start_time, end_time, 1/fs, alpha, positive_order_n_list, object_omega)
    positive_iq_wave_from_fft = remove_negative_frequency(iq_wave)
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(wave_times, positive_iq_wave_from_fft.real, alpha=0.5)
    ax5.plot(wave_times, positive_iq_wave_from_fft.imag, alpha=0.5)
    ax5.set_ylim(min_value, max_value)

    plt.show()


def compare_positive_and_all_frequencies():
    object_freq = 1.
    object_omega = 2. * np.pi * object_freq
    object_displacement = 0.0005  # [m]
    start_time = 0.
    n_wave = 3
    end_time = n_wave / object_freq
    fs = 100
    alpha = 2 * WAVE_NUMBER * object_displacement
    # wave freq
    start_n = 3
    end_n = 6
    order_n_list = np.arange(-end_n, end_n+1)
    order_n_list = [int(value) for value in order_n_list if abs(value) >= start_n]
    # positive_order_n_list = [value for value in order_n_list if value >= 0]

    # original wave
    iq_wave, wave_times = generate_bessel_wave(start_time, end_time, 1/fs, alpha, order_n_list, object_omega)
    positive_iq_wave = remove_negative_frequency(iq_wave)

    # amps
    iq_amp = np.abs(iq_wave)
    positive_amp = np.abs(positive_iq_wave)

    # plot
    fig = plt.figure(figsize=(12, 6))
    gs= GridSpec(2, 2, figure=fig)

    # #
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iq_wave.real, iq_wave.imag, alpha=0.5)
    ax1.set_xlim(-2., 2.)
    ax1.set_ylim(-2., 2.)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(positive_iq_wave.real, positive_iq_wave.imag, alpha=0.5)
    ax2.set_xlim(-2., 2.)
    ax2.set_ylim(-2., 2.)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(wave_times, iq_amp, alpha=0.5)
    ax3.plot(wave_times, positive_amp, alpha=0.5)

    plt.show()


def main():
    object_freq = 1.
    object_omega = 2. * np.pi * object_freq
    object_displacement = 0.001  # [m]
    start_time = 0.
    end_time = 10 / object_freq
    step_time = 0.025
    start_omega = 0. * (2 * np.pi)
    end_omega = 10. * (2 * np.pi)
    step_omega = 0.025 * (2 * np.pi)
    fs = 1000
    minus_omega = False
    omegas, times = generate_omega_and_time_map(start_time, end_time, step_time, start_omega, end_omega, step_omega, minus_omega=minus_omega)
    # stft
    initial_phase = np.deg2rad(30.)
    stft_time_rage = 0.5 * 1 / object_freq
    # order_n_list = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    # order_n_list = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]
    order_n_list = [-6, -5, -4, -3, 3, 4, 5, 6]
    # order_n_list = [-4, -3, 3, 4]
    # order_n_list = [-6, -5, -4, 4, 5, 6]
    positive_order_n_list = [value for value in order_n_list if value >= 0]
    alpha = 2 * WAVE_NUMBER * object_displacement

    iq_wave, wave_times = generate_bessel_wave(start_time, end_time, 1/fs, alpha, order_n_list, object_omega)
    positive_iq_wave = remove_negative_frequency(iq_wave)
    iq_powers, power_times = calculate_iq_power(wave_times, positive_iq_wave, 5, 100)
    stft_values = calculate_stft(alpha, order_n_list, object_omega, omegas, times, stft_time_rage, initial_phase)
    power_stft_values = stft_values * stft_values.conj()
    power_stft_values = np.abs(power_stft_values.T)  # (horizon: time, vertical: freq)
    power_at_time = np.sum(power_stft_values, axis=0)
    one_time = times[:, 0]

    # natural_positive_iq_wave, _ = generate_bessel_wave(start_time, end_time, 1/fs, alpha, positive_order_n_list, object_omega)

    # plt.plot(wave_times, iq_wave.real, alpha=0.5)
    # plt.plot(wave_times, iq_wave.imag, alpha=0.5)
    # plt.show()
    # return

    # fft_values = np.fft.fftshift(np.fft.fft(positive_iq_wave))
    # fft_freq = np.fft.fftshift(np.fft.fftfreq(len(fft_values), 0.001))
    # plt.plot(fft_freq, np.abs(fft_values))

    # plt.plot(wave_times, positive_iq_wave.real, alpha=0.5)
    # plt.plot(wave_times, positive_iq_wave.imag, alpha=0.5)
    # plt.plot(wave_times, natural_positive_iq_wave.real, alpha=0.5)
    # plt.plot(wave_times, natural_positive_iq_wave.imag, alpha=0.5)
    # plt.show()

    plt.plot(power_times, iq_powers, c='C0', label='Power')
    plt.plot(one_time, power_at_time, c='C1', label='STFT')
    plt.legend()
    plt.show()

    omega_to_freq = 1. / (2. * np.pi)
    omega_min = omega_to_freq * omegas.min() if minus_omega else 0.
    plt_extent = (one_time.min(), one_time.max(), omega_min, omega_to_freq * omegas.max())
    _, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(power_stft_values, extent=plt_extent, origin='lower', aspect='auto')
    ax2 = ax.twinx()
    alpha = 0. if minus_omega else 0.5
    ax2.plot(one_time, power_at_time, c='white', alpha=alpha)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Freq [Hz]')
    plt.colorbar(im, ax=ax, label='Power')
    plt.show()


if __name__ == '__main__':
    # remove_negative_frequency_trial()
    compare_positive_and_all_frequencies()
    # main()
