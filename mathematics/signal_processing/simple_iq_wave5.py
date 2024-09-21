
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from numpy.typing import NDArray


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


def calculate_integraration(
        base_frequency: float, coeff_n: int, target_coeff_freq: float, stft_time: float) -> complex:
    if coeff_n == target_coeff_freq:
        return stft_time
    # main
    _tmp_freq = (coeff_n - target_coeff_freq) * base_frequency
    numerator = -1.j * (
        np.exp(1.j * stft_time * _tmp_freq) - 1.)
    denominator = _tmp_freq
    return numerator / denominator


def calculate_bessel_coeff(n_degree: int, amplitude: float) -> float:
    return jn(n_degree, ALPHA * amplitude)


def simple_harmonic_theoretical_stft(
        displacement: float, base_frequency: float, target_coeff_freq: float, stft_time: float,
        bessel_max_degree: int, power_degree: int) -> NDArray:
    # for short
    def _calc_bessel(degree: int) -> float:
        return calculate_bessel_coeff(degree, displacement)

    def _calc_gn(coeff_degree: int) -> complex:
        return calculate_integraration(base_frequency, coeff_degree, target_coeff_freq, stft_time)

    # prepare
    bessel_max_degree = max(1, abs(bessel_max_degree))
    power_degree = abs(power_degree)
    # main
    power_list = []
    for n_freq in range(power_degree):
        # power = 0.
        tmp_power = 0.j
        for i in range(-bessel_max_degree, bessel_max_degree + 1):
            # g_n * (g_n_l).conj
            g_n = _calc_gn(i)
            g_n_l = _calc_gn(i-n_freq)
            tmp = g_n * np.conj(g_n_l)
            # tmp1 = 2. * np.abs(g_n * np.conj(g_n_l))
            # jn * j(n-l)
            tmp *= _calc_bessel(i) * _calc_bessel(i - n_freq)
            tmp_power += tmp
            # power
            # power += tmp1 * tmp2
        power = 2. * np.abs(tmp_power)
        if n_freq == 0:
            power /= 2.
        power_list.append(power)
    return np.array(power_list)


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


def main():
    # amplitudes, frequency list, delta phase list
    displacement = 200.e-6  # heart [m]
    # displacement = 2.e-3  # respiration [m]
    # displacement = 0.1  # respiration [m]
    base_freq = 1.  # [Hz]
    stft_time = (1/base_freq) * 0.5

    stft_freq_coeff = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    for i, coeff in enumerate(stft_freq_coeff):
        power_list = simple_harmonic_theoretical_stft(
            displacement, base_freq, coeff, stft_time, bessel_max_degree=40, power_degree=10)
        # power_list /= power_list[1]
        plt.scatter(
            base_freq * np.arange(len(power_list)), power_list,
            c=f'C{i}', label=f'{coeff * base_freq:.2f} [Hz]', s=10)
        # plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
