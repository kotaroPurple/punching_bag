
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

LIGHT_OF_SPEED = 3.e8
SENSOR_FREQUENCY = 24.e9
COEFF_ALPHA = 2 * SENSOR_FREQUENCY / LIGHT_OF_SPEED
SAMPLE_RATE = 1000  # [Hz]


def calculate_frequency(
        times: NDArray, f_coeff: float, radius: float, f_object: float, delta: float = 0.) -> NDArray:
    freq = 2 * np.pi * f_coeff * radius * f_object * np.cos(2 * np.pi * f_object * times - delta)
    return freq


def calculate_phase(
        times: NDArray, f_coeff: float, radius: float, f_object: float, delta: float = 0.) -> NDArray:
    freq = 2 * np.pi * f_coeff * radius \
        * (np.sin(2 * np.pi * f_object * times - delta) + np.sin(delta))
    return freq


def calculate_iq_waves(
        times: NDArray, f_coeff: float, radius: float, f_object: float, delta: float = 0.) \
        -> tuple[NDArray, NDArray, NDArray, NDArray]:
    freq = calculate_frequency(times, f_coeff, radius, f_object, delta)
    theta_i = calculate_phase(times, f_coeff, radius, f_object, delta)
    theta_q = theta_i - np.pi / 2
    i_data = 1. * np.cos(theta_i)
    q_data = 1. * np.cos(theta_q)
    return np.c_[i_data, q_data], freq, theta_i, theta_q


def main():
    radius = 6.e-3  # [m]
    f_object = 1.  # [Hz]
    max_time = 2.  # [sec]
    delta = np.pi / 6
    times = np.linspace(0., max_time, int(max_time * SAMPLE_RATE))
    iq_data, iq_freq, theta_i, theta_q = calculate_iq_waves(times, COEFF_ALPHA, radius, f_object, delta=delta)

    # fft
    chunk = 2048
    fft_result = np.fft.fft(iq_data[:chunk, 0])
    fft_freq = np.fft.fftfreq(chunk, 1. / SAMPLE_RATE)
    fft_abs = np.abs(fft_result)
    fft_abs = fft_abs[:chunk//2]
    fft_freq = fft_freq[:chunk//2]

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(times, iq_data[:, 0], alpha=0.5, label='I')
    axes[0, 0].plot(times, iq_data[:, 1], alpha=0.5, label='Q')
    axes[0, 1].plot(iq_data[:, 0], iq_data[:, 1], c='C2', alpha=0.5, label='I+iQ')
    axes[1, 0].plot(fft_freq, fft_abs, c='C3', label='Freq Amp')
    axes[1, 0].vlines(f_object, ymin=0., ymax=np.max(fft_abs), color='gray', alpha=0.8)
    axes[1, 1].plot(times, iq_freq, label='freq.')

    axes[0, 1].set_xlim((-1.5, 1.5))
    axes[0, 1].set_ylim((-1.5, 1.5))
    axes[1, 1].set_ylim((-2*np.pi, 2*np.pi))
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 0].set_xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
