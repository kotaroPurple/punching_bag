
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import jn
from numpy.typing import NDArray


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


def generate_iq_from_displacement(
        displacements: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Generate IQ signal from displacement data."""
    return np.exp(2j * WAVE_NUMBER * displacements)


def generate_displacement(
        amp: float, frequency: float, duration: float, sample_rate: float, start: float = 0.) \
            -> NDArray[np.floating]:
    """Generate a sinusoidal displacement signal."""
    t = np.arange(0, duration, 1 / sample_rate)
    displacement = amp * np.sin(2 * np.pi * frequency * t) + start
    return displacement


def generate_iq_with_theory(
        displacement_amp: float, frequency: float, duration: float, sample_rate: float,
        order_list: list[int]) -> NDArray[np.complex128]:
    """Generate IQ signal using theoretical model."""
    t = np.arange(0, duration, 1 / sample_rate)
    iq_signal = np.zeros(len(t), np.complex128)
    for n in order_list:
        bessel_coeff = jn(n, 2 * WAVE_NUMBER * displacement_amp)
        iq_signal += bessel_coeff * np.exp(1j * 2 * np.pi * n * frequency * t)
    return iq_signal


def generate_gaussian_displacement(
        amp: float, frequency: float, center_time: float, sigma: float, duration: float, sample_rate: float) -> NDArray[np.floating]:
    """Generate a Gaussian-shaped displacement signal."""
    # 正規分布を周期的になるように生成
    t = np.arange(0, duration, 1 / sample_rate)
    data = np.zeros_like(t)
    n_blocks = int(duration * frequency) + 1
    period = 1 / frequency
    for n in range(n_blocks):
        # if n != 5:
        #     continue
        block_center = center_time + n * period
        data += generate_single_gaussian(amp, block_center, sigma, t)
    return data


def generate_single_gaussian(
        amp: float, center_time: float, sigma: float, t: NDArray[np.floating]) -> NDArray[np.floating]:
    """Generate a single Gaussian-shaped displacement signal."""
    gaussian = amp * np.exp(-0.5 * ((t - center_time) / sigma) ** 2)
    return gaussian


def apply_highpass_filter(
        data: NDArray, cutoff: float, fs: int, order: int=5) -> NDArray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered


def main() -> None:
    # displacement parameters
    # amplitudes = [0.000_2, 0.001]  # [m]
    # frequencies = [1.1, 0.2]  # [Hz]
    # amplitudes = [0.000_2]  # [m]
    # frequencies = [1.0]  # [Hz]
    # duration = 20.0  # [s]

    amplitudes = [0.020]  # [m]
    frequencies = [1 / 30.]  # [Hz]
    duration = 60.0  # [s]

    sample_rate = 100.0  # [Hz]
    # generate displacement
    total_displacement = np.zeros(int(duration * sample_rate))
    for amp, freq in zip(amplitudes, frequencies):
        displacement = generate_displacement(amp, freq, duration, sample_rate, start=0.001)
        total_displacement += displacement
    total_displacement += generate_gaussian_displacement(-0.000_1, frequencies[0], center_time=0.125, sigma=0.05, duration=duration, sample_rate=sample_rate)
    total_displacement += generate_gaussian_displacement(0.000_07, frequencies[0], center_time=0.625, sigma=0.05, duration=duration, sample_rate=sample_rate)
    # generate IQ signal
    iq_signal = generate_iq_from_displacement(total_displacement)
    # iq_theory = generate_iq_with_theory(
    #     max(amplitudes), frequencies[0], duration, sample_rate, order_list=list(range(-10, 11)))

    # apply filter
    cutoff_frequency = 2.55  # [Hz]
    int_cutoff = round(cutoff_frequency)
    max_order = 10
    order_list = list(range(-max_order, -int_cutoff + 1, 1)) + list(range(int_cutoff, max_order + 1, 1))
    filtered_iq_signal = apply_highpass_filter(iq_signal, cutoff_frequency, int(sample_rate), order=7)
    filtered_iq_theory = generate_iq_with_theory(
        max(amplitudes), frequencies[0], duration, sample_rate, order_list=order_list)

    filtered_iq_signal[:int(sample_rate)] = 0
    filtered_iq_signal[-int(sample_rate):] = 0

    n_conv = int(0.3 * sample_rate)
    abs_filtered = np.abs(filtered_iq_signal)
    mv_abs_filtered = np.convolve(abs_filtered, np.ones(n_conv) / n_conv, mode='same')

    # plot
    _, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    time = np.arange(0, duration, 1 / sample_rate)
    axes[0].plot(time, total_displacement)
    axes[0].set_title("Displacement Signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Displacement [m]")

    axes[1].plot(time, iq_signal.real, label='I Component', alpha=0.5)
    axes[1].plot(time, iq_signal.imag, label='Q Component', alpha=0.5)
    axes[1].set_title("IQ Signal")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    axes[2].plot(time, filtered_iq_signal.real, label='I Component', alpha=0.5)
    axes[2].plot(time, filtered_iq_signal.imag, label='Q Component', alpha=0.5)
    axes[2].plot(time, abs_filtered, label='Magnitude', alpha=0.5, c='gray')
    axes[2].plot(time, mv_abs_filtered, label='Magnitude (Moving Avg)', alpha=0.8, c='black')
    # axes[2].plot(time, filtered_iq_theory.real, label='I Component (Theory)', alpha=0.5, linestyle='--')
    # axes[2].plot(time, filtered_iq_theory.imag, label='Q Component (Theory)', alpha=0.5, linestyle='--')
    axes[2].set_title("Filtered IQ Signal")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
