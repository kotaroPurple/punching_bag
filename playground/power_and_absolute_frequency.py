
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


LIGHT_SPEED = 3e8
SENSOR_FREQUENCY = 24e9
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQUENCY / LIGHT_SPEED


def generate_iq_wave(
        displacements: NDArray[np.floating], wave_number: float, amp: float = 1.) -> NDArray[np.complex128]:
    return amp * np.exp(2j * wave_number * displacements).astype(np.complex128)


def analyze_frequency(data: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    """
    Analyze the frequency content of the input data.

    Args:
        data (NDArray): Input data array.
        fs (float): Sampling frequency.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing the frequency array and the corresponding power spectrum.
    """
    n = len(data)
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    fft_result = np.fft.fftshift(np.fft.fft(data * np.hamming(len(data))))
    power_spectrum = np.abs(fft_result) ** 2 / n
    return freq, power_spectrum


def main():
    # displacements
    # base_frequency_list = [1.0, 0.15]  # [Hz]
    # base_move_list = [0.000_1000, 0.002_1]  # [mm]
    # deltas = [0.0, np.deg2rad(30)]
    base_frequency_list = [1.0]  # [Hz]
    base_move_list = [0.000_1000]  # [mm]
    deltas = [0.0]
    max_times = 40.
    sample_rate = 100.  # [Hz]
    times = np.arange(0., max_times, 1. / sample_rate)
    displacements = np.zeros(len(times), dtype=np.float64)
    for freq, move, delta in zip(base_frequency_list, base_move_list, deltas):
        displacements += move * np.sin(2 * np.pi * freq * times + delta)

    # iq wave
    iq_wave = generate_iq_wave(displacements, WAVE_NUMBER)
    iq_minus_dc = iq_wave - np.mean(iq_wave)
    iq_power = (iq_minus_dc * iq_minus_dc.conj()).real
    abs_iq = np.sqrt(iq_power)

    iq_power -= np.mean(iq_power)
    abs_iq -= np.mean(abs_iq)

    freq, power_spectrum = analyze_frequency(iq_power, sample_rate)
    _, abs_spectrum = analyze_frequency(abs_iq, sample_rate)

    power_spectrum /= np.max(power_spectrum)
    abs_spectrum /= np.max(abs_spectrum)

    plt.figure(figsize=(12, 6))
    plt.plot(times, iq_power, alpha=0.7)
    plt.plot(times, abs_iq, alpha=0.7)
    # plt.plot(times, iq_minus_dc.real, label="real")
    # plt.plot(times, iq_minus_dc.imag, label="imag")

    plt.figure(figsize=(12, 6))
    plt.plot(freq, power_spectrum, alpha=0.7)
    plt.plot(freq, abs_spectrum, alpha=0.7)
    plt.xlim(-5.5, 5.5)
    plt.yscale("log")

    plt.show()


if __name__ == '__main__':
    main()
