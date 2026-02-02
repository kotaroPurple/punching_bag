
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
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


def bandpass_filter(x: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if not (0.0 < low_hz < high_hz < fs_hz * 0.5):
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist.")
    b, a = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz)
    return signal.filtfilt(b, a, x)


def triangle_wave(t: np.ndarray, freq_hz: float, peak_pos: float, amp: float = 1.0, start: float = 0.0) -> np.ndarray:
    """Asymmetric triangle wave in [-1, 1] with peak position in (0, 1)."""
    if not (0.0 < peak_pos < 1.0):
        raise ValueError("peak_pos must be between 0 and 1 (exclusive).")
    return start + amp * signal.sawtooth(2.0 * np.pi * freq_hz * t, width=peak_pos)


def main() -> None:
    duration = 20.0
    sample_rate = 100.0
    # d1 = generate_displacement(amp=0.000_2, frequency=1.0, duration=duration, sample_rate=sample_rate)
    # d2 = generate_displacement(amp=0.000_2, frequency=1.0, duration=duration, sample_rate=sample_rate, start=0.001_0)
    times = np.arange(0, duration, 1 / sample_rate)
    d1 = triangle_wave(times, freq_hz=1.0, peak_pos=0.2, amp=0.000_2, start=0.000_0)
    d2 = triangle_wave(times, freq_hz=1.0, peak_pos=0.7, amp=0.000_2, start=0.003_0)

    iq1 = generate_iq_from_displacement(d1)
    iq2 = generate_iq_from_displacement(d2)
    iq_sum = iq1 + iq2

    filtered_iq = bandpass_filter(iq_sum, fs_hz=sample_rate, low_hz=3.0, high_hz=6.0, order=4)

    plt.figure(figsize=(6, 6))
    plt.plot(times, d1, label="Object 1", alpha=0.7)
    plt.plot(times, d2, label="Object 2", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Quadrature (Q)")
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(6, 6))
    plt.plot(iq1.real, iq1.imag, label="Object 1", alpha=0.7)
    plt.plot(iq2.real, iq2.imag, label="Object 2", alpha=0.7)
    plt.plot(iq_sum.real, iq_sum.imag, label="Sum", alpha=0.7)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, iq1.real, label="I1", c='blue', alpha=0.7)
    plt.plot(times, iq1.imag, label="Q1", c='blue', linestyle='--', alpha=0.7)
    plt.plot(times, iq2.real, label="I2", c='red', alpha=0.7)
    plt.plot(times, iq2.imag, label="Q2", c='red', linestyle='--', alpha=0.7)
    plt.plot(times, iq_sum.real, label="I sum", c='green', alpha=0.7)
    plt.plot(times, iq_sum.imag, label="Q sum", c='green', linestyle='--', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    _, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    ax.plot(times, iq_sum.real, label="I sum", c='green', alpha=0.7)
    ax.plot(times, iq_sum.imag, label="Q sum", c='green', linestyle='--', alpha=0.7)
    ax2.plot(times, filtered_iq.real, label="I filtered", c='orange', alpha=0.7)
    ax2.plot(times, filtered_iq.imag, label="Q filtered", c='orange', linestyle='--', alpha=0.7)
    ax2.plot(times, np.abs(filtered_iq), label="Magnitude filtered", c='purple', alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax2.set_ylabel("Filtered Amplitude")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
