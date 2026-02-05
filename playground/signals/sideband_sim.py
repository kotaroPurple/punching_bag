
"""Sideband Frequency Simulation."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


def make_sine_like_wave(
    freq: float,
    sample_rate: float,
    duration: float,
    first_half_amplitude: float,
    second_half_amplitude: float,
) -> NDArray[np.float64]:
    """Generate a sine-like wave with variable amplitude.

    Args:
        freq: Base frequency of the sine wave in Hz.
        sample_rate: Sampling rate in Hz.
        duration: Duration of the signal in seconds.
        first_half_amplitude: Amplitude of the first half of the signal.
        second_half_amplitude: Amplitude of the second half of the signal.

    Returns:
        A numpy array containing the sine wave with variable amplitude.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    phase = 2 * np.pi * freq * t
    # 0-2.pi: first_half_amplitude, 2.pi-4.pi: second_half_amplitude
    amplitude = np.where(np.fmod(phase, 4 * np.pi) < 2 * np.pi, first_half_amplitude, second_half_amplitude)
    wave = amplitude * np.sin(phase)
    return wave


def apply_hilbert_transform(signal_wave: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Apply Hilbert transform to the signal.

    Args:
        signal_wave: Input signal wave.

    Returns:
        The analytic signal obtained from the Hilbert transform.
    """
    analytic_signal = signal.hilbert(signal_wave)
    return analytic_signal


def main() -> None:
    """Main function to simulate and plot sideband frequencies."""
    sample_rate = 100  # Hz
    duration = 20.0  # seconds
    base_freq = 1.0  # Hz

    signal_wave = make_sine_like_wave(
        freq=base_freq,
        sample_rate=sample_rate,
        duration=duration,
        first_half_amplitude=1.0,
        second_half_amplitude=0.5,
    )

    t = np.arange(0, duration, 1 / sample_rate)

    analytic_signal = apply_hilbert_transform(signal_wave)

    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_wave, label="Original Signal", c='gray', alpha=0.5)
    plt.plot(t, np.real(analytic_signal), label="Hilbert Transform (Real Part)", alpha=0.5)
    plt.plot(t, np.imag(analytic_signal), label="Hilbert Transform (Imaginary Part)", alpha=0.5)
    plt.plot(t, np.abs(analytic_signal), label="Envelope", c='red', linewidth=2)
    plt.title("Sine-like Wave with Sideband Frequencies")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.xlim(0, duration)

    _range = int(1.8 / base_freq * sample_rate)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(analytic_signal[:_range]), np.imag(analytic_signal[:_range]), c=t[:_range], cmap='viridis')
    plt.title("Analytic Signal in Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.colorbar(label="Time (s)")
    plt.grid()

    plt.figure(figsize=(12, 6))
    # inst_freq = np.diff(np.unwrap(np.angle(analytic_signal))) * (sample_rate / (2.0 * np.pi))
    inst_phase = np.unwrap(np.angle(analytic_signal))
    # inst_freq = np.diff(inst_phase) * (sample_rate / (2.0 * np.pi))
    plt.plot(t, inst_phase / (2 * np.pi), label="Instantaneous Phase", c='blue')
    plt.title("Instantaneous Phase of Analytic Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
