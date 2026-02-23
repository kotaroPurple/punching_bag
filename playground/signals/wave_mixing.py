"""Beat (wave mixing) simulation with SciPy filtering."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


def generate_beat_signal(
    sample_rate: float,
    duration: float,
    freq1: float,
    freq2: float,
    amp1: float = 1.0,
    amp2: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a beat signal from sin + sin."""
    t = np.arange(0, duration, 1 / sample_rate)
    wave1 = amp1 * np.sin(2 * np.pi * freq1 * t)
    wave2 = amp2 * np.sin(2 * np.pi * freq2 * t)
    beat = wave1 + wave2
    return t, beat, wave1, wave2


def lowpass_filter(
    x: NDArray[np.float64],
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.float64]:
    """Apply zero-phase low-pass filter to a signal."""
    sos = signal.butter(order, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


def lowpass_filter_with_fft(
    x: NDArray[np.float64],
    sample_rate: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Apply low-pass filter using FFT."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    fft_coeffs = np.fft.rfft(x)
    fft_coeffs[freqs > cutoff_hz] = 0
    return np.fft.irfft(fft_coeffs, n)


def highpass_filter(
    x: NDArray[np.float64],
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.float64]:
    """Apply zero-phase high-pass filter to a signal."""
    sos = signal.butter(order, cutoff_hz, btype="high", fs=sample_rate, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


def highpass_filter_with_fft(
    x: NDArray[np.float64],
    sample_rate: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Apply high-pass filter using FFT."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    fft_coeffs = np.fft.rfft(x)
    fft_coeffs[freqs < cutoff_hz] = 0
    return np.fft.irfft(fft_coeffs, n)


def hilbert_envelope(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute amplitude envelope from the analytic signal."""
    analytic = signal.hilbert(x)
    return np.abs(analytic)


def synchronous_demodulate(
    x: NDArray[np.float64],
    t: NDArray[np.float64],
    sample_rate: float,
    target_freq: float,
    baseband_cutoff_hz: float,
    order: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract one tone by synchronous (I/Q) detection."""
    lo_cos = np.cos(2 * np.pi * target_freq * t)
    lo_sin = np.sin(2 * np.pi * target_freq * t)

    # 2x scaling keeps amplitude on I/Q channels easy to interpret.
    i_mixed = 2.0 * x * lo_cos
    q_mixed = -2.0 * x * lo_sin

    i_baseband = lowpass_filter(
        i_mixed,
        sample_rate=sample_rate,
        cutoff_hz=baseband_cutoff_hz,
        order=order,
        zero_phase=True,
    )
    q_baseband = lowpass_filter(
        q_mixed,
        sample_rate=sample_rate,
        cutoff_hz=baseband_cutoff_hz,
        order=order,
        zero_phase=True,
    )

    amplitude = np.sqrt(i_baseband ** 2 + q_baseband ** 2)
    reconstructed = i_baseband * lo_cos - q_baseband * lo_sin
    return i_baseband, q_baseband, amplitude, reconstructed


def main() -> None:
    """Create beat signal, filter it, and visualize."""
    sample_rate = 200.0
    duration = 5.0
    freq1 = 6.5
    freq2 = 9.0
    amp1 = 2.0
    amp2 = 1.0
    low_cutoff_hz = 7.5
    high_cutoff_hz = 8.0
    sync_target_hz = freq1
    sync_baseband_cutoff_hz = 0.8

    t, beat, wave1, wave2 = generate_beat_signal(
        sample_rate=sample_rate,
        duration=duration,
        freq1=freq1,
        freq2=freq2,
        amp1=amp1,
        amp2=amp2,
    )

    low_filtered = lowpass_filter(
        beat, sample_rate=sample_rate, cutoff_hz=low_cutoff_hz, order=4, zero_phase=True)
    low_filtered_fft = lowpass_filter_with_fft(
        beat, sample_rate=sample_rate, cutoff_hz=low_cutoff_hz)
    high_filtered = highpass_filter(
        beat, sample_rate=sample_rate, cutoff_hz=high_cutoff_hz, order=4, zero_phase=True)
    high_filtered_fft = highpass_filter_with_fft(
        beat, sample_rate=sample_rate, cutoff_hz=high_cutoff_hz)
    envelope = np.abs(2.0 * np.cos(np.pi * (freq2 - freq1) * t))
    # envelope = np.abs(beat)

    # low_filtered = low_filtered_fft
    # high_filtered = high_filtered_fft

    low_filtered_envelope = np.abs(low_filtered)
    high_filtered_envelope = np.abs(high_filtered)
    beat_hilbert_envelope = hilbert_envelope(high_filtered)
    i_baseband, q_baseband, sync_amplitude, sync_reconstructed = synchronous_demodulate(
        beat,
        t,
        sample_rate=sample_rate,
        target_freq=sync_target_hz,
        baseband_cutoff_hz=sync_baseband_cutoff_hz,
        order=4,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, beat, c="gray", linewidth=1., label="Beat signal: sin(2πf1t) + sin(2πf2t)", alpha=0.7)
    # axes[0].plot(t, wave1, c="tab:orange", linewidth=1.0, label=f"sin(2πf1t) ({freq1:.1f} Hz)", alpha=0.7)
    # axes[0].plot(t, wave2, c="tab:green", linewidth=1.0, label=f"sin(2πf2t) ({freq2:.1f} Hz)", alpha=0.7)
    axes[0].set_title(f"Beat signal (f1={freq1:.1f} Hz, f2={freq2:.1f} Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(False)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, low_filtered, c="tab:blue", linewidth=1.0, label=f"Low-pass filtered ({low_cutoff_hz:.1f} Hz)")
    axes[1].plot(t, high_filtered, c="tab:purple", linewidth=1.0, label=f"High-pass filtered ({high_cutoff_hz:.1f} Hz)")
    axes[1].plot(t, low_filtered_fft, c="tab:blue", linewidth=1.0, label=f"Low-pass filtered (FFT) ({low_cutoff_hz:.1f} Hz)", alpha=0.5)
    axes[1].plot(t, high_filtered_fft, c="tab:purple", linewidth=1.0, label=f"High-pass filtered (FFT) ({high_cutoff_hz:.1f} Hz)", alpha=0.5)
    # axes[1].plot(t, wave1, c="tab:orange", linewidth=1.0, label=f"sin(2πf1t) ({freq1:.1f} Hz)", alpha=0.7)
    # axes[1].plot(t, wave2, c="tab:green", linewidth=1.0, label=f"sin(2πf2t) ({freq2:.1f} Hz)", alpha=0.7)
    axes[1].set_title("Filtered result (SciPy Butterworth + filtfilt)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(False)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, beat, c="lightgray", linewidth=1.0, label="Beat signal")
    axes[2].plot(t, envelope, c="tab:red", linewidth=1.5, label="Theoretical envelope: 2cos(πΔft)", alpha=0.7)
    axes[2].plot(t, low_filtered_envelope, c="tab:blue", linewidth=1.5, label=f"Low-pass filtered envelope ({low_cutoff_hz:.1f} Hz)", alpha=0.7)
    axes[2].plot(t, high_filtered_envelope, c="tab:purple", linewidth=1.5, label=f"High-pass filtered envelope ({high_cutoff_hz:.1f} Hz)", alpha=0.7)
    axes[2].plot(t, beat_hilbert_envelope, c="tab:green", linewidth=2.0, label="Hilbert envelope", alpha=0.9)
    axes[2].set_title("Beat signal and envelope")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(False)
    axes[2].legend(loc="upper right")

    axes[3].plot(t, beat, c="lightgray", linewidth=0.8, label="Beat signal", alpha=0.8)
    axes[3].plot(t, sync_reconstructed, c="tab:orange", linewidth=1.6, label=f"Synchronous detected tone ({sync_target_hz:.1f} Hz)")
    axes[3].plot(t, sync_amplitude, c="tab:brown", linewidth=1.8, label="Synchronous detected amplitude")
    axes[3].plot(t, i_baseband, c="tab:blue", linewidth=1.0, label="I baseband", alpha=0.7)
    axes[3].plot(t, q_baseband, c="tab:green", linewidth=1.0, label="Q baseband", alpha=0.7)
    axes[3].set_title(f"Synchronous Detection (baseband LPF: {sync_baseband_cutoff_hz:.1f} Hz)")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(False)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
