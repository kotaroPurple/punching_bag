"""Generate displacement with sine base and periodic Gaussian-like pulses."""

import matplotlib.pyplot as plt
import numpy as np
from enzanlab.signal.filters.sideband import SidebandFilter
from numpy.typing import NDArray
from scipy import signal

C = 299_792_458.0  # Speed of light [m/s]


def apply_bandpass_filter(
    data: NDArray[np.float64|np.complex128],
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    filter_order: int = 6,
) -> NDArray[np.float64]:
    """Apply a Butterworth band-pass filter to the input signal.

    Args:
        data: Input signal array.
        sample_rate: Sampling frequency in Hz.
        low_cutoff: Lower cutoff frequency in Hz.
        high_cutoff: Upper cutoff frequency in Hz.
        filter_order: Order of the Butterworth filter (default: 6).
    Returns:
        Filtered signal array.
    """
    nyquist = 0.5 * sample_rate
    if low_cutoff <= 0 or high_cutoff >= nyquist or low_cutoff >= high_cutoff:
        raise ValueError("Invalid cutoff frequencies. Must satisfy 0 < low_cutoff < high_cutoff < Nyquist.")
    sos = signal.butter(filter_order, (low_cutoff / nyquist, high_cutoff / nyquist), btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data)


def generate_sine_with_gaussian_pulse(
    freq: float,
    sample_rate: float,
    duration: float,
    sine_amplitude: float = 1.0,
    pulse_amplitude: float = 0.5,
    pulse_phase_center: float = 0.0,
    pulse_phase_sigma: float = 0.35,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate displacement made of a sine wave and same-period Gaussian-like pulses.

    Args:
        freq: Base frequency in Hz.
        sample_rate: Sampling rate in Hz.
        duration: Signal duration in seconds.
        sine_amplitude: Amplitude of the sine component.
        pulse_amplitude: Peak amplitude of the Gaussian pulse component.
        pulse_phase_center: Pulse center in phase [rad] within each period.
        pulse_phase_sigma: Pulse width in phase [rad].

    Returns:
        t: Time array.
        sine_component: Sine component.
        pulse_component: Periodic Gaussian-like pulse component.
        displacement: Sum of sine and pulse components.
    """
    t = np.arange(0.0, duration, 1.0 / sample_rate, dtype=np.float64)
    phase = 2.0 * np.pi * freq * t

    sine_component = sine_amplitude * np.sin(phase)

    # Wrap phase error to [-pi, pi) so one Gaussian pulse appears in each period.
    wrapped_phase = (phase - pulse_phase_center + np.pi) % (2.0 * np.pi) - np.pi
    pulse_component = pulse_amplitude * np.exp(-0.5 * (wrapped_phase / pulse_phase_sigma) ** 2)

    displacement = sine_component + pulse_component
    return t, sine_component, pulse_component, displacement


def generate_cw_doppler_iq(
    displacement: NDArray[np.float64],
    carrier_freq_hz: float = 24e9,
) -> NDArray[np.complex128]:
    """Generate CW Doppler IQ signal from displacement.

    IQ(t) = exp(2j * k * d(t)), where k = 2*pi*fc/c.

    Args:
        displacement: Displacement d(t) in meters.
        carrier_freq_hz: CW carrier frequency in Hz.

    Returns:
        Complex IQ signal.
    """
    k = 2.0 * np.pi * carrier_freq_hz / C
    return np.exp(2.0j * k * displacement)


def main() -> None:
    """Create and plot the displacement signal."""
    sample_rate = 500.0  # Sample rate in Hz
    t, sine_component, pulse_component, displacement = generate_sine_with_gaussian_pulse(
        freq=4.0,
        sample_rate=sample_rate,
        duration=5.0,
        sine_amplitude=2.0e-4,
        pulse_amplitude=0.5e-4,
        pulse_phase_center=np.pi / 4.0,
        pulse_phase_sigma=0.12,
    )
    iq_signal = generate_cw_doppler_iq(displacement=displacement, carrier_freq_hz=24e9)

    # sideband filter
    band_cutoffs = (20.0, 50.0)
    sideband_filter = SidebandFilter(sample_rate=sample_rate, band=band_cutoffs, zero_phase=True)
    sideband_negative_filter = SidebandFilter(sample_rate=sample_rate, band=(-band_cutoffs[1], -band_cutoffs[0]), zero_phase=True)
    iq_sideband_positive = sideband_filter.filter(iq_signal)
    iq_sideband_negative = sideband_negative_filter.filter(iq_signal) 

    # normal filter for comparison
    iq_normal_filtered = apply_bandpass_filter(
        data=iq_signal,
        sample_rate=sample_rate,
        low_cutoff=band_cutoffs[0],
        high_cutoff=band_cutoffs[1],
        filter_order=6,
    )

    # clear edges for better visualization
    edge_left = int(0.5 * sample_rate)  # 0.5 seconds worth of samples
    edge_right = int(0.5 * sample_rate)
    iq_sideband_positive[:edge_left] = 0.0
    iq_sideband_positive[-edge_right:] = 0.0
    iq_sideband_negative[:edge_left] = 0.0
    iq_sideband_negative[-edge_right:] = 0.0
    iq_normal_filtered[:edge_left] = 0.0
    iq_normal_filtered[-edge_right:] = 0.0

    # diff
    iq_diff_sideband = np.r_[iq_sideband_positive[1:] - iq_sideband_positive[:-1], 0.+0.j]
    iq_diff_normal = np.r_[iq_normal_filtered[1:] - iq_normal_filtered[:-1], 0.+0.j]

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, sine_component, label="Sine component", color="tab:blue", alpha=0.7)
    plt.plot(t, pulse_component, label="Gaussian pulse component", color="tab:orange", alpha=0.9)
    plt.plot(t, displacement, label="Displacement (sum)", color="tab:red", linewidth=2.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Displacement: Sine Base + Same-Period Gaussian Pulse")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12, 4))
    plt.plot(t, np.real(iq_signal), label="I (real)", color="tab:green", alpha=0.7)
    plt.plot(t, np.imag(iq_signal), label="Q (imag)", color="tab:purple", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("CW Doppler IQ (24 GHz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12, 4))
    plt.plot(t, np.real(iq_normal_filtered), label="Normal Filtered I (real)", color="tab:green", alpha=0.7)
    plt.plot(t, np.imag(iq_normal_filtered), label="Normal Filtered Q (imag)", color="tab:purple", alpha=0.7)
    plt.plot(t, np.real(iq_sideband_positive), label="Sideband Filtered I (real)", color="tab:blue", alpha=0.7)
    plt.plot(t, np.imag(iq_sideband_positive), label="Sideband Filtered Q (imag)", color="tab:orange", alpha=0.7)
    plt.plot(t, np.real(iq_sideband_negative), label="Negative Sideband Filtered I (real)", color="tab:cyan", alpha=0.7)
    plt.plot(t, np.imag(iq_sideband_negative), label="Negative Sideband Filtered Q (imag)", color="tab:brown", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Sideband Filtered CW Doppler IQ (24 GHz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # plt.figure(figsize=(12, 4))
    # plt.plot(t, np.real(iq_diff_normal), label="Normal Filtered IQ Diff (real)", color="tab:green", alpha=0.7)
    # plt.plot(t, np.imag(iq_diff_normal), label="Normal Filtered IQ Diff (imag)", color="tab:purple", alpha=0.7)
    # plt.plot(t, np.real(iq_diff_sideband), label="Sideband Filtered IQ Diff (real)", color="tab:blue", alpha=0.7)
    # plt.plot(t, np.imag(iq_diff_sideband), label="Sideband Filtered IQ Diff (imag)", color="tab:orange", alpha=0.7)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.title("Difference of Filtered IQ Signals")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
