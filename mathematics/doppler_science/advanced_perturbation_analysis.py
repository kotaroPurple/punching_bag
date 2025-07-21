import numpy as np
from scipy import special, signal
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd

def generate_perturbed_displacement(t, heart_rate_hz, d0, dp, n=4, phi=-np.pi/2):
    """
    Generate heartbeat displacement with perturbation
    """
    omega = 2 * np.pi * heart_rate_hz
    d = d0 * np.sin(omega * t) + dp * np.sin(n * omega * t + phi)
    return d

def generate_realistic_heartbeat(t, heart_rate_hz, d0, systolic_ratio=0.3):
    """
    Generate more realistic heartbeat waveform

    Parameters:
    -----------
    t : array
        Time array (s)
    heart_rate_hz : float
        Heart rate (Hz)
    d0 : float
        Amplitude (m)
    systolic_ratio : float
        Systolic phase ratio

    Returns:
    --------
    d : array
        Displacement
    """
    period = 1 / heart_rate_hz
    d = np.zeros_like(t)

    for i, time in enumerate(t):
        # Position within period (0-1)
        phase = (time % period) / period

        if phase < systolic_ratio:
            # Systolic phase: rapid rise and small bump
            normalized_phase = phase / systolic_ratio
            if normalized_phase < 0.5:
                # Rapid rise
                d[i] = d0 * np.sin(np.pi * normalized_phase)
            else:
                # Small bump
                d[i] = d0 * (0.8 + 0.2 * np.sin(np.pi * (normalized_phase - 0.5)))
        else:
            # Diastolic phase: gradual decline
            normalized_phase = (phase - systolic_ratio) / (1 - systolic_ratio)
            d[i] = d0 * (1 - normalized_phase)

    return d

def analyze_time_frequency(signal, fs, window_size=256, overlap=0.75):
    """
    Time-frequency analysis using Short-Time Fourier Transform

    Parameters:
    -----------
    signal : array
        Signal to analyze
    fs : float
        Sampling frequency (Hz)
    window_size : int
        Window size
    overlap : float
        Overlap ratio

    Returns:
    --------
    f : array
        Frequency array
    t : array
        Time array
    Sxx : array
        Spectrogram
    """
    f, t, Sxx = signal.spectrogram(
        signal,
        fs=fs,
        window=('tukey', 0.25),
        nperseg=window_size,
        noverlap=int(window_size * overlap),
        detrend='constant',
        scaling='spectrum'
    )
    return f, t, Sxx

def wavelet_analysis(signal, fs, scales=None):
    """
    Analysis using Continuous Wavelet Transform

    Parameters:
    -----------
    signal : array
        Signal to analyze
    fs : float
        Sampling frequency (Hz)
    scales : array or None
        Wavelet scales

    Returns:
    --------
    scales : array
        Scale array
    frequencies : array
        Corresponding frequencies
    cwt : array
        Wavelet coefficients
    """
    from scipy import signal as sg

    # Default scales
    if scales is None:
        scales = np.arange(1, 128)

    # Continuous wavelet transform
    cwt = sg.cwt(signal, sg.morlet2, scales)

    # Convert scales to frequencies
    frequencies = sg.scale2frequency(sg.morlet2, scales) * fs

    return scales, frequencies, cwt

def harmonic_analysis(freqs, spectrum, fundamental_freq, n_harmonics=5):
    """
    Harmonic analysis

    Parameters:
    -----------
    freqs : array
        Frequency array
    spectrum : array
        Spectrum
    fundamental_freq : float
        Fundamental frequency
    n_harmonics : int
        Number of harmonics to analyze

    Returns:
    --------
    harmonic_freqs : array
        Harmonic frequencies
    harmonic_powers : array
        Harmonic powers
    harmonic_ratios : array
        Ratios of harmonics to fundamental
    """
    harmonic_freqs = np.zeros(n_harmonics)
    harmonic_powers = np.zeros(n_harmonics)
    harmonic_ratios = np.zeros(n_harmonics)

    # Get power of fundamental frequency
    idx_fundamental = np.argmin(np.abs(freqs - fundamental_freq))
    fundamental_power = np.abs(spectrum[idx_fundamental])

    # Calculate power of each harmonic
    for i in range(n_harmonics):
        harmonic_freq = fundamental_freq * (i + 1)
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        harmonic_freqs[i] = freqs[idx]
        harmonic_powers[i] = np.abs(spectrum[idx])
        harmonic_ratios[i] = harmonic_powers[i] / fundamental_power

    return harmonic_freqs, harmonic_powers, harmonic_ratios

def compare_waveforms(heart_rate_hz, d0, dp_ratio=0.2, duration=5, fs=100):
    """
    Comparison of different heartbeat waveforms

    Parameters:
    -----------
    heart_rate_hz : float
        Heart rate (Hz)
    d0 : float
        Base amplitude (m)
    dp_ratio : float
        Perturbation amplitude ratio
    duration : float
        Signal duration (s)
    fs : float
        Sampling frequency (Hz)
    """
    t = np.arange(0, duration, 1/fs)
    k = 500  # Wave number

    # 1. Simple sine wave
    d_sine = d0 * np.sin(2 * np.pi * heart_rate_hz * t)

    # 2. Perturbed sine wave
    dp = d0 * dp_ratio
    d_perturbed = generate_perturbed_displacement(t, heart_rate_hz, d0, dp)

    # 3. Realistic heartbeat waveform
    d_realistic = generate_realistic_heartbeat(t, heart_rate_hz, d0)

    # Generate IQ signals
    s_sine = np.exp(2j * k * d_sine)
    s_perturbed = np.exp(2j * k * d_perturbed)
    s_realistic = np.exp(2j * k * d_realistic)

    # Calculate intensity
    intensity_sine = np.abs(s_sine - np.mean(s_sine))**2
    intensity_perturbed = np.abs(s_perturbed - np.mean(s_perturbed))**2
    intensity_realistic = np.abs(s_realistic - np.mean(s_realistic))**2

    # Spectral analysis
    freqs_sine, spectrum_sine = analyze_spectrum(intensity_sine, fs, max_freq=10*heart_rate_hz)
    freqs_perturbed, spectrum_perturbed = analyze_spectrum(intensity_perturbed, fs, max_freq=10*heart_rate_hz)
    freqs_realistic, spectrum_realistic = analyze_spectrum(intensity_realistic, fs, max_freq=10*heart_rate_hz)

    # Harmonic analysis
    fundamental_freq = 2 * heart_rate_hz  # Fundamental frequency of intensity is twice the heart rate
    _, _, harmonic_ratios_sine = harmonic_analysis(freqs_sine, spectrum_sine, fundamental_freq)
    _, _, harmonic_ratios_perturbed = harmonic_analysis(freqs_perturbed, spectrum_perturbed, fundamental_freq)
    _, _, harmonic_ratios_realistic = harmonic_analysis(freqs_realistic, spectrum_realistic, fundamental_freq)

    # Plot
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    # Waveforms
    axs[0, 0].plot(t[:int(2/heart_rate_hz*fs)], d_sine[:int(2/heart_rate_hz*fs)])
    axs[0, 0].set_title('Simple Sine Wave')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Displacement (m)')
    axs[0, 0].grid(True)

    axs[0, 1].plot(t[:int(2/heart_rate_hz*fs)], d_perturbed[:int(2/heart_rate_hz*fs)])
    axs[0, 1].set_title('Perturbed Sine Wave')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Displacement (m)')
    axs[0, 1].grid(True)

    axs[0, 2].plot(t[:int(2/heart_rate_hz*fs)], d_realistic[:int(2/heart_rate_hz*fs)])
    axs[0, 2].set_title('Realistic Heartbeat Waveform')
    axs[0, 2].set_xlabel('Time (s)')
    axs[0, 2].set_ylabel('Displacement (m)')
    axs[0, 2].grid(True)

    # Intensity
    axs[1, 0].plot(t[:int(2/heart_rate_hz*fs)], intensity_sine[:int(2/heart_rate_hz*fs)])
    axs[1, 0].set_title('Simple Sine Wave Intensity')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Intensity')
    axs[1, 0].grid(True)

    axs[1, 1].plot(t[:int(2/heart_rate_hz*fs)], intensity_perturbed[:int(2/heart_rate_hz*fs)])
    axs[1, 1].set_title('Perturbed Sine Wave Intensity')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].grid(True)

    axs[1, 2].plot(t[:int(2/heart_rate_hz*fs)], intensity_realistic[:int(2/heart_rate_hz*fs)])
    axs[1, 2].set_title('Realistic Heartbeat Intensity')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Intensity')
    axs[1, 2].grid(True)

    # Spectrum
    axs[2, 0].plot(freqs_sine, np.abs(spectrum_sine))
    axs[2, 0].set_title('Simple Sine Wave Spectrum')
    axs[2, 0].set_xlabel('Frequency (Hz)')
    axs[2, 0].set_ylabel('Amplitude')
    axs[2, 0].set_xlim(0, 10*heart_rate_hz)
    axs[2, 0].grid(True)

    axs[2, 1].plot(freqs_perturbed, np.abs(spectrum_perturbed))
    axs[2, 1].set_title('Perturbed Sine Wave Spectrum')
    axs[2, 1].set_xlabel('Frequency (Hz)')
    axs[2, 1].set_ylabel('Amplitude')
    axs[2, 1].set_xlim(0, 10*heart_rate_hz)
    axs[2, 1].grid(True)

    axs[2, 2].plot(freqs_realistic, np.abs(spectrum_realistic))
    axs[2, 2].set_title('Realistic Heartbeat Spectrum')
    axs[2, 2].set_xlabel('Frequency (Hz)')
    axs[2, 2].set_ylabel('Amplitude')
    axs[2, 2].set_xlim(0, 10*heart_rate_hz)
    axs[2, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Compare harmonic ratios
    plt.figure(figsize=(10, 6))
    x = np.arange(len(harmonic_ratios_sine))
    width = 0.25

    plt.bar(x - width, harmonic_ratios_sine, width, label='Simple Sine Wave')
    plt.bar(x, harmonic_ratios_perturbed, width, label='Perturbed Sine Wave')
    plt.bar(x + width, harmonic_ratios_realistic, width, label='Realistic Heartbeat')

    plt.xlabel('Harmonic Order')
    plt.ylabel('Ratio to Fundamental')
    plt.title('Comparison of Harmonic Structure')
    plt.xticks(x, [f'{i+1}th' for i in range(len(harmonic_ratios_sine))])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Time-frequency analysis
    plt.figure(figsize=(15, 10))

    # Simple sine wave
    plt.subplot(3, 1, 1)
    f, t_spec, Sxx = analyze_time_frequency(intensity_sine, fs)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Simple Sine Wave Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 10*heart_rate_hz)
    plt.colorbar(label='Power (dB)')

    # Perturbed sine wave
    plt.subplot(3, 1, 2)
    f, t_spec, Sxx = analyze_time_frequency(intensity_perturbed, fs)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Perturbed Sine Wave Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 10*heart_rate_hz)
    plt.colorbar(label='Power (dB)')

    # Realistic heartbeat waveform
    plt.subplot(3, 1, 3)
    f, t_spec, Sxx = analyze_time_frequency(intensity_realistic, fs)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Realistic Heartbeat Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 10*heart_rate_hz)
    plt.colorbar(label='Power (dB)')

    plt.tight_layout()
    plt.show()

def analyze_spectrum(signal, fs, max_freq=None):
    """
    Spectral analysis of signal
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)

    if max_freq is not None:
        idx = freqs <= max_freq
        freqs = freqs[idx]
        spectrum = spectrum[idx]

    return freqs, spectrum

def main():
    # Parameter settings
    k = 500  # Wave number (24GHz)
    d0 = 0.0001  # Base amplitude (0.1mm)
    heart_rate_hz = 1.2  # Heart rate (Hz) = 72 bpm

    # Compare different heartbeat waveforms
    compare_waveforms(heart_rate_hz, d0)

if __name__ == "__main__":
    main()