import numpy as np
from scipy import special, signal
import matplotlib.pyplot as plt

def generate_perturbed_displacement(t, heart_rate_hz, d0, dp, n=4, phi=-np.pi/2):
    """
    Generate heartbeat displacement with perturbation

    Parameters:
    -----------
    t : array
        Time array (s)
    heart_rate_hz : float
        Heart rate (Hz)
    d0 : float
        Base amplitude (m)
    dp : float
        Perturbation amplitude (m)
    n : int
        Perturbation frequency multiplier
    phi : float
        Perturbation phase shift (rad)

    Returns:
    --------
    d : array
        Displacement
    """
    omega = 2 * np.pi * heart_rate_hz

    # Base displacement + perturbation
    d = d0 * np.sin(omega * t) + dp * np.sin(n * omega * t + phi)

    return d

def generate_iq_signal(k, d):
    """
    Generate IQ signal from displacement

    Parameters:
    -----------
    k : float
        Wave number (rad/m)
    d : array
        Displacement (m)

    Returns:
    --------
    s : array
        IQ signal
    """
    s = np.exp(2j * k * d)
    return s

def analyze_spectrum(signal, fs, max_freq=None):
    """
    Spectral analysis of signal

    Parameters:
    -----------
    signal : array
        Signal to analyze
    fs : float
        Sampling frequency (Hz)
    max_freq : float or None
        Maximum frequency to display (Hz)

    Returns:
    --------
    freqs : array
        Frequency array
    spectrum : array
        Spectrum
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)

    if max_freq is not None:
        idx = freqs <= max_freq
        freqs = freqs[idx]
        spectrum = spectrum[idx]

    return freqs, spectrum

def find_peaks(freqs, spectrum, min_height=None, min_distance=None):
    """
    Detect peaks in spectrum

    Parameters:
    -----------
    freqs : array
        Frequency array
    spectrum : array
        Spectrum
    min_height : float or None
        Minimum peak height
    min_distance : float or None
        Minimum distance between peaks (Hz)

    Returns:
    --------
    peak_freqs : array
        Peak frequencies
    peak_heights : array
        Peak heights
    """
    # Absolute value of spectrum
    abs_spectrum = np.abs(spectrum)

    # Set minimum height
    if min_height is None:
        min_height = 0.1 * np.max(abs_spectrum)

    # Set minimum distance (in index units)
    if min_distance is not None:
        min_distance_idx = int(min_distance / (freqs[1] - freqs[0]))
    else:
        min_distance_idx = 1

    # Detect peaks
    peak_indices = signal.find_peaks(abs_spectrum, height=min_height, distance=min_distance_idx)[0]
    peak_freqs = freqs[peak_indices]
    peak_heights = abs_spectrum[peak_indices]

    return peak_freqs, peak_heights

def compare_perturbation_effects(heart_rate_hz, d0, dp_ratios, k=500, duration=10, fs=100):
    """
    Compare effects of perturbation

    Parameters:
    -----------
    heart_rate_hz : float
        Heart rate (Hz)
    d0 : float
        Base amplitude (m)
    dp_ratios : list
        List of perturbation amplitude to base amplitude ratios
    k : float
        Wave number (rad/m)
    duration : float
        Signal duration (s)
    fs : float
        Sampling frequency (Hz)
    """
    t = np.arange(0, duration, 1/fs)
    omega = 2 * np.pi * heart_rate_hz

    # Without perturbation
    d_normal = d0 * np.sin(omega * t)
    s_normal = generate_iq_signal(k, d_normal)
    intensity_normal = np.abs(s_normal - np.mean(s_normal))**2

    # Lists to store results
    displacements = [d_normal]
    intensities = [intensity_normal]
    spectra = []
    peak_freqs_list = []

    # Spectral analysis without perturbation
    freqs_normal, spectrum_normal = analyze_spectrum(intensity_normal, fs, max_freq=10*heart_rate_hz)
    spectra.append((freqs_normal, np.abs(spectrum_normal)))
    peak_freqs, peak_heights = find_peaks(freqs_normal, spectrum_normal)
    peak_freqs_list.append(peak_freqs)

    # With perturbation
    for dp_ratio in dp_ratios:
        dp = d0 * dp_ratio
        d_perturbed = generate_perturbed_displacement(t, heart_rate_hz, d0, dp)
        s_perturbed = generate_iq_signal(k, d_perturbed)
        intensity_perturbed = np.abs(s_perturbed - np.mean(s_perturbed))**2

        displacements.append(d_perturbed)
        intensities.append(intensity_perturbed)

        freqs_perturbed, spectrum_perturbed = analyze_spectrum(intensity_perturbed, fs, max_freq=10*heart_rate_hz)
        spectra.append((freqs_perturbed, np.abs(spectrum_perturbed)))
        peak_freqs, peak_heights = find_peaks(freqs_perturbed, spectrum_perturbed)
        peak_freqs_list.append(peak_freqs)

    # Plot
    fig, axs = plt.subplots(3, len(dp_ratios) + 1, figsize=(15, 10))

    titles = ['No Perturbation'] + [f'Perturbation Ratio {ratio:.2f}' for ratio in dp_ratios]

    for i, title in enumerate(titles):
        # Displacement
        axs[0, i].plot(t[:int(2/heart_rate_hz*fs)], displacements[i][:int(2/heart_rate_hz*fs)])
        axs[0, i].set_title(f'{title} - Displacement')
        axs[0, i].set_xlabel('Time (s)')
        axs[0, i].set_ylabel('Displacement (m)')
        axs[0, i].grid(True)

        # Intensity
        axs[1, i].plot(t[:int(2/heart_rate_hz*fs)], intensities[i][:int(2/heart_rate_hz*fs)])
        axs[1, i].set_title(f'{title} - Intensity')
        axs[1, i].set_xlabel('Time (s)')
        axs[1, i].set_ylabel('Intensity')
        axs[1, i].grid(True)

        # Spectrum
        freqs, spectrum = spectra[i]
        axs[2, i].plot(freqs, spectrum)

        # Display fundamental frequency and harmonics
        for j in range(1, 11):
            axs[2, i].axvline(x=j*heart_rate_hz, color='r', linestyle='--', alpha=0.3)

        # Display peaks
        peak_freqs = peak_freqs_list[i]
        for freq in peak_freqs:
            axs[2, i].axvline(x=freq, color='g', linestyle='-', alpha=0.5)
            # Frequency label
            axs[2, i].text(freq, 0.8*np.max(spectrum), f'{freq:.2f}Hz',
                          rotation=90, verticalalignment='center')

        axs[2, i].set_title(f'{title} - Spectrum')
        axs[2, i].set_xlabel('Frequency (Hz)')
        axs[2, i].set_ylabel('Amplitude')
        axs[2, i].set_xlim(0, 10*heart_rate_hz)
        axs[2, i].grid(True)

    plt.tight_layout()
    plt.show()

    # Detailed analysis of perturbation effects
    print("Changes in frequency components due to perturbation:")
    print(f"Basic heart rate: {heart_rate_hz:.2f} Hz")
    print(f"Expected main frequency: {2*heart_rate_hz:.2f} Hz")

    for i, title in enumerate(titles):
        print(f"\n{title}:")
        peak_freqs = peak_freqs_list[i]
        for freq in peak_freqs:
            # Analyze relationship with fundamental frequency
            ratio = freq / heart_rate_hz
            ratio_rounded = round(ratio * 2) / 2  # Round to nearest 0.5
            print(f"  {freq:.2f} Hz (approx. {ratio:.2f}x heart rate, ≈{ratio_rounded}x)")

def analyze_heart_rate_estimation(heart_rate_hz, d0, dp_ratios, k=500, duration=30, fs=100, trials=10):
    """
    Analyze the effect of perturbation on heart rate estimation

    Parameters:
    -----------
    heart_rate_hz : float
        Heart rate (Hz)
    d0 : float
        Base amplitude (m)
    dp_ratios : list
        List of perturbation amplitude to base amplitude ratios
    k : float
        Wave number (rad/m)
    duration : float
        Signal duration (s)
    fs : float
        Sampling frequency (Hz)
    trials : int
        Number of trials for each condition
    """
    t = np.arange(0, duration, 1/fs)

    # Arrays to store results
    errors = np.zeros((len(dp_ratios) + 1, trials))
    estimated_rates = np.zeros((len(dp_ratios) + 1, trials))

    for trial in range(trials):
        # Without perturbation
        d_normal = d0 * np.sin(2 * np.pi * heart_rate_hz * t)
        s_normal = generate_iq_signal(k, d_normal)
        intensity_normal = np.abs(s_normal - np.mean(s_normal))**2

        # Add noise
        noise = np.random.normal(0, 0.01 * np.std(intensity_normal), len(intensity_normal))
        intensity_normal += noise

        # Heart rate estimation
        freqs, spectrum = analyze_spectrum(intensity_normal, fs)
        valid_idx = (freqs >= 0.5) & (freqs <= 4.0)
        max_idx = np.argmax(np.abs(spectrum[valid_idx]))
        est_freq = freqs[valid_idx][max_idx]
        est_hr = est_freq / 2  # Divide by 2 because the frequency appears at twice the heart rate

        errors[0, trial] = abs(est_hr - heart_rate_hz)
        estimated_rates[0, trial] = est_hr

        # With perturbation
        for i, dp_ratio in enumerate(dp_ratios):
            dp = d0 * dp_ratio
            d_perturbed = generate_perturbed_displacement(t, heart_rate_hz, d0, dp)
            s_perturbed = generate_iq_signal(k, d_perturbed)
            intensity_perturbed = np.abs(s_perturbed - np.mean(s_perturbed))**2

            # Add noise
            noise = np.random.normal(0, 0.01 * np.std(intensity_perturbed), len(intensity_perturbed))
            intensity_perturbed += noise

            # Heart rate estimation
            freqs, spectrum = analyze_spectrum(intensity_perturbed, fs)
            valid_idx = (freqs >= 0.5) & (freqs <= 4.0)
            max_idx = np.argmax(np.abs(spectrum[valid_idx]))
            est_freq = freqs[valid_idx][max_idx]
            est_hr = est_freq / 2  # Divide by 2 because the frequency appears at twice the heart rate

            errors[i+1, trial] = abs(est_hr - heart_rate_hz)
            estimated_rates[i+1, trial] = est_hr

    # Mean error and standard deviation
    mean_errors = np.mean(errors, axis=1)
    std_errors = np.std(errors, axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    labels = ['No Perturbation'] + [f'Perturbation Ratio {ratio:.2f}' for ratio in dp_ratios]
    plt.bar(np.arange(len(labels)), mean_errors, yerr=std_errors, capsize=5)
    plt.xlabel('Perturbation Condition')
    plt.ylabel('Heart Rate Estimation Error (Hz)')
    plt.title('Effect of Perturbation on Heart Rate Estimation')
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Detailed results
    print("\nHeart Rate Estimation Results:")
    print(f"True heart rate: {heart_rate_hz:.2f} Hz ({heart_rate_hz*60:.1f} bpm)")

    for i, label in enumerate(labels):
        mean_est = np.mean(estimated_rates[i])
        std_est = np.std(estimated_rates[i])
        print(f"\n{label}:")
        print(f"  Mean estimated heart rate: {mean_est:.2f} Hz ({mean_est*60:.1f} bpm)")
        print(f"  Standard deviation: {std_est:.4f} Hz")
        print(f"  Mean error: {mean_errors[i]:.4f} Hz ({mean_errors[i]*60:.1f} bpm)")

def main():
    # Parameter settings
    k = 500  # Wave number (24GHz)
    d0 = 0.0001  # Base amplitude (0.1mm)
    heart_rate_hz = 1.2  # Heart rate (Hz) = 72 bpm

    # Perturbation amplitude ratios
    dp_ratios = [0.1, 0.2, 0.3]

    # Compare effects of perturbation
    compare_perturbation_effects(heart_rate_hz, d0, dp_ratios)

    # Analyze effect on heart rate estimation
    analyze_heart_rate_estimation(heart_rate_hz, d0, dp_ratios)

if __name__ == "__main__":
    main()