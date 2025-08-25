"""Simple DMD example with visualization."""

import numpy as np
import matplotlib.pyplot as plt
from svd_online_dmd import SvdOnlineDmd


def simple_dmd_example():
    """Simple example showing DMD on a two-frequency signal."""
    # Create simple test signal: 1Hz + 3Hz
    dt = 0.02
    t = np.arange(0, 3, dt)
    signal = np.sin(2*np.pi*1*t) + 0.6*np.sin(2*np.pi*3*t) + 0.05*np.random.randn(len(t))
    number = len(signal)

    # Apply DMD
    window_size = 15
    init_size = 80
    dmd = SvdOnlineDmd(window_size=window_size, max_rank=6, forgetting_factor=0.99)
    dmd.initialize(signal[:init_size])

    # Update with more data
    for i in range(init_size, number):
        dmd.update(signal[i])

    # Get results
    frequencies = dmd.get_mode_frequencies(dt=dt)
    amplitudes = dmd.get_mode_amplitudes()
    x_init = signal[:window_size]
    total_recon, mode_signals = dmd.decompose_by_modes(steps=number, x_init=x_init)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Simple DMD Example: 1Hz + 3Hz Signal')

    # Original signal
    axes[0,0].plot(t, signal, 'b-', alpha=0.7, label='Original')
    axes[0,0].set_title('Original Signal')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].grid(True, alpha=0.3)

    # Reconstruction
    t_recon = np.arange(len(total_recon)) * dt
    axes[0,1].plot(t, signal, 'b-', alpha=0.3, label='Original')
    axes[0,1].plot(t_recon, total_recon, 'r-', linewidth=2, label='DMD Reconstruction')
    axes[0,1].set_title('DMD Reconstruction')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].grid(True, alpha=0.3)

    # Individual modes
    for i, mode in enumerate(mode_signals[:]):
        axes[1,0].plot(t_recon, np.real(mode), label=f'Mode {i+1}: {frequencies[i]:.1f}Hz')
    axes[1,0].set_title('Individual Modes')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Frequency spectrum
    axes[1,1].stem(frequencies, np.abs(amplitudes), basefmt=' ')
    axes[1,1].set_title('Frequency Spectrum')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('Amplitude')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(-5, 5)

    plt.tight_layout()
    plt.show()

    # Print results
    print("Detected frequencies and amplitudes:")
    for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        if abs(amp) > 0.1:  # Only show significant modes
            print(f"  Mode {i+1}: {freq:6.2f} Hz, Amplitude: {abs(amp):6.3f}")


if __name__ == "__main__":
    simple_dmd_example()
