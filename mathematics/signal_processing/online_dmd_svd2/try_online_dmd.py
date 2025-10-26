"""Test Online DMD stability and frequency tracking."""

import numpy as np
import matplotlib.pyplot as plt
from online_dmd import OnlineDMD
from signal_generator import SignalGenerator
from hankel import HankelSignal, array_to_hankel_matrix


def test_chirp_tracking():
    """Test DMD tracking of chirp signal."""
    print("Testing Online DMD with Chirp Signal...")

    # Parameters
    sample_rate = 100
    dt = 1.0 / sample_rate
    duration = 10.0
    samples = int(duration * sample_rate)

    # Generate chirp signal (1Hz to 5Hz over 8 seconds)
    generator = SignalGenerator(sample_rate)
    _signal_data = []
    _time_data = []

    initial_size_rate = 1.5

    f0 = 1.0
    f1 = 1.2
    a0 = 1.0
    a1 = 1.3
    for i in range(samples):
        sample = generator.generate_chirp(f0=f0, f1=f1, a0=a0, a1=a1, duration=duration, noise_level=0.05)
        _signal_data.append(sample)
        _time_data.append(i * dt)

    signal_data = np.array(_signal_data)
    time_data = np.array(_time_data)

    # Test different DMD configurations
    configs = [
        {"window_size": 100, "max_rank": 4, "forgetting_factor": 1.0, "tau": 0.01, "name": "Standard DMD"},
        {"window_size": 100, "max_rank": 4, "forgetting_factor": 0.99, "tau": 0.01, "name": "Forgetting λ=0.99"},
        {"window_size": 100, "max_rank": 4, "forgetting_factor": 0.97, "tau": 0.01, "name": "Forgetting λ=0.97"},
    ]

    results = {}

    for config in configs:
        print(f"Testing {config['name']}...")

        # Initialize DMD
        window_size = config['window_size']
        dmd = OnlineDMD(
            n_dim=window_size, r_max=config['max_rank'], lambda_=config['forgetting_factor'], tau_add=config['tau']
        )

        # Initialize with first portion
        # init_length = config['window_size'] + 10
        init_length = int(initial_size_rate * sample_rate)
        init_data = array_to_hankel_matrix(signal_data[:init_length], window_size)
        dmd.initialize(init_data)

        hankel = HankelSignal(window_size)
        hankel.initialize(init_data[:, -1])

        # Track frequency evolution
        freq_evolution = []
        time_points = []
        growth_rates = []
        amps = []

        # Process signal and record dominant frequency
        short_samples = 10
        for i in range(init_length, samples, short_samples):  # Every 10 samples (0.1s)
            # new sample and make x vector
            for value in signal_data[i:i + short_samples]:
                new_vector = hankel.update(value)
                dmd.update(new_vector)

            # Get current analysis
            try:
                frequencies = dmd.get_mode_frequencies(dt=dt)
                amplitudes = dmd.get_mode_amplitudes()
                growth = dmd.get_mode_growth_rates(dt=dt)

                if len(frequencies) > 0 and len(amplitudes) > 0:
                    # Find dominant frequency
                    amp_magnitudes = np.abs(amplitudes)
                    dominant_idx = np.argmax(amp_magnitudes)

                    freq_evolution.append(abs(frequencies[dominant_idx]))
                    time_points.append(i * dt)
                    growth_rates.append(growth[dominant_idx])
                    amps.append(amp_magnitudes[dominant_idx])

            except Exception as e:
                print(f"Analysis failed at t={i*dt:.1f}s: {e}")

        results[config['name']] = {
            'time': time_points,
            'frequency': freq_evolution,
            'growth_rates': growth_rates,
            'amplitudes': amps,
            'config': config
        }

    # Theoretical chirp frequency
    theoretical_time = np.linspace(0, duration, 100)
    theoretical_freq = []
    for t in theoretical_time:
        if t <= duration:
            f_inst = f0 + (f1 - f0) * t / duration  # Linear sweep
        else:
            f_inst = 5.0
        theoretical_freq.append(f_inst)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Online DMD Frequency Tracking Performance', fontsize=16)

    # Original chirp signal
    axes[0, 0].plot(time_data, signal_data)
    axes[0, 0].set_title('Chirp Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Frequency tracking comparison
    axes[0, 1].plot(theoretical_time, theoretical_freq, 'k--', linewidth=1, label='True Frequency')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['frequency']:
            axes[0, 1].plot(
                result['time'], result['frequency'],
                color=colors[i % len(colors)], marker='o', markersize=3, label=name, alpha=0.5)

    axes[0, 1].set_title('Frequency Tracking Comparison')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Detected Frequency (Hz)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(f0 * 0.8, f1 * 1.2)

    # Amplitudes
    axes[1, 0].set_title('Mode Amplitudes')
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['amplitudes']:
            axes[1, 0].plot(
                result['time'], result['amplitudes'],
                color=colors[i % len(colors)], marker='o', markersize=3,
                label=name, alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Tracking error analysis
    axes[1, 1].set_title('Tracking Error Analysis')
    for i, (name, result) in enumerate(results.items()):
        if result['time'] and result['frequency']:
            # Interpolate theoretical frequency at measurement times
            theoretical_interp = np.interp(result['time'], theoretical_time, theoretical_freq)
            tracking_error = np.abs(np.array(result['frequency']) - theoretical_interp)

            axes[1, 1].plot(
                result['time'], tracking_error,
                color=colors[i % len(colors)], marker='o', markersize=3,
                label=f"{name} (avg: {np.mean(tracking_error):.2f}Hz)", alpha=0.8)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Tracking Error (Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print analysis summary
    print("\n=== Online DMD Stability Analysis ===")
    for name, result in results.items():
        if result['time'] and result['frequency']:
            theoretical_interp = np.interp(result['time'], theoretical_time, theoretical_freq)
            tracking_error = np.abs(np.array(result['frequency']) - theoretical_interp)
            avg_growth = np.mean(np.abs(result['growth_rates'])) if result['growth_rates'] else 0

            print(f"\n{name}:")
            print(f"  Average tracking error: {np.mean(tracking_error):.3f} Hz")
            print(f"  Max tracking error: {np.max(tracking_error):.3f} Hz")
            print(f"  Average |growth rate|: {avg_growth:.3f}")
            print(f"  Stability: {'Good' if avg_growth < 1.0 else 'Poor'}")

    return results


if __name__ == "__main__":
    results = test_chirp_tracking()
