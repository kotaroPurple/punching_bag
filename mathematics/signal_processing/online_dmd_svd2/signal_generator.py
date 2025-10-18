"""Signal generator for DMD viewer."""

import numpy as np
from typing import List


class SignalGenerator:
    """Generate various test signals for DMD analysis."""

    def __init__(self, sample_rate: float = 100.0):
        """Initialize signal generator.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.time = 0.0

    def reset_time(self):
        """Reset time counter."""
        self.time = 0.0

    def generate_sine_wave(self, frequency: float, amplitude: float, noise_level: float = 0.0) -> float:
        """Generate single sine wave sample.

        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        signal = amplitude * np.sin(2 * np.pi * frequency * self.time)
        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_multi_frequency(self, frequencies: List[float], amplitudes: List[float], noise_level: float = 0.0) -> float:
        """Generate multi-frequency signal sample.

        Args:
            frequencies: List of frequencies in Hz
            amplitudes: List of amplitudes
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        signal = 0.0
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * self.time)

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_time_varying(self, noise_level: float = 0.0) -> float:
        """Generate time-varying signal sample.

        Args:
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        t = self.time

        if t < 3:
            # First 3 seconds: 1Hz + 2Hz
            signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t)
        elif t < 6:
            # Next 3 seconds: 1Hz + 3Hz
            signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*3*t)
        else:
            # After 6 seconds: 1Hz + 2Hz + 4Hz
            signal = (np.sin(2*np.pi*1*t) +
                     0.5*np.sin(2*np.pi*2*t) +
                     0.3*np.sin(2*np.pi*4*t))

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def generate_chirp(self, f0: float = 1.0, f1: float = 2.5, a0: float = 1.0, a1: float = 1.0, duration: float = 10.0, noise_level: float = 0.0) -> float:
        """Generate chirp signal sample (linear frequency sweep).

        Args:
            f0: Starting frequency in Hz
            f1: Ending frequency in Hz
            duration: Total duration for sweep
            noise_level: Noise level (0-1)

        Returns:
            Signal sample
        """
        t = self.time

        # Linear frequency sweep
        amp = a0 + (a1 - a0) * (t / duration)
        if t <= duration:
            # Instantaneous frequency: f(t) = f0 + (f1-f0)*t/duration
            # Phase: φ(t) = 2π * [f0*t + (f1-f0)*t²/(2*duration)]
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
            signal = amp * np.sin(phase)
        else:
            # After duration, use final frequency
            signal = np.sin(2 * np.pi * f1 * t)

        if noise_level > 0:
            signal += noise_level * np.random.randn()

        self.time += self.dt
        return signal

    def get_current_time(self) -> float:
        """Get current time."""
        return self.time
