"""DTW-based interval estimation for slowly drifting sinusoidal signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class GeneratedSignal:
    """Container for generated signals."""

    time: np.ndarray
    values: np.ndarray
    instantaneous_freq: np.ndarray
    clean_values: np.ndarray | None = None


def generate_timebase(duration: float, sample_rate: float) -> np.ndarray:
    """Return an evenly spaced time vector."""

    step = 1.0 / sample_rate
    return np.arange(0.0, duration, step, dtype=float)


def generate_template(time: np.ndarray, base_freq: float) -> np.ndarray:
    """Create a constant-frequency sine template."""

    return np.sin(2.0 * np.pi * base_freq * time)


def generate_slowly_drifting_signal(
    time: np.ndarray,
    sample_rate: float,
    base_freq: float,
    freq_drift: float,
    noise_std: float = 0.0,
    pulse_count: int = 0,
    pulse_width: float = 0.01,
    pulse_amplitude: float = 0.8,
    pulse_times: np.ndarray | None = None,
    seed: int | None = 0,
) -> GeneratedSignal:
    """Generate a signal whose instantaneous frequency drifts linearly."""

    # Normalized time between 0 and 1 for the drift profile
    indices = np.arange(time.size, dtype=float)
    normalized = indices / indices[-1]
    instantaneous_freq = base_freq * (1.0 + freq_drift * normalized)

    # Integrate instantaneous frequency to obtain the phase
    phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    clean_values = np.sin(phase)

    rng = np.random.default_rng(seed)
    noisy_values = clean_values.copy()
    if noise_std > 0.0:
        noisy_values += rng.normal(scale=noise_std, size=noisy_values.shape)

    if (pulse_count > 0 or (pulse_times is not None and pulse_times.size > 0)) and pulse_width > 0.0 and pulse_amplitude != 0.0:
        pulses = _generate_sparse_pulses(
            num_samples=time.size,
            sample_rate=sample_rate,
            rng=rng,
            pulse_count=pulse_count,
            pulse_width=pulse_width,
            pulse_amplitude=pulse_amplitude,
            pulse_times=pulse_times,
        )
        noisy_values += pulses

    return GeneratedSignal(
        time=time,
        values=noisy_values,
        instantaneous_freq=instantaneous_freq,
        clean_values=clean_values,
    )


def _generate_sparse_pulses(
    num_samples: int,
    sample_rate: float,
    rng: np.random.Generator,
    pulse_count: int,
    pulse_width: float,
    pulse_amplitude: float,
    pulse_times: np.ndarray | None = None,
) -> np.ndarray:
    """Create sparse transient pulses to mimic measurement artifacts."""

    pulses = np.zeros(num_samples, dtype=float)
    width_samples = max(1, int(round(pulse_width * sample_rate)))
    window_length = max(2, 2 * width_samples)
    window = np.hanning(window_length)

    if num_samples <= window_length:
        pulses += pulse_amplitude * window[:num_samples]
        return pulses

    if pulse_times is not None and pulse_times.size > 0:
        centers = np.clip((pulse_times * sample_rate).astype(int), width_samples, num_samples - width_samples - 1)
    else:
        valid_indices = np.arange(width_samples, num_samples - width_samples)
        if valid_indices.size == 0 or pulse_count <= 0:
            return pulses

        replace = pulse_count > valid_indices.size
        centers = rng.choice(valid_indices, size=pulse_count, replace=replace)

    half_window = window_length // 2
    for center in centers:
        start = max(0, center - half_window)
        end = min(num_samples, center + half_window)
        window_start = half_window - (center - start)
        window_end = window_start + (end - start)
        pulses[start:end] += pulse_amplitude * window[window_start:window_end]

    return pulses


def _dtw_path(template: np.ndarray, signal: np.ndarray) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Compute a DTW warping path between template and signal."""

    distances = cdist(template[:, None], signal[:, None], metric="euclidean")
    n, m = distances.shape
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = distances[i - 1, j - 1] + best_prev

    i, j = n, m
    path_template: list[int] = []
    path_signal: list[int] = []
    while i > 0 and j > 0:
        path_template.append(i - 1)
        path_signal.append(j - 1)
        prev_steps = (
            cost[i - 1, j],
            cost[i, j - 1],
            cost[i - 1, j - 1],
        )
        direction = int(np.argmin(prev_steps))
        if direction == 0:
            i -= 1
        elif direction == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    return cost[n, m], (np.array(path_template[::-1]), np.array(path_signal[::-1]))


def _build_template_to_signal_map(path: Tuple[np.ndarray, np.ndarray], template_len: int) -> np.ndarray:
    """Convert a DTW path into a mapping from template indices to signal indices."""

    sums = np.zeros(template_len, dtype=float)
    counts = np.zeros(template_len, dtype=float)
    template_indices, signal_indices = path
    for t_idx, s_idx in zip(template_indices, signal_indices, strict=False):
        sums[t_idx] += s_idx
        counts[t_idx] += 1.0

    valid = counts > 0
    averaged = np.empty_like(sums)
    averaged[valid] = sums[valid] / counts[valid]
    if not np.all(valid):
        averaged[~valid] = np.interp(
            np.flatnonzero(~valid),
            np.flatnonzero(valid),
            averaged[valid],
        )
    return averaged


def estimate_cycle_intervals(
    template: np.ndarray,
    signal: np.ndarray,
    sample_rate: float,
    base_freq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate cycle start times and intervals using DTW alignment."""

    _, path = _dtw_path(template, signal)
    mapping = _build_template_to_signal_map(path, template_len=template.size)

    template_time = np.arange(template.size, dtype=float) / sample_rate
    max_time = template_time[-1]
    cycle_template_times = np.arange(0.0, max_time, 1.0 / base_freq)
    cycle_template_indices = cycle_template_times * sample_rate

    mapped_signal_indices = np.interp(
        cycle_template_indices,
        np.arange(template.size, dtype=float),
        mapping,
    )
    cycle_start_times = mapped_signal_indices / sample_rate
    cycle_intervals = np.diff(cycle_start_times)
    return cycle_start_times, cycle_intervals


def run_demo() -> None:
    """Small demo that prints DTW-based interval estimates."""

    sample_rate = 2000.0  # Hz
    duration = 3.0  # seconds
    base_freq = 5.0  # Hz
    drift = 0.35  # +35% frequency increase over the interval

    time = generate_timebase(duration, sample_rate)
    template = generate_template(time, base_freq)
    generated = generate_slowly_drifting_signal(
        time,
        sample_rate,
        base_freq,
        drift,
        noise_std=0.08,
        pulse_count=0,
        pulse_width=0.002,
        pulse_amplitude=0.5,
        pulse_times=np.array([0.45, 1.65]),
        seed=42,
    )

    cycle_start_times, intervals = estimate_cycle_intervals(
        template,
        generated.values,
        sample_rate,
        base_freq,
    )

    estimated_frequency = 1.0 / intervals
    interval_midpoints = cycle_start_times[:-1] + intervals / 2.0
    true_frequency = np.interp(interval_midpoints, generated.time, generated.instantaneous_freq)

    print("DTW interval estimation for a slowly drifting sinusoid")
    print("cycle | start [s] | interval [s] | est freq [Hz] | true freq [Hz]")
    for idx, start, interval, f_est, f_true in zip(
        range(intervals.size),
        cycle_start_times[:-1],
        intervals,
        estimated_frequency,
        true_frequency,
        strict=False,
    ):
        print(f"{idx:5d} | {start:9.4f} | {interval:12.6f} | {f_est:13.4f} | {f_true:13.4f}")

    plot_interval_estimates(
        generated.time,
        generated.values,
        cycle_start_times,
        intervals,
        interval_midpoints,
        true_frequency,
        estimated_frequency,
    )


def plot_interval_estimates(
    time: np.ndarray,
    signal: np.ndarray,
    cycle_start_times: np.ndarray,
    intervals: np.ndarray,
    interval_midpoints: np.ndarray,
    true_frequency: np.ndarray,
    estimated_frequency: np.ndarray,
) -> None:
    """Visualize the signal, detected intervals, and frequency estimates."""

    fig, (ax_signal, ax_freq) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax_signal.plot(time, signal, label="drifting signal", linewidth=1.2)
    for start in cycle_start_times[:-1]:
        ax_signal.axvline(start, color="tab:red", alpha=0.3, linestyle="--")
    ax_signal.set_title("DTW-aligned intervals on drifting sinusoid")
    ax_signal.set_xlabel("Time [s]")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.legend(loc="upper right")

    ax_freq.plot(interval_midpoints, true_frequency, label="true inst. freq", color="tab:green")
    ax_freq.plot(interval_midpoints, estimated_frequency, label="DTW estimate", color="tab:blue", marker="o")
    ax_freq.set_xlabel("Time [s]")
    ax_freq.set_ylabel("Frequency [Hz]")
    ax_freq.set_title("Instantaneous frequency vs DTW-based estimate")
    ax_freq.legend(loc="upper left")
    ax_freq.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
