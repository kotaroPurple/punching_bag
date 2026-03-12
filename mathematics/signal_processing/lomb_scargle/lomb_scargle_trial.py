"""Lomb-Scargle with missing segments: NumPy, SciPy, and Astropy examples.

Run:
    uv run mathematics/signal_processing/lomb_scargle/lomb_scargle_trial.py
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:  # pragma: no cover - optional dependency
    scipy_signal = None

try:
    from astropy.timeseries import LombScargle as AstropyLombScargle
except ImportError:  # pragma: no cover - optional dependency
    AstropyLombScargle = None


@dataclass
class SpectrumResult:
    method: str
    freqs_hz: np.ndarray
    power_raw: np.ndarray
    power_norm: np.ndarray
    peak_freq_hz: float
    peak_power_raw: float


@dataclass
class SignalData:
    t_full: np.ndarray
    y_full: np.ndarray
    t_obs: np.ndarray
    y_obs: np.ndarray
    gap_ranges_s: list[tuple[float, float]]
    truth: dict[str, float]


def generate_signal_with_gaps(
    duration_s: float = 8.0,
    fs_nominal_hz: float = 80.0,
    gap_ranges_s: list[tuple[float, float]] | None = None,
    noise_std: float = 0.35,
    seed: int = 7,
) -> SignalData:
    """Create unevenly sampled observations by removing two (or more) time gaps."""
    rng = np.random.default_rng(seed)
    t_full = np.arange(0.0, duration_s, 1.0 / fs_nominal_hz)
    if gap_ranges_s is None:
        gap_ranges_s = [(1.8, 2.7), (4.8, 5.9)]

    f1_hz = 1.8
    f2_hz = 4.7
    phase2 = 0.35 * math.pi
    y_full = (
        1.0 * np.sin(2.0 * np.pi * f1_hz * t_full)
        + 0.65 * np.sin(2.0 * np.pi * f2_hz * t_full + phase2)
        + noise_std * rng.standard_normal(t_full.size)
    )

    keep = np.ones_like(t_full, dtype=bool)
    for gap_start_s, gap_end_s in gap_ranges_s:
        keep &= (t_full < gap_start_s) | (t_full > gap_end_s)

    t_obs = t_full[keep]
    y_obs = y_full[keep]
    y_obs = y_obs - np.mean(y_obs)

    truth = {"f1_hz": f1_hz, "f2_hz": f2_hz}
    return SignalData(
        t_full=t_full,
        y_full=y_full,
        t_obs=t_obs,
        y_obs=y_obs,
        gap_ranges_s=gap_ranges_s,
        truth=truth,
    )


def lomb_scargle_numpy(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """Classic Lomb-Scargle implementation with NumPy only (Press-style form)."""
    y = y - np.mean(y)
    var = np.var(y)
    if var <= 0.0:
        raise ValueError("Signal variance is zero; Lomb-Scargle is undefined.")

    omega = 2.0 * np.pi * freqs_hz
    power = np.empty_like(freqs_hz, dtype=float)

    for i, w in enumerate(omega):
        wt = w * t
        sin2wt = np.sin(2.0 * wt)
        cos2wt = np.cos(2.0 * wt)
        tau = 0.5 * math.atan2(np.sum(sin2wt), np.sum(cos2wt)) / w

        arg = w * (t - tau)
        c = np.cos(arg)
        s = np.sin(arg)
        c2 = np.sum(c * c)
        s2 = np.sum(s * s)

        term_c = (np.sum(y * c) ** 2) / c2 if c2 > 0.0 else 0.0
        term_s = (np.sum(y * s) ** 2) / s2 if s2 > 0.0 else 0.0
        power[i] = 0.5 * (term_c + term_s) / var

    return power


def normalize_power_unit_peak(power: np.ndarray) -> np.ndarray:
    max_power = float(np.max(power))
    if max_power <= 0.0:
        return np.zeros_like(power)
    return power / max_power


def analyze_with_numpy(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray) -> SpectrumResult:
    power_raw = lomb_scargle_numpy(t, y, freqs_hz)
    power_norm = normalize_power_unit_peak(power_raw)
    peak_idx = int(np.argmax(power_raw))
    peak = float(freqs_hz[peak_idx])
    return SpectrumResult("NumPy (custom)", freqs_hz, power_raw, power_norm, peak, float(power_raw[peak_idx]))


def analyze_with_scipy(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray) -> SpectrumResult | None:
    if scipy_signal is None:
        return None
    omega = 2.0 * np.pi * freqs_hz
    power_raw = scipy_signal.lombscargle(t, y - np.mean(y), omega, normalize=True, precenter=False)
    power_norm = normalize_power_unit_peak(power_raw)
    peak_idx = int(np.argmax(power_raw))
    peak = float(freqs_hz[peak_idx])
    return SpectrumResult("SciPy", freqs_hz, power_raw, power_norm, peak, float(power_raw[peak_idx]))


def analyze_with_astropy(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray) -> SpectrumResult | None:
    if AstropyLombScargle is None:
        return None
    # Match the custom/SciPy treatment: remove mean beforehand and avoid extra mean fitting.
    ls = AstropyLombScargle(t, y - np.mean(y), fit_mean=False, center_data=False)
    power_raw = ls.power(freqs_hz, normalization="standard")
    power_norm = normalize_power_unit_peak(power_raw)
    peak_idx = int(np.argmax(power_raw))
    peak = float(freqs_hz[peak_idx])
    return SpectrumResult("Astropy", freqs_hz, power_raw, power_norm, peak, float(power_raw[peak_idx]))


def print_summary(results: list[SpectrumResult], truth: dict[str, float]) -> None:
    print("=== Ground truth frequencies ===")
    print(f"f1 = {truth['f1_hz']:.3f} Hz, f2 = {truth['f2_hz']:.3f} Hz")
    print()
    print("=== Estimated dominant frequency (largest peak) ===")
    for r in results:
        print(f"{r.method:16s}: {r.peak_freq_hz:.4f} Hz")
    print()
    print("=== Raw peak power (method-dependent normalization) ===")
    for r in results:
        print(f"{r.method:16s}: {r.peak_power_raw:.6f}")
    print("\nPlot uses common normalization: power / max(power) for each method.")


def maybe_plot(
    signal_data: SignalData,
    results: list[SpectrumResult],
    output_png: str = "mathematics/signal_processing/lomb_scargle/lomb_scargle_trial.png",
    show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skip plotting.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].plot(signal_data.t_full, signal_data.y_full, lw=1.2, color="#4c78a8", label="full signal")
    for i, (gap_start_s, gap_end_s) in enumerate(signal_data.gap_ranges_s):
        axes[0].axvspan(
            gap_start_s,
            gap_end_s,
            color="#e45756",
            alpha=0.2,
            label="missing segments" if i == 0 else None,
        )
    axes[0].set_title("Original Signal and Missing Segments (2 gaps)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].scatter(signal_data.t_obs, signal_data.y_obs, s=9, alpha=0.85, label="observed samples")
    axes[1].set_title("Observed Samples (Unevenly Observed)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for r in results:
        axes[2].plot(r.freqs_hz, r.power_norm, label=f"{r.method} (peak {r.peak_freq_hz:.3f} Hz)")

    axes[2].axvline(signal_data.truth["f1_hz"], color="k", ls="--", lw=1.0, alpha=0.8, label="true freqs")
    axes[2].axvline(signal_data.truth["f2_hz"], color="k", ls="--", lw=1.0, alpha=0.8)
    axes[2].set_title("Lomb-Scargle Periodogram (Unit-Peak Normalized)")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Normalized Power")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.savefig(output_png, dpi=150)
    print(f"\nSaved plot: {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def estimate_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate y ~ x^p from log-log linear fit."""
    mask = (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    p, _b = np.polyfit(lx, ly, 1)
    return float(p)


def benchmark_runtime(
    n_values: np.ndarray,
    repeats: int,
    f_min_hz: float,
    f_max_hz: float,
    n_freqs: int,
) -> dict[str, np.ndarray]:
    timings: dict[str, list[float]] = {"numpy_custom": []}
    if scipy_signal is not None:
        timings["scipy"] = []
    if AstropyLombScargle is not None:
        timings["astropy_auto"] = []
        timings["astropy_fast"] = []
        timings["astropy_scipy"] = []

    for n_target in n_values:
        duration_s = float(n_target) / 80.0
        signal_data = generate_signal_with_gaps(duration_s=duration_s)
        t = signal_data.t_obs
        y = signal_data.y_obs
        freqs_hz = np.linspace(f_min_hz, f_max_hz, n_freqs)

        # Warm-up
        _ = lomb_scargle_numpy(t, y, freqs_hz)
        if scipy_signal is not None:
            _ = analyze_with_scipy(t, y, freqs_hz)
        if AstropyLombScargle is not None:
            _ = analyze_with_astropy(t, y, freqs_hz)

        tmp: dict[str, list[float]] = {k: [] for k in timings}
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = lomb_scargle_numpy(t, y, freqs_hz)
            tmp["numpy_custom"].append(time.perf_counter() - t0)

            if scipy_signal is not None:
                t0 = time.perf_counter()
                _ = analyze_with_scipy(t, y, freqs_hz)
                tmp["scipy"].append(time.perf_counter() - t0)

            if AstropyLombScargle is not None:
                ls = AstropyLombScargle(t, y - np.mean(y), fit_mean=False, center_data=False)

                t0 = time.perf_counter()
                _ = ls.power(freqs_hz, normalization="standard", method="auto")
                tmp["astropy_auto"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                _ = ls.power(freqs_hz, normalization="standard", method="fast")
                tmp["astropy_fast"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                _ = ls.power(freqs_hz, normalization="standard", method="scipy")
                tmp["astropy_scipy"].append(time.perf_counter() - t0)

        for k in timings:
            timings[k].append(float(np.median(tmp[k])))

    return {k: np.array(v) for k, v in timings.items()}


def benchmark_astropy_fast_scaled_m(
    n_values: np.ndarray,
    repeats: int,
    f_min_hz: float,
    f_max_hz: float,
    min_freqs: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if AstropyLombScargle is None:
        return np.array([]), np.array([])

    times: list[float] = []
    m_values: list[int] = []
    for n_target in n_values:
        duration_s = float(n_target) / 80.0
        signal_data = generate_signal_with_gaps(duration_s=duration_s)
        t = signal_data.t_obs
        y = signal_data.y_obs

        nfreqs = max(min_freqs, int(n_target))
        m_values.append(nfreqs)
        freqs_hz = np.linspace(f_min_hz, f_max_hz, nfreqs)
        ls = AstropyLombScargle(t, y - np.mean(y), fit_mean=False, center_data=False)

        _ = ls.power(freqs_hz, normalization="standard", method="fast")
        run_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = ls.power(freqs_hz, normalization="standard", method="fast")
            run_times.append(time.perf_counter() - t0)
        times.append(float(np.median(run_times)))

    return np.array(m_values, dtype=float), np.array(times, dtype=float)


def maybe_plot_benchmark(
    n_values: np.ndarray,
    timings: dict[str, np.ndarray],
    output_png: str,
    show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skip benchmark plotting.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    for name, arr in timings.items():
        axes[0].plot(n_values, arr, marker="o", lw=1.3, label=name)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Runtime vs Data Length N (log-log)")
    axes[0].set_xlabel("N (target total samples before removing gaps)")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    nlogn = n_values * np.log2(n_values)
    for name, arr in timings.items():
        axes[1].plot(n_values, arr / nlogn, marker="o", lw=1.3, label=f"{name} / (N log2 N)")
    axes[1].set_xscale("log")
    axes[1].set_title("N log N Normalized Runtime")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Runtime / (N log2 N)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.savefig(output_png, dpi=150)
    print(f"\nSaved benchmark plot: {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> None:
    n_values = np.array([400, 800, 1600, 3200, 6400, 12800], dtype=int)
    timings = benchmark_runtime(
        n_values=n_values,
        repeats=args.benchmark_repeats,
        f_min_hz=args.fmin,
        f_max_hz=args.fmax,
        n_freqs=args.nfreqs,
    )

    print("=== Runtime Benchmark (median seconds) ===")
    print("N values:", n_values.tolist())
    for name, arr in timings.items():
        slope = estimate_loglog_slope(n_values.astype(float), arr)
        print(f"{name:14s}: {arr.tolist()}  log-log slope ~ {slope:.3f}")

    if "astropy_fast" in timings:
        slope_fast = estimate_loglog_slope(n_values.astype(float), timings["astropy_fast"])
        print(
            f"\nAstropy fast empirical slope ~ {slope_fast:.3f} (1.0 ~ O(N), 1.0-1.2 often consistent with O(N log N) in finite range)"
        )
        print("Note: this slope is with fixed frequency grid size M, so it may look flatter than theory.")

    if AstropyLombScargle is not None:
        m_values, t_fast_scaled = benchmark_astropy_fast_scaled_m(
            n_values=n_values,
            repeats=args.benchmark_repeats,
            f_min_hz=args.fmin,
            f_max_hz=args.fmax,
            min_freqs=512,
        )
        slope_scaled = estimate_loglog_slope(n_values.astype(float), t_fast_scaled)
        slope_nlogn = estimate_loglog_slope((n_values * np.log2(n_values)).astype(float), t_fast_scaled)
        print("\n=== Astropy fast with scaled frequency-grid size (M ~= N) ===")
        print(f"M values: {m_values.astype(int).tolist()}")
        print(f"times: {t_fast_scaled.tolist()}")
        print(f"log-log slope vs N: {slope_scaled:.3f}")
        print(f"log-log slope vs N log2 N: {slope_nlogn:.3f} (close to 1 suggests O(N log N))")

    maybe_plot_benchmark(n_values, timings, output_png=args.benchmark_output, show=args.show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lomb-Scargle demo with two missing signal segments.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib window in addition to saving PNG.",
    )
    parser.add_argument(
        "--output",
        default="mathematics/signal_processing/lomb_scargle/lomb_scargle_trial.png",
        help="Path to output PNG file.",
    )
    parser.add_argument("--fmin", type=float, default=0.1, help="Minimum frequency [Hz].")
    parser.add_argument("--fmax", type=float, default=15.0, help="Maximum frequency [Hz].")
    parser.add_argument("--nfreqs", type=int, default=4000, help="Number of frequency grid points.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run runtime benchmark for varying data lengths and save benchmark plot.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Number of repeats per N for median runtime.",
    )
    parser.add_argument(
        "--benchmark-output",
        default="mathematics/signal_processing/lomb_scargle/lomb_scargle_benchmark.png",
        help="Path to benchmark PNG file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark:
        run_benchmark(args)
        return

    signal_data = generate_signal_with_gaps()
    t = signal_data.t_obs
    y = signal_data.y_obs
    truth = signal_data.truth

    freqs_hz = np.linspace(args.fmin, args.fmax, args.nfreqs)

    results: list[SpectrumResult] = [analyze_with_numpy(t, y, freqs_hz)]

    scipy_result = analyze_with_scipy(t, y, freqs_hz)
    if scipy_result is not None:
        results.append(scipy_result)
    else:
        print("SciPy is not installed; skipped SciPy example.")

    astropy_result = analyze_with_astropy(t, y, freqs_hz)
    if astropy_result is not None:
        results.append(astropy_result)
    else:
        print("Astropy is not installed; skipped Astropy example.")

    print_summary(results, truth)
    maybe_plot(signal_data, results, output_png=args.output, show=args.show)


if __name__ == "__main__":
    main()
