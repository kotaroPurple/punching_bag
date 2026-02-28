"""1D bump detection filter trial.

Goal:
- Suppress beat-like oscillation response
- Keep localized bump response

Candidates:
- LoG (scale-normalized)
- Bi-scale LoG (LoG_small - alpha * LoG_large)
- DoG wavelet
- Mexican hat (Ricker) wavelet
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


@dataclass
class FilterSpec:
    name: str
    kernel: np.ndarray


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def normalize_kernel(h: np.ndarray) -> np.ndarray:
    """Force zero-mean and unit L2 norm for fair comparison."""
    h = h - np.mean(h)
    n = np.linalg.norm(h)
    return h / (n + 1e-12)


def center_pad_to_same_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad shorter kernel so both arrays share center and length."""
    m = max(len(a), len(b))

    def _pad(x: np.ndarray) -> np.ndarray:
        d = m - len(x)
        left = d // 2
        right = d - left
        return np.pad(x, (left, right))

    return _pad(a), _pad(b)


def gaussian_kernel(fs: float, sigma_s: float, trunc_sigma: float = 4.0) -> np.ndarray:
    sigma_n = sigma_s * fs
    radius = int(np.ceil(trunc_sigma * sigma_n))
    n = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-0.5 * (n / sigma_n) ** 2)
    g /= np.sum(g)
    return g


def log_kernel(fs: float, sigma_s: float, trunc_sigma: float = 4.0) -> np.ndarray:
    """Scale-normalized 1D LoG kernel (discrete)."""
    sigma_n = sigma_s * fs
    radius = int(np.ceil(trunc_sigma * sigma_n))
    n = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-0.5 * (n / sigma_n) ** 2)
    # second derivative of gaussian (up to constant)
    d2g = ((n * n - sigma_n * sigma_n) / (sigma_n**4)) * g
    # scale-normalization for multi-scale comparability
    return normalize_kernel(-(sigma_n**2) * d2g)


def dog_kernel(fs: float, sigma_small_s: float, k: float = 1.6) -> np.ndarray:
    g1 = gaussian_kernel(fs, sigma_small_s)
    g2 = gaussian_kernel(fs, sigma_small_s * k)
    p1, p2 = center_pad_to_same_length(g1, g2)
    return normalize_kernel(p1 - p2)


def ricker_kernel(fs: float, a_s: float, trunc_sigma: float = 4.0) -> np.ndarray:
    """Mexican hat wavelet kernel."""
    a_n = a_s * fs
    radius = int(np.ceil(trunc_sigma * a_n))
    n = np.arange(-radius, radius + 1, dtype=float)
    h = (1.0 - (n * n) / (a_n * a_n)) * np.exp(-0.5 * (n / a_n) ** 2)
    return normalize_kernel(h)


def morlet_kernel(fs: float, sigma_s: float, f0_hz: float, trunc_sigma: float = 4.0) -> np.ndarray:
    """Real Morlet-like wavelet (Gaussian-windowed cosine, zero-mean)."""
    sigma_n = sigma_s * fs
    radius = int(np.ceil(trunc_sigma * sigma_n))
    n = np.arange(-radius, radius + 1, dtype=float)
    t = n / fs
    h = np.exp(-0.5 * (n / sigma_n) ** 2) * np.cos(2.0 * np.pi * f0_hz * t)
    return normalize_kernel(h)


def multicycle_dog_comb_kernel(
    fs: float,
    period_s: float,
    side_sigma_s: float,
    n_side: int,
    decay: float = 0.65,
) -> np.ndarray:
    """Central positive lobe + alternating side lobes at +/-k*period.

    This creates a multi-lobe shape that cancels periodic components near 1/period.
    """
    period_n = period_s * fs
    side_sigma_n = side_sigma_s * fs
    radius = int(np.ceil((n_side + 1.5) * period_n))
    n = np.arange(-radius, radius + 1, dtype=float)

    # start with narrow center bump
    h = np.exp(-0.5 * (n / (0.75 * side_sigma_n)) ** 2)

    # alternating signs on side lobes: - + - + ...
    for k in range(1, n_side + 1):
        sign = -1.0 if (k % 2 == 1) else 1.0
        amp = (decay ** (k - 1)) * sign
        c = k * period_n
        lobe_pos = np.exp(-0.5 * ((n - c) / side_sigma_n) ** 2)
        lobe_neg = np.exp(-0.5 * ((n + c) / side_sigma_n) ** 2)
        h += amp * (lobe_pos + lobe_neg)

    return normalize_kernel(h)


def multilobe_comb_bandstop_kernel(
    fs: float,
    f_low_hz: float,
    f_high_hz: float,
    n_freqs: int = 9,
    n_side: int = 3,
    side_sigma_ratio: float = 0.22,
    decay: float = 0.68,
) -> np.ndarray:
    """Band-stop style MultiLobeComb by summing comb kernels tuned to multiple periods.

    Each sub-kernel is tuned to one center frequency in [f_low_hz, f_high_hz].
    Summation creates multiple shallow/deep notches over the target band.
    """
    freqs = np.linspace(f_low_hz, f_high_hz, n_freqs)
    # emphasize middle of the band slightly
    center = 0.5 * (f_low_hz + f_high_hz)
    span = max(1e-6, 0.5 * (f_high_hz - f_low_hz))
    weights = np.exp(-0.5 * ((freqs - center) / (0.7 * span)) ** 2)
    weights /= np.sum(weights)

    h_sum: np.ndarray | None = None
    for w, f0 in zip(weights, freqs):
        period_s = 1.0 / f0
        side_sigma_s = side_sigma_ratio * period_s
        h_i = multicycle_dog_comb_kernel(
            fs=fs,
            period_s=period_s,
            side_sigma_s=side_sigma_s,
            n_side=n_side,
            decay=decay,
        )
        if h_sum is None:
            h_sum = w * h_i
        else:
            h_sum, h_i = center_pad_to_same_length(h_sum, h_i)
            h_sum = h_sum + w * h_i

    assert h_sum is not None
    return normalize_kernel(h_sum)


def apply_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return fftconvolve(x, h, mode="same")


def build_signals(fs: float, duration_s: float) -> tuple[np.ndarray, dict[str, np.ndarray], float, float]:
    t = np.arange(0.0, duration_s, 1.0 / fs)

    base_freq_hz = 1.0
    sine = 0.5 * np.sin(2.0 * np.pi * base_freq_hz * t)

    bump_center_s = duration_s * 0.55
    bump_sigma_s = 0.030
    bump_amp = 1.0
    bump = bump_amp * np.exp(-0.5 * ((t - bump_center_s) / bump_sigma_s) ** 2)
    sine_plus_bump = sine + bump

    # Repetitive component spanning 3-10 Hz band
    beat_freqs = [6.0, 7.5]
    beat_amps = [0.9, 0.8]
    beat_phases = [0.0, 0.6]
    beat = np.zeros_like(t)
    for a, f, p in zip(beat_amps, beat_freqs, beat_phases):
        beat += a * np.sin(2.0 * np.pi * f * t + p)
    beat_plus_sine_bump = beat + sine_plus_bump

    signals = {
        "sine": sine,
        "sine + bump": sine_plus_bump,
        "beat": beat,
        "beat + sine + bump": beat_plus_sine_bump,
    }
    return t, signals, bump_center_s, bump_sigma_s


def make_candidates(fs: float) -> list[FilterSpec]:
    h_log = log_kernel(fs, sigma_s=0.05)
    h_log_small = log_kernel(fs, sigma_s=0.04)
    h_log_large = log_kernel(fs, sigma_s=0.11)
    h_log_small, h_log_large = center_pad_to_same_length(h_log_small, h_log_large)
    h_bilog = normalize_kernel(h_log_small - 0.70 * h_log_large)
    h_dog = dog_kernel(fs, sigma_small_s=0.04, k=1.8)
    h_ricker = ricker_kernel(fs, a_s=0.05)
    h_morlet_2cy = morlet_kernel(fs, sigma_s=0.16, f0_hz=5.4)
    h_morlet_4cy = morlet_kernel(fs, sigma_s=0.30, f0_hz=5.4)
    h_comb_3 = multicycle_dog_comb_kernel(fs, period_s=1.0 / 5.4, side_sigma_s=0.045, n_side=3)
    h_comb_5 = multicycle_dog_comb_kernel(fs, period_s=1.0 / 5.4, side_sigma_s=0.040, n_side=5)
    h_band_comb_3 = multilobe_comb_bandstop_kernel(
        fs=fs,
        f_low_hz=3.0,
        f_high_hz=10.0,
        n_freqs=9,
        n_side=3,
        side_sigma_ratio=0.22,
        decay=0.84,
    )
    h_band_comb_5 = multilobe_comb_bandstop_kernel(
        fs=fs,
        f_low_hz=3.0,
        f_high_hz=10.0,
        n_freqs=11,
        n_side=3,
        side_sigma_ratio=0.22,
        decay=0.84,
    )

    return [
        # FilterSpec("LoG(s=50ms)", h_log),
        # FilterSpec("Bi-LoG(40ms,110ms)", h_bilog),
        # FilterSpec("DoG(40ms,k=1.8)", h_dog),
        # FilterSpec("MexHat(50ms)", h_ricker),
        FilterSpec("Morlet(~2cy,5.4Hz)", h_morlet_2cy),
        FilterSpec("Morlet(~4cy,5.4Hz)", h_morlet_4cy),
        FilterSpec("MultiLobeComb(3)", h_comb_3),
        FilterSpec("MultiLobeComb(5)", h_comb_5),
        FilterSpec("BandMultiLobeComb(best)", h_band_comb_3),
        FilterSpec("BandMultiLobeComb(best-nearby)", h_band_comb_5),
    ]


def search_best_band_comb(
    fs: float,
    signals: dict[str, np.ndarray],
    t: np.ndarray,
    bump_center_s: float,
    bump_sigma_s: float,
    top_k: int = 10,
) -> list[tuple[float, float, float, float, int, int, float, float]]:
    """Grid-search band MultiLobeComb parameters.

    Returns rows sorted by score_mix desc:
    (score_mix, score_beat, bump_peak, beat_rms, n_freqs, n_side, side_sigma_ratio, decay)
    """
    bump_window = np.abs(t - bump_center_s) <= 3.0 * bump_sigma_s
    non_bump_window = ~bump_window

    n_freqs_grid = [7, 9, 11, 13, 15]
    n_side_grid = [3, 5, 7]
    side_sigma_ratio_grid = [0.12, 0.15, 0.18, 0.22]
    decay_grid = [0.60, 0.66, 0.72, 0.78, 0.84]

    rows: list[tuple[float, float, float, float, int, int, float, float]] = []
    for n_freqs, n_side, side_sigma_ratio, decay in product(
        n_freqs_grid,
        n_side_grid,
        side_sigma_ratio_grid,
        decay_grid,
    ):
        h = multilobe_comb_bandstop_kernel(
            fs=fs,
            f_low_hz=3.0,
            f_high_hz=10.0,
            n_freqs=n_freqs,
            n_side=n_side,
            side_sigma_ratio=side_sigma_ratio,
            decay=decay,
        )
        y_sine = apply_filter(signals["sine"], h)
        y_sine_bump = apply_filter(signals["sine + bump"], h)
        y_beat = apply_filter(signals["beat"], h)
        y_mix = apply_filter(signals["beat + sine + bump"], h)

        bump_only_resp = y_sine_bump - y_sine
        bump_peak = float(np.max(np.abs(bump_only_resp[bump_window])))
        beat_rms = rms(y_beat)
        mixed_bg_rms = rms(y_mix[non_bump_window])
        score_beat = bump_peak / (beat_rms + 1e-12)
        score_mix = bump_peak / (mixed_bg_rms + 1e-12)

        rows.append((score_mix, score_beat, bump_peak, beat_rms, n_freqs, n_side, side_sigma_ratio, decay))

    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:top_k]


def main() -> None:
    fs = 200.0
    duration_s = 8.0
    t, signals, bump_center_s, bump_sigma_s = build_signals(fs, duration_s)
    filters = make_candidates(fs)

    bump_window = np.abs(t - bump_center_s) <= 3.0 * bump_sigma_s
    non_bump_window = ~bump_window

    rows = []
    responses: dict[str, dict[str, np.ndarray]] = {}

    for f in filters:
        y = {name: apply_filter(x, f.kernel) for name, x in signals.items()}
        responses[f.name] = y

        bump_only_resp = y["sine + bump"] - y["sine"]
        bump_peak = float(np.max(np.abs(bump_only_resp[bump_window])))
        beat_leak_rms = rms(y["beat"])
        mixed_bg_rms = rms(y["beat + sine + bump"][non_bump_window])

        rows.append(
            (
                f.name,
                bump_peak,
                beat_leak_rms,
                mixed_bg_rms,
                bump_peak / (beat_leak_rms + 1e-12),
                bump_peak / (mixed_bg_rms + 1e-12),
            )
        )

    rows.sort(key=lambda x: x[5], reverse=True)

    print(f"fs={fs:.1f} Hz, duration={duration_s:.1f} s")
    print("score_beat = bump_peak / beat_leak_rms")
    print("score_mix  = bump_peak / mixed_bg_rms (outside bump window)")
    print()
    print("name                      bump_peak  beat_rms  mixed_bg_rms  score_beat  score_mix")
    print("-" * 86)
    for r in rows:
        print(f"{r[0]:24s}  {r[1]:8.4f}  {r[2]:8.4f}  {r[3]:12.4f}  {r[4]:10.4f}  {r[5]:9.4f}")

    best_name = rows[0][0]
    print(f"\nBest by score_mix: {best_name}")

    top = search_best_band_comb(
        fs=fs,
        signals=signals,
        t=t,
        bump_center_s=bump_center_s,
        bump_sigma_s=bump_sigma_s,
        top_k=10,
    )
    print("\nTop BandMultiLobeComb params by score_mix")
    print("score_mix  score_beat  bump_peak  beat_rms  n_freqs  n_side  side_sigma_ratio  decay")
    print("-" * 90)
    for r in top:
        print(f"{r[0]:9.4f}  {r[1]:10.4f}  {r[2]:9.4f}  {r[3]:8.4f}  {r[4]:7d}  {r[5]:6d}  {r[6]:16.3f}  {r[7]:5.2f}")

    n_filters = len(filters)
    fig, axes = plt.subplots(n_filters, 3, figsize=(15, 1.5 * n_filters), sharex=False)
    fig.suptitle("Bump-oriented filter comparison (LoG variants + wavelet-like)", fontsize=13)

    for i, f in enumerate(filters):
        y = responses[f.name]
        bump_only_resp = y["sine + bump"] - y["sine"]

        n = np.arange(-(len(f.kernel) // 2), len(f.kernel) // 2 + 1)
        tk = n / fs
        axes[i, 0].plot(tk, f.kernel, lw=1.2, color="tab:blue")
        axes[i, 0].axhline(0.0, color="0.4", lw=0.8)
        axes[i, 0].set_ylabel(f.name)
        axes[i, 0].set_title("Kernel")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(t, y["beat"], lw=1.0, color="tab:orange", label="beat response")
        axes[i, 1].set_title("Response to beat")
        axes[i, 1].grid(alpha=0.3)

        axes[i, 2].plot(t, y["beat + sine + bump"], lw=1.0, color="0.4", label="mixed response")
        axes[i, 2].plot(t, bump_only_resp, lw=1.1, color="tab:red", label="bump-only component")
        axes[i, 2].axvspan(bump_center_s - 3 * bump_sigma_s, bump_center_s + 3 * bump_sigma_s, color="tab:green", alpha=0.1)
        axes[i, 2].set_title("Mixed response and bump component")
        axes[i, 2].grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[-1, 2].set_xlabel("Time [s]")
    axes[0, 2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    fig2, axes2 = plt.subplots(n_filters, 2, figsize=(15, 1.5 * n_filters), sharey=False)
    fig2.suptitle("Original vs filtered (input: beat + sine + bump)", fontsize=13)

    x_mixed = signals["beat + sine + bump"]
    zoom_half_width_s = 0.8
    zoom_mask = np.abs(t - bump_center_s) <= zoom_half_width_s

    for i, f in enumerate(filters):
        y_mixed = responses[f.name]["beat + sine + bump"]

        axes2[i, 0].plot(t, x_mixed, lw=1.0, color="0.7", label="original")
        axes2[i, 0].plot(t, y_mixed, lw=1.0, color="tab:red", label="filtered")
        axes2[i, 0].axvspan(
            bump_center_s - 3 * bump_sigma_s,
            bump_center_s + 3 * bump_sigma_s,
            color="tab:green",
            alpha=0.1,
        )
        axes2[i, 0].set_ylabel(f.name)
        axes2[i, 0].set_title("Full range")
        axes2[i, 0].grid(alpha=0.3)

        axes2[i, 1].plot(t[zoom_mask], x_mixed[zoom_mask], lw=1.0, color="0.7", label="original")
        axes2[i, 1].plot(t[zoom_mask], y_mixed[zoom_mask], lw=1.0, color="tab:red", label="filtered")
        axes2[i, 1].axvspan(
            bump_center_s - 3 * bump_sigma_s,
            bump_center_s + 3 * bump_sigma_s,
            color="tab:green",
            alpha=0.1,
        )
        axes2[i, 1].set_title("Around bump")
        axes2[i, 1].grid(alpha=0.3)

    axes2[-1, 0].set_xlabel("Time [s]")
    axes2[-1, 1].set_xlabel("Time [s]")
    axes2[0, 0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
