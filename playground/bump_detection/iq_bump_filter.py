"""IQ bump extraction demo with BandMultiLobeComb.

Scenario:
- Target displacement: 1 Hz, 0.2 mm sine + one bump per cycle
- Interference displacement: large mm-order motion from 6 Hz and 7 Hz
- Sensor receives sum of both IQ vectors

Pipeline:
1) Generate displacements
2) Convert each to IQ and sum
3) Apply sensor-like DC cut (0.1 Hz HPF) on I/Q
4) Make analytic (complex) BandMultiLobeComb by Hilbert pair and filter IQ directly
5) Detect bump peaks from filtered output
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, fftconvolve, find_peaks, hilbert, sosfiltfilt

LIGHT_SPEED = 3e8
SENSOR_FREQUENCY = 24e9
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQUENCY / LIGHT_SPEED


def generate_iq_wave(
    displacements_m: NDArray[np.floating],
    wave_number: float,
    amp: float = 1.0,
    phase_offset_rad: float = 0.0,
) -> NDArray[np.complex128]:
    return (amp * np.exp(1j * (2.0 * wave_number * displacements_m + phase_offset_rad))).astype(np.complex128)


def normalize_kernel(h: NDArray[np.floating]) -> NDArray[np.floating]:
    h = h - np.mean(h)
    n = np.linalg.norm(h)
    return h / (n + 1e-12)


def center_pad_to_same_length(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    m = max(len(a), len(b))

    def _pad(x: NDArray[np.floating]) -> NDArray[np.floating]:
        d = m - len(x)
        left = d // 2
        right = d - left
        return np.pad(x, (left, right))

    return _pad(a), _pad(b)


def multicycle_dog_comb_kernel(
    fs: float,
    period_s: float,
    side_sigma_s: float,
    n_side: int,
    decay: float,
) -> NDArray[np.floating]:
    period_n = period_s * fs
    side_sigma_n = side_sigma_s * fs
    radius = int(np.ceil((n_side + 1.5) * period_n))
    n = np.arange(-radius, radius + 1, dtype=float)

    h = np.exp(-0.5 * (n / (0.75 * side_sigma_n)) ** 2)
    for k in range(1, n_side + 1):
        sign = -1.0 if (k % 2 == 1) else 1.0
        amp = (decay ** (k - 1)) * sign
        c = k * period_n
        h += amp * np.exp(-0.5 * ((n - c) / side_sigma_n) ** 2)
        h += amp * np.exp(-0.5 * ((n + c) / side_sigma_n) ** 2)
    return normalize_kernel(h)


def band_multilobe_comb_kernel(
    fs: float,
    f_low_hz: float,
    f_high_hz: float,
    n_freqs: int = 9,
    n_side: int = 3,
    side_sigma_ratio: float = 0.22,
    decay: float = 0.84,
) -> NDArray[np.floating]:
    freqs = np.linspace(f_low_hz, f_high_hz, n_freqs)
    center = 0.5 * (f_low_hz + f_high_hz)
    span = max(1e-6, 0.5 * (f_high_hz - f_low_hz))
    weights = np.exp(-0.5 * ((freqs - center) / (0.7 * span)) ** 2)
    weights /= np.sum(weights)

    h_sum: NDArray[np.floating] | None = None
    for w, f0 in zip(weights, freqs):
        h_i = multicycle_dog_comb_kernel(
            fs=fs,
            period_s=1.0 / f0,
            side_sigma_s=side_sigma_ratio / f0,
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


def apply_dccut_hpf(
    x: NDArray[np.floating],
    fs: float,
    cutoff_hz: float = 0.1,
    order: int = 2,
) -> NDArray[np.floating]:
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def analytic_kernel_from_real(h_real: NDArray[np.floating]) -> NDArray[np.complex128]:
    # h_a = h + j*H{h}.  H{h} is Hilbert transform of h.
    h_a = hilbert(h_real)
    return h_a.astype(np.complex128)


def complex_iq_filter_response(
    i_hp: NDArray[np.floating],
    q_hp: NDArray[np.floating],
    h_real: NDArray[np.floating],
) -> tuple[NDArray[np.complex128], NDArray[np.floating]]:
    z = i_hp + 1j * q_hp
    h_a = analytic_kernel_from_real(h_real)
    y = fftconvolve(z, h_a, mode="same")
    # Magnitude is robust to global IQ phase rotation.
    score = np.abs(y)
    return y, score


def log_enhance_1d(x: NDArray[np.floating], sigma_samples: float) -> NDArray[np.floating]:
    # Scale-normalized 2nd derivative of Gaussian for bump-like local structure.
    return -(sigma_samples**2) * gaussian_filter1d(x, sigma=sigma_samples, order=2, mode="reflect")


def complex_log_score(
    y_complex: NDArray[np.complex128],
    sigma_samples: float,
    baseline_sigma_samples: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Apply LoG on real/imag separately, then combine and remove slow baseline."""
    y_r = log_enhance_1d(y_complex.real, sigma_samples=sigma_samples)
    y_i = log_enhance_1d(y_complex.imag, sigma_samples=sigma_samples)
    score_raw = np.sqrt(y_r * y_r + y_i * y_i)
    baseline = gaussian_filter1d(score_raw, sigma=baseline_sigma_samples, mode="reflect")
    score = np.clip(score_raw - baseline, 0.0, None)
    return score, score_raw, baseline


def bump_bg_ratio(score: NDArray[np.floating], t: NDArray[np.floating], duration_s: float, bump_phase_s: float, win_s: float) -> tuple[float, float, float]:
    bump_peaks = []
    mask = np.zeros_like(score, dtype=bool)
    for sec in range(int(np.floor(duration_s))):
        c = sec + bump_phase_s
        w = np.abs(t - c) <= win_s
        if np.any(w):
            bump_peaks.append(float(np.max(score[w])))
            mask |= w
    bump_mean_peak = float(np.mean(bump_peaks)) if bump_peaks else 0.0
    bg_rms = float(np.sqrt(np.mean(score[~mask] ** 2))) if np.any(~mask) else 0.0
    ratio = bump_mean_peak / (bg_rms + 1e-12)
    return ratio, bump_mean_peak, bg_rms


def build_filter_kernels(fs: float) -> dict[str, NDArray[np.floating]]:
    """Filter-only candidates without frequency estimation/cancellation."""
    h_base = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=3.0,
        f_high_hz=10.0,
        n_freqs=9,
        n_side=3,
        side_sigma_ratio=0.22,
        decay=0.84,
    )
    h_strong = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=2.5,
        f_high_hz=11.0,
        n_freqs=15,
        n_side=5,
        side_sigma_ratio=0.24,
        decay=0.90,
    )
    h_mid = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=3.0,
        f_high_hz=10.0,
        n_freqs=13,
        n_side=5,
        side_sigma_ratio=0.20,
        decay=0.88,
    )
    h_cascade_1 = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=2.8,
        f_high_hz=8.5,
        n_freqs=11,
        n_side=4,
        side_sigma_ratio=0.22,
        decay=0.86,
    )
    h_cascade_2 = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=6.0,
        f_high_hz=11.5,
        n_freqs=11,
        n_side=4,
        side_sigma_ratio=0.22,
        decay=0.86,
    )
    h_cascade, h_cascade_2 = center_pad_to_same_length(h_cascade_1, h_cascade_2)
    h_cascade = normalize_kernel(fftconvolve(h_cascade, h_cascade_2, mode="full"))

    return {
        "baseline-comb": h_base,
        "strong-comb": h_strong,
        "dense-comb": h_mid,
        "cascade-comb": h_cascade,
    }


def build_displacements(fs: float, duration_s: float) -> tuple[NDArray[np.floating], dict[str, NDArray[np.floating]]]:
    t = np.arange(0.0, duration_s, 1.0 / fs)

    # Target: 1 Hz, 0.2 mm + one localized bump per 1-second cycle
    target_base = 0.2e-3 * np.sin(2.0 * np.pi * 1.0 * t)
    bump = np.zeros_like(t)
    bump_sigma_s = 0.01
    bump_amp_m = 0.50e-3
    for sec in range(int(np.floor(duration_s))):
        center = sec + 0.4
        bump += bump_amp_m * np.exp(-0.5 * ((t - center) / bump_sigma_s) ** 2)
    target = target_base + bump

    # Interference: large mm-order motion around 6-7 Hz
    interference = 0.5e-3 * np.sin(2.0 * np.pi * 6.0 * t + 0.4) + 1.1e-3 * np.sin(2.0 * np.pi * 7.3 * t + 1.3)

    return t, {
        "target_base": target_base,
        "target_bump_only": bump,
        "target_total": target,
        "interference": interference,
        "mixed_displacement_sum": target + interference,
    }


def main() -> None:
    fs = 200.0
    duration_s = 12.0
    t, disp = build_displacements(fs, duration_s)

    iq_target = generate_iq_wave(disp["target_total"], WAVE_NUMBER, amp=1.0, phase_offset_rad=0.0)
    iq_interference = generate_iq_wave(disp["interference"], WAVE_NUMBER, amp=1.0, phase_offset_rad=0.9)
    iq_mix = iq_target + iq_interference

    i_raw = iq_mix.real
    q_raw = iq_mix.imag
    i_hp = apply_dccut_hpf(i_raw, fs=fs, cutoff_hz=0.1, order=2)
    q_hp = apply_dccut_hpf(q_raw, fs=fs, cutoff_hz=0.1, order=2)

    kernels = build_filter_kernels(fs)

    method_out: dict[str, dict[str, NDArray[np.floating] | NDArray[np.complex128] | float]] = {}
    for name, h in kernels.items():
        y_complex, _ = complex_iq_filter_response(i_hp, q_hp, h)
        score, score_raw, baseline = complex_log_score(
            y_complex,
            sigma_samples=0.030 * fs,
            baseline_sigma_samples=0.25 * fs,
        )
        ratio, bump_peak, bg_rms = bump_bg_ratio(score, t, duration_s=duration_s, bump_phase_s=0.40, win_s=0.08)
        method_out[name] = {
            "kernel": h,
            "y_complex": y_complex,
            "score_raw": score_raw,
            "baseline": baseline,
            "score": score,
            "ratio": ratio,
            "bump_peak": bump_peak,
            "bg_rms": bg_rms,
        }

    best_method = max(method_out, key=lambda k: float(method_out[k]["ratio"]))
    y_complex = method_out[best_method]["y_complex"]  # type: ignore[assignment]
    baseline = method_out[best_method]["baseline"]  # type: ignore[assignment]
    score = method_out[best_method]["score"]  # type: ignore[assignment]

    # Envelope-like measure for bump timing
    min_dist = int(0.6 * fs)
    threshold = np.mean(score) + 1.3 * np.std(score)
    peaks_global, _ = find_peaks(score, distance=min_dist, height=threshold)

    # Phase-locked pick: one peak per 1-second cycle near expected bump phase.
    bump_phase_s = 0.40
    bump_search_half_window_s = 0.25
    peaks_phase_locked: list[int] = []
    for sec in range(int(np.floor(duration_s))):
        left_t = sec + bump_phase_s - bump_search_half_window_s
        right_t = sec + bump_phase_s + bump_search_half_window_s
        if right_t <= 0.0 or left_t >= duration_s:
            continue
        mask = (t >= left_t) & (t <= right_t)
        if not np.any(mask):
            continue
        idxs = np.flatnonzero(mask)
        k = idxs[np.argmax(score[idxs])]
        if score[k] >= 0.7 * threshold:
            peaks_phase_locked.append(int(k))

    peaks = np.asarray(peaks_phase_locked, dtype=int)

    print(f"fs={fs:.1f} Hz, duration={duration_s:.1f} s")
    print(f"selected kernel_len={len(method_out[best_method]['kernel'])}")
    print("method ranking (higher ratio is better):")
    sorted_methods = sorted(method_out.items(), key=lambda kv: float(kv[1]["ratio"]), reverse=True)
    for name, out in sorted_methods:
        print(
            f"  {name:22s} ratio={float(out['ratio']):.3f} "
            f"bump_peak={float(out['bump_peak']):.4f} bg_rms={float(out['bg_rms']):.4f}"
        )
    print(f"selected method={best_method}")
    print(f"global-peak detections={len(peaks_global)} at times (s): {np.round(t[peaks_global], 3)}")
    print(f"phase-locked detections={len(peaks)} at times (s): {np.round(t[peaks], 3)}")

    fig, axes = plt.subplots(8, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("IQ bump extraction: filter-only alternatives for periodic suppression", fontsize=14)

    axes[0].plot(t, disp["target_total"] * 1e3, label="target (1Hz + bump)", lw=1.2)
    axes[0].plot(t, disp["interference"] * 1e3, label="interference (6Hz+7Hz)", lw=1.0)
    axes[0].set_ylabel("Disp [mm]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iq_mix.real, iq_mix.imag, lw=0.8, color="tab:purple")
    axes[1].set_title("Mixed IQ trajectory")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, i_hp, lw=1.0, color="tab:blue", label="I HPF")
    axes[2].plot(t, q_hp, lw=1.0, color="tab:orange", label="Q HPF")
    axes[2].set_ylabel("I/Q HPF")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    method_colors = {
        "baseline-comb": "0.5",
        "strong-comb": "tab:blue",
        "dense-comb": "tab:orange",
        "cascade-comb": "tab:green",
    }
    for name in kernels:
        axes[3].plot(
            t,
            method_out[name]["score_raw"],
            lw=0.9,
            label=name,
            color=method_colors.get(name, None),
        )
    axes[3].set_ylabel("Raw score")
    axes[3].set_title("Method comparison (complex LoG combined)")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    h_best = method_out[best_method]["kernel"]
    n = np.arange(-(len(h_best) // 2), len(h_best) // 2 + 1)
    axes[4].plot(n / fs, h_best, color="tab:blue", lw=1.2)
    axes[4].axhline(0.0, color="0.3", lw=0.8)
    axes[4].set_ylabel("Kernel")
    axes[4].set_title("BandMultiLobeComb kernel (3-10 Hz suppression)")
    axes[4].grid(alpha=0.3)

    axes[5].plot(t, np.abs(y_complex), lw=1.0, color="tab:red")
    axes[5].set_ylabel("|Complex out|")
    axes[5].set_title(f"Selected: {best_method}")
    axes[5].grid(alpha=0.3)

    axes[6].plot(t, y_complex.real, lw=1.0, color="tab:red", label="Re{y}")
    axes[6].plot(t, y_complex.imag, lw=1.0, color="tab:purple", label="Im{y}")
    axes[6].set_ylabel("Complex out")
    axes[6].legend(loc="upper right")
    axes[6].grid(alpha=0.3)

    axes[7].plot(t, method_out[best_method]["score_raw"], lw=1.0, color="tab:brown", label="raw score")
    axes[7].plot(t, baseline, lw=1.0, color="tab:blue", label="local baseline")
    axes[7].plot(t, score, lw=1.0, color="tab:orange", label="score (baseline removed)")
    axes[7].axhline(threshold, color="tab:green", ls="--", lw=1.0, label="threshold")
    axes[7].plot(t[peaks_global], score[peaks_global], "x", color="0.35", ms=5, label="global peaks")
    axes[7].plot(t[peaks], score[peaks], "ko", ms=4, label="phase-locked bumps")
    for sec in range(int(np.floor(duration_s))):
        center = sec + bump_phase_s
        axes[7].axvspan(
            center - bump_search_half_window_s,
            center + bump_search_half_window_s,
            color="tab:green",
            alpha=0.06,
        )
    axes[7].set_ylabel("LoG score")
    axes[7].set_xlabel("Time [s]")
    axes[7].legend(loc="upper right")
    axes[7].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
