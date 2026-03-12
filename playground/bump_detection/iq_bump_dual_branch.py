"""Multi-branch bump detector on IQ signals.

Idea:
- Bump branch B(t): responds to local spike-like structure
- Repetition branch R(t): responds to periodic 3-10 Hz oscillation
- Cycle-similarity branch C(t): penalizes short-period repeated spikes
- Final score S(t) = alpha * B_norm(t) - beta(t) * R_norm(t) - gamma * C_norm(t)
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


def apply_dccut_hpf(
    x: NDArray[np.floating],
    fs: float,
    cutoff_hz: float = 0.1,
    order: int = 2,
) -> NDArray[np.floating]:
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def apply_bandpass(
    x: NDArray[np.floating],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> NDArray[np.floating]:
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


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
    n_freqs: int,
    n_side: int,
    side_sigma_ratio: float,
    decay: float,
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


def analytic_kernel_from_real(h_real: NDArray[np.floating]) -> NDArray[np.complex128]:
    return hilbert(h_real).astype(np.complex128)


def log_enhance_1d(x: NDArray[np.floating], sigma_samples: float) -> NDArray[np.floating]:
    return -(sigma_samples**2) * gaussian_filter1d(x, sigma=sigma_samples, order=2, mode="reflect")


def robust_positive_scale(x: NDArray[np.floating]) -> NDArray[np.floating]:
    q50 = np.percentile(x, 50)
    q90 = np.percentile(x, 90)
    return np.clip((x - q50) / (q90 - q50 + 1e-12), 0.0, None)


def agc_normalize_complex(
    z: NDArray[np.complex128],
    fs: float,
    sigma_s: float = 0.20,
    floor_ratio: float = 0.25,
) -> NDArray[np.complex128]:
    """Slow AGC normalization to suppress dominant large-amplitude beating."""
    env = gaussian_filter1d(np.abs(z), sigma=sigma_s * fs, mode="reflect")
    floor = floor_ratio * np.percentile(env, 50)
    env = np.maximum(env, floor)
    return (z / env).astype(np.complex128)


def soft_clip_complex(z: NDArray[np.complex128], clip_level: float) -> NDArray[np.complex128]:
    """Magnitude soft clipping to limit outlier dominance while preserving phase."""
    mag = np.abs(z)
    gain = np.tanh(mag / (clip_level + 1e-12)) / (mag / (clip_level + 1e-12) + 1e-12)
    return (z * gain).astype(np.complex128)


def short_period_similarity_penalty(
    x: NDArray[np.floating],
    fs: float,
    f_low_hz: float = 3.0,
    f_high_hz: float = 10.0,
    n_freqs: int = 9,
) -> NDArray[np.floating]:
    """Penalize structures repeating at short periods (3-10 Hz)."""
    freqs = np.linspace(f_low_hz, f_high_hz, n_freqs)
    c = np.zeros_like(x)
    x_n = robust_positive_scale(x)
    for f in freqs:
        lag = max(1, int(round(fs / f)))
        p = np.abs(x_n * np.roll(x_n, lag))
        p[:lag] = 0.0
        c += gaussian_filter1d(p, sigma=0.03 * fs, mode="reflect")
    c /= len(freqs)
    return robust_positive_scale(c)


def optimize_branch_weights(
    t: NDArray[np.floating],
    b_n: NDArray[np.floating],
    r_n: NDArray[np.floating],
    c_n: NDArray[np.floating],
    duration_s: float,
    bump_phase_s: float,
) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float, float, float]]]:
    """Grid-search (alpha, beta0, kappa, gamma) maximizing bump/background separation.

    Returns:
    - best tuple (alpha, beta0, kappa, gamma)
    - top rows [(score, alpha, beta0, kappa, gamma), ...]
    """
    alpha_grid = [0.8, 1.0, 1.2]
    beta0_grid = [0.4, 0.6, 0.8, 1.0]
    kappa_grid = [0.0, 0.2, 0.4, 0.6]
    gamma_grid = [0.0, 0.3, 0.6, 0.9]

    bump_half_win = 0.07
    mask = np.zeros_like(t, dtype=bool)
    for sec in range(int(np.floor(duration_s))):
        c = sec + bump_phase_s
        mask |= np.abs(t - c) <= bump_half_win

    rows: list[tuple[float, float, float, float, float]] = []
    for alpha in alpha_grid:
        for beta0 in beta0_grid:
            for kappa in kappa_grid:
                beta_t = beta0 * (1.0 + kappa * r_n)
                for gamma in gamma_grid:
                    s = alpha * b_n - beta_t * r_n - gamma * c_n
                    score = np.clip(s - np.median(s), 0.0, None)
                    bump_peak = float(np.mean(score[mask])) if np.any(mask) else 0.0
                    bg_rms = float(np.sqrt(np.mean(score[~mask] ** 2))) if np.any(~mask) else 0.0
                    # prefer high bump-vs-background and avoid vanishing score
                    metric = bump_peak / (bg_rms + 1e-12)
                    rows.append((metric, alpha, beta0, kappa, gamma))

    rows.sort(key=lambda x: x[0], reverse=True)
    best = rows[0]
    return (best[1], best[2], best[3], best[4]), rows[:10]


def build_displacements(fs: float, duration_s: float) -> tuple[NDArray[np.floating], dict[str, NDArray[np.floating]]]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    target_base = 0.2e-3 * np.sin(2.0 * np.pi * 1.0 * t)
    bump = np.zeros_like(t)
    bump_sigma_s = 0.010
    bump_amp_m = 0.20e-3
    for sec in range(int(np.floor(duration_s))):
        center = sec + 0.40
        bump += bump_amp_m * np.exp(-0.5 * ((t - center) / bump_sigma_s) ** 2)
    target = target_base + bump
    interference = 0.5e-3 * np.sin(2.0 * np.pi * 6.0 * t + 0.4) + 1.1e-3 * np.sin(2.0 * np.pi * 7.3 * t + 1.3)
    return t, {
        "target_total": target,
        "target_bump_only": bump,
        "interference": interference,
    }


def main() -> None:
    fs = 200.0
    duration_s = 12.0
    t, disp = build_displacements(fs, duration_s)

    iq_target = generate_iq_wave(disp["target_total"], WAVE_NUMBER, amp=1.0, phase_offset_rad=0.0)
    iq_interf = generate_iq_wave(disp["interference"], WAVE_NUMBER, amp=1.0, phase_offset_rad=0.9)
    iq_mix = iq_target + iq_interf
    i_hp = apply_dccut_hpf(iq_mix.real, fs)
    q_hp = apply_dccut_hpf(iq_mix.imag, fs)
    z_raw = (i_hp + 1j * q_hp).astype(np.complex128)
    z_agc = agc_normalize_complex(z_raw, fs=fs, sigma_s=0.20, floor_ratio=0.25)
    clip_level = 2.5 * np.percentile(np.abs(z_agc), 50)
    z = soft_clip_complex(z_agc, clip_level=clip_level)

    h_bump = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=3.0,
        f_high_hz=10.0,
        n_freqs=13,
        n_side=5,
        side_sigma_ratio=0.20,
        decay=0.88,
    )
    h_rep = band_multilobe_comb_kernel(
        fs=fs,
        f_low_hz=4.0,
        f_high_hz=8.5,
        n_freqs=15,
        n_side=5,
        side_sigma_ratio=0.24,
        decay=0.92,
    )
    h_bump_a = analytic_kernel_from_real(h_bump)
    h_rep_a = analytic_kernel_from_real(h_rep)

    # Branch B: spike sensitivity
    y_b = fftconvolve(z, h_bump_a, mode="same")
    b_raw = np.sqrt(log_enhance_1d(y_b.real, 0.022 * fs) ** 2 + log_enhance_1d(y_b.imag, 0.022 * fs) ** 2)
    b = np.clip(b_raw - gaussian_filter1d(b_raw, sigma=0.20 * fs, mode="reflect"), 0.0, None)

    # Branch R: periodicity sensitivity
    # Use IQ trajectory speed, then 3-10Hz band energy envelope.
    di = np.diff(z.real, prepend=z.real[0]) * fs
    dq = np.diff(z.imag, prepend=z.imag[0]) * fs
    iq_speed = np.sqrt(di * di + dq * dq)
    # log-compression further reduces dominance of large beating swings.
    iq_speed = np.log1p(iq_speed / (np.percentile(iq_speed, 50) + 1e-12))
    speed_bp = apply_bandpass(iq_speed, fs=fs, low_hz=3.0, high_hz=10.0, order=3)
    r = gaussian_filter1d(np.abs(speed_bp), sigma=0.08 * fs, mode="reflect")

    b_n = robust_positive_scale(b)
    r_n = robust_positive_scale(r)
    c_n = short_period_similarity_penalty(b_n, fs=fs, f_low_hz=3.0, f_high_hz=10.0, n_freqs=9)

    (alpha, beta0, kappa, gamma), top_rows = optimize_branch_weights(
        t=t,
        b_n=b_n,
        r_n=r_n,
        c_n=c_n,
        duration_s=duration_s,
        bump_phase_s=0.40,
    )
    beta_t = beta0 * (1.0 + kappa * r_n)
    s = alpha * b_n - beta_t * r_n - gamma * c_n

    score = np.clip(s - np.median(s), 0.0, None)
    threshold = np.mean(score) + 1.2 * np.std(score)
    peaks_global, _ = find_peaks(score, distance=int(0.6 * fs), height=threshold)

    bump_phase_s = 0.40
    bump_win = 0.25
    peaks_cycle: list[int] = []
    for sec in range(int(np.floor(duration_s))):
        mask = np.abs(t - (sec + bump_phase_s)) <= bump_win
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            continue
        k = idxs[np.argmax(score[idxs])]
        if score[k] >= 0.7 * threshold:
            peaks_cycle.append(int(k))
    peaks = np.asarray(peaks_cycle, dtype=int)

    print(f"global peaks={len(peaks_global)} times={np.round(t[peaks_global], 3)}")
    print(f"cycle peaks ={len(peaks)} times={np.round(t[peaks], 3)}")
    print(f"best weights: alpha={alpha:.2f}, beta0={beta0:.2f}, kappa={kappa:.2f}, gamma={gamma:.2f}")
    print("top weight candidates (metric, alpha, beta0, kappa, gamma):")
    for row in top_rows[:5]:
        print(f"  {row[0]:.3f}, {row[1]:.2f}, {row[2]:.2f}, {row[3]:.2f}, {row[4]:.2f}")
    print(
        "term magnitudes: "
        f"mean|alpha*B|={np.mean(np.abs(alpha * b_n)):.3f}, "
        f"mean|beta*R|={np.mean(np.abs(beta_t * r_n)):.3f}, "
        f"mean|gamma*C|={np.mean(np.abs(gamma * c_n)):.3f}"
    )

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex="col")
    ax = axes.ravel()

    ax[0].plot(t, disp["target_total"] * 1e3, lw=1.2, label="target")
    ax[0].plot(t, disp["interference"] * 1e3, lw=1.0, label="interference")
    ax[0].set_ylabel("Disp [mm]")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)

    ax[1].plot(z_raw.real, z_raw.imag, lw=0.6, color="0.7", label="after DCCut")
    ax[1].plot(z.real, z.imag, lw=0.8, color="tab:purple", label="after AGC+clip")
    ax[1].set_title("IQ trajectory preprocessing")
    ax[1].set_xlabel("I")
    ax[1].set_ylabel("Q")
    ax[1].axis("equal")
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, b_n, lw=1.0, color="tab:red", label="B norm")
    ax[2].set_ylabel("B(t)")
    ax[2].legend(loc="upper right")
    ax[2].grid(alpha=0.3)

    ax[3].plot(t, r_n, lw=1.0, color="tab:blue", label="R norm")
    ax[3].plot(t, c_n, lw=1.0, color="tab:purple", label="C norm")
    ax[3].plot(t, beta_t, lw=0.9, color="tab:green", label="beta(t)")
    ax[3].set_ylabel("R(t), C(t), beta")
    ax[3].legend(loc="upper right")
    ax[3].grid(alpha=0.3)

    ax[4].plot(t, alpha * b_n, lw=1.0, color="tab:red", label="alpha*B")
    ax[4].plot(t, beta_t * r_n, lw=1.0, color="tab:blue", label="beta*R")
    ax[4].plot(t, gamma * c_n, lw=1.0, color="tab:purple", label="gamma*C")
    ax[4].set_ylabel("Branch terms")
    ax[4].legend(loc="upper right")
    ax[4].grid(alpha=0.3)

    ax[5].plot(t, s, lw=1.0, color="tab:brown", label="S=alpha*B-beta*R-gamma*C")
    ax[5].set_ylabel("S(t)")
    ax[5].legend(loc="upper right")
    ax[5].grid(alpha=0.3)

    ax[6].plot(t, score, lw=1.0, color="tab:orange", label="score")
    ax[6].axhline(threshold, color="tab:green", ls="--", lw=1.0, label="threshold")
    ax[6].plot(t[peaks_global], score[peaks_global], "x", color="0.35", ms=5, label="global peaks")
    ax[6].plot(t[peaks], score[peaks], "ko", ms=4, label="cycle peaks")
    ax[6].set_ylabel("Detection")
    ax[6].legend(loc="upper right")
    ax[6].grid(alpha=0.3)

    ax[7].plot(t, disp["target_bump_only"] * 1e3, lw=1.0, color="tab:gray", label="true bump")
    ax[7].plot(t, score / (np.max(score) + 1e-12), lw=1.0, color="tab:orange", label="normalized score")
    ax[7].set_ylabel("Bump vs score")
    ax[7].legend(loc="upper right")
    ax[7].grid(alpha=0.3)

    ax[6].set_xlabel("Time [s]")
    ax[7].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
