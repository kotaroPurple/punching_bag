"""Generalized beat-like envelope demo.

Theory sketch:
Let x(t) = s(t) + r(t), and define analytic signals z_s, z_r, z_x.
Then the envelope is E_x(t) = |z_x(t)| and

    E_x(t)^2 = |z_s|^2 + |z_r|^2 + 2|z_s||z_r|cos(Delta phi),

where Delta phi = angle(z_r) - angle(z_s).

For two close sinusoids, Delta phi changes almost linearly at the difference
frequency, so the envelope oscillates periodically (classical "beat").
For a general r(t), Delta phi can be irregular, but the same interference term
still controls envelope reinforcement/cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, chirp, hilbert, sosfiltfilt, square


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Case:
    name: str
    partner: FloatArray


def analytic(x: FloatArray) -> NDArray[np.complex128]:
    return hilbert(x).astype(np.complex128)


def bandpass_noise(
    t: FloatArray,
    fs: float,
    low_hz: float,
    high_hz: float,
    seed: int,
    gain: float,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(t.shape[0]).astype(np.float64)
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return (gain * sosfiltfilt(sos, white)).astype(np.float64)


def build_cases(t: FloatArray, fs: float, f0_hz: float) -> list[Case]:
    center = 0.5 * (t[0] + t[-1])
    gaussian_narrow = 1.1 * np.exp(-0.5 * ((t - center) / 0.08) ** 2)
    gaussian_wide = 1.1 * np.exp(-0.5 * ((t - center) / 0.60) ** 2)

    return [
        Case(
            name="Near sine (classic beat)",
            partner=0.9 * np.sin(2.0 * np.pi * (f0_hz + 1.2) * t + 0.3),
        ),
        Case(
            name="Linear chirp near f0",
            partner=0.9 * chirp(t, f0=f0_hz - 1.5, f1=f0_hz + 1.5, t1=t[-1], method="linear"),
        ),
        Case(
            name="Square wave near f0",
            partner=0.5 * square(2.0 * np.pi * (f0_hz + 0.9) * t + 0.2),
        ),
        Case(
            name="Band-limited noise near f0",
            partner=bandpass_noise(
                t=t,
                fs=fs,
                low_hz=f0_hz - 2.0,
                high_hz=f0_hz + 2.0,
                seed=7,
                gain=0.65,
            ),
        ),
        Case(
            name="Narrow Gaussian-like pulse (small variance)",
            partner=gaussian_narrow,
        ),
        Case(
            name="Wide Gaussian-like pulse (large variance)",
            partner=gaussian_wide,
        ),
    ]


def plot_case(
    ax_row: NDArray[np.object_],
    t: FloatArray,
    s: FloatArray,
    r: FloatArray,
    title: str,
) -> None:
    mix = s + r
    z_s = analytic(s)
    z_r = analytic(r)
    z_mix = analytic(mix)

    env_mix = np.abs(z_mix)
    env2_mix = env_mix**2
    base = np.abs(z_s) ** 2 + np.abs(z_r) ** 2
    dphi = np.unwrap(np.angle(z_r) - np.angle(z_s))
    cross = 2.0 * np.abs(z_s) * np.abs(z_r) * np.cos(dphi)

    ax0, ax1, ax2 = ax_row
    ax0.plot(t, s, lw=1.0, label="s(t)=A sin")
    ax0.plot(t, r, lw=1.0, label="r(t)")
    ax0.plot(t, mix, lw=1.0, alpha=0.8, label="x(t)=s+r")
    ax0.set_title(title, fontsize=10)
    ax0.set_ylabel("Amplitude")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    ax1.plot(t, env_mix, lw=1.2, label="|z_x| (envelope)")
    ax1.plot(t, np.sqrt(base), lw=1.0, ls="--", label="sqrt(|z_s|^2+|z_r|^2)")
    ax1.set_ylabel("Envelope")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t, env2_mix, lw=1.2, label="|z_x|^2")
    ax2.plot(t, base, lw=1.0, ls="--", label="|z_s|^2+|z_r|^2")
    ax2.plot(t, cross, lw=1.0, ls=":", label="interference term")
    ax2.set_ylabel("Energy-like")
    ax2.set_xlabel("Time [s]")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)


def main() -> None:
    fs = 400.0
    duration_s = 6.0
    t = np.arange(0.0, duration_s, 1.0 / fs, dtype=np.float64)

    f0_hz = 12.0
    s = 1.0 * np.sin(2.0 * np.pi * f0_hz * t)

    cases = build_cases(t, fs, f0_hz)

    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 10), sharex=True)
    if len(cases) == 1:
        axes = np.array([axes], dtype=object)

    for i, case in enumerate(cases):
        plot_case(axes[i], t, s, case.partner, case.name)

    fig.suptitle("Generalized beat-like envelopes from s(t)+r(t)", fontsize=14)
    fig.tight_layout()
    out_path = Path(__file__).with_name("beat_like_signal_overview.png")
    fig.savefig(out_path, dpi=160)
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    else:
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
