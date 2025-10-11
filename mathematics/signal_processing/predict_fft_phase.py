# Fix: the complex lock-in phase corresponds to the phase at n=0.
# Convert to window-center reference by adding + 2π f_hat * tc.

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from math import pi
import pandas as pd
# from caas_jupyter_tools import display_dataframe_to_user


@dataclass
class SynthSpec:
    fs: float = 1000.0      # サンプリング周波数 [Hz]
    N: int = 4096           # 窓長（サンプル数）
    f0: float = 7.3         # 基本周波数（真値）
    amps: np.ndarray = None # 倍音1..Mの振幅
    phases0: np.ndarray = None  # 各倍音の初期位相（n=0基準, [-π, π]）
    rng_seed: int = 42      # 乱数シード（再現性）


def generate_harmonic_signal(spec: SynthSpec, snr_db: float = 25.0):
    """
    基本波＋倍音(1..M)の実信号を生成し、近接妨害＋白色雑音を加える。
    さらに、窓中心での真の位相（各倍音）も返す。

    Returns
    -------
    x : (N,) ndarray
        観測信号（実数）
    t : (N,) ndarray
        時間軸 [s]
    w : (N,) ndarray
        Hann窓（解析で使う想定）
    true : dict
        {
            'f0_true': float,
            'amps_true': (M,),
            'phases0_true': (M,),          # n=0基準
            'phi_center_true': (M,),       # 窓中心時刻の真の位相（[-π, π]）
            'tc': float,                   # 窓中心時刻 [s]
            'interferers': list[(amp, freq, phase)]
        }
    """
    rng = np.random.default_rng(spec.rng_seed)

    fs, N, f0 = spec.fs, spec.N, spec.f0
    t = np.arange(N) / fs
    tc = (N - 1) / (2 * fs)  # 窓中心時刻

    # 倍音振幅・初期位相
    if spec.amps is None:
        amps_true = np.array([1.00, 0.45, 0.30, 0.22, 0.18], dtype=float)  # M=5
    else:
        amps_true = np.asarray(spec.amps, dtype=float)

    M = len(amps_true)
    if spec.phases0 is None:
        phases0_true = rng.uniform(-pi, pi, size=M)
    else:
        phases0_true = np.asarray(spec.phases0, dtype=float)

    # --- 基本波＋倍音 ---
    x = np.zeros(N, dtype=float)
    for m in range(1, M + 1):
        x += amps_true[m - 1] * np.cos(2 * np.pi * (m * f0) * t + phases0_true[m - 1])

    # --- 近接妨害（再現用に固定）---
    #   1) 1.02 * f0, amp=0.10, phase=+0.7
    #   2) 2*0.985 * f0, amp=0.07, phase=-1.2
    #   3) 3*1.012 * f0, amp=0.05, phase=+2.1
    interferers = [
        (0.10, 1.02 * f0,  +0.7),
        (0.07, 2.0 * 0.985 * f0, -1.2),
        (0.05, 3.0 * 1.012 * f0, +2.1),
    ]
    for a, fI, ph in interferers:
        x += a * np.cos(2 * np.pi * fI * t + ph)

    # --- 白色雑音（SNR指定）---
    sig_power = np.mean(x**2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    x += rng.normal(scale=np.sqrt(noise_power), size=N)

    # --- 窓（解析用に返しておく）---
    w = np.hanning(N)

    # --- 真の位相（窓中心時刻の値）---
    # phi_m(tc) = phases0_m + 2π * m * f0 * tc, を [-π, π]に正規化
    phi_center_true = (phases0_true + 2 * np.pi * np.arange(1, M + 1) * f0 * tc + np.pi) % (2 * np.pi) - np.pi

    true = dict(
        f0_true=f0,
        amps_true=amps_true,
        phases0_true=phases0_true,
        phi_center_true=phi_center_true,
        tc=tc,
        interferers=interferers,
        snr_db=snr_db,
        fs=fs,
        N=N,
    )
    return x, t, w, true


# Reuse previously defined variables: x, fs, N, xw, Wsum, f0_true, M, amps_true, phases0_true, tc
def jacobsen_3pt_delta(Xm1, X0, Xp1):
    alpha = (Xp1 - Xm1) / (2*X0 - Xm1 - Xp1)
    return np.real(alpha)


def principal_coeff_phase_center(x, fs, f_search=None, nperseg=None, windowed_signal=None, window_sum=None, tc=None):
    N = len(x) if nperseg is None else nperseg
    if len(x) != N:
        x = x[:N]

    if windowed_signal is None or window_sum is None:
        w = np.hanning(N)
        xw = w * x
        Wsum = np.sum(w)
    else:
        xw = windowed_signal
        Wsum = window_sum

    X = rfft(xw)
    f = rfftfreq(N, 1/fs)

    if f_search is None:
        k0 = int(np.argmax(np.abs(X)))
    else:
        k0 = int(np.argmin(np.abs(f - f_search)))

    k0 = int(np.clip(k0, 1, len(X)-2))
    delta = jacobsen_3pt_delta(X[k0-1], X[k0], X[k0+1])
    f_hat = (k0 + np.clip(delta, -0.5, 0.5)) * fs / N

    n = np.arange(N)
    z = xw * np.exp(-1j*2*np.pi*f_hat*n/fs)
    A_hat = (2.0 / Wsum) * np.sum(z)

    phi_0 = np.angle(A_hat)
    if tc is None:
        tc_loc = (N-1)/(2*fs)
    else:
        tc_loc = tc
    phi_center = np.angle(np.exp(1j*(phi_0 + 2*np.pi*f_hat*tc_loc)))
    return f_hat, A_hat, phi_center

# True center phases
spec = SynthSpec()  # 既定値＝今回の検証と同じ条件
fs = spec.fs
f0_true = spec.f0
x, t, w, info = generate_harmonic_signal(spec, snr_db=25.0)
phi_center_true = info['phi_center_true']
# phi_center_true = np.mod(phases0_true + 2*np.pi*np.arange(1, M+1)*f0_true*tc + np.pi, 2*np.pi) - np.pi

# Estimate
results = []
f0_est, _, _ = principal_coeff_phase_center(x, fs, f_search=f0_true, nperseg=None, windowed_signal=None, window_sum=None, tc=None)
M = 2

for m in range(1, M+1):
    fm_guess = m * f0_est
    fm_hat, A_hat, phi_hat_c = principal_coeff_phase_center(
        x, fs, f_search=fm_guess, nperseg=None, windowed_signal=None, window_sum=None, tc=None)
    phi_true = phi_center_true[m-1]
    err = np.angle(np.exp(1j*(phi_hat_c - phi_true)))
    amp_hat = np.abs(A_hat)
    results.append({
        "harmonic_m": m,
        "f_true_Hz": m*f0_true,
        "f_hat_Hz": fm_hat,
        # "amp_true": amps_true[m-1],
        "|A_hat|": amp_hat,
        "phi_true_center_rad": phi_true,
        "phi_hat_center_rad": phi_hat_c,
        "phase_err_deg": np.degrees(err),
    })

df2 = pd.DataFrame(results)
df2["phi_hat_center_deg"] = np.degrees(df2["phi_hat_center_rad"])

# Baseline again for comparison
X = rfft(x * w)
f = rfftfreq(len(x), 1/fs)
phase_errs_baseline = []
tc = info["tc"]
for m in range(1, M+1):
    k = int(np.argmin(np.abs(f - m*f0_true)))
    phi_hat_bin = np.angle(X[k])
    phi_hat_center_bin = np.angle(np.exp(1j*(phi_hat_bin + 2*np.pi*f[k]*tc)))
    phi_true = phi_center_true[m-1]
    err = np.angle(np.exp(1j*(phi_hat_center_bin - phi_true)))
    phase_errs_baseline.append(np.degrees(err))

df2["baseline_fftbin_phase_err_deg"] = phase_errs_baseline

# Annotated spectrum highlighting estimated phases
plt.figure()
plt.plot(f, np.abs(X), label="Windowed spectrum")
plt.scatter(df2["f_hat_Hz"].values, df2["|A_hat|"].values, color="red", zorder=3, label="Estimated harmonics")
for _, row in df2.iterrows():
    plt.annotate(f"{row['phi_hat_center_deg']:.1f}°",
                 (row["f_hat_Hz"], row["|A_hat|"]),
                 textcoords="offset points", xytext=(0, 8),
                 ha="center", color="red")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Harmonic peaks with phase annotations (window center)")
plt.xlim(0, (M + 1.5) * f0_true)
plt.legend()
plt.grid(True)

# Phase error view retained for diagnostics
plt.figure()
plt.plot(df2["harmonic_m"].values, df2["phase_err_deg"].values, marker="o", label="Proposed (sub-bin + lock-in, center-corrected)")
# plt.plot(df2["harmonic_m"].values, df2["baseline_fftbin_phase_err_deg"].values, marker="x", linestyle="--", label="Naive FFT-bin")
plt.xlabel("Harmonic order m")
plt.ylabel("Phase error [deg]")
plt.title("Phase estimation error by harmonic (center-corrected)")
plt.legend()
plt.grid(True)

# Reconstruct signal from estimated amplitudes and center phases
x_recon = np.zeros_like(x)
for _, row in df2.iterrows():
    freq = row["f_hat_Hz"]
    amp = row["|A_hat|"]
    phi_center = row["phi_hat_center_rad"]
    x_recon += amp * np.cos(2 * np.pi * freq * (t - tc) + phi_center)

plt.figure()
plt.plot(t, x, label="Original signal", alpha=0.7)
plt.plot(t, x_recon, label="Reconstructed from estimated phases", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Signal reconstruction using estimated harmonic phases")
plt.legend()
plt.grid(True)

plt.show()
