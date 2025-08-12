
# Re-run the visualization after a reset.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from numpy.fft import fft, ifft, rfft, rfftfreq, fftfreq
from numpy.typing import NDArray

# ---------- Parameters ----------
fs = 200
T  = 120.0
t  = np.arange(0, T, 1/fs)
N  = len(t)

# Rates
f_H = 1.25   # heart
f_R = 0.25  # respiration

# Displacements and wavelength
lam_mm = 12.5
dH_mm  = 0.2
dR_mm  = 4.0

beta_H = 4*np.pi * (dH_mm / lam_mm)
beta_R = 4*np.pi * (dR_mm / lam_mm)

phi_H = np.deg2rad(0.)
phi_R = np.deg2rad(30.)

# remove low frequency
abs_fmin = 0.6

print(f"beta_H (heart) ≈ {beta_H:.3f}, beta_R (resp) ≈ {beta_R:.3f}")

# ---------- Helper functions ----------
def normalize_signal(x: NDArray) -> NDArray:
    """Normalize 1D signal to range [-1, 1]"""
    x_min = np.min(x)
    x_max = np.max(x)
    return 2 * (x - x_min) / (x_max - x_min) - 1


def split_pos_neg(x: NDArray[np.complex128], fs: int, abs_fmin: float = 0.0):
    X = fft(x)
    f = fftfreq(len(x), 1/fs)
    abs_fmin = abs(abs_fmin)
    Xpos = np.where(f >= abs_fmin, X, 0)
    Xneg = np.where(f <= -abs_fmin, X, 0)
    x_pos = ifft(Xpos)
    x_neg = ifft(Xneg)
    return x_pos, x_neg


def power_spectrum(x, fs, zp=4, window="hann"):
    N = len(x)
    win = get_window(window, N)
    X = rfft(x*win, n=zp*N)
    freqs = rfftfreq(zp*N, 1/fs)
    P = (np.abs(X)**2) / (np.sum(win**2))
    return freqs, P


def plot_spectrum(freqs: NDArray[np.float64], P: NDArray[np.float64], title: str, fmarks: list[float] = []):
    plt.figure()
    plt.plot(freqs, 10*np.log10(P + 1e-16))
    for fm in fmarks:
        plt.axvline(fm, linestyle='--')
    plt.xlim(0, 5.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.title(title)
    plt.tight_layout()


def plot_spectra(freqs: NDArray[np.float64], Ps: list[NDArray[np.float64]], title: str, fmarks: list[float] = [], labels: list[str] = []):
    plt.figure(figsize=(12, 6))
    if len(labels) < len(Ps):
        labels.extend([''] * (len(Ps) - len(labels)))
    for P, label in zip(Ps, labels):
        plt.plot(freqs, 10*np.log10(P + 1e-16), label=label, alpha=0.4)
    for fm in fmarks:
        plt.axvline(fm, linestyle='--', alpha=0.3, c='black')
    plt.xlim(0, 5.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_time(x, fs, title, showing_range: float = 10.):
    plt.figure()
    tt = np.arange(len(x))/fs
    keep = tt >= (tt[-1]-showing_range)
    plt.plot(tt[keep], x[keep].real if np.iscomplexobj(x) else x[keep])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title + f" (last {showing_range:.1f} s)")
    plt.tight_layout()


def plot_times(
        xs: list[NDArray[np.complex128]], fs: int, title: str, labels: list[str] = [],
        showing_range: float = 10.):
    plt.figure(figsize=(12, 4))
    tt = np.arange(len(xs[0]))/fs
    keep = tt >= (tt[-1]-showing_range)

    if len(labels) < len(xs):
        labels.extend([''] * (len(xs) - len(labels)))

    for x, label in zip(xs, labels):
        plt.plot(tt[keep], x[keep].real if np.iscomplexobj(x) else x[keep], label=label, alpha=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title + f" (last {showing_range:.1f} s)")
    plt.legend()
    plt.tight_layout()


# ---------- Build signals ----------
s_heart = np.exp(1j * (beta_H * np.sin(2*np.pi*f_H*t + phi_H)))
s_both  = np.exp(
    1j * (beta_R * np.sin(2*np.pi*f_R*t + phi_R) \
    + beta_H * np.sin(2*np.pi*f_H*t + phi_H)))

# DC cut
sH_hp = s_heart - np.mean(s_heart)
sB_hp = s_both - np.mean(s_both)

# Split
sH_pos, sH_neg = split_pos_neg(sH_hp, fs, abs_fmin=abs_fmin)
sB_pos, sB_neg = split_pos_neg(sB_hp, fs, abs_fmin=abs_fmin)

# Power sequences
pH_all = np.abs(sH_hp)**2
pH_pos = np.abs(sH_pos)**2
pH_neg = np.abs(sH_neg)**2
pH_sum  = pH_pos + pH_neg
pH_diff = pH_pos - pH_neg

pB_all = np.abs(sB_hp)**2
pB_pos = np.abs(sB_pos)**2
pB_neg = np.abs(sB_neg)**2
pB_sum = pB_pos + pB_neg
pB_diff = pB_pos - pB_neg

# ---------- Time plots ----------
showing_range = 15.0
plot_times(
    [normalize_signal(pH_all), normalize_signal(pH_diff), normalize_signal(pH_pos)], fs, "Heart-only",
    labels=['|s - DC|^2 (even-only ~ 2 f_H)', '|s_+|^2 - |s_-|^2 (odd-only; shows f_H)', '|s_+|^2'], showing_range=showing_range)
    # labels=['|s - DC|^2 (even-only ~ 2 f_H)', '|s_+|^2 - |s_-|^2 (odd-only; shows f_H)', '|s_+|^2 + |s_-|^2 (odd-only; shows f_H)'])
plot_times(
    [normalize_signal(pB_all), normalize_signal(pB_diff), normalize_signal(pB_pos)], fs, "Resp+Heart",
    labels=['|s - DC|^2 (even + odd-odd; 2 f_R, f_R±f_H)', '|s_+|^2 - |s_-|^2 (odd-only; f_R and f_H)', '|s_+|^2'], showing_range=showing_range)
    # labels=['|s - DC|^2 (even + odd-odd; 2 f_R, f_R±f_H)', '|s_+|^2 - |s_-|^2 (odd-only; f_R and f_H)', '|s_+|^2 + |s_-|^2 (odd-only; f_R and f_H)'])

# ---------- Spectra ----------
zp = 2
fH_all,  PH_all  = power_spectrum(normalize_signal(pH_all), fs, zp=zp)
fH_diff, PH_diff = power_spectrum(normalize_signal(pH_diff), fs, zp=zp)
fH_pos, PH_pos = power_spectrum(normalize_signal(pH_pos), fs, zp=zp)
fB_all,  PB_all  = power_spectrum(normalize_signal(pB_all), fs, zp=zp)
fB_diff, PB_diff = power_spectrum(normalize_signal(pB_diff), fs, zp=zp)
fB_pos, PB_pos = power_spectrum(normalize_signal(pB_pos), fs, zp=zp)

plot_spectra(
    fH_all, [PH_all, PH_diff, PH_pos], "Heart-only spectrum",
    fmarks=[f_H, 2*f_H, 3*f_H], labels=['|s - DC|^2', '|s_+|^2 - |s_-|^2', '|s_+|^2'])

plot_spectra(
    fB_all, [PB_all, PB_diff, PB_pos], "Resp+Heart spectrum",
    fmarks=[f_R, 2*f_R, f_H, abs(f_R-f_H), f_R+f_H], labels=['|s - DC|^2', '|s_+|^2 - |s_-|^2', '|s_+|^2'])
# plot_spectra(
#     fB_all, [PB_all, PB_diff], "Resp+Heart spectrum",
#     fmarks=[f_R, 2*f_R, f_H, abs(f_R-f_H), f_R+f_H], labels=['|s - DC|^2', '|s_+|^2 - |s_-|^2'])

plt.show()
