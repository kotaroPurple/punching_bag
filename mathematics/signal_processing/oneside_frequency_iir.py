
# one_sided_iir.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal, optimize


def spectrum(x, fs, nfft=8192):
    X = np.fft.fftshift(np.fft.fft(x, nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))
    mag = np.abs(X) + 1e-12
    return f, 20*np.log10(mag/mag.max())


def _a_from_fc(fc: float, fs: float) -> float:
    """Analog all-pass A(s)=(s-w0)/(s+w0) -> bilinear -> H(z)=(a+z^-1)/(1+a z^-1)."""
    k = np.tan(np.pi * float(fc) / float(fs))
    a = (1 - k) / (1 + k)
    return float(np.clip(a, -0.999, 0.999))  # stay strictly inside unit circle


def _allpass_freqz(a: float, w: np.ndarray) -> np.ndarray:
    z = np.exp(1j * w)
    return (a + z**-1) / (1 + a * z**-1)


def _cascade_freqz(a_list: np.ndarray, w: np.ndarray) -> np.ndarray:
    H = np.ones_like(w, dtype=complex)
    for a in a_list:
        H *= _allpass_freqz(float(a), w)
    return H


def _fit_allpass_pair(
    fs: float, f1: float, f2: float, nsec: int, wpts: int = 512, init_ratio: float = np.sqrt(2.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    最小二乗で 2 本のオールパス群 (a0, a1) を設計：
    Δ位相 = ∠H1 - ∠H0 ≈ +90° （[f1,f2] で）。
    """
    # 初期値：帯域を等比配置、片方を少しシフト
    f0 = np.geomspace(f1, f2, nsec)
    f1s = np.clip(f0 * init_ratio, f1 * 1.05, f2 / 1.05)
    a0 = np.array([_a_from_fc(fc, fs) for fc in f0], dtype=float)
    a1 = np.array([_a_from_fc(fc, fs) for fc in f1s], dtype=float)

    w = np.linspace(2 * np.pi * f1 / fs, 2 * np.pi * f2 / fs, wpts)
    target = -(np.pi / 2) * np.ones_like(w)
    x0 = np.concatenate([a0, a1])

    def resid(x):
        a0x, a1x = x[:nsec], x[nsec:]
        H0 = _cascade_freqz(a0x, w)
        H1 = _cascade_freqz(a1x, w)
        dphi = np.unwrap(np.angle(H1) - np.angle(H0))
        return dphi - target

    lb = -0.995 * np.ones_like(x0)
    ub = +0.995 * np.ones_like(x0)
    res = optimize.least_squares(resid, x0, bounds=(lb, ub), max_nfev=4000)
    return res.x[:nsec].astype(float), res.x[nsec:].astype(float)


@dataclass
class OneSidedIIR:
    """
    IIR 片側通過フィルタ（正 or 負）。ストリーミング対応（内部状態保持）。

    y_pos ≈ 0.5*(H0*x + j*H1*x)
    y_neg ≈ 0.5*(H0*x - j*H1*x)
    """
    fs: float
    band: tuple[float, float] = (0.3, 10.0)  # 設計帯域（例：ドップラーIQは ≤10 Hz）
    nsec: int = 6                            # 1次オールパス段数（片経路）
    mode: str = "pos"                        # "pos" or "neg"
    design: str = "fit"                      # "fit"（推奨） or "geometric"
    wpts: int = 512                          # 設計・診断の周波数サンプル

    def __post_init__(self):
        self.set_mode(self.mode)
        # self._design_filters()
        self.reset_state()

    # ---------- design ----------
    def _design_filters(self):
        f1, f2 = self.band
        if self.design == "fit":
            self.a0, self.a1 = _fit_allpass_pair(self.fs, f1, f2, self.nsec, self.wpts)
        elif self.design == "geometric":
            f0 = np.geomspace(f1, f2, self.nsec)
            f1s = np.clip(f0 * np.sqrt(2.0), f1 * 1.05, f2 / 1.05)
            self.a0 = np.array([_a_from_fc(fc, self.fs) for fc in f0], dtype=float)
            self.a1 = np.array([_a_from_fc(fc, self.fs) for fc in f1s], dtype=float)
        else:
            raise ValueError("design must be 'fit' or 'geometric'")
        # 係数（b,a）を事前生成
        self._ba0 = [(np.array([a, 1.0], float), np.array([1.0, a], float)) for a in self.a0]
        self._ba1 = [(np.array([a, 1.0], float), np.array([1.0, a], float)) for a in self.a1]

    # ---------- mode ----------
    def set_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ("pos", "neg"):
            raise ValueError("mode must be 'pos' or 'neg'")
        self._sgn = +1.0 if mode == "pos" else -1.0
        self.mode = mode
        self._design_filters()

    # ---------- state ----------
    def reset_state(self):
        # 1次IIRなので各段 zi は1サンプル。複素対応。
        self._zi0 = [np.zeros(1, dtype=complex) for _ in range(self.nsec)]
        self._zi1 = [np.zeros(1, dtype=complex) for _ in range(self.nsec)]

    # ---------- processing ----------
    def _cascade_lfilter(self, x: np.ndarray, ba_list, zi_list):
        y = x
        for i, (b, a) in enumerate(ba_list):
            y, zi_list[i] = signal.lfilter(b, a, y, zi=zi_list[i])
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: 1-D 複素IQ。チャンク処理可（状態保持）。"""
        x = np.asarray(x).astype(complex, copy=False)
        y0 = self._cascade_lfilter(x, self._ba0, self._zi0)
        y1 = self._cascade_lfilter(x, self._ba1, self._zi1)
        return 0.5 * (y0 + (1j * self._sgn) * y1)

    # ---------- analysis ----------
    def response(self, nfft: int = 4096):
        """複素周波数応答を返す: (freq_Hz, H_complex)"""
        f = np.linspace(-0.5 * self.fs, 0.5 * self.fs, nfft, endpoint=False)
        w = 2 * np.pi * f / self.fs
        H0 = _cascade_freqz(self.a0, w)
        H1 = _cascade_freqz(self.a1, w)
        H = 0.5 * (H0 + (1j * self._sgn) * H1)
        return f, H

    def phase_diff(self, npts: int = 256):
        """設計帯域内の Δ位相（deg）: (freq_Hz, Δphase_deg)"""
        f1, f2 = self.band
        f = np.linspace(f1, f2, npts)
        w = 2 * np.pi * f / self.fs
        H0 = _cascade_freqz(self.a0, w)
        H1 = _cascade_freqz(self.a1, w)
        dphi = np.unwrap(np.angle(H1) - np.angle(H0))
        return f, np.rad2deg(dphi)


# Desired positive tones at 1.5 Hz and 6 Hz, plus mirror at -1.5 Hz (image)
fs = 100.0
total_time = 10.
t = np.arange(int(total_time*fs))/fs  # 30 seconds
x = 0.8*np.exp(1j*2*np.pi*1.5*t) + 0.5*np.exp(1j*2*np.pi*6.0*t) + 0.5*np.exp(-1j*2*np.pi*3.0*t)
x += 0.01*(np.random.default_rng(0).standard_normal(len(t)) + 1j*np.random.default_rng(1).standard_normal(len(t)))

flt = OneSidedIIR(fs, band=(0.3, 10.0), nsec=6, mode="pos", design="fit")
y_pos = flt.process(x)

flt.set_mode("neg")
y_neg = flt.process(x)

# plt.plot(t, y_pos.real, alpha=0.5)
# plt.plot(t, y_pos.imag, alpha=0.5)
# plt.show()

# Spectra
f_x, X_db = spectrum(x, fs, 32768)
f_x, Y_db = spectrum(y_pos, fs, 32768)
f_x, Y_neg_db = spectrum(y_neg, fs, 32768)

# plt.figure(figsize=(7, 4))
# plt.plot(f_x, X_db, alpha=0.4)
# plt.plot(f_x, Y_db, alpha=0.4)
# plt.plot(f_x, Y_neg_db, alpha=0.4)
# plt.xlim(-15, 15)
# plt.ylim(-80, 3)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.title("Before: baseband IQ (≤10 Hz)")
# plt.grid(True)
# plt.show()



# ------- パラメータ -------
FPS = 20
DURATION = len(t) / fs

# ------- 時間軸と位相 -------
frames = int(FPS * DURATION)

# ------- 図の準備 -------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4))
for ax, title in [
        (axL, "Original"), (axR, "Positive: Blue, Negative: Red")]:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)

# 正側のアーティスト
gt_pos,  = axL.plot([], [], marker="o", markersize=5, c='gray')

# 負側のアーティスト
tip_pos,  = axR.plot([], [], marker="o", markersize=5, c='blue')
tip_neg,  = axR.plot([], [], marker="o", markersize=5, c='red')

artists = (gt_pos, tip_pos, tip_neg)

# ------- アニメーション関数 -------
def init():
    gt_pos.set_data([], [])
    tip_pos.set_data([], [])
    tip_neg.set_data([], [])
    return artists


def update(i):
    # 正（+ω）
    gt_x, gt_y = x[i].real, x[i].imag
    x_p = y_pos[i].real
    y_p = y_pos[i].imag
    gt_pos.set_data([gt_x], [gt_y])
    tip_pos.set_data([x_p], [y_p])

    # 負
    x_n = y_neg[i].real
    y_n = y_neg[i].imag
    tip_neg.set_data([x_n], [y_n])

    return artists

# ------- 実行 -------
anim = FuncAnimation(
    fig, update, frames=frames, init_func=init,
    interval=1000 / FPS, blit=True, repeat=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
    # anim.save('./test.gif')
