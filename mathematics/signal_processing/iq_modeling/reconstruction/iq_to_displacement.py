
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import hilbert
from scipy.special import jn


# ==== 共通の設定 ====
fs = 500                   # サンプリング周波数 [Hz]
t = np.linspace(0, 10, fs, endpoint=False)  # 時間軸
f1 = 1                       # 位相変調の基本周波数 [Hz]
f2 = 0.2
a = 0.0001  # 2.4048                  # 変調指数 (例：J0(a)=0 となる値)
b = 0.001
k = 500.

# IQ 信号を Jacobi–Anger 展開に基づく形で生成
# s(t) = exp[i * a * sin(2*pi*f*t)]
s = np.exp(1j * 2 * k * (a * np.sin(2*np.pi * f1 * t) + b * np.cos(2*np.pi * f2 * t)))
s = s * np.exp(1j * 2 * np.pi * 0.)
s_modified = s - np.mean(s)
# s = s - jn(0, 2 * k * a)
# s = s - np.mean(s)

# IQ 信号から瞬時位相を抽出 (unwrap して連続化)
theta_original = np.unwrap(np.angle(s))
theta = np.unwrap(np.angle(s_modified))
# 理想的には theta(t) = a*sin(2*pi*f*t) となる（平均は 0 です）
print("Mean of theta: {:.3f}".format(np.mean(theta)))

# === 方法 1: DC オフセット付加による方法 ===
theta0 = 10.0  # 適当な定数オフセット (例として1.0)
theta_offset = theta + theta0

# scipy.signal.hilbert を用いて解析信号を得る
# ※ ヒルベルト変換は線形で定数成分は 0 となるため、
#    実際の直交成分は H{theta_offset} = H{theta} となります。
analytic_theta_offset = cast(NDArray, hilbert(theta_offset))
analytic_theta_offset = analytic_theta_offset - complex(theta0, 0.)
hilbert_theta_method1 = np.real(analytic_theta_offset)

# === 方法 2: 周波数シフトによる方法 ===
# まず、位相信号に周波数シフトを掛ける
omega0 = 2 * np.pi * 1.0  # 例: 1 [Hz]分のシフト (rad/s)
# 周波数シフト: 時間領域での乗算により、
# theta_shift(t) = theta(t) * exp(i*omega0*t)
theta_shift = theta * np.exp(1j * omega0 * t)

# 複素信号にも適用できる FFT ベースの解析信号 (ヒルベルト変換) を自作します
def analytic_signal_complex(x):
    """
    入力 x: 長さ N の複素信号 (numpy 配列)
    標準的な FFT ベースの手法により、x の解析信号 (負の周波数成分を 0 にする) を返します。
    """
    N = len(x)
    X = np.fft.fft(x)
    H_filter = np.zeros(N)
    if N % 2 == 0:
        # N が偶数の場合
        H_filter[0] = 1
        H_filter[1:N//2] = 2
        H_filter[N//2] = 1
        H_filter[N//2+1:] = 0
    else:
        # N が奇数の場合
        H_filter[0] = 1
        H_filter[1:(N+1)//2] = 2
        H_filter[(N+1)//2:] = 0
    return np.fft.ifft(X * H_filter)

# theta_shift の解析信号を構成
theta_shift_analytic = analytic_signal_complex(theta_shift)
# その後、周波数シフトを打ち消す
theta_analytic_method2 = theta_shift_analytic * np.exp(-1j * omega0 * t)
# 得られた解析信号の虚部が、元の theta(t) のヒルベルト変換 H{theta(t)} です。
hilbert_theta_method2 = np.real(theta_analytic_method2)

# ==== プロットして比較 ====
plt.figure(figsize=(12, 8))

# (1) 元の位相信号 theta(t)
plt.subplot(3, 1, 1)
plt.plot(t, theta_original, label='Original $\\theta(t)$', color='black', alpha=0.5)
plt.plot(t, theta, label='$\\theta(t)$', c='C0', alpha=0.5)
plt.plot(t, analytic_theta_offset.real, label='(R) (Method 1)', c='orange', alpha=0.5, ls='--')
plt.plot(t, theta_analytic_method2.real, label='(R) (Method 2)', c='green', alpha=0.5, ls='--')
plt.title('instantaneous phase $\\theta(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Phase [rad]')
plt.legend()

# (2) 方法 1: DC オフセット付加によるヒルベルト変換
plt.subplot(3, 1, 2)
plt.plot(t, analytic_theta_offset.real, label='(R) (Method 1)', color='orange')
plt.plot(t, analytic_theta_offset.imag, label='(I) (Method 1)', color='orange', alpha=0.5)
plt.plot(t, np.abs(analytic_theta_offset), label='Envelope (Method 1)', color='black', alpha=0.5)
plt.title('Method 1: DC Offset')
plt.xlabel('Time [s]')
plt.ylabel('Hilbert Transform')
plt.legend()

# (3) 方法 2: 周波数シフトによるヒルベルト変換
plt.subplot(3, 1, 3)
plt.plot(t, theta_analytic_method2.real, label='(R) (Method 2)', color='green')
plt.plot(t, theta_analytic_method2.imag, label='(I) (Method 2)', color='green', alpha=0.5)
plt.plot(t, np.abs(theta_analytic_method2), label='Envelope (Method 2)', color='black', alpha=0.5)
plt.title('Method2: Phase Shift')
plt.xlabel('Time [s]')
plt.ylabel('Hilbert Transform')
plt.legend()

plt.tight_layout()
plt.show()
