
import numpy as np
import matplotlib.pyplot as plt

# 信号パラメータ
fs = 1000  # サンプリング周波数 (Hz)
N = 2048  # 信号のサンプル数
t = np.arange(N) / fs  # 時間ベクトル

# 擬似信号の生成（50 Hz と 120 Hz の正弦波 + ノイズ）
freq1 = 50  # Hz
freq2 = 120  # Hz
amplitude1 = 1.0
amplitude2 = 0.5
noise_power = 0.2

signal = amplitude1 * np.sin(2 * np.pi * freq1 * t) + amplitude2 * np.sin(2 * np.pi * freq2 * t)
noise = np.random.normal(scale=np.sqrt(noise_power), size=t.shape)
x = signal + noise

# 全データに対してFFTを適用
X = np.fft.rfft(x)
Pxx_fft = (1 / (fs * N)) * np.abs(X) ** 2
Pxx_fft[1:-1] *= 2  # 単方向スペクトルへの変換
f_fft = np.fft.rfftfreq(N, 1 / fs)

# Welch 法のパラメータを設定
L = 512  # セグメントの長さ
Overlap = 256  # オーバーラップの長さ
window = np.hanning(L)  # ハニング窓

# セグメントの数を計算
K = (N - Overlap) // (L - Overlap)

# 各セグメントのフーリエ変換を計算し、パワースペクトルを求める
Pxx_welch = np.zeros(L // 2 + 1)
for k in range(K):
    start = k * (L - Overlap)
    end = start + L
    segment = x[start:end] * window
    Xk = np.fft.rfft(segment)
    Pxx_welch += np.abs(Xk) ** 2

# パワースペクトルを平均化し、正規化
Pxx_welch /= (K * L * np.sum(window ** 2))
f_welch = np.fft.rfftfreq(L, 1 / fs)

# 結果をプロット
plt.figure(figsize=(14, 8))

plt.semilogy(f_fft, Pxx_fft, label='fft')
plt.semilogy(f_welch, Pxx_welch, label='welch')
plt.grid()
plt.legend()
plt.show()
