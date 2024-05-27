
import numpy as np
import matplotlib.pyplot as plt

# サンプル信号を生成
# サンプル数
N = 500
# サンプリング周波数 (Hz)
fs = 500.0
T = 1.0 / fs
# 時間軸
x = np.linspace(0.0, N*T, N, endpoint=False)
# 信号生成: 50Hz の正弦波 + 80Hz の正弦波
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
y = np.cos(4. * 2.0*np.pi*x)

# 離散フーリエ変換を実行
yf = np.fft.fft(y)
# 周波数軸
xf = np.fft.fftfreq(N, T)[:N//2]

# 振幅スペクトルの計算
amplitude_spectrum = 2.0/N * np.abs(yf[:N//2])

# 結果をプロット
plt.figure(figsize=(12, 6))

# 時間領域の信号
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Time Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# 周波数領域の信号
plt.subplot(2, 1, 2)
plt.plot(xf, amplitude_spectrum)
plt.title("Frequency Domain Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
