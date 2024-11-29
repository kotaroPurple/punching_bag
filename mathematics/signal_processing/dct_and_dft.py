
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, fft

# サンプル信号の作成
N = 128          # サンプル数
fs = 100         # サンプリング周波数 [Hz]
T = 1.0 / fs     # サンプリング間隔 [秒]
t = np.arange(N) * T  # 時間軸 [秒]
# 低周波成分（1Hz相当）と高周波成分（10Hz相当）の合成信号
f_low = 1
f_high = 20
signal = np.cos(2 * np.pi * f_low * t) + 0.5 * np.cos(2 * np.pi * f_high * t)

# DCTの適用（DCT-IIを使用）
dct_coefficients = dct(signal, type=2, norm='ortho')

# DFTの適用
fft_coefficients = fft(signal)
# DFTの絶対値を取って正規化
fft_magnitude = np.abs(fft_coefficients) / N

# 周波数軸の作成
freqs = np.fft.fftfreq(N, d=1.0/fs)  # サンプリング間隔d=1と仮定

# DCTの周波数軸（0からNyquist周波数まで）
# DCTの周波数軸（0からNyquist周波数まで）
dct_freqs = np.linspace(0, fs/2, N)

# DFTの正の周波数部分のみを取得
positive_freqs = freqs[:N//2]
positive_fft_magnitude = fft_magnitude[:N//2] + fft_magnitude[N//2:][::-1]

# DCTの絶対値
dct_magnitude = np.abs(dct_coefficients)

# プロットの作成
plt.figure(figsize=(14, 6))

# 元の信号のプロット
plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title('original')

# DCTのスペクトル
plt.subplot(2, 2, 2)
plt.stem(dct_freqs, dct_magnitude, basefmt=" ")  # use_line_collection=True)
plt.title('DCT')
plt.xlabel('Normalized Freq.')
plt.ylabel('abs DCT')
# plt.xlim(0, 0.5)

# DFTのスペクトル
plt.subplot(2, 2, 4)
plt.stem(positive_freqs, positive_fft_magnitude[:N//2], basefmt=" ")  # , use_line_collection=True)
plt.title('DFT')
plt.xlabel('freq')
plt.ylabel('abs DFT')
# plt.xlim(0, 0.5)

# DCTとDFTのスペクトルを比較するための拡大図
plt.subplot(2, 2, 3)
plt.stem(dct_freqs, dct_magnitude, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='DCT')  # , use_line_collection=True)
plt.stem(positive_freqs, positive_fft_magnitude[:N//2], linefmt='C1-', markerfmt='C1x', basefmt=" ", label='DFT')  # , use_line_collection=True)
plt.title('DCT vs DFT')
plt.xlabel('Normalized Freq.')
plt.ylabel('abs')
plt.legend()
# plt.xlim(0, 15/N)
plt.tight_layout()
plt.show()
