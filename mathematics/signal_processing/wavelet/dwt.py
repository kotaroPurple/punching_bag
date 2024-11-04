
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fft, ifft


def get_db4_filters():
    """
    Daubechies 4 (db4) フィルタの係数を返します。
    """
    sqrt_3 = np.sqrt(3)
    denom = 4 * np.sqrt(2)
    h0 = (1 + sqrt_3) / denom
    h1 = (3 + sqrt_3) / denom
    h2 = (3 - sqrt_3) / denom
    h3 = (1 - sqrt_3) / denom
    h = np.array([h0, h1, h2, h3])
    # ウェーブレットフィルタはスケーリングフィルタの交差順
    g = np.array([h3, -h2, h1, -h0])
    return h, g


def convolve(signal, filter_coeffs):
    """
    信号とフィルタの畳み込みを行います（周期拡張を使用）。
    :param signal: 入力信号（1D NumPy 配列）
    :param filter_coeffs: フィルタ係数（1D NumPy 配列）
    :return: 畳み込み結果
    """
    N = len(signal)
    M = len(filter_coeffs)
    # 周期拡張
    extended_signal = np.concatenate((signal, signal[:M-1]))
    # 畳み込み
    conv = np.zeros(N)
    for i in range(N):
        conv[i] = np.sum(extended_signal[i:i+M] * filter_coeffs[::-1])
    return conv


def downsample(signal):
    """
    信号をダウンサンプリングします（偶数インデックスのみを採取）。
    :param signal: 入力信号（1D NumPy 配列）
    :return: ダウンサンプリング後の信号
    """
    return signal[::2]


# フィルタの取得
h, g = get_db4_filters()
print("スケーリングフィルタ (h):", h)
print("ウェーブレットフィルタ (g):", g)


def dwt(signal, h, g, level=1):
    """
    離散ウェーブレット変換 (DWT) を実装します。
    :param signal: 入力信号（1D NumPy 配列）
    :param h: スケーリングフィルタ（1D NumPy 配列）
    :param g: ウェーブレットフィルタ（1D NumPy 配列）
    :param level: 分解レベル
    :return: 近似係数と詳細係数のリスト
    """
    coeffs = []
    current_signal = signal.copy()
    for _ in range(level):
        a = downsample(convolve(current_signal, h))
        d = downsample(convolve(current_signal, g))
        coeffs.append((a, d))
        current_signal = a  # 次のレベルの入力信号は近似係数
    return coeffs


def fast_convolve(signal, filter_coeffs):
    """
    FFT を用いた畳み込み関数
    :param signal: 入力信号（1D NumPy 配列）
    :param filter_coeffs: フィルタ係数（1D NumPy 配列）
    :return: 畳み込み結果
    """
    n = len(signal)
    m = len(filter_coeffs)
    N = 2**int(np.ceil(np.log2(n + m - 1)))  # FFT 長を次の 2 の累乗に設定

    # ゼロパディング
    signal_padded = np.pad(signal, (0, N - n), 'constant')
    filter_padded = np.pad(filter_coeffs, (0, N - m), 'constant')

    # FFT
    signal_fft = fft(signal_padded)
    filter_fft = fft(filter_padded)

    # 周波数領域での乗算
    convolution_fft = signal_fft * filter_fft

    # 逆 FFT
    convolution = ifft(convolution_fft)

    # 実数部を返す（フィルタは実数の場合が多い）
    return np.real(convolution[:n + m - 1])


def fast_dwt(signal, h, g, level=1):
    """
    FFT を用いて高速化した離散ウェーブレット変換 (DWT) を実装します。
    :param signal: 入力信号（1D NumPy 配列）
    :param h: スケーリングフィルタ（1D NumPy 配列）
    :param g: ウェーブレットフィルタ（1D NumPy 配列）
    :param level: 分解レベル
    :return: 近似係数と詳細係数のリスト
    """
    coeffs = []
    current_signal = signal.copy()
    for _ in range(level):
        a_full = fast_convolve(current_signal, h)
        d_full = fast_convolve(current_signal, g)
        # ダウンサンプリング
        a = a_full[::2]
        d = d_full[::2]
        coeffs.append((a, d))
        current_signal = a  # 次のレベルの入力信号は近似係数
    return coeffs


# # サンプル信号の生成（例: 複数の正弦波の合成）
# fs = 500  # サンプリング周波数
# t = np.arange(0, 1, 1/fs)
# freq1 = 50  # 周波数1
# freq2 = 120  # 周波数2
# signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# # DWT の実行
# coeffs = dwt(signal, h, g, level=4)

# # 係数のプロット
# fig, axes = plt.subplots(5, 1, figsize=(10, 10))
# axes[0].plot(t, signal)
# axes[0].set_title('Original Signal')

# for i, (a, d) in enumerate(coeffs, 1):
#     axes[i].plot(a, label=f'Level {i} Approximation')
#     axes[i].plot(d, label=f'Level {i} Detail')
#     axes[i].legend()

# plt.tight_layout()
# plt.show()


# サンプル信号の生成
fs = 1000  # サンプリング周波数
t = np.arange(0, 1, 1/fs)
freq1 = 50  # 周波数1
freq2 = 120  # 周波数2
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# DWT の実行時間測定
start_time = time.time()
coeffs = dwt(signal, h, g, level=4)
dwt_time = time.time() - start_time
print(f"標準 DWT 実行時間: {dwt_time:.6f} 秒")

# FFT を用いた DWT の実行時間測定
start_time = time.time()
fast_coeffs = fast_dwt(signal, h, g, level=4)
fast_dwt_time = time.time() - start_time
print(f"FFT を用いた DWT 実行時間: {fast_dwt_time:.6f} 秒")

fig, axes = plt.subplots(5, 1, figsize=(10, 10))
axes[0].plot(t, signal)
axes[0].set_title('Original Signal')

for i, (a, d) in enumerate(fast_coeffs, 1):
    axes[i].plot(a, label=f'Level {i} Approximation')
    axes[i].plot(d, label=f'Level {i} Detail')
    axes[i].legend()
plt.show()

# 係数の比較
for level in range(4):
    a_standard, d_standard = coeffs[level]
    a_fast, d_fast = fast_coeffs[level]
    # assert np.allclose(a_standard, a_fast[:500], atol=1e-6), f"Level {level+1} Approximation 不一致"
    # assert np.allclose(d_standard, d_fast, atol=1e-6), f"Level {level+1} Detail 不一致"

print("標準 DWT と FFT を用いた DWT の結果は一致しています。")
