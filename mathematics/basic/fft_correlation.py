
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate


def autocorrelation(x):
    """
    信号 x の自己相関を FFT を用いて計算する関数

    Parameters:
        x : 1次元の numpy.array
            対象となる信号

    Returns:
        ac : 1次元の numpy.array
            非負ラグにおける自己相関関数
    """
    # 信号を numpy 配列に変換し、ゼロ平均化
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)

    # 円相関（循環相関）を回避するため、十分な長さにゼロパディング
    # ここでは最小長さとして 2*n-1 を用い、効率のために次の2の冪乗に丸める
    nfft = 2 ** int(np.ceil(np.log2(2*n - 1)))

    # FFT を計算
    X = np.fft.fft(x, n=nfft)
    # パワースペクトルを求める（複素共役との積）
    S = X * X.conj()
    # 逆 FFT により自己相関を求める
    result = np.fft.ifft(S).real
    # result は長さ nfft の配列となっており、循環相関が計算されるので、
    # 正のラグ（0～n-1）の部分を取り出す
    ac = result[:n]

    # ラグが大きくなるとサンプル数が減るため、正規化を行う（オプション）
    norm = np.arange(n, 0, -1)
    ac = ac / norm

    return ac


# 使用例:
if __name__ == "__main__":
    # 例: サイン波に雑音を加えた信号
    t = np.linspace(0, 1, 2**12, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(t.size)
    signal_without_dc = signal - signal.mean()

    n_process = 100

    start_time = time.time()
    for _ in range(n_process):
        ac = autocorrelation(signal)
    print(f'numpy fft proces time: {time.time() - start_time:.3f} [sec]')

    start_time = time.time()
    for _ in range(n_process):
        np_ac = np.correlate(signal_without_dc, signal_without_dc, mode='same')
    print(f'numpy proces time: {time.time() - start_time:.3f} [sec]')

    start_time = time.time()
    for _ in range(n_process):
        sc_ac = correlate(signal_without_dc, signal_without_dc, mode='same', method='fft')
    print(f'scipy proces time: {time.time() - start_time:.3f} [sec]')

    print(ac.shape)
    print(np_ac.shape)
    print(sc_ac.shape)

    lags = np.arange(len(ac))
    plt.subplot(3, 1, 1)
    plt.stem(lags, ac)
    plt.title('Autocorrelation via FFT')

    plt.subplot(3, 1, 2)
    plt.stem(lags, np_ac)
    plt.title('Numpy Correlate')

    plt.subplot(3, 1, 3)
    plt.stem(lags, sc_ac)
    plt.title('Scipy Correlate (FFT)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.tight_layout()
    plt.show()
