
import numpy as np
import matplotlib.pyplot as plt


def morlet_wavelet(t, omega0=5.0):
    """
    Morlet ウェーブレットの定義
    :param t: 時間配列
    :param omega0: 中心周波数
    :return: Morlet ウェーブレット
    """
    return (np.pi**(-0.25)) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)


def cwt(signal, scales, wavelet, dt=1.0):
    """
    連続ウェーブレット変換 (CWT)
    :param signal: 入力信号
    :param scales: スケールの配列
    :param wavelet: マザーウェーブレット関数
    :param dt: サンプリング間隔
    :return: CWT の結果
    """
    n = len(signal)
    t = np.arange(-n//2, n//2) * dt
    cwt_matrix = np.zeros((len(scales), n), dtype=complex)

    for idx, scale in enumerate(scales):
        # ウェーブレットのスケーリングと平行移動
        scaled_wavelet = wavelet(t / scale)
        # 畳み込み
        convolution = np.convolve(signal, scaled_wavelet[::-1], mode='same') * dt / np.sqrt(scale)
        cwt_matrix[idx, :] = convolution

    return cwt_matrix


# サンプル信号の生成（例: 複数の正弦波の合成）
fs = 500  # サンプリング周波数
t = np.arange(0, 1, 1/fs)
freq1 = 50  # 周波数1
freq2 = 120  # 周波数2
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# スケールの設定
scales = np.arange(1, 128)

# CWT の実行
cwt_result = cwt(signal, scales, lambda t: morlet_wavelet(t, omega0=5.0), dt=1/fs)

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(t, signal)

plt.subplot(212)
plt.imshow(
    np.abs(cwt_result), extent=(0., 1., float(scales[-1]), float(scales[0])),  # extent: 軸目盛り
    aspect='auto', cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.title('Continuous Wavelet Transform (Morlet)')
plt.colorbar(label='Magnitude')
plt.show()
