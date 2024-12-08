
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def find_extrema(data: NDArray) -> tuple[NDArray, NDArray]:
    first_is_maxima = data[0] >= data[1]
    last_is_maxima = data[-1] >= data[-2]
    maxima, _ = signal.find_peaks(data)
    minima, _ = signal.find_peaks(-data)
    if first_is_maxima:
        maxima = np.concatenate(([0], maxima))
    else:
        minima = np.concatenate(([0], minima))
    if last_is_maxima:
        maxima = np.concatenate((maxima, [len(data) - 1]))
    else:
        minima = np.concatenate((minima, [len(data) - 1]))
    return maxima, minima


def interpolate_envelope(x, y, N):
    """
    局所的な極値をスプライン補間して包絡線を生成します。
    """
    if len(x) < 2:
        # 極値が2つ未満の場合、線形補間
        if len(x) == 1:
            return np.full(N, y[0])
        else:
            return np.zeros(N)
    cs = CubicSpline(x, y, bc_type='natural')
    return cs(np.arange(N))


def sift(signal, max_sifts=10, tol=1e-3):
    """
    スィフティングプロセスを実行し、IMFを抽出します。
    """
    imfs = []
    residue = signal.copy()
    number = len(signal)
    for _ in range(max_sifts):
        h = residue.copy()
        for _ in range(10):  # 各IMF抽出に対する最大スィフティング回数
            maxima, minima = find_extrema(h)
            if len(maxima) < 2 or len(minima) < 2:
                break  # スプライン補間が不可能な場合
            upper_env = interpolate_envelope(maxima, h[maxima], number)
            lower_env = interpolate_envelope(minima, h[minima], number)
            mean_env = (upper_env + lower_env) / 2
            prev_h = h.copy()
            h = h - mean_env
            # 収束判定
            if np.all(np.abs(h - prev_h) < tol):
                break
        imfs.append(h)
        residue = residue - h
        # 停止条件
        if np.all(np.abs(residue) < tol):
            break
    imfs.append(residue)
    return imfs


def ceemdan(signal, ensemble_size=100, noise_width=0.05, max_sifts=10, tol=1e-3):
    """
    CEEMDANを実行し、IMFを抽出します。

    Parameters:
    - signal: 分解対象の信号
    - ensemble_size: エンセムブルの数
    - noise_width: 追加するホワイトノイズの標準偏差の割合（信号の標準偏差に対する比率）
    - max_sifts: 各EMD実行における最大スィフティング回数
    - tol: スィフティングの収束基準

    Returns:
    - imfs: CEEMDANで抽出されたIMFのリスト
    """
    imfs = []
    residue = signal.copy()
    number = len(signal)

    counter = 0
    while True:
        imf_ensemble = []
        noise_std = noise_width * np.std(residue)
        for _ in range(ensemble_size):
            # ホワイトノイズの生成
            noise = np.random.normal(0, noise_std, number)
            noisy_signal = residue + noise
            # EMDの実行
            extracted_imfs = sift(noisy_signal, max_sifts=max_sifts, tol=tol)
            if len(extracted_imfs) == 0:
                continue
            imf_ensemble.append(extracted_imfs[0])  # 最初のIMFのみを抽出
        if not imf_ensemble:
            break
        # アンサンブル平均
        imf_avg = np.mean(imf_ensemble, axis=0)
        imfs.append(imf_avg)
        # 残差の更新
        residue = residue - imf_avg
        # 停止条件
        if np.std(residue) < tol:
            break
        # stop
        counter += 1
        print(counter)
        if counter >= max_sifts:
            break
    imfs.append(residue)
    return imfs


# サンプル信号の生成
t = np.linspace(0, 1, 1000)
# 例: 複数の周波数を持つ信号
data = (np.sin(2 * np.pi * 5 * t) +
          0.5 * np.sin(2 * np.pi * 20 * t) +
          0.2 * np.sin(2 * np.pi * 50 * t) +
          0.1 * np.random.randn(len(t)))  # ノイズを追加

# CEEMDANの適用
ensemble_size = 100  # エンセムブルの数
noise_width = 0.05    # ノイズ幅の設定
imfs = ceemdan(data, ensemble_size=ensemble_size, noise_width=noise_width)

# 結果のプロット
plt.figure(figsize=(12, 9))
plt.subplot(len(imfs)+1, 1, 1)
plt.plot(t, data, 'r')
plt.title("Original Signal")
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs)+1, 1, i+2)
    plt.plot(t, imf)
    plt.title(f"CEEMDAN IMF {i+1}")
plt.tight_layout()
plt.show()
