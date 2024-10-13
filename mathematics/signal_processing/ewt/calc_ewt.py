
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d


def calculate_complex_ewt(signal, sampling_rate=1.0, tau=0.05):
    N = len(signal)
    # フーリエ変換と周波数軸の取得
    fft_signal = fft(signal)
    freqs = fftfreq(N, d=1.0 / sampling_rate)
    abs_fft_signal = np.abs(fft_signal)
    # スペクトルの平滑化（必要に応じて）
    abs_fft_signal_smoothed = gaussian_filter1d(abs_fft_signal, sigma=5)
    # スペクトルのピーク検出
    peaks, _ = find_peaks(abs_fft_signal_smoothed)
    boundaries = freqs[peaks]
    # 境界周波数を昇順にソート
    boundaries = np.sort(boundaries)
    # 境界周波数を正の周波数のみで設定
    boundaries = boundaries[boundaries > 0]
    # 始端と終端を追加
    boundaries = np.concatenate(([0], boundaries, [sampling_rate / 2]))
    # フィルタの構築と適用
    modes = []
    for n in range(len(boundaries) - 1):
        wn = boundaries[n]
        wn1 = boundaries[n + 1]
        filter = create_symmetric_filter(freqs, wn, wn1, tau)
        # フィルタの適用
        filtered_fft = fft_signal * filter
        mode = ifft(filtered_fft)
        modes.append(mode)
    return modes


def create_symmetric_filter(freqs, wn, wn1, tau):
    gamma = tau * (wn1 - wn)
    filter = np.zeros_like(freqs, dtype=np.float64)
    # 通過帯域（正・負の周波数）
    idx_pass = ((freqs >= wn + gamma) & (freqs <= wn1 - gamma)) | ((freqs <= -wn - gamma) & (freqs >= -wn1 + gamma))
    filter[idx_pass] = 1
    # 遷移帯域（正の周波数）
    idx_trans_up = (freqs >= wn - gamma) & (freqs < wn + gamma)
    filter[idx_trans_up] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_trans_up] - wn) / (2 * gamma)))
    idx_trans_down = (freqs > wn1 - gamma) & (freqs <= wn1 + gamma)
    filter[idx_trans_down] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_trans_down] - wn1) / (2 * gamma)))
    # 遷移帯域（負の周波数）
    idx_trans_up_neg = (freqs <= -wn + gamma) & (freqs > -wn - gamma)
    filter[idx_trans_up_neg] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_trans_up_neg] + wn) / (2 * gamma)))
    idx_trans_down_neg = (freqs < -wn1 + gamma) & (freqs >= -wn1 - gamma)
    filter[idx_trans_down_neg] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_trans_down_neg] + wn1) / (2 * gamma)))
    return filter
