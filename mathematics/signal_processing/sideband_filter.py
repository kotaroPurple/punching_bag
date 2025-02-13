
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz
from numpy.typing import NDArray


def extract_band_positive(
        data: NDArray, fs: int, f_low: float, f_high: float, order: int = 4) -> NDArray:
    # 時刻軸の生成
    t = np.arange(len(data)) / fs

    # 帯域の中心周波数と半帯域幅の計算
    fc = (f_low + f_high) / 2.0

    # (1) 搬送波による周波数シフト：対象帯域をベースバンドに移動
    shifted = data * np.exp(-1j * 2 * np.pi * fc * t)

    # (2) 低域通過 IIR フィルタの設計：カットオフ周波数を半帯域幅に設定
    sos = _make_filter(fs, f_low, f_high, order)

    # フィルタ適用
    filtered = sosfiltfilt(sos, shifted)

    # (3) 逆搬送波で元の周波数位置に戻す
    result = filtered * np.exp(1j * 2 * np.pi * fc * t)
    return result


def sideband_freqz(fs: int, f_low: float, f_high: float, order: int):
    sos = _make_filter(fs, f_low, f_high, order)
    w, h = sosfreqz(sos, worN=2000, fs=fs)
    w2 = np.r_[-w[::-1][:-1], w]
    h2 = np.r_[h[::-1][:-1], h]
    w2 = w2 + (f_low + f_high) / 2.0
    return w2, h2


def _make_filter(fs: int, f_low: float, f_high: float, order: int):
    nyquist = fs / 2.0
    bw_half = (f_high - f_low) / 2.0
    norm_cutoff = bw_half / nyquist  # 正規化カットオフ周波数
    sos = butter(order, norm_cutoff, btype='lowpass', output='sos')
    return sos


# --- 使用例 ---
if __name__ == '__main__':
    fs = 2000  # サンプリング周波数 [Hz]
    t = np.arange(0, 1.0, 1/fs)

    # 合成信号：100 Hz, 300 Hz, 500 Hz の成分を含む実信号
    sig = np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*300*t) + 0.3*np.sin(2*np.pi*500*t)

    # 例：300 Hz ± 50 Hz の帯域（250 Hz～350 Hz）の成分を抽出する
    f_low = 250
    f_high = 350
    band_extracted = extract_band_positive(sig, fs, f_low, f_high, order=4)

    # freqz
    w, h = sideband_freqz(fs, f_low, f_high, order=4)

    # 結果のスペクトルプロット
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))

    # 元の信号のスペクトル
    X_orig = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1/fs))
    axs[0, 0].plot(freqs, np.abs(X_orig))
    axs[0, 0].set_title("Original Signal Spectrum")
    axs[0, 0].set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel("Magnitude")

    # 抽出後の信号のスペクトル
    X_band = np.fft.fftshift(np.fft.fft(band_extracted))
    axs[0, 1].plot(freqs, np.abs(X_band))
    axs[0, 1].set_title("Extracted Band ({}-{} Hz) Spectrum".format(f_low, f_high))
    axs[0, 1].set_xlabel("Frequency [Hz]")

    axs[1, 0].plot(w, 20 * np.log10(np.abs(h) + 1e-6))

    plt.tight_layout()
    plt.show()
