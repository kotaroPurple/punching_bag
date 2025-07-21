import numpy as np
from scipy import special, signal
import matplotlib.pyplot as plt

def generate_iq_signal(k, d0, heart_rate_hz, duration, fs):
    """
    心拍に対応する単振動からのIQ信号を生成

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        振幅 (m)
    heart_rate_hz : float
        心拍数 (Hz)
    duration : float
        信号の長さ (s)
    fs : float
        サンプリング周波数 (Hz)

    Returns:
    --------
    t : array
        時間配列
    s : array
        IQ信号
    """
    t = np.arange(0, duration, 1/fs)
    omega = 2 * np.pi * heart_rate_hz

    # 単振動の変位
    d = d0 * np.sin(omega * t)

    # IQ信号
    s = np.exp(2j * k * d)

    return t, s

def estimate_heart_rate(s, fs, filter_signal=True, window_size=None):
    """
    IQ信号から心拍数を推定

    Parameters:
    -----------
    s : array
        IQ信号
    fs : float
        サンプリング周波数 (Hz)
    filter_signal : bool
        信号をフィルタリングするかどうか
    window_size : int or None
        窓関数のサイズ

    Returns:
    --------
    heart_rate_hz : float
        推定心拍数 (Hz)
    heart_rate_bpm : float
        推定心拍数 (bpm)
    freqs : array
        周波数配列
    spectrum : array
        スペクトル
    """
    # DC成分の除去
    s_dc = s - np.mean(s)

    # 強度計算
    intensity = np.abs(s_dc)**2

    # フィルタリング (0.5-4.0 Hz, 心拍数の範囲に対応)
    if filter_signal:
        b, a = signal.butter(4, [0.5, 4.0], btype='bandpass', fs=fs)  # type: ignore
        intensity = signal.filtfilt(b, a, intensity)

    # 窓関数の適用
    if window_size is not None:
        window = signal.windows.hann(window_size)
        intensity = intensity * window

    # FFTによるスペクトル計算
    spectrum = np.abs(np.fft.rfft(intensity))
    freqs = np.fft.rfftfreq(len(intensity), 1/fs)

    # 有効な周波数範囲 (0.5-4.0 Hz)
    valid_idx = (freqs >= 0.5) & (freqs <= 4.0)
    valid_spectrum = spectrum[valid_idx]
    valid_freqs = freqs[valid_idx]

    # 最大ピークの検出
    max_idx = np.argmax(valid_spectrum)
    peak_freq = valid_freqs[max_idx]

    # 心拍数の推定 (ピーク周波数の半分)
    heart_rate_hz = peak_freq / 2
    heart_rate_bpm = heart_rate_hz * 60

    return heart_rate_hz, heart_rate_bpm, freqs, spectrum

def add_noise(s, snr_db):
    """
    IQ信号にノイズを追加

    Parameters:
    -----------
    s : array
        IQ信号
    snr_db : float
        信号対雑音比 (dB)

    Returns:
    --------
    s_noisy : array
        ノイズ付きIQ信号
    """
    # 信号の電力
    signal_power = np.mean(np.abs(s)**2)

    # ノイズの電力
    noise_power = signal_power / (10**(snr_db/10))

    # 複素ガウスノイズの生成
    noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, len(s)) + 1j * np.random.normal(0, 1, len(s)))

    # ノイズ付き信号
    s_noisy = s + noise

    return s_noisy

def plot_results(t, s, intensity, freqs, spectrum, heart_rate_hz, heart_rate_bpm):
    """
    結果のプロット
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # IQ信号
    axs[0].plot(t, np.real(s), label='Real')
    axs[0].plot(t, np.imag(s), label='Imag')
    axs[0].set_title('IQ Signal')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    # 強度
    axs[1].plot(t, intensity)
    axs[1].set_title('Signal Intensity')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Intensity')
    axs[1].grid(True)

    # スペクトル
    axs[2].plot(freqs, spectrum)
    axs[2].set_title('Frequency Spectrum')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Amplitude')
    axs[2].axvline(x=heart_rate_hz*2, color='r', linestyle='--',
                   label=f'Peak: {heart_rate_hz*2:.2f} Hz')
    axs[2].legend()
    axs[2].grid(True)

    plt.suptitle(f'Heart Rate Estimation: {heart_rate_hz:.2f} Hz ({heart_rate_bpm:.1f} bpm)')
    plt.tight_layout()
    plt.show()

def main():
    # パラメータ設定
    k = 500  # 波数 (24GHz)
    d0 = 0.0001  # 振幅 (0.1mm)
    true_heart_rate_hz = 1.2  # 真の心拍数 (Hz) = 72 bpm

    # 信号生成
    fs = 100  # サンプリング周波数 (Hz)
    duration = 10  # 信号の長さ (s)
    t, s = generate_iq_signal(k, d0, true_heart_rate_hz, duration, fs)

    # ノイズ付き信号
    s_noisy = add_noise(s, 10)  # SNR = 10dB

    # 心拍数推定
    heart_rate_hz, heart_rate_bpm, freqs, spectrum = estimate_heart_rate(s_noisy, fs)

    # DC成分を除いた強度
    s_dc = s_noisy - np.mean(s_noisy)
    intensity = np.abs(s_dc)**2

    # 結果のプロット
    plot_results(t, s_noisy, intensity, freqs, spectrum, heart_rate_hz, heart_rate_bpm)

    print(f"真の心拍数: {true_heart_rate_hz:.2f} Hz ({true_heart_rate_hz*60:.1f} bpm)")
    print(f"推定心拍数: {heart_rate_hz:.2f} Hz ({heart_rate_bpm:.1f} bpm)")
    print(f"誤差: {abs(heart_rate_hz - true_heart_rate_hz):.4f} Hz ({abs(heart_rate_bpm - true_heart_rate_hz*60):.1f} bpm)")

if __name__ == "__main__":
    main()