import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def simplified_intensity(k, d0, omega, t, max_terms=3):
    """
    ベッセル関数の性質を利用して簡略化したIQ波形の強度変化を計算

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        振幅 (m)
    omega : float
        角周波数 (rad/s)
    t : array
        時間配列 (s)
    max_terms : int
        考慮する項の数

    Returns:
    --------
    intensity : array
        簡略化した強度変化
    """
    # 小さな引数に対するベッセル関数の近似
    if 2*k*d0 < 0.3:  # 十分小さい場合は近似式を使用
        J0 = 1 - (k*d0)**2
        # J2 = (k*d0)**2 / 2

        # 簡略化した強度変化（定数項と2ω項のみ）
        intensity = 2*(k*d0)**2 - 2*(k*d0)**2 * np.cos(2*omega*t)
    else:
        # 有限項での計算
        J0 = special.jv(0, 2*k*d0)
        intensity = 1 - J0**2

        for n in range(1, max_terms+1):
            J2n = special.jv(2*n, 2*k*d0)
            intensity -= 4*J0*J2n*np.cos(2*n*omega*t)

    return intensity

def simplified_positive_freq_intensity(k, d0, omega, t, max_terms=3):
    """
    ベッセル関数の性質を利用して簡略化した片側周波数の強度変化を計算

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        振幅 (m)
    omega : float
        角周波数 (rad/s)
    t : array
        時間配列 (s)
    max_terms : int
        考慮する項の数

    Returns:
    --------
    intensity : array
        簡略化した強度変化
    """
    # 小さな引数に対するベッセル関数の近似
    if 2*k*d0 < 0.3:  # 十分小さい場合は近似式を使用
        # 簡略化した強度変化（定数項とω項のみ）
        intensity = 1 + 2*k*d0*np.cos(omega*t)
    else:
        # 有限項での計算
        intensity = 0

        # 定数項
        for n in range(max_terms+1):
            Jn = special.jv(n, 2*k*d0)
            intensity += Jn**2

        # 周波数成分
        for k_idx in range(1, max_terms+1):
            coef = 0
            for n in range(max_terms+1-k_idx):
                Jn = special.jv(n, 2*k*d0)
                Jnk = special.jv(n+k_idx, 2*k*d0)
                coef += Jn * Jnk
            intensity += 2 * coef * np.cos(k_idx*omega*t)

    return intensity

def estimate_heart_rate_simplified(k, d0, heart_rate_hz, duration, fs, snr_db=10, use_positive_freq=False):
    """
    簡略化した計算を用いて心拍数を推定

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        振幅 (m)
    heart_rate_hz : float
        真の心拍数 (Hz)
    duration : float
        信号の長さ (s)
    fs : float
        サンプリング周波数 (Hz)
    snr_db : float
        信号対雑音比 (dB)
    use_positive_freq : bool
        片側周波数を使用するかどうか

    Returns:
    --------
    estimated_hr : float
        推定心拍数 (Hz)
    """
    t = np.arange(0, duration, 1/fs)
    omega = 2 * np.pi * heart_rate_hz

    # 簡略化した強度変化を計算
    if use_positive_freq:
        intensity = simplified_positive_freq_intensity(k, d0, omega, t)
        freq_factor = 1  # 片側周波数の場合、検出周波数がそのまま心拍数
    else:
        intensity = simplified_intensity(k, d0, omega, t)
        freq_factor = 2  # 全周波数の場合、検出周波数の半分が心拍数

    # ノイズ追加
    noise_power = np.var(intensity) / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(intensity))
    noisy_intensity = intensity + noise

    # スペクトル計算
    spectrum = np.abs(np.fft.rfft(noisy_intensity))
    freqs = np.fft.rfftfreq(len(noisy_intensity), 1/fs)

    # 有効な周波数範囲 (0.5-4.0 Hz)
    valid_idx = (freqs >= 0.5) & (freqs <= 4.0)
    valid_spectrum = spectrum[valid_idx]
    valid_freqs = freqs[valid_idx]

    # 最大ピークの検出
    max_idx = np.argmax(valid_spectrum)
    peak_freq = valid_freqs[max_idx]

    # 心拍数の推定
    estimated_hr = peak_freq / freq_factor

    return estimated_hr, freqs, spectrum

def compare_methods(k, d0, heart_rate_hz, duration=10, fs=100, snr_db=10):
    """
    異なる計算方法を比較
    """
    t = np.arange(0, duration, 1/fs)
    omega = 2 * np.pi * heart_rate_hz

    # 1. 全周波数の強度変化（簡略化）
    intensity_full = simplified_intensity(k, d0, omega, t)
    hr_full, freqs_full, spectrum_full = estimate_heart_rate_simplified(
        k, d0, heart_rate_hz, duration, fs, snr_db, use_positive_freq=False)

    # 2. 片側周波数の強度変化（簡略化）
    intensity_pos = simplified_positive_freq_intensity(k, d0, omega, t)
    hr_pos, freqs_pos, spectrum_pos = estimate_heart_rate_simplified(
        k, d0, heart_rate_hz, duration, fs, snr_db, use_positive_freq=True)

    # 結果表示
    print(f"真の心拍数: {heart_rate_hz:.2f} Hz ({heart_rate_hz*60:.1f} bpm)")
    print(f"全周波数による推定: {hr_full:.2f} Hz ({hr_full*60:.1f} bpm), 誤差: {abs(hr_full-heart_rate_hz):.4f} Hz")
    print(f"片側周波数による推定: {hr_pos:.2f} Hz ({hr_pos*60:.1f} bpm), 誤差: {abs(hr_pos-heart_rate_hz):.4f} Hz")

    # プロット
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 強度変化
    axs[0, 0].plot(t[:200], intensity_full[:200])
    axs[0, 0].set_title('全周波数の強度変化')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Intensity')
    axs[0, 0].grid(True)

    axs[0, 1].plot(t[:200], intensity_pos[:200])
    axs[0, 1].set_title('片側周波数の強度変化')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Intensity')
    axs[0, 1].grid(True)

    # スペクトル
    axs[1, 0].plot(freqs_full, spectrum_full)
    axs[1, 0].axvline(x=heart_rate_hz*2, color='r', linestyle='--',
                     label=f'Expected: {heart_rate_hz*2:.2f} Hz')
    axs[1, 0].axvline(x=hr_full*2, color='g', linestyle='--',
                     label=f'Detected: {hr_full*2:.2f} Hz')
    axs[1, 0].set_title('全周波数のスペクトル')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(freqs_pos, spectrum_pos)
    axs[1, 1].axvline(x=heart_rate_hz, color='r', linestyle='--',
                     label=f'Expected: {heart_rate_hz:.2f} Hz')
    axs[1, 1].axvline(x=hr_pos, color='g', linestyle='--',
                     label=f'Detected: {hr_pos:.2f} Hz')
    axs[1, 1].set_title('片側周波数のスペクトル')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return hr_full, hr_pos

def main():
    # パラメータ設定
    k = 500  # 波数 (24GHz)
    d0 = 0.0001  # 振幅 (0.1mm)
    heart_rate_hz = 1.2  # 心拍数 (Hz) = 72 bpm

    # 異なる計算方法の比較
    hr_full, hr_pos = compare_methods(k, d0, heart_rate_hz)

    # 振幅の影響を調査
    d0_values = np.logspace(-5, -3, 5)  # 0.01mm から 1mm まで
    hr_errors_full = []
    hr_errors_pos = []

    for d in d0_values:
        hr_f, hr_p = compare_methods(k, d, heart_rate_hz)
        hr_errors_full.append(abs(hr_f - heart_rate_hz))
        hr_errors_pos.append(abs(hr_p - heart_rate_hz))

    # 誤差のプロット
    plt.figure(figsize=(10, 6))
    plt.loglog(d0_values*1000, hr_errors_full, 'o-', label='全周波数')
    plt.loglog(d0_values*1000, hr_errors_pos, 's-', label='片側周波数')
    plt.xlabel('振幅 (mm)')
    plt.ylabel('心拍数推定誤差 (Hz)')
    plt.title('振幅と心拍数推定誤差の関係')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()