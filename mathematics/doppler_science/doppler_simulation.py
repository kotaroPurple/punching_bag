import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def simulate_iq_waveform(k, d0, omega, t, max_n=10):
    """
    単振動する物体からのIQ波形をシミュレーション

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
    max_n : int
        ベッセル関数の最大次数

    Returns:
    --------
    s : array
        IQ波形
    """
    s = np.zeros(len(t), dtype=complex)

    # ベッセル関数の計算
    for n in range(-max_n, max_n+1):
        J_n = special.jv(n, 2*k*d0)
        s += J_n * np.exp(1j * n * omega * t)

    return s

def calculate_intensity(s, dc_removed=False, J0=None):
    """
    IQ波形の強度を計算

    Parameters:
    -----------
    s : array
        IQ波形
    dc_removed : bool
        DCを除去するかどうか
    J0 : float
        J_0(2kd0)の値

    Returns:
    --------
    intensity : array
        強度
    """
    if dc_removed and J0 is not None:
        # DCを除去した強度
        s_prime = s - J0
        intensity = np.abs(s_prime)**2
    else:
        # 通常の強度
        intensity = np.abs(s)**2

    return intensity

def calculate_positive_freq_intensity(k, d0, omega, t, max_n=10, remove_dc=False, remove_up_to=0):
    """
    片側周波数のIQ波形の強度を計算

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
    max_n : int
        ベッセル関数の最大次数
    remove_dc : bool
        DCを除去するかどうか
    remove_up_to : int
        n < remove_up_toまでの周波数成分を除去

    Returns:
    --------
    s_plus : array
        片側周波数のIQ波形
    intensity : array
        強度
    """
    s_plus = np.zeros(len(t), dtype=complex)

    # 片側周波数のIQ波形
    start_n = max(1 if remove_dc else 0, remove_up_to)
    for n in range(start_n, max_n+1):
        J_n = special.jv(n, 2*k*d0)
        s_plus += J_n * np.exp(1j * n * omega * t)

    # 強度計算
    intensity = np.abs(s_plus)**2

    return s_plus, intensity

def theoretical_intensity_spectrum(k, d0, omega_hz, max_n=10):
    """
    理論的な強度スペクトルを計算

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        振幅 (m)
    omega_hz : float
        基本周波数 (Hz)
    max_n : int
        最大次数

    Returns:
    --------
    freqs : array
        周波数配列 (Hz)
    amplitudes : array
        各周波数の振幅
    """
    # DCを除いた場合の理論的なスペクトル
    J0 = special.jv(0, 2*k*d0)

    freqs = []
    amplitudes = []

    # DC成分
    freqs.append(0)
    amplitudes.append(1 - J0**2)

    # 偶数倍の周波数成分
    for n in range(1, max_n+1):
        J_2n = special.jv(2*n, 2*k*d0)
        freqs.append(2*n*omega_hz)
        amplitudes.append(-4*J0*J_2n)

    return np.array(freqs), np.array(amplitudes)

def plot_simulation_results(t, s, intensity, freqs, spectrum, title):
    """
    シミュレーション結果をプロット
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # IQ波形
    axs[0].plot(t, np.real(s), label='Real')
    axs[0].plot(t, np.imag(s), label='Imag')
    axs[0].set_title('IQ Waveform')
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
    axs[2].stem(freqs, np.abs(spectrum))
    axs[2].set_title('Frequency Spectrum')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # パラメータ設定
    k = 500  # 波数 (24GHz)
    d0 = 0.0001  # 振幅 (0.1mm)
    f_heart = 1.2  # 心拍数 (Hz) = 72 bpm
    omega = 2 * np.pi * f_heart  # 角周波数 (rad/s)

    # 時間配列
    fs = 100  # サンプリング周波数 (Hz)
    duration = 5  # シミュレーション時間 (s)
    t = np.arange(0, duration, 1/fs)

    # IQ波形のシミュレーション
    s = simulate_iq_waveform(k, d0, omega, t)

    # DCを除いた強度計算
    J0 = special.jv(0, 2*k*d0)
    intensity = calculate_intensity(s, dc_removed=True, J0=J0)

    # 周波数スペクトル
    spectrum = np.fft.fft(intensity)
    freqs = np.fft.fftfreq(len(t), 1/fs)

    # 理論的なスペクトル
    theo_freqs, theo_amps = theoretical_intensity_spectrum(k, d0, f_heart)

    # 結果のプロット
    plot_simulation_results(t, s, intensity, freqs[:len(freqs)//2],
                           spectrum[:len(spectrum)//2],
                           f'Heart Rate Simulation: {f_heart} Hz ({f_heart*60} bpm)')

    # 片側周波数の強度
    s_plus, intensity_plus = calculate_positive_freq_intensity(k, d0, omega, t)

    # DCを除いた片側周波数の強度
    s_plus_dc, intensity_plus_dc = calculate_positive_freq_intensity(k, d0, omega, t, remove_dc=True)

    # DCと基本周波数の逓倍までを除いた片側周波数の強度
    s_plus_M, intensity_plus_M = calculate_positive_freq_intensity(k, d0, omega, t, remove_up_to=3)

    print(f"心拍数: {f_heart} Hz ({f_heart*60} bpm)")
    print(f"予想される強度変化の主要周波数: {2*f_heart} Hz")

    # 各種強度のスペクトル
    spectrum_plus = np.fft.fft(intensity_plus)
    spectrum_plus_dc = np.fft.fft(intensity_plus_dc)
    spectrum_plus_M = np.fft.fft(intensity_plus_M)

    # 最大周波数成分の検出
    pos_freqs = freqs[:len(freqs)//2]
    pos_spectrum = np.abs(spectrum[:len(spectrum)//2])
    max_idx = np.argmax(pos_spectrum[1:]) + 1  # DCを除く
    detected_freq = pos_freqs[max_idx]

    print(f"検出された強度変化の主要周波数: {detected_freq:.2f} Hz")
    print(f"推定心拍数: {detected_freq/2:.2f} Hz ({detected_freq/2*60:.1f} bpm)")

if __name__ == "__main__":
    main()
