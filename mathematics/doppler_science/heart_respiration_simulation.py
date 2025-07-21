import numpy as np
from scipy import special, signal
import matplotlib.pyplot as plt

def generate_combined_iq_signal(k, d0, d1, heart_rate_hz, resp_rate_hz, delta=0, duration=10, fs=100):
    """
    心拍と呼吸の複合変位によるIQ信号を生成

    Parameters:
    -----------
    k : float
        波数 (rad/m)
    d0 : float
        心拍による振幅 (m)
    d1 : float
        呼吸による振幅 (m)
    heart_rate_hz : float
        心拍数 (Hz)
    resp_rate_hz : float
        呼吸数 (Hz)
    delta : float
        心拍と呼吸の位相差 (rad)
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
    omega0 = 2 * np.pi * heart_rate_hz
    omega1 = 2 * np.pi * resp_rate_hz

    # 複合変位
    d = d0 * np.sin(omega0 * t) + d1 * np.sin(omega1 * t - delta)

    # IQ信号
    s = np.exp(2j * k * d)

    return t, s

def analyze_frequency_components(s, fs, max_freq=5.0):
    """
    IQ信号の周波数成分を解析

    Parameters:
    -----------
    s : array
        IQ信号
    fs : float
        サンプリング周波数 (Hz)
    max_freq : float
        表示する最大周波数 (Hz)

    Returns:
    --------
    freqs : array
        周波数配列
    spectrum : array
        スペクトル
    """
    # FFTによるスペクトル計算
    spectrum = np.fft.fft(s)
    freqs = np.fft.fftfreq(len(s), 1/fs)

    # 正の周波数のみ表示
    positive_idx = freqs > 0
    freqs = freqs[positive_idx]
    spectrum = np.abs(spectrum[positive_idx])

    # 最大周波数までの表示
    max_idx = freqs <= max_freq
    freqs = freqs[max_idx]
    spectrum = spectrum[max_idx]

    return freqs, spectrum

def separate_heart_respiration(s, fs, heart_band=(0.8, 2.0), resp_band=(0.1, 0.5)):
    """
    IQ信号から心拍と呼吸の成分を分離

    Parameters:
    -----------
    s : array
        IQ信号
    fs : float
        サンプリング周波数 (Hz)
    heart_band : tuple
        心拍のバンドパスフィルタ範囲 (Hz)
    resp_band : tuple
        呼吸のバンドパスフィルタ範囲 (Hz)

    Returns:
    --------
    heart_component : array
        心拍成分
    resp_component : array
        呼吸成分
    """
    # DC成分の除去
    s_dc = s - np.mean(s)

    # 強度計算
    intensity = np.abs(s_dc)**2

    # 心拍成分のフィルタリング
    b_heart, a_heart = signal.butter(4, [heart_band[0], heart_band[1]], btype='bandpass', fs=fs)
    heart_component = signal.filtfilt(b_heart, a_heart, intensity)

    # 呼吸成分のフィルタリング
    b_resp, a_resp = signal.butter(4, [resp_band[0], resp_band[1]], btype='bandpass', fs=fs)
    resp_component = signal.filtfilt(b_resp, a_resp, intensity)

    return heart_component, resp_component

def estimate_rates(heart_component, resp_component, fs):
    """
    心拍数と呼吸数を推定

    Parameters:
    -----------
    heart_component : array
        心拍成分
    resp_component : array
        呼吸成分
    fs : float
        サンプリング周波数 (Hz)

    Returns:
    --------
    heart_rate_hz : float
        推定心拍数 (Hz)
    resp_rate_hz : float
        推定呼吸数 (Hz)
    """
    # 心拍数の推定
    heart_spectrum = np.abs(np.fft.rfft(heart_component))
    heart_freqs = np.fft.rfftfreq(len(heart_component), 1/fs)

    # 有効な周波数範囲 (0.8-2.0 Hz)
    heart_valid_idx = (heart_freqs >= 0.8) & (heart_freqs <= 2.0)
    heart_valid_spectrum = heart_spectrum[heart_valid_idx]
    heart_valid_freqs = heart_freqs[heart_valid_idx]

    # 最大ピークの検出
    heart_max_idx = np.argmax(heart_valid_spectrum)
    heart_rate_hz = heart_valid_freqs[heart_max_idx] / 2  # 2倍の周波数が現れるため

    # 呼吸数の推定
    resp_spectrum = np.abs(np.fft.rfft(resp_component))
    resp_freqs = np.fft.rfftfreq(len(resp_component), 1/fs)

    # 有効な周波数範囲 (0.1-0.5 Hz)
    resp_valid_idx = (resp_freqs >= 0.1) & (resp_freqs <= 0.5)
    resp_valid_spectrum = resp_spectrum[resp_valid_idx]
    resp_valid_freqs = resp_freqs[resp_valid_idx]

    # 最大ピークの検出
    resp_max_idx = np.argmax(resp_valid_spectrum)
    resp_rate_hz = resp_valid_freqs[resp_max_idx]

    return heart_rate_hz, resp_rate_hz

def analyze_intermodulation(s, fs, heart_rate_hz, resp_rate_hz):
    """
    相互変調成分の解析

    Parameters:
    -----------
    s : array
        IQ信号
    fs : float
        サンプリング周波数 (Hz)
    heart_rate_hz : float
        心拍数 (Hz)
    resp_rate_hz : float
        呼吸数 (Hz)

    Returns:
    --------
    intermod_freqs : list
        相互変調周波数のリスト
    intermod_powers : list
        相互変調成分のパワーのリスト
    """
    # DC成分の除去
    s_dc = s - np.mean(s)

    # 強度計算
    intensity = np.abs(s_dc)**2

    # スペクトル計算
    spectrum = np.abs(np.fft.rfft(intensity))
    freqs = np.fft.rfftfreq(len(intensity), 1/fs)

    # 相互変調周波数
    intermod_freqs = [
        heart_rate_hz * 2,  # 心拍の2倍
        resp_rate_hz,       # 呼吸
        heart_rate_hz + resp_rate_hz,  # 和
        heart_rate_hz - resp_rate_hz,  # 差
        heart_rate_hz * 2 + resp_rate_hz,  # 心拍の2倍 + 呼吸
        heart_rate_hz * 2 - resp_rate_hz   # 心拍の2倍 - 呼吸
    ]

    # 各周波数のパワーを計算
    intermod_powers = []
    for freq in intermod_freqs:
        # 最も近い周波数インデックスを検索
        idx = np.argmin(np.abs(freqs - freq))
        # 周辺のパワーの平均を計算
        window = 3  # 周辺のインデックス数
        power = np.mean(spectrum[max(0, idx-window):min(len(spectrum), idx+window+1)])
        intermod_powers.append(power)

    return intermod_freqs, intermod_powers

def plot_results(t, s, heart_component, resp_component, fs, true_heart_rate, true_resp_rate, est_heart_rate, est_resp_rate):
    """
    結果のプロット
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # IQ信号
    axs[0].plot(t, np.real(s), label='Real')
    axs[0].plot(t, np.imag(s), label='Imag')
    axs[0].set_title('IQ Signal')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    # 強度
    intensity = np.abs(s - np.mean(s))**2
    axs[1].plot(t, intensity)
    axs[1].set_title('Signal Intensity')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Intensity')
    axs[1].grid(True)

    # 心拍成分
    axs[2].plot(t, heart_component)
    axs[2].set_title('Heart Rate Component')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)

    # 呼吸成分
    axs[3].plot(t, resp_component)
    axs[3].set_title('Respiration Rate Component')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Amplitude')
    axs[3].grid(True)

    plt.suptitle(f'Heart Rate: {est_heart_rate:.2f} Hz ({est_heart_rate*60:.1f} bpm) [True: {true_heart_rate:.2f} Hz]\n'
                f'Respiration Rate: {est_resp_rate:.2f} Hz ({est_resp_rate*60:.1f} bpm) [True: {true_resp_rate:.2f} Hz]')
    plt.tight_layout()

    # 周波数スペクトル
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 12))

    # 全体のスペクトル
    intensity = np.abs(s - np.mean(s))**2
    spectrum = np.abs(np.fft.rfft(intensity))
    freqs = np.fft.rfftfreq(len(intensity), 1/fs)

    # 相互変調成分の解析
    intermod_freqs, intermod_powers = analyze_intermodulation(s, fs, true_heart_rate, true_resp_rate)
    intermod_labels = [
        f'2HR: {intermod_freqs[0]:.2f} Hz',
        f'RR: {intermod_freqs[1]:.2f} Hz',
        f'HR+RR: {intermod_freqs[2]:.2f} Hz',
        f'HR-RR: {intermod_freqs[3]:.2f} Hz',
        f'2HR+RR: {intermod_freqs[4]:.2f} Hz',
        f'2HR-RR: {intermod_freqs[5]:.2f} Hz'
    ]

    axs2[0].plot(freqs, spectrum)
    for i, (freq, label) in enumerate(zip(intermod_freqs, intermod_labels)):
        axs2[0].axvline(x=freq, color=f'C{i+1}', linestyle='--', label=label)
    axs2[0].set_title('Intensity Spectrum with Intermodulation Components')
    axs2[0].set_xlabel('Frequency (Hz)')
    axs2[0].set_ylabel('Amplitude')
    axs2[0].set_xlim(0, 5)
    axs2[0].legend()
    axs2[0].grid(True)

    # 心拍成分のスペクトル
    heart_freqs = np.fft.rfftfreq(len(heart_component), 1/fs)
    heart_spectrum = np.abs(np.fft.rfft(heart_component))
    axs2[1].plot(heart_freqs, heart_spectrum)
    axs2[1].axvline(x=true_heart_rate*2, color='r', linestyle='--',
                   label=f'True: {true_heart_rate*2:.2f} Hz')
    axs2[1].axvline(x=est_heart_rate*2, color='g', linestyle='--',
                   label=f'Estimated: {est_heart_rate*2:.2f} Hz')
    axs2[1].set_title('Heart Rate Spectrum')
    axs2[1].set_xlabel('Frequency (Hz)')
    axs2[1].set_ylabel('Amplitude')
    axs2[1].set_xlim(0, 5)
    axs2[1].legend()
    axs2[1].grid(True)

    # 呼吸成分のスペクトル
    resp_freqs = np.fft.rfftfreq(len(resp_component), 1/fs)
    resp_spectrum = np.abs(np.fft.rfft(resp_component))
    axs2[2].plot(resp_freqs, resp_spectrum)
    axs2[2].axvline(x=true_resp_rate, color='r', linestyle='--',
                   label=f'True: {true_resp_rate:.2f} Hz')
    axs2[2].axvline(x=est_resp_rate, color='g', linestyle='--',
                   label=f'Estimated: {est_resp_rate:.2f} Hz')
    axs2[2].set_title('Respiration Rate Spectrum')
    axs2[2].set_xlabel('Frequency (Hz)')
    axs2[2].set_ylabel('Amplitude')
    axs2[2].set_xlim(0, 1)
    axs2[2].legend()
    axs2[2].grid(True)

    plt.tight_layout()
    plt.show()

    # 相互変調成分のパワー比較
    plt.figure(figsize=(10, 6))
    plt.bar(intermod_labels, intermod_powers)
    plt.title('Power of Intermodulation Components')
    plt.ylabel('Power')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # パラメータ設定
    k = 500  # 波数 (24GHz)
    d0 = 0.0001  # 心拍による振幅 (0.1mm)
    d1 = 0.002  # 呼吸による振幅 (2mm) - 修正値
    heart_rate_hz = 1.2  # 心拍数 (Hz) = 72 bpm
    resp_rate_hz = 0.3  # 呼吸数 (Hz) = 18 bpm

    # 信号生成
    fs = 100  # サンプリング周波数 (Hz)
    duration = 30  # 信号の長さ (s)
    t, s = generate_combined_iq_signal(k, d0, d1, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)

    # 心拍と呼吸の分離
    heart_component, resp_component = separate_heart_respiration(s, fs)

    # 心拍数と呼吸数の推定
    est_heart_rate, est_resp_rate = estimate_rates(heart_component, resp_component, fs)

    # 結果のプロット
    plot_results(t, s, heart_component, resp_component, fs,
                heart_rate_hz, resp_rate_hz, est_heart_rate, est_resp_rate)

    # 結果の表示
    print(f"真の心拍数: {heart_rate_hz:.2f} Hz ({heart_rate_hz*60:.1f} bpm)")
    print(f"推定心拍数: {est_heart_rate:.2f} Hz ({est_heart_rate*60:.1f} bpm)")
    print(f"心拍数誤差: {abs(est_heart_rate - heart_rate_hz):.4f} Hz ({abs(est_heart_rate*60 - heart_rate_hz*60):.1f} bpm)")
    print()
    print(f"真の呼吸数: {resp_rate_hz:.2f} Hz ({resp_rate_hz*60:.1f} bpm)")
    print(f"推定呼吸数: {est_resp_rate:.2f} Hz ({est_resp_rate*60:.1f} bpm)")
    print(f"呼吸数誤差: {abs(est_resp_rate - resp_rate_hz):.4f} Hz ({abs(est_resp_rate*60 - resp_rate_hz*60):.1f} bpm)")

    # 振幅比の影響を調査
    print("\n振幅比の影響を調査:")
    amplitude_ratios = [5, 10, 20, 50]

    heart_errors = []
    resp_errors = []

    for ratio in amplitude_ratios:
        d1_test = d0 * ratio
        t, s = generate_combined_iq_signal(k, d0, d1_test, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)
        heart_component, resp_component = separate_heart_respiration(s, fs)
        est_heart_rate, est_resp_rate = estimate_rates(heart_component, resp_component, fs)

        heart_error = abs(est_heart_rate - heart_rate_hz)
        resp_error = abs(est_resp_rate - resp_rate_hz)
        heart_errors.append(heart_error)
        resp_errors.append(resp_error)

        print(f"振幅比 d1/d0 = {ratio}:")
        print(f"  心拍数誤差: {heart_error:.4f} Hz")
        print(f"  呼吸数誤差: {resp_error:.4f} Hz")

    # 振幅比と誤差の関係をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(amplitude_ratios, heart_errors, 'o-', label='心拍数誤差')
    plt.plot(amplitude_ratios, resp_errors, 's-', label='呼吸数誤差')
    plt.xlabel('振幅比 (d1/d0)')
    plt.ylabel('誤差 (Hz)')
    plt.title('振幅比と推定誤差の関係')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()