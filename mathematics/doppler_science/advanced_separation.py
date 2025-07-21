import numpy as np
from scipy import special, signal
import matplotlib.pyplot as plt
from scipy import linalg

def generate_combined_iq_signal(k, d0, d1, heart_rate_hz, resp_rate_hz, delta=0, duration=10, fs=100, noise_level=0.01):
    """
    心拍と呼吸の複合変位によるIQ信号を生成（ノイズ付き）
    """
    t = np.arange(0, duration, 1/fs)
    omega0 = 2 * np.pi * heart_rate_hz
    omega1 = 2 * np.pi * resp_rate_hz

    # 複合変位
    d = d0 * np.sin(omega0 * t) + d1 * np.sin(omega1 * t - delta)

    # IQ信号
    s = np.exp(2j * k * d)

    # ノイズ追加
    noise = noise_level * (np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t)))
    s_noisy = s + noise

    return t, s_noisy

def empirical_mode_decomposition(signal, max_imf=10, max_iter=100, tol=1e-6):
    """
    経験的モード分解（EMD）による信号の分解

    Parameters:
    -----------
    signal : array
        入力信号
    max_imf : int
        最大IMF数
    max_iter : int
        各IMFの最大反復回数
    tol : float
        収束判定の閾値

    Returns:
    --------
    imfs : list
        内在的モード関数（IMF）のリスト
    """
    # 簡易版EMD実装
    signal = signal.copy()
    imfs = []

    for _ in range(max_imf):
        h = signal.copy()

        # IMFの抽出
        for _ in range(max_iter):
            # 極大値と極小値の検出
            max_idx = signal.argrelmax(h)[0]
            min_idx = signal.argrelmin(h)[0]

            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            # スプライン補間
            t = np.arange(len(h))
            max_env = np.interp(t, max_idx, h[max_idx])
            min_env = np.interp(t, min_idx, h[min_idx])

            # 平均エンベロープ
            mean_env = (max_env + min_env) / 2

            # IMF候補の更新
            h_prev = h.copy()
            h = h - mean_env

            # 収束判定
            if np.sum((h - h_prev)**2) / np.sum(h_prev**2) < tol:
                break

        # IMFの保存
        imfs.append(h)

        # 残差の更新
        signal = signal - h

        # 残差が単調になったら終了
        if len(signal.argrelmax()[0]) <= 1 or len(signal.argrelmin()[0]) <= 1:
            break

    # 残差を追加
    imfs.append(signal)

    return imfs

def wavelet_decomposition(signal, fs, scales=None):
    """
    連続ウェーブレット変換による時間-周波数解析

    Parameters:
    -----------
    signal : array
        入力信号
    fs : float
        サンプリング周波数
    scales : array or None
        ウェーブレットスケール

    Returns:
    --------
    coef : array
        ウェーブレット係数
    freqs : array
        対応する周波数
    """
    from scipy import signal as sg

    # デフォルトスケール
    if scales is None:
        scales = np.arange(1, 128)

    # 連続ウェーブレット変換
    coef, freqs = sg.cwt(signal, sg.morlet2, scales)

    # スケールを周波数に変換
    freqs = freqs * fs

    return coef, freqs

def adaptive_filter(signal, reference, filter_length=32, mu=0.01):
    """
    適応フィルタによる信号分離

    Parameters:
    -----------
    signal : array
        入力信号
    reference : array
        参照信号
    filter_length : int
        フィルタ長
    mu : float
        適応ステップサイズ

    Returns:
    --------
    output : array
        フィルタ出力
    error : array
        誤差信号（目的の信号）
    """
    N = len(signal)
    w = np.zeros(filter_length)  # フィルタ係数
    output = np.zeros(N)
    error = np.zeros(N)

    # LMSアルゴリズム
    for n in range(filter_length, N):
        x = reference[n:n-filter_length:-1]
        output[n] = np.dot(w, x)
        error[n] = signal[n] - output[n]
        w = w + mu * error[n] * x

    return output, error

def independent_component_analysis(X, n_components=2):
    """
    独立成分分析（ICA）による信号分離

    Parameters:
    -----------
    X : array
        混合信号（行：信号、列：サンプル）
    n_components : int
        分離する成分数

    Returns:
    --------
    S : array
        分離された信号
    """
    from sklearn.decomposition import FastICA

    # ICAの適用
    ica = FastICA(n_components=n_components)
    S = ica.fit_transform(X.T).T

    return S

def advanced_heart_respiration_separation(s, fs, heart_band=(0.8, 2.0), resp_band=(0.1, 0.5), method='bandpass'):
    """
    高度な信号処理手法による心拍と呼吸の分離

    Parameters:
    -----------
    s : array
        IQ信号
    fs : float
        サンプリング周波数
    heart_band : tuple
        心拍のバンドパス範囲
    resp_band : tuple
        呼吸のバンドパス範囲
    method : str
        分離手法（'bandpass', 'adaptive', 'ica'）

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

    if method == 'bandpass':
        # バンドパスフィルタによる分離
        b_heart, a_heart = signal.butter(4, [heart_band[0], heart_band[1]], btype='bandpass', fs=fs)
        heart_component = signal.filtfilt(b_heart, a_heart, intensity)

        b_resp, a_resp = signal.butter(4, [resp_band[0], resp_band[1]], btype='bandpass', fs=fs)
        resp_component = signal.filtfilt(b_resp, a_resp, intensity)

    elif method == 'adaptive':
        # 適応フィルタによる分離
        # まず呼吸成分を抽出
        b_resp, a_resp = signal.butter(4, [resp_band[0], resp_band[1]], btype='bandpass', fs=fs)
        resp_reference = signal.filtfilt(b_resp, a_resp, intensity)

        # 適応フィルタで呼吸成分を除去
        _, heart_component = adaptive_filter(intensity, resp_reference)
        resp_component = resp_reference

        # 心拍成分をさらにフィルタリング
        b_heart, a_heart = signal.butter(4, [heart_band[0], heart_band[1]], btype='bandpass', fs=fs)
        heart_component = signal.filtfilt(b_heart, a_heart, heart_component)

    elif method == 'ica':
        # ICAによる分離
        # 位相情報も利用
        X = np.vstack([np.real(s_dc), np.imag(s_dc)])
        S = independent_component_analysis(X)

        # 各成分をフィルタリングして心拍と呼吸に分類
        components = []
        for i in range(S.shape[0]):
            # スペクトル計算
            spectrum = np.abs(np.fft.rfft(S[i]))
            freqs = np.fft.rfftfreq(len(S[i]), 1/fs)

            # 心拍帯域のエネルギー
            heart_idx = (freqs >= heart_band[0]) & (freqs <= heart_band[1])
            heart_energy = np.sum(spectrum[heart_idx]**2)

            # 呼吸帯域のエネルギー
            resp_idx = (freqs >= resp_band[0]) & (freqs <= resp_band[1])
            resp_energy = np.sum(spectrum[resp_idx]**2)

            components.append((S[i], heart_energy, resp_energy))

        # エネルギーに基づいて分類
        components.sort(key=lambda x: x[1]/x[2])  # 心拍/呼吸エネルギー比

        # 呼吸成分（比が小さい）
        resp_component = components[0][0]

        # 心拍成分（比が大きい）
        heart_component = components[-1][0]

        # さらにバンドパスフィルタを適用
        b_heart, a_heart = signal.butter(4, [heart_band[0], heart_band[1]], btype='bandpass', fs=fs)
        heart_component = signal.filtfilt(b_heart, a_heart, heart_component)

        b_resp, a_resp = signal.butter(4, [resp_band[0], resp_band[1]], btype='bandpass', fs=fs)
        resp_component = signal.filtfilt(b_resp, a_resp, resp_component)

    else:
        raise ValueError(f"Unknown method: {method}")

    return heart_component, resp_component

def compare_separation_methods(k, d0, d1, heart_rate_hz, resp_rate_hz, duration=30, fs=100):
    """
    異なる信号分離手法の比較
    """
    # 信号生成
    t, s = generate_combined_iq_signal(k, d0, d1, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)

    # 各手法による分離
    methods = ['bandpass', 'adaptive', 'ica']
    results = {}

    for method in methods:
        try:
            heart_component, resp_component = advanced_heart_respiration_separation(s, fs, method=method)

            # 心拍数と呼吸数の推定
            heart_spectrum = np.abs(np.fft.rfft(heart_component))
            heart_freqs = np.fft.rfftfreq(len(heart_component), 1/fs)

            heart_valid_idx = (heart_freqs >= 0.8) & (heart_freqs <= 2.0)
            heart_max_idx = np.argmax(heart_spectrum[heart_valid_idx])
            est_heart_rate = heart_freqs[heart_valid_idx][heart_max_idx] / 2

            resp_spectrum = np.abs(np.fft.rfft(resp_component))
            resp_freqs = np.fft.rfftfreq(len(resp_component), 1/fs)

            resp_valid_idx = (resp_freqs >= 0.1) & (resp_freqs <= 0.5)
            resp_max_idx = np.argmax(resp_spectrum[resp_valid_idx])
            est_resp_rate = resp_freqs[resp_valid_idx][resp_max_idx]

            results[method] = {
                'heart_component': heart_component,
                'resp_component': resp_component,
                'est_heart_rate': est_heart_rate,
                'est_resp_rate': est_resp_rate,
                'heart_error': abs(est_heart_rate - heart_rate_hz),
                'resp_error': abs(est_resp_rate - resp_rate_hz)
            }
        except Exception as e:
            print(f"Method {method} failed: {e}")

    # 結果の表示
    print("分離手法の比較:")
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  推定心拍数: {result['est_heart_rate']:.2f} Hz ({result['est_heart_rate']*60:.1f} bpm)")
        print(f"  心拍数誤差: {result['heart_error']:.4f} Hz")
        print(f"  推定呼吸数: {result['est_resp_rate']:.2f} Hz ({result['est_resp_rate']*60:.1f} bpm)")
        print(f"  呼吸数誤差: {result['resp_error']:.4f} Hz")

    # プロット
    fig, axs = plt.subplots(len(methods), 2, figsize=(12, 4*len(methods)))

    for i, method in enumerate(methods):
        if method in results:
            # 心拍成分
            axs[i, 0].plot(t, results[method]['heart_component'])
            axs[i, 0].set_title(f'{method.upper()}: Heart Component')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].grid(True)

            # 呼吸成分
            axs[i, 1].plot(t, results[method]['resp_component'])
            axs[i, 1].set_title(f'{method.upper()}: Respiration Component')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return results

def main():
    # パラメータ設定
    k = 500  # 波数 (24GHz)
    d0 = 0.0001  # 心拍による振幅 (0.1mm)
    d1 = 0.002  # 呼吸による振幅 (2mm)
    heart_rate_hz = 1.2  # 心拍数 (Hz) = 72 bpm
    resp_rate_hz = 0.3  # 呼吸数 (Hz) = 18 bpm

    # 信号生成
    fs = 100  # サンプリング周波数 (Hz)
    duration = 30  # 信号の長さ (s)
    t, s = generate_combined_iq_signal(k, d0, d1, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)

    # 異なる分離手法の比較
    results = compare_separation_methods(k, d0, d1, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)

    # 振幅比の影響を調査
    print("\n振幅比の影響を調査:")
    amplitude_ratios = [5, 10, 20, 50]

    for ratio in amplitude_ratios:
        d1_test = d0 * ratio
        print(f"\n振幅比 d1/d0 = {ratio}:")
        _ = compare_separation_methods(k, d0, d1_test, heart_rate_hz, resp_rate_hz, duration=duration, fs=fs)

if __name__ == "__main__":
    main()