
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import coherence


def signed_coherence_from_fft(x, fs):
    N = len(x)
    if N % 2 != 0:
        raise ValueError("信号長は偶数である必要があります。")

    # FFT を計算
    X = np.fft.fft(x * np.hanning(N))
    # 周波数軸の計算
    freqs = np.fft.fftfreq(N, d=1/fs)

    # 正の周波数に対応するインデックス：k = 1, 2, ..., N//2 - 1
    pos_indices = np.arange(1, N//2)
    # 対応する負の周波数はインデックス N - k
    neg_indices = N - pos_indices

    # 正の成分と対応する負の成分
    X_pos = X[pos_indices]
    X_neg = X[neg_indices]
    abs_x_pos = np.abs(X_pos)
    abs_x_neg = np.abs(X_neg)

    # ピークのみ求める
    valid_index1 = np.r_[(abs_x_pos[:-1] > abs_x_pos[1:]), False]
    valid_index2 = np.r_[False, (abs_x_pos[1:] > abs_x_pos[:-1])]
    valid_index = valid_index1 * valid_index2
    # 複素正規化クロススペクトル
    C_complex = (X_pos[valid_index] * np.conjugate(X_neg[valid_index])) / (abs_x_pos[valid_index] * abs_x_neg[valid_index])

    # signed coherence はその実部
    C_real = np.real(C_complex)
    # 位相差 (radians)
    phase = np.angle(C_complex)

    # 対応する周波数軸: f = k * fs / N
    f = freqs[pos_indices][valid_index]

    return f, C_real, phase


if __name__ == '__main__':
    fs = 1000  # サンプリング周波数 1000 Hz
    t = np.arange(0, 1.0, 1/fs)
    N = len(t)

    # 例: 50 Hz の正弦波成分と、180°位相反転した50 Hz 成分を含む複素信号
    sig1 = np.exp(1j * 2*np.pi*50*t) + np.exp(1j * 2*np.pi*25*t)        # 50 Hz, 位相 0
    sig2 = 0.8 * np.exp(1j * (-2*np.pi*50*t + 1 * np.pi)) + 0.5 * np.exp(-1j *(2*np.pi*25*t + 0.25 * np.pi))   # 50 Hz, 180°ずれ
    # sig1 = np.cos(2*np.pi*50*t)          # 50 Hz, 位相 0
    # sig2 = 0.8 * np.cos(2*np.pi*50*t + 0.25 * np.pi)  # 50 Hz, 180°ずれ
    x = sig1 + sig2  # 合成信号

    # Welch や ifft を使わずに、全信号 FFT から直接 signed coherence を計算
    f, C_real, phase = signed_coherence_from_fft(x, fs)

    plt.figure(figsize=(8, 4))
    # plt.plot(f, C_real, 'o-', label='Signed Coherence (Real part)')
    plt.scatter(f, C_real, label='Signed Coherence (Real part)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signed Coherence')
    plt.title('Signed Coherence between +f and -f components')
    plt.grid(True)
    plt.legend()
    plt.show()
