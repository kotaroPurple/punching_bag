
import numpy as np
import matplotlib.pyplot as plt


def morlet_wavelet(t, scale, omega0=6.):
    """
    Morlet ウェーブレットの定義

    ψ(t) = π^(-1/4) * exp(i * ω₀ * t) * exp(-t²/2)

    ただし、スケール変換を考慮して ψ((t)/scale)/√(scale) とします。

    Parameters
    ----------
    t : ndarray
        時間軸の配列
    scale : float
        スケール（拡大・縮小のパラメータ）
    omega0 : float, optional
        中心周波数 (デフォルトは 6)

    Returns
    -------
    psi : ndarray
        スケールを考慮した Morlet ウェーブレット（複素数）
    """
    return (np.pi**(-0.25) / np.sqrt(scale)) * np.exp(1j * omega0 * (t / scale)) * np.exp(-(t / scale)**2 / 2)


def cwt(signal, scales, dt, omega0=6):
    """
    連続ウェーブレット変換 (CWT) の計算

    CWT の定義は
        W(scale, b) = ∫ x(t) ψ*[(t - b)/scale] dt
    となります（ψ* は ψ の複素共役）。
    ここでは離散信号に対して、np.convolve を用いた畳み込みで実装しています。

    Parameters
    ----------
    signal : ndarray
        入力信号（1 次元の NumPy 配列）
    scales : ndarray
        使用するスケールの配列
    dt : float
        サンプリング間隔
    omega0 : float, optional
        Morlet ウェーブレットの中心周波数 (デフォルトは 6)

    Returns
    -------
    cwt_coeffs : ndarray
        ウェーブレット係数（shape: [len(scales), len(signal)]）
    """
    n = len(signal)
    # ウェーブレット用の時間軸を信号長に合わせて [-T/2, T/2] とする
    t = np.arange(-n//2, n//2) * dt
    cwt_coeffs = np.zeros((len(scales), n), dtype=complex)

    for idx, scale in enumerate(scales):
        # 現在のスケールに対応する Morlet ウェーブレットを生成
        psi = morlet_wavelet(t, scale, omega0)
        # CWT の定義では wavelet の複素共役を用いるので注意
        psi_conj = np.conjugate(psi)
        # 畳み込みを実施（'same' モードで信号と同じ長さに）
        cwt_coeffs[idx, :] = np.convolve(signal, psi_conj, mode='same')
    return cwt_coeffs


def main():
    # サンプル信号の作成：2 つの正弦波（50 Hz と 120 Hz）の合成
    dt = 0.001           # サンプリング間隔 [s]
    t = np.arange(0, 1, dt)  # 時間軸
    freq1 = 1
    freq2 = 100
    signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)

    # func = morlet_wavelet(t, scale=0.5, omega0=6.)
    # print(func.shape)
    # plt.plot(t, func.real, alpha=0.5)
    # plt.plot(t, func.imag, alpha=0.5)
    # plt.show()

    # return

    # 解析に使用するスケールを設定（適宜調整してください）
    # scales = np.linspace(1, 100, 100)
    scales = np.linspace(0.1, 10, 100)

    # CWT の計算
    cwt_result = cwt(signal, scales, dt, omega0=6)

    # 結果の可視化：スケール軸 vs. 時間軸のスカログラム
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(cwt_result), extent=(t[0], t[-1], scales[0], scales[-1]),
        origin='lower', aspect='auto', cmap='jet')
    plt.xlabel("Time [s]")
    plt.ylabel("Scale")
    plt.title("Morlet Wavelet Transform (CWT)")
    plt.colorbar(label="Magnitude")
    plt.show()


if __name__ == "__main__":
    main()
