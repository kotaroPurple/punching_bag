
import numpy as np
import matplotlib.pyplot as plt


def fft_convolve(x, y):
    """
    FFT を用いた線形畳み込み（'same' モード）。

    Parameters:
        x: 1次元の入力配列（信号）
        y: 1次元のフィルタカーネル（ウィンドウ関数）

    Returns:
        畳み込み結果の中央部（入力信号と同じ長さ）
    """
    N = len(x)
    M = len(y)
    # 線形畳み込みの長さは N + M - 1
    L = N + M - 1
    # FFT のサイズは L 以上の最小の 2 の冪乗
    L_fft = 2**int(np.ceil(np.log2(L)))

    X = np.fft.fft(x, n=L_fft)
    Y = np.fft.fft(y, n=L_fft)
    conv = np.fft.ifft(X * Y).real
    # 'same' モード: 畳み込み結果の中央 N 個のサンプルを抽出
    start = (M - 1) // 2
    end = start + N
    return conv[start:end]


def fft_convolve_reflect(x, y):
    """
    反射パディングを行った上で、FFT を用いて線形畳み込みを計算します。
    'same' モードで、元の信号と同じ長さの結果を返します。

    Parameters:
        x: 1次元の入力信号
        y: 1次元のフィルタカーネル（ウィンドウ関数）

    Returns:
        畳み込み結果（元の信号と同じ長さ）
    """
    N = len(x)
    M = len(y)
    # pad_width = (M - 1) // 2  # 両端に追加するパディングの幅
    pad_width = int(0.1 * N)
    # 反射パディングを適用
    x_padded = np.pad(x, pad_width, mode='reflect')

    # padded signal の長さ
    N_padded = len(x_padded)
    L = N_padded + M - 1  # 線形畳み込みの全長
    L_fft = 2**int(np.ceil(np.log2(L)))

    # FFT 畳み込み
    X = np.fft.fft(x_padded, n=L_fft)
    Y = np.fft.fft(y, n=L_fft)
    conv = np.fft.ifft(X * Y).real

    # 畳み込み結果から 'same' モードの中央部を抽出
    start = (M - 1) // 2
    conv_same = conv[start : start + N_padded]
    # パディング部分を取り除く
    return conv_same[pad_width:pad_width+N]


def iterative_filter(f, kernel, epsilon=1e-6, max_iter=1000):
    """
    反復的に局所平均を除去して IMF（Intrinsic Mode Function）を抽出する。

    反復更新式:
      f^(k+1)(t) = f^(k)(t) - (f^(k) * w)(t)

    Parameters:
        f: 入力信号（1次元 numpy 配列）
        kernel: フィルタカーネル（ウィンドウ関数、1次元 numpy 配列）
        epsilon: 収束判定用の閾値
        max_iter: 反復の最大回数

    Returns:
        f_filtered: 収束した IMF とみなす成分
        n_iter: 反復回数
    """
    f_current = f.copy()
    for i in range(max_iter):
        # conv = fft_convolve(f_current, kernel)
        conv = fft_convolve_reflect(f_current, kernel)
        f_next = f_current - conv
        # 収束判定: 連続する更新間の変化のノルムが epsilon 以下なら終了
        if np.linalg.norm(f_next - f_current) < epsilon:
            break
        f_current = f_next
    return f_current, i+1


def uniform_kernel(length):
    """
    一様ウィンドウ（矩形窓）を作成し、正規化する。

    Parameters:
        length: ウィンドウのサンプル数（奇数であると対称性が保たれやすい）

    Returns:
        正規化された一様ウィンドウ
    """
    kernel = np.ones(length)
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_kernel(length, sigma=None):
    """
    ガウシアン窓を生成し、正規化する。

    Parameters:
        length: ウィンドウのサンプル数（奇数が望ましい）
        sigma: ガウシアンの標準偏差。指定しない場合は length/6 とする。

    Returns:
        正規化されたガウシアン窓
    """
    if sigma is None:
        sigma = length / 6.0
    # 中心を 0 とするための x 軸
    half_len = (length - 1) // 2
    x = np.arange(-half_len, half_len + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    # 正規化：窓の和が 1 になるように
    kernel = kernel / np.sum(kernel)
    return kernel


def predict_kernel_size(data) -> float:
    diff = np.diff(data)
    diff_sign = ((diff[:-1] * diff[1:]) < 0) & (diff[:-1] > 0)
    indices, = np.where(diff_sign)
    period_diff = indices[1:] - indices[:-1]
    return float(np.mean(period_diff)) / 2.


def FIF_decomposition(signal, kernel_length=51, epsilon=1e-6, max_iter=1000, max_imfs=10):
    """
    FIF を用いた信号の分解。反復フィルタリングにより IMF を抽出し、残差信号を計算する。

    Parameters:
        signal: 入力信号（1次元 numpy 配列）
        kernel_length: フィルタカーネルの長さ
        epsilon: iterative_filter の収束判定用閾値
        max_iter: iterative_filter の最大反復回数
        max_imfs: 抽出する IMF の最大個数

    Returns:
        imfs: 抽出した IMF のリスト
        residual: 最終的な残差信号
    """
    imfs = []
    residual = signal.copy()
    for imf_index in range(max_imfs):
        # 停止条件：残差信号が単調になる、または極大・極小点の数が十分少なくなった場合
        # 簡易な例として、極大・極小の数が 2 以下なら停止する
        num_extrema = np.sum(np.diff(np.sign(np.diff(residual))) != 0)
        if num_extrema < 2:
            break
        # kernel
        _kernel_size = predict_kernel_size(residual)
        # print(_kernel_size)
        # kernel = uniform_kernel(kernel_length)
        kernel = gaussian_kernel(max(int(1 * _kernel_size), 1))
        imf, n_iter = iterative_filter(residual, kernel, epsilon, max_iter)
        imfs.append(imf)
        residual = residual - imf
    return imfs, residual


if __name__ == "__main__":
    # サンプル信号の生成（2 つの正弦波の和にノイズを加えた信号）
    t = np.linspace(0, 1, 1000)
    freq1 = 50  # 50 Hz
    freq2 = 80  # 80 Hz
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    signal = signal + 0. * np.random.randn(len(t))

    # signal = signal * np.hanning(len(signal))

    # FIF による分解
    imfs, residual = FIF_decomposition(signal, kernel_length=27, epsilon=1e-6, max_iter=1, max_imfs=5)

    # 結果のプロット
    plt.figure(figsize=(12, 8))
    plt.subplot(len(imfs) + 2, 1, 1)
    plt.plot(t, signal)
    plt.title("Original Signal")

    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 2, 1, i + 2)
        plt.plot(t, imf)
        plt.title(f"IMF {i + 1}")

    plt.subplot(len(imfs) + 2, 1, len(imfs) + 2)
    plt.plot(t, residual)
    plt.title("Residual")
    plt.tight_layout()
    plt.show()
