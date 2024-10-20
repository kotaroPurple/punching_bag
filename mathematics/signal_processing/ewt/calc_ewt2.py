
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import NDArray


def ewt1d(
    signal: NDArray,
    max_scales: int = 5,
    use_log: bool = False,
    detection_method: str = "locmax",
    try_completion: bool = False,
    regularization: str = 'average',
    length_filter: int = 10,
    sigma_filter: float = 5.0
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Perform the Empirical Wavelet Transform of a 1D signal over specified scales.

    Parameters:
        signal (NDArray): The 1D input signal.
        max_scales (int, optional): Maximum number of supports (modes or signal components). Default is 5.
        use_log (bool, optional): Whether to work with the log spectrum. Default is False.
        detection_method (str, optional): Method for boundary detection ('locmax', 'locmaxmin', 'locmaxminf'). Default is "locmax".
        try_completion (bool, optional): Whether to complete the number of modes if fewer than max_scales are detected. Default is False.
        regularization (str, optional): Regularization method ('none', 'gaussian', 'average'). Default is 'average'.
        length_filter (int, optional): Width of the filters (Gaussian or average). Default is 10.
        sigma_filter (float, optional): Standard deviation of the Gaussian filter. Default is 5.0.

    Returns:
        tuple[NDArray, NDArray, NDArray]:
            - ewt: First the low frequency component, then the successive frequency subbands.
            - mfb: The filter bank in the Fourier domain.
            - boundaries: Vector containing the set of boundaries corresponding to the Fourier line segmentation (normalized between 0 and π).
    """
    # Compute one-sided magnitude spectrum
    spectrum = np.fft.fft(signal)
    spectrum = np.abs(spectrum[:int(np.ceil(spectrum.size / 2))])

    # Detect boundaries of Fourier segments
    boundaries = ewt_boundaries_detect(
        spectrum,
        use_log,
        detection_method,
        max_scales,
        regularization,
        length_filter,
        sigma_filter
    )
    boundaries = boundaries * np.pi / round(spectrum.size)

    # Complete boundaries if required
    if try_completion and len(boundaries) < max_scales - 1:
        boundaries = ewt_boundaries_completion(boundaries, max_scales - 1)

    # Extend the signal by mirroring to handle boundaries
    half_length = int(np.ceil(signal.size / 2))
    mirrored_signal = np.concatenate([
        np.flip(signal[:half_length - 1], axis=0),
        signal,
        np.flip(signal[-half_length + 1:], axis=0)
    ])
    mirrored_spectrum = np.fft.fft(mirrored_signal)

    # Build the corresponding filter bank
    filter_bank = ewt_meyer_filter_bank(boundaries, mirrored_spectrum.size)

    # Filter the signal to extract each subband
    ewt = np.zeros(filter_bank.shape)
    for k in range(filter_bank.shape[1]):
        ewt[:, k] = np.real(np.fft.ifft(np.conj(filter_bank[:, k]) * mirrored_spectrum))
    ewt = ewt[half_length - 1:-half_length, :]

    return ewt, filter_bank, boundaries


def ewt_boundaries_detect(
    spectrum: NDArray,
    use_log: bool,
    detection_method: str,
    max_scales: int,
    regularization: str,
    length_filter: int,
    sigma_filter: float
) -> NDArray:
    """
    Segment the spectrum into a specified number of supports using different techniques.

    Parameters:
        spectrum (NDArray): The spectrum to segment.
        use_log (bool): Whether to work with the log of the spectrum.
        detection_method (str): Method for boundary detection ('locmax', 'locmaxmin', 'locmaxminf').
        max_scales (int): Maximum number of supports.
        regularization (str): Regularization method ('none', 'gaussian', 'average').
        length_filter (int): Width of the filters.
        sigma_filter (float): Standard deviation of the Gaussian filter.

    Returns:
        NDArray: List of detected boundaries.
    """
    # Apply logarithm if required
    if use_log:
        processed_spectrum = np.log(spectrum)
    else:
        processed_spectrum = spectrum.copy()

    # Regularization
    if regularization == 'average':
        reg_filter = np.ones(length_filter) / length_filter
        presignal = np.convolve(processed_spectrum, reg_filter, mode='same')
    elif regularization == 'gaussian':
        reg_filter = np.zeros(length_filter)
        reg_filter[length_filter // 2] = 1  # Prefer odd filter lengths
        gaussian_reg_filter = gaussian_filter(reg_filter, sigma_filter)
        presignal = np.convolve(processed_spectrum, gaussian_reg_filter, mode='same')
    else:
        presignal = processed_spectrum

    # Boundary detection
    if detection_method == "locmax":
        boundaries = _local_max(presignal, max_scales)
    elif detection_method == "locmaxmin":
        boundaries = _local_max_min(presignal, max_scales)
    elif detection_method == "locmaxminf":
        boundaries = _local_max_min(presignal, max_scales, original_spectrum=spectrum)
    else:
        raise ValueError(f"Unsupported detection method: {detection_method}")

    return boundaries + 1


def _local_max(spectrum: NDArray, max_scales: int) -> NDArray:
    """
    Segment the spectrum by taking the midpoint between the N largest local maxima.

    Parameters:
        spectrum (NDArray): The spectrum to segment.
        max_scales (int): Maximum number of bands.

    Returns:
        NDArray: List of detected boundaries.
    """
    num_boundaries = max_scales - 1
    local_maxima = np.zeros(spectrum.size)
    local_minima = np.full(spectrum.size, np.max(spectrum))

    # Detect local maxima and minima
    for i in range(1, spectrum.size - 1):
        if spectrum[i - 1] < spectrum[i] > spectrum[i + 1]:
            local_maxima[i] = spectrum[i]
        if spectrum[i - 1] > spectrum[i] <= spectrum[i + 1]:
            local_minima[i] = spectrum[i]

    # Get indices of the top N maxima
    max_indices = np.argsort(local_maxima)[-num_boundaries:]
    max_indices = np.sort(max_indices)

    # Compute midpoints between consecutive maxima
    boundaries = np.zeros(num_boundaries)
    for i in range(num_boundaries):
        if i == 0:
            a = 0
        else:
            a = max_indices[i - 1]
        boundaries[i] = (a + max_indices[i]) / 2

    return boundaries


def _local_max_min(
    regularized_spectrum: NDArray,
    max_scales: int,
    original_spectrum: NDArray|None = None
) -> NDArray:
    """
    Segment the spectrum by detecting the lowest local minima between the N largest local maxima.

    Parameters:
        regularized_spectrum (NDArray): The regularized spectrum for detecting maxima.
        max_scales (int): Maximum number of bands.
        original_spectrum (NDArray, optional): The original spectrum for detecting minima. If None, use regularized_spectrum.

    Returns:
        NDArray: List of detected boundaries.
    """
    num_boundaries = max_scales - 1
    local_maxima = np.zeros(regularized_spectrum.size)
    if original_spectrum is not None:
        minima_source = original_spectrum
    else:
        minima_source = regularized_spectrum

    local_minima = np.full(minima_source.size, np.max(minima_source))

    # Detect local maxima and minima
    for i in range(1, regularized_spectrum.size - 1):
        if regularized_spectrum[i - 1] < regularized_spectrum[i] > regularized_spectrum[i + 1]:
            local_maxima[i] = regularized_spectrum[i]
        if minima_source[i - 1] > minima_source[i] < minima_source[i + 1]:
            local_minima[i] = minima_source[i]

    # Get indices of the top N maxima
    top_max_indices = np.argsort(local_maxima)[-num_boundaries:]
    top_max_indices = np.sort(top_max_indices)

    # Detect the lowest minima between consecutive maxima
    boundaries = np.zeros(num_boundaries)
    for i in range(num_boundaries):
        if i == 0:
            start = 1
        else:
            start = top_max_indices[i - 1]
        end = top_max_indices[i]
        interval_minima = local_minima[start:end]
        if interval_minima.size == 0:
            boundaries[i] = start
            continue
        min_value = np.min(interval_minima)
        min_indices = np.where(interval_minima == min_value)[0]
        # Choose the middle index if multiple minima have the same value
        middle_index = min_indices[len(min_indices) // 2]
        boundaries[i] = start + middle_index - 1

    return boundaries


def ewt_boundaries_completion(boundaries: NDArray, total_boundaries: int) -> NDArray:
    """
    Complete the boundaries vector to reach the desired number of boundaries by equally splitting the last band.

    Parameters:
        boundaries (NDArray): The existing boundaries.
        total_boundaries (int): The desired total number of boundaries.

    Returns:
        NDArray: The completed boundaries vector.
    """
    num_additional = total_boundaries - len(boundaries)
    delta_w = (np.pi - boundaries[-1]) / (num_additional + 1)
    additional_boundaries = boundaries[-1] + delta_w * (np.arange(1, num_additional + 1))
    completed_boundaries = np.concatenate([boundaries, additional_boundaries])
    return completed_boundaries


def ewt_meyer_filter_bank(boundaries: NDArray, signal_length: int) -> NDArray:
    """
    Generate the Meyer filter bank corresponding to the provided set of frequency segments.

    Parameters:
        boundaries (NDArray): Boundaries of frequency segments (0 and π are not included).
        signal_length (int): Length of the signal.

    Returns:
        NDArray: Filter bank in the Fourier domain, with the scaling function first followed by wavelets.
    """
    num_filters = len(boundaries) + 1

    # Compute gamma
    gamma = 1.0
    for k in range(len(boundaries) - 1):
        r = (boundaries[k + 1] - boundaries[k]) / (boundaries[k + 1] + boundaries[k])
        if r < gamma:
            gamma = r
    r = (np.pi - boundaries[-1]) / (np.pi + boundaries[-1])
    if r < gamma:
        gamma = r
    gamma = (1 - 1 / signal_length) * gamma  # Ensure gamma is strictly less than the minimum

    filter_bank = np.zeros((signal_length, num_filters))

    # Generate scaling function
    scaling_function = ewt_meyer_scaling(boundaries[0], gamma, signal_length)
    filter_bank[:, 0] = scaling_function

    # Generate wavelets
    for k in range(len(boundaries) - 1):
        wavelet = ewt_meyer_wavelet(boundaries[k], boundaries[k + 1], gamma, signal_length)
        filter_bank[:, k + 1] = wavelet

    # Last wavelet
    last_wavelet = ewt_meyer_wavelet(boundaries[-1], np.pi, gamma, signal_length)
    filter_bank[:, -1] = last_wavelet

    return filter_bank


def ewt_meyer_scaling(
    lower_boundary: float,
    gamma: float,
    signal_length: int
) -> NDArray:
    """
    Generate the scaling function for the Meyer filter bank.

    Parameters:
        lower_boundary (float): Lower boundary frequency.
        gamma (float): Transition ratio.
        signal_length (int): Length of the signal.

    Returns:
        NDArray: Scaling function in the Fourier domain.
    """
    mi = signal_length // 2
    frequencies = np.fft.fftshift(np.linspace(0, 2 * np.pi - 2 * np.pi / signal_length, num=signal_length))
    frequencies[:mi] -= 2 * np.pi
    abs_frequencies = np.abs(frequencies)
    scaling = np.zeros(signal_length)
    a_n = 1.0 / (2 * gamma * lower_boundary)
    p_b = (1.0 + gamma) * lower_boundary
    m_b = (1.0 - gamma) * lower_boundary

    for k in range(signal_length):
        if abs_frequencies[k] <= m_b:
            scaling[k] = 1
        elif m_b < abs_frequencies[k] <= p_b:
            scaling[k] = np.cos(np.pi * _ewt_beta(a_n * (abs_frequencies[k] - m_b)) / 2)
        else:
            scaling[k] = 0

    scaling = np.fft.ifftshift(scaling)
    return scaling


def _ewt_beta(x: float) -> float:
    """
    Function used in the construction of Meyer's wavelet.

    Parameters:
        x (float): Input value.

    Returns:
        float: Computed beta value.
    """
    if x < 0:
        return 0.0
    elif x > 1:
        return 1.0
    else:
        return (x ** 4) * (35.0 - 84.0 * x + 70.0 * (x ** 2) - 20.0 * (x ** 3))


def ewt_meyer_wavelet(
    lower_boundary: float,
    upper_boundary: float,
    gamma: float,
    signal_length: int
) -> NDArray:
    """
    Generate a 1D Meyer wavelet in the Fourier domain associated with a specific frequency band.

    Parameters:
        lower_boundary (float): Lower boundary frequency.
        upper_boundary (float): Upper boundary frequency.
        gamma (float): Transition ratio.
        signal_length (int): Length of the signal.

    Returns:
        NDArray: Wavelet in the Fourier domain.
    """
    mi = signal_length // 2
    frequencies = np.fft.fftshift(np.linspace(0, 2 * np.pi - 2 * np.pi / signal_length, num=signal_length))
    frequencies[:mi] -= 2 * np.pi
    abs_frequencies = np.abs(frequencies)
    wavelet = np.zeros(signal_length)

    a_n = 1.0 / (2 * gamma * lower_boundary)
    a_m = 1.0 / (2 * gamma * upper_boundary)
    p_b_n = (1.0 + gamma) * lower_boundary
    m_b_n = (1.0 - gamma) * lower_boundary
    p_b_m = (1.0 + gamma) * upper_boundary
    m_b_m = (1.0 - gamma) * upper_boundary

    for k in range(signal_length):
        if p_b_n <= abs_frequencies[k] <= m_b_m:
            wavelet[k] = 1
        elif m_b_m < abs_frequencies[k] <= p_b_m:
            wavelet[k] = np.cos(np.pi * _ewt_beta(a_m * (abs_frequencies[k] - m_b_m)) / 2)
        elif m_b_n <= abs_frequencies[k] <= p_b_n:
            wavelet[k] = np.sin(np.pi * _ewt_beta(a_n * (abs_frequencies[k] - m_b_n)) / 2)
        else:
            wavelet[k] = 0

    wavelet = np.fft.ifftshift(wavelet)
    return wavelet


def generate_test_signal(N: int, fs: float) -> np.ndarray:
    """
    複数の周波数成分を持つテスト信号を生成します。

    Parameters:
        N (int): サンプル数。
        fs (float): サンプリング周波数（Hz）。

    Returns:
        np.ndarray: 生成された信号。
    """
    t = np.arange(N) / fs
    # 5 Hz, 15 Hz, 30 Hz の正弦波の合成
    signal = (
        np.sin(2 * np.pi * 5 * t) +          # 5 Hz
        0.5 * np.sin(2 * np.pi * 15 * t) +   # 15 Hz
        0.2 * np.sin(2 * np.pi * 30 * t)     # 30 Hz
    )
    # ノイズの追加
    noise = 0.05 * np.random.randn(N)
    signal += noise
    return signal


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # パラメータ設定
    fs = 100.0  # サンプリング周波数 (Hz)
    N = 1024    # サンプル数

    # テスト信号の生成
    signal = generate_test_signal(N, fs)

    # EWTの適用
    ewt_result, filter_bank, boundaries = ewt1d(
        signal,
        max_scales=4,
        use_log=False,
        detection_method="locmax",
        try_completion=True,
        regularization='average',
        length_filter=10,
        sigma_filter=5.0
    )

    # フーリエスペクトルの計算
    spectrum = np.abs(np.fft.fft(signal))[:N // 2]
    freq = np.linspace(0, fs / 2, len(spectrum))

    # 境界点の周波数への変換
    boundary_freq = boundaries * fs / np.pi / 2

    # フィルタバンクの周波数軸
    filter_freq = np.linspace(0, fs / 2, filter_bank.shape[0] // 2)
    half_size = filter_bank.shape[0] // 2

    # プロットの作成
    plt.figure(figsize=(14, 12))

    # 1. 元の信号
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(N) / fs, signal, color='blue')
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 2. スペクトルと境界点
    plt.subplot(4, 1, 2)
    plt.plot(freq, spectrum, color='black', label='Spectrum')
    for b in boundary_freq:
        plt.axvline(x=b, color='red', linestyle='--', label='Boundary' if b == boundary_freq[0] else "")

    ax2 = plt.twinx()
    for k in range(filter_bank.shape[1]):
        ax2.plot(filter_freq, filter_bank[:half_size, k], label=f'Filter {k}', alpha=0.7)  # NOQA
    plt.title("Spectrum with Boundaries and Filter Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # 4. EWTによるサブバンド
    plt.subplot(4, 1, 3)
    for k in range(ewt_result.shape[1]):
        plt.plot(np.arange(N-1) / fs, ewt_result[:, k], label=f'EWT Component {k}', alpha=0.5)
    plt.title("EWT Components")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # 4. EWTによるサブバンド
    plt.subplot(4, 1, 4)
    sum_wave = np.sum(ewt_result, axis=1)
    plt.plot(np.arange(N) / fs, signal, color='C0', alpha=0.5)
    plt.plot(np.arange(N-1) / fs, sum_wave, color='C1', alpha=0.5)
    plt.title("Raw and Sum of EWTs")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
