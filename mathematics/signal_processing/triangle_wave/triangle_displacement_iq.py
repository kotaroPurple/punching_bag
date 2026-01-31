
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


def _power_spectrum_from_signal(
    x: np.ndarray,
    axis: int = -1,
    window: np.ndarray | None = None,
    nfft: int | None = None,
    onesided: bool = True,
) -> np.ndarray:
    """
    Compute (one-sided or two-sided) power spectrum from time-domain signal using NumPy FFT.

    Parameters
    ----------
    x : np.ndarray
        Time-domain signal. Can be real or complex.
    axis : int
        Time axis.
    window : np.ndarray | None
        Optional window with length matching x.shape[axis].
    nfft : int | None
        FFT length. If None, use signal length along axis.
    onesided : bool
        If True and x is real, returns rfft-based one-sided spectrum.
        If x is complex, onesided is ignored and full fft is used.

    Returns
    -------
    P : np.ndarray
        Power spectrum |X[k]|^2 (not normalized by N).
    """
    x = np.asarray(x)
    n = x.shape[axis]
    if window is not None:
        w = np.asarray(window)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError("window must be 1D and match the signal length along 'axis'.")
        shape = [1] * x.ndim
        shape[axis] = n
        x = x * w.reshape(shape)

    if nfft is None:
        nfft = n

    is_real = np.isrealobj(x)
    if onesided and is_real:
        X = np.fft.rfft(x, n=nfft, axis=axis)
    else:
        X = np.fft.fft(x, n=nfft, axis=axis)

    P = (X.real * X.real) + (X.imag * X.imag)  # |X|^2 without np.abs for speed
    return P


def spectrum_flatness_measure(
    x_or_P: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
    from_signal: bool = False,
    window: np.ndarray | None = None,
    nfft: int | None = None,
    onesided: bool = True,
) -> np.ndarray:
    """
    Spectrum Flatness Measure (SFM): geometric mean / arithmetic mean.

    If from_signal=False, input is interpreted as a nonnegative power spectrum P[k].
    If from_signal=True, input is interpreted as a time-domain signal and FFT -> power is computed.

    Parameters
    ----------
    x_or_P : np.ndarray
        Power spectrum P[k] (nonnegative) or time signal.
    axis : int
        Frequency axis (if P) or time axis (if signal).
    eps : float
        Floor value to avoid log(0).
    from_signal : bool
        If True, treat x_or_P as time-domain signal.
    window : np.ndarray | None
        Optional window for FFT when from_signal=True.
    nfft : int | None
        FFT length when from_signal=True.
    onesided : bool
        Use one-sided spectrum for real signal when from_signal=True.

    Returns
    -------
    sfm : np.ndarray
        SFM in [0, 1] (theoretical). With eps-flooring, practical lower bound > 0.
    """
    if from_signal:
        P = _power_spectrum_from_signal(x_or_P, axis=axis, window=window, nfft=nfft, onesided=onesided)
    else:
        P = np.asarray(x_or_P)

    # Ensure nonnegativity and numerical safety
    P = np.maximum(P, 0.0)
    Pf = np.maximum(P, eps)

    # log-geom mean: exp(mean(log(P)))
    log_gm = np.mean(np.log(Pf), axis=axis)
    gm = np.exp(log_gm)
    am = np.mean(Pf, axis=axis)

    sfm = gm / am
    # clamp for numerical noise
    return np.clip(sfm, 0.0, 1.0)


def npwe(
    x_or_P: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
    from_signal: bool = False,
    window: np.ndarray | None = None,
    nfft: int | None = None,
    onesided: bool = True,
) -> np.ndarray:
    """
    Normalized Power Weighted Entropy (NPWE).
    This is the normalized Shannon entropy of the normalized power spectrum:

        p[k] = P[k] / sum_k P[k]
        NPWE = - (1/log K) * sum_k p[k] log p[k]

    If from_signal=False, input is interpreted as a nonnegative power spectrum P[k].
    If from_signal=True, input is interpreted as time-domain signal and FFT -> power is computed.

    Parameters
    ----------
    x_or_P : np.ndarray
        Power spectrum P[k] (nonnegative) or time signal.
    axis : int
        Frequency axis (if P) or time axis (if signal).
    eps : float
        Floor value to avoid log(0) and divide-by-zero.
    from_signal : bool
        If True, treat x_or_P as time-domain signal.
    window : np.ndarray | None
        Optional window for FFT when from_signal=True.
    nfft : int | None
        FFT length when from_signal=True.
    onesided : bool
        Use one-sided spectrum for real signal when from_signal=True.

    Returns
    -------
    npwe_value : np.ndarray
        NPWE in [0, 1]. 0 means perfectly concentrated (one bin), 1 means perfectly flat.
    """
    if from_signal:
        P = _power_spectrum_from_signal(x_or_P, axis=axis, window=window, nfft=nfft, onesided=onesided)
    else:
        P = np.asarray(x_or_P)

    P = np.maximum(P, 0.0)
    Pf = np.maximum(P, eps)

    # Normalize power to probability
    S = np.sum(Pf, axis=axis, keepdims=True)
    S = np.maximum(S, eps)
    p = Pf / S

    # Shannon entropy
    H = -np.sum(p * np.log(np.maximum(p, eps)), axis=axis)

    # Normalize by log(K)
    K = Pf.shape[axis]
    # If K == 1, entropy normalization is degenerate; define NPWE = 0
    if K <= 1:
        return np.zeros_like(H)

    npwe_value = H / np.log(K)
    return np.clip(npwe_value, 0.0, 1.0)


def generate_iq_from_displacement(
        displacements: NDArray[np.floating], wave_number: float, amp: float = 1.0, init_phase: float = 0.0) -> NDArray[np.complex128]:
    """Generate IQ signal from displacement using phase modulation."""
    phase = 2.0 * wave_number * displacements + init_phase
    iq_signal = amp * np.exp(1j * phase)
    return iq_signal


def triangle_wave(t: np.ndarray, freq_hz: float, peak_pos: float, amp: float = 1.0) -> np.ndarray:
    """Asymmetric triangle wave in [-1, 1] with peak position in (0, 1)."""
    if not (0.0 < peak_pos < 1.0):
        raise ValueError("peak_pos must be between 0 and 1 (exclusive).")
    return amp * signal.sawtooth(2.0 * np.pi * freq_hz * t, width=peak_pos)


def bandpass_filter(x: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if not (0.0 < low_hz < high_hz < fs_hz * 0.5):
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist.")
    b, a = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz)
    return signal.filtfilt(b, a, x)


def extract_side_band_frequency(
        data: NDArray, fs: int, f_low: float, f_high: float, order: int = 4) \
            -> tuple[NDArray, NDArray, NDArray]:
    # 時刻軸の生成
    t = np.arange(len(data)) / fs

    # 帯域の中心周波数と半帯域幅の計算
    fc = (f_low + f_high) / 2.0

    # (1) 搬送波による周波数シフト：対象帯域をベースバンドに移動
    shifted = data * np.exp(-1j * 2 * np.pi * fc * t)

    # (2) 低域通過 IIR フィルタの設計：カットオフ周波数を半帯域幅に設定
    nyquist = fs / 2.0
    bw_half = (f_high - f_low) / 2.0
    norm_cutoff = bw_half / nyquist  # 正規化カットオフ周波数
    sos = signal.butter(order, norm_cutoff, btype='lowpass', output='sos')

    # フィルタ適用
    filtered = signal.sosfiltfilt(sos, shifted)

    # (3) 逆搬送波で元の周波数位置に戻す
    result = filtered * np.exp(1j * 2 * np.pi * fc * t)
    return shifted, filtered, result


def calculate_fft_magnitude(data: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    """Calculate FFT spectrum of the input signal."""
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spectrum = np.fft.fft(data)
    # fft shift
    freqs = np.fft.fftshift(freqs)
    spectrum = np.fft.fftshift(spectrum)
    return freqs, np.abs(spectrum)


def main() -> None:
    # Signal parameters
    fs_hz = 100.0
    duration_s = 10.0
    freq_hz = 1.0
    peak_pos = 0.15  # 0.0-1.0: 0.5 is symmetric

    # Band-pass parameters
    low_hz = 3.0
    high_hz = 10.0
    d_max = 0.000_2  # [m]

    # Generate triangle wave displacement and IQ signal
    light_speed = 299_792_458.0  # [m/s]
    radar_frequency = 24.0e9  # [Hz]
    wave_number = 2.0 * np.pi * radar_frequency / light_speed

    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    displacements = triangle_wave(t, freq_hz=freq_hz, peak_pos=peak_pos, amp=d_max)
    iq_data = generate_iq_from_displacement(displacements, wave_number, amp=1.0, init_phase=0.0)

    filtered_iq = bandpass_filter(iq_data, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, order=4)
    freq_shifted_positive, lowpassed_positive, side_iq_positive = extract_side_band_frequency(
        iq_data, fs=int(fs_hz), f_low=low_hz, f_high=high_hz, order=4)
    freq_shifted_negative, lowpassed_negative, side_iq_negative = extract_side_band_frequency(
        iq_data, fs=int(fs_hz), f_low=-high_hz, f_high=-low_hz, order=4)

    # fix edges for transient effects
    ignore_index = (t < 0.50) + (t > duration_s - 0.50)
    iq_data[ignore_index] = 0.0 + 0.0j
    filtered_iq[ignore_index] = 0.0 + 0.0j
    lowpassed_positive[ignore_index] = 0.0 + 0.0j
    side_iq_positive[ignore_index] = 0.0 + 0.0j
    lowpassed_negative[ignore_index] = 0.0 + 0.0j
    side_iq_negative[ignore_index] = 0.0 + 0.0j

    # calculate FFT spectrum
    freqs_pos, spectrum_pos = calculate_fft_magnitude(np.abs(side_iq_positive) - np.mean(np.abs(side_iq_positive)), fs=fs_hz)
    freqs_neg, spectrum_neg = calculate_fft_magnitude(np.abs(side_iq_negative) - np.mean(np.abs(side_iq_negative)), fs=fs_hz)

    # calculate SFM and NPWE
    sfm_positive = spectrum_flatness_measure(
        spectrum_pos, axis=0, from_signal=False, eps=1e-12)
    npwe_positive = npwe(
        spectrum_pos, axis=0, from_signal=False, eps=1e-12)
    sfm_negative = spectrum_flatness_measure(
        spectrum_neg, axis=0, from_signal=False, eps=1e-12)
    npwe_negative = npwe(
        spectrum_neg, axis=0, from_signal=False, eps=1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    axes[0, 0].plot(t, iq_data.real, label="I", alpha=0.7)
    axes[0, 0].plot(t, iq_data.imag, label="Q", alpha=0.7)
    axes[0, 0].set_title("Original")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[1, 0].plot(t, filtered_iq.real, color="C0", label="Band-pass I", alpha=0.7)
    axes[1, 0].plot(t, filtered_iq.imag, color="C1", label="Band-pass Q", alpha=0.7)
    axes[1, 0].plot(t, np.abs(filtered_iq), color="gray", label="Magnitude", alpha=0.7)
    axes[1, 0].set_title("After band-pass filter")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True)

    axes[0, 1].plot(t, side_iq_positive.real, label="side_iq_positive I", alpha=0.5)
    axes[0, 1].plot(t, side_iq_positive.imag, label="side_iq_positive Q", alpha=0.5)
    axes[0, 1].plot(t, np.abs(side_iq_positive), label="Magnitude", alpha=0.7, c="gray")
    axes[0, 1].set_title("Filtered I and Q channels during positive side-band extraction")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(False)

    axes[1, 1].plot(t, side_iq_negative.real, label="side_iq_negative I", alpha=0.5)
    axes[1, 1].plot(t, side_iq_negative.imag, label="side_iq_negative Q", alpha=0.5)
    axes[1, 1].plot(t, np.abs(side_iq_negative), label="Magnitude", alpha=0.7, c="gray")
    axes[1, 1].set_title("Filtered I and Q channels during negative side-band extraction")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(False)

    for ax in axes.flat:
        ax.legend()

    plt.tight_layout()

    # plot IQ plane
    time_range = (t >= 2.5) & (t <= 3.5)
    sub_side_iq_positive = side_iq_positive[time_range]
    sub_side_iq_negative = side_iq_negative[time_range]
    sub_t = t[time_range] - t[time_range][0]
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sc = ax2.scatter(sub_side_iq_positive.real, sub_side_iq_positive.imag, label="Side-band IQ", alpha=0.5, c=sub_t, s=15)
    sc = ax2.scatter(sub_side_iq_negative.real, sub_side_iq_negative.imag, label="Side-band IQ", alpha=0.5, c=sub_t, s=15)
    ax2.plot(sub_side_iq_positive.real, sub_side_iq_positive.imag, alpha=0.3, c="gray")
    ax2.plot(sub_side_iq_negative.real, sub_side_iq_negative.imag, alpha=0.3, c="purple")
    ax2.set_title("IQ Plane")
    ax2.set_xlabel("In-phase")
    ax2.set_ylabel("Quadrature")
    ax2.grid(True)
    plt.colorbar(sc, label="Time [s]")
    plt.tight_layout()

    # plot FFT spectrum
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(freqs_pos, spectrum_pos, label="Positive Side-band IQ Spectrum", alpha=0.7)
    ax3.plot(freqs_neg, spectrum_neg, label="Negative Side-band IQ Spectrum", alpha=0.7)
    ax3.set_title(f"FFT Spectrum of Side-band IQ Signals\nSFM: {sfm_positive:.2f} (positive), {sfm_negative:.2f} (negative)\nNPWE: {npwe_positive:.2f} (positive), {npwe_negative:.2f} (negative)")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Magnitude")
    ax3.set_xlim(0, 20)
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
