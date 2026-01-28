
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


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


def extract_positive_band_fft(
        data: NDArray,
        fs: float,
        f_low: float,
        f_high: float,
        analytic: bool = True,
) -> NDArray:
    """Keep only positive-frequency components in [f_low, f_high] via FFT masking."""
    if not (0.0 <= f_low < f_high <= fs * 0.5):
        raise ValueError("Require 0 <= f_low < f_high <= Nyquist.")
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spectrum = np.fft.fft(data)

    mask = (freqs >= f_low) & (freqs <= f_high)
    spectrum *= mask

    # If input is real, doubling positive frequencies keeps amplitude of the analytic signal.
    if analytic and not np.iscomplexobj(data):
        nyq = fs * 0.5
        pos_mask = (freqs > 0.0) & (freqs < nyq)
        spectrum[pos_mask] *= 2.0

    return np.fft.ifft(spectrum)


def main() -> None:
    # Signal parameters
    fs_hz = 100.0
    duration_s = 10.0
    freq_hz = 1.0
    peak_pos = 0.1  # 0.0-1.0: 0.5 is symmetric

    # Band-pass parameters
    low_hz = 3.0
    high_hz = 10.0

    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    x = triangle_wave(t, freq_hz=freq_hz, peak_pos=peak_pos, amp=1.0)
    y = triangle_wave(t, freq_hz=freq_hz, peak_pos=peak_pos, amp=0.5)
    iq_data = x + 1j * y
    iq_data += -0.1 + 0.1j
    filtered_iq = bandpass_filter(iq_data, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, order=4)
    freq_shifted, lowpassed, side_iq = extract_side_band_frequency(
        iq_data, fs=int(fs_hz), f_low=low_hz, f_high=high_hz, order=4)
    pos_band_iq = extract_positive_band_fft(
        iq_data, fs=fs_hz, f_low=0., f_high=1.5, analytic=False)

    ignore_index = (t < 0.35) + (t > duration_s - 0.35)
    iq_data[ignore_index] = 0.0 + 0.0j
    filtered_iq[ignore_index] = 0.0 + 0.0j
    lowpassed[ignore_index] = 0.0 + 0.0j
    side_iq[ignore_index] = 0.0 + 0.0j
    pos_band_iq[ignore_index] = 0.0 + 0.0j

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

    # axes[0, 1].plot(t, freq_shifted.real, label="freq shifted I", alpha=0.2)
    # axes[0, 1].plot(t, lowpassed.real, label="lowpassed I", alpha=0.5)
    axes[0, 1].plot(t, side_iq.real, label="side_iq I", alpha=0.5)
    axes[0, 1].plot(t, np.abs(side_iq), label="Magnitude", alpha=0.5, c="gray")
    axes[0, 1].plot(t, np.abs(pos_band_iq), label="Pos-only Mag (FFT)", alpha=0.5, c="tab:green")
    axes[0, 1].set_title("Filtered I channel during side-band extraction")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(False)

    # axes[1, 1].plot(t, freq_shifted.imag, label="freq shifted Q", alpha=0.2)
    # axes[1, 1].plot(t, lowpassed.imag, label="lowpassed Q", alpha=0.5)
    axes[1, 1].plot(t, side_iq.real, label="side_iq I", alpha=0.5)
    axes[1, 1].plot(t, side_iq.imag, label="side_iq Q", alpha=0.5)
    axes[1, 1].plot(t, np.abs(side_iq), label="Magnitude", alpha=0.5, c="gray")
    axes[1, 1].set_title("Filtered Q channel during side-band extraction")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(False)

    for ax in axes.flat:
        ax.legend()

    plt.tight_layout()

    # plot IQ plane
    time_range = (t >= 2.5) & (t <= 3.5)
    sub_side_iq = side_iq[time_range]
    sub_t = t[time_range]
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(sub_side_iq.real, sub_side_iq.imag, label="Side-band IQ", alpha=0.5, c=sub_t, s=15)
    ax2.plot(sub_side_iq.real, sub_side_iq.imag, alpha=0.3, c="gray")
    ax2.set_title("IQ Plane")
    ax2.set_xlabel("In-phase")
    ax2.set_ylabel("Quadrature")
    ax2.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
