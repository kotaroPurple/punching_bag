
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.special import jn
from scipy.signal import butter, sosfilt, sosfiltfilt


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


@dataclass(slots=True)
class SidebandFilter:
    """Provide a single-sideband band-pass filter via complex demodulation.

    The filter isolates a narrow frequency band by translating the selected sideband to
    baseband, applying an IIR low-pass filter, and (optionally) shifting it back to the
    original center frequency.

    Args:
        sample_rate (float): Sampling frequency in Hz.
        band (tuple[float, float]): Lower and upper frequency bounds in Hz. Values must
            satisfy -sample_rate / 2 < band[0] < band[1] < sample_rate / 2.
        zero_phase (bool): If True, keep the demodulated (baseband) signal instead of
            shifting it back to the original center frequency.
        filter_order (int): Order of the Butterworth low-pass filter applied after
            demodulation (default: 6).
        remodulate (bool): If True, shift the filtered baseband signal back to the
            original center frequency (default: True).

    Example:
        >>> fs = 1_000.0
        >>> filt = SidebandFilter(sample_rate=fs, band=(90.0, 110.0))
        >>> t = np.arange(2_048) / fs
        >>> x = np.exp(1j * 2 * np.pi * 100.0 * t)
        >>> y = filt.filter(x)
        >>> y.shape
        (2048,)
    """

    sample_rate: float
    band: tuple[float, float]
    zero_phase: bool = False
    filter_order: int = 6
    remodulate: bool = True
    _sos: NDArray[np.float64] = field(init=False, repr=False)
    _center_frequency: float = field(init=False, repr=False)
    _transient_samples: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and design the prototype low-pass filter."""
        self.sample_rate = float(self.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if len(self.band) != 2:
            raise ValueError("band must contain exactly two frequency bounds.")
        low, high = (float(self.band[0]), float(self.band[1]))
        if low >= high:
            raise ValueError("band must satisfy band[0] < band[1].")
        nyquist = 0.5 * self.sample_rate
        if low <= -nyquist or high >= nyquist:
            raise ValueError(
                "Band edges must lie strictly within (-sample_rate/2, sample_rate/2)."
            )

        self.band = (low, high)
        if not isinstance(self.zero_phase, (bool, np.bool_)):
            raise TypeError("zero_phase must be a boolean.")
        self.zero_phase = bool(self.zero_phase)
        if not isinstance(self.remodulate, (bool, np.bool_)):
            raise TypeError("remodulate must be a boolean.")
        self.remodulate = bool(self.remodulate)

        self.filter_order = int(self.filter_order)
        if self.filter_order < 1:
            raise ValueError("filter_order must be greater than or equal to 1.")

        self._center_frequency = 0.5 * (low + high)
        cutoff = 0.5 * (high - low)
        if cutoff <= 0:
            raise ValueError("Computed cutoff frequency must be positive.")

        self._sos = butter(
            N=self.filter_order,
            Wn=cutoff,
            btype="low",
            fs=self.sample_rate,
            output="sos",
        )  # type: ignore
        self._transient_samples = max(1, 4 * self.filter_order)

    def filter(self, signal: NDArray[np.complex128], axis: int = -1) -> NDArray[np.complex128]:
        """Apply the sideband filter to a signal.

        Args:
            signal (np.ndarray): Input signal containing the targeted sideband. The array
                can be real or complex and of arbitrary shape.
            axis (int): Axis along which the time series is stored. Defaults to the last
                axis.

        Returns:
            np.ndarray: Complex output with the same shape as ``signal`` where only the
            selected sideband remains.

        Raises:
            ValueError: If ``signal`` has zero length along the filtering axis.

        Example:
            >>> filt = SidebandFilter(sample_rate=1_000.0, band=(90.0, 110.0))
            >>> t = np.arange(1_024) / 1_000.0
            >>> x = np.cos(2 * np.pi * 100.0 * t)
            >>> y = filt.filter(x)
            >>> y.dtype
            dtype('complex128')
        """
        data = np.asarray(signal, dtype=np.complex128)
        if data.shape == ():
            raise ValueError("signal must not be scalar.")

        data = np.moveaxis(data, axis, -1)
        n_samples = data.shape[-1]
        if n_samples == 0:
            raise ValueError("signal must contain at least one sample along the target axis.")

        time = np.arange(n_samples, dtype=np.float64) / self.sample_rate
        shift_frequency = self._center_frequency
        demod_phase = np.exp(-1j * 2.0 * np.pi * shift_frequency * time)
        baseband = data * demod_phase

        if self.zero_phase:
            baseband_filtered = sosfiltfilt(self._sos, baseband, axis=-1)
        else:
            baseband_filtered = sosfilt(self._sos, baseband, axis=-1)

        filtered = baseband_filtered
        if self.remodulate:
            remod_phase = np.exp(1j * 2.0 * np.pi * shift_frequency * time)
            filtered = baseband_filtered * remod_phase

        filtered = np.moveaxis(filtered, -1, axis)  # type: ignore
        return filtered  # type: ignore

    @property
    def sos(self) -> NDArray[np.float64]:
        """Return a copy of the IIR low-pass section coefficients."""
        return self._sos.copy()


def generate_iq_from_displacement(
        displacements: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Generate IQ signal from displacement data."""
    return np.exp(2j * WAVE_NUMBER * displacements)


def generate_displacement(
        amp: float, frequency: float, duration: float, sample_rate: float, start: float = 0.) \
            -> NDArray[np.floating]:
    """Generate a sinusoidal displacement signal."""
    t = np.arange(0, duration, 1 / sample_rate)
    displacement = amp * np.sin(2 * np.pi * frequency * t) + start
    return displacement


def generate_iq_with_theory(
        displacement_amp: float, frequency: float, duration: float, sample_rate: float,
        order_list: list[int]) -> NDArray[np.complex128]:
    """Generate IQ signal using theoretical model."""
    t = np.arange(0, duration, 1 / sample_rate)
    iq_signal = np.zeros(len(t), np.complex128)
    for n in order_list:
        bessel_coeff = jn(n, 2 * WAVE_NUMBER * displacement_amp)
        iq_signal += bessel_coeff * np.exp(1j * 2 * np.pi * n * frequency * t)
    return iq_signal


def generate_gaussian_displacement(
        amp: float, frequency: float, center_time: float, sigma: float, duration: float, sample_rate: float) -> NDArray[np.floating]:
    """Generate a Gaussian-shaped displacement signal."""
    # 正規分布を周期的になるように生成
    t = np.arange(0, duration, 1 / sample_rate)
    data = np.zeros_like(t)
    n_blocks = int(duration * frequency) + 1
    period = 1 / frequency
    for n in range(n_blocks):
        # if n != 5:
        #     continue
        block_center = center_time + n * period
        data += generate_single_gaussian(amp, block_center, sigma, t)
    return data


def generate_single_gaussian(
        amp: float, center_time: float, sigma: float, t: NDArray[np.floating]) -> NDArray[np.floating]:
    """Generate a single Gaussian-shaped displacement signal."""
    gaussian = amp * np.exp(-0.5 * ((t - center_time) / sigma) ** 2)
    return gaussian


def apply_highpass_filter(
        data: NDArray, cutoff: float, fs: int, order: int=5) -> NDArray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered


def main() -> None:
    # displacement parameters
    amplitudes = [0.000_2]  # [m]
    frequencies = [1.0]  # [Hz]
    duration = 10.0  # [s]

    sample_rate = 50.0  # [Hz]
    # generate displacement
    total_displacement = np.zeros(int(duration * sample_rate))
    for amp, freq in zip(amplitudes, frequencies):
        displacement = generate_displacement(amp, freq, duration, sample_rate, start=0.001)
        total_displacement += displacement
    total_displacement += generate_gaussian_displacement(-0.000_1, frequencies[0], center_time=0.125, sigma=0.05, duration=duration, sample_rate=sample_rate)
    total_displacement += generate_gaussian_displacement(0.000_07, frequencies[0], center_time=0.625, sigma=0.05, duration=duration, sample_rate=sample_rate)
    # generate IQ signal
    iq_signal = generate_iq_from_displacement(total_displacement)

    # apply filter
    cutoff_frequency = 2.55  # [Hz]
    filtered_iq_signal = apply_highpass_filter(iq_signal, cutoff_frequency, int(sample_rate), order=7)

    # apply sideband filter
    side_cutoff = (cutoff_frequency, 20.0)
    sideband_filter = SidebandFilter(sample_rate, side_cutoff, zero_phase=True, filter_order=6, remodulate=True)
    sideband_filtered_iq = sideband_filter.filter(iq_signal)

    filtered_iq_signal[:int(sample_rate)] = 0
    filtered_iq_signal[-int(sample_rate):] = 0
    sideband_filtered_iq[:int(sample_rate)] = 0
    sideband_filtered_iq[-int(sample_rate):] = 0

    n_conv = int(0.2 * sample_rate)
    abs_filtered = np.abs(filtered_iq_signal)
    mv_abs_filtered = np.convolve(abs_filtered, np.ones(n_conv) / n_conv, mode='same')
    abs_sideband_filtered = np.abs(sideband_filtered_iq)
    mv_abs_sideband_filtered = np.convolve(abs_sideband_filtered, np.ones(n_conv) / n_conv, mode='same')

    # plot
    _, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    time = np.arange(0, duration, 1 / sample_rate)
    axes[0].plot(time, total_displacement)
    axes[0].set_title("Displacement Signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Displacement [m]")

    axes[1].plot(time, iq_signal.real, label='I Component', alpha=0.5)
    axes[1].plot(time, iq_signal.imag, label='Q Component', alpha=0.5)
    axes[1].set_title("IQ Signal")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    axes[2].plot(time, filtered_iq_signal.real, label='I Component', alpha=0.5)
    axes[2].plot(time, filtered_iq_signal.imag, label='Q Component', alpha=0.5)
    axes[2].plot(time, sideband_filtered_iq.real, label='I Side', alpha=0.5)
    axes[2].plot(time, sideband_filtered_iq.imag, label='Q Side', alpha=0.5)
    axes[2].set_title("Filtered IQ Signal")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()

    axes[3].plot(time, abs_filtered, label='Magnitude', alpha=0.5, c='blue')
    axes[3].plot(time, abs_sideband_filtered, label='Side Magnituide', alpha=0.5, c='red')
    axes[3].plot(time, mv_abs_filtered, alpha=0.3, c='blue', linestyle='--')
    axes[3].plot(time, mv_abs_sideband_filtered, alpha=0.3, c='red', linestyle='--')
    axes[3].set_title("Filtered IQ Signal")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Amplitude")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
