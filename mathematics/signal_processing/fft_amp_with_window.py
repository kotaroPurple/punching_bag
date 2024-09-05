
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from numpy.typing import NDArray


def make_wave(fs: int, time_range: float, freq: float, delta: float) -> tuple[NDArray, NDArray]:
    length = int(fs * time_range)
    times = np.arange(length) / fs
    x = np.sin(2 * np.pi * freq * times - delta)
    return times, x


def get_fft(fs: int, data: NDArray, window: int) -> tuple[NDArray, NDArray]:
    fft_result = np.fft.rfft(data, window)
    freqs = np.fft.rfftfreq(window, 1/fs)
    return fft_result, freqs


def main():
    fs = 500
    time_range = 10.0
    freqs = np.arange(1., 70, 0.2)
    deltas = np.zeros_like(freqs)

    peak_values = []

    for freq, delta in zip(freqs, deltas):
        _, x = make_wave(fs, time_range, freq, delta)
        fft_window = 2**(int(np.log2(len(x))))
        x_with_window = x[:fft_window] * np.hanning(fft_window)
        fft_result, fft_freq = get_fft(fs, x_with_window, fft_window)
        fft_amp = np.abs(fft_result) / len(fft_result)
        peak_value = np.max(fft_amp)
        peak_values.append(peak_value)

    #
    plt.plot(freqs, peak_values)
    plt.show()


if __name__ == '__main__':
    main()
