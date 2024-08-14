
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def generate_wave(
        fs: int, time_range: float, freq: float, step_freq: float) -> tuple[NDArray, NDArray]:
    # short sine curve
    short_time = np.arange(int(1. / freq * fs + 0.5)) / fs
    one_sine = np.sin(2 * np.pi * freq * short_time)
    # step length
    step_length = int(1. / step_freq * fs + 0.5) - len(short_time)
    sine_and_zero = np.r_[np.zeros(step_length), one_sine]
    # generate wave
    length = int(time_range * fs + 0.5)
    times = np.arange(length) / fs
    data = np.zeros(length)
    for i in range(length // len(sine_and_zero)):
        data[i * len(sine_and_zero):(i + 1) * len(sine_and_zero)] = sine_and_zero.copy()
    return times, data


# Lowpass Filter (8 Hz 以下の低周波ノイズ除去)
def lowpass_filter(data, cutoff=8, fs=100, order=5, zero_phase: bool = True):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    if zero_phase:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


# Highpass Filter (17 Hz 以上の高周波ノイズ除去)
def highpass_filter(data, cutoff=17, fs=100, order=5, zero_phase: bool = True):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    if zero_phase:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(data, lowcut, highcut, fs, order=5, zero_phase: bool = True):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if zero_phase:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def precise_butter_bandpass(
        data, samplerate, fp: list[float], fs: list[float], gpass: float, gstop: float, order: int, zero_phase: bool = True) \
        -> NDArray:
    fn = samplerate / 2
    wp = np.array(fp) / fn
    ws = np.array(fs) / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(min(order, N), Wn, "band")
    if zero_phase:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def main():
    fs = 500
    all_time = 30.
    sin_freq = 4.
    step_freq = 0.2
    times, data = generate_wave(fs, all_time, sin_freq, step_freq)
    # lowpass
    zero_phase = False
    low_cutoff = 1.
    lowpassed = lowpass_filter(data, low_cutoff, fs, order=5, zero_phase=zero_phase)
    x_minus_low = data - lowpassed
    # highpass
    high_cutoff = 5.
    highpassed = highpass_filter(x_minus_low, high_cutoff, fs, order=5, zero_phase=zero_phase)
    x_minus_high_low = x_minus_low - highpassed
    _, axes = plt.subplots(nrows=3, figsize=(10, 8))

    # bandpass
    # bandpassed = butter_bandpass(data, low_cutoff, high_cutoff, fs, order=31, zero_phase=zero_phase)
    bandpassed = precise_butter_bandpass(data, fs, [low_cutoff, high_cutoff], [0.5, high_cutoff+3.], 10, 50, order=4, zero_phase=True)

    axes[0].plot(times, data, alpha=0.5)
    axes[0].plot(times, lowpassed, alpha=0.5)
    axes[0].plot(times, x_minus_low, alpha=0.5)

    axes[1].plot(times, data, alpha=0.5)
    axes[1].plot(times, highpassed, alpha=0.5)
    axes[1].plot(times, x_minus_high_low, alpha=0.5)
    axes[1].plot(times[:-fs * 1], bandpassed[:-fs * 1], alpha=0.5)

    axes[2].plot(times[:-1], x_minus_high_low[1:] - x_minus_high_low[:-1])
    axes[2].set_ylim(-1., 1.)
    plt.show()


if __name__ == '__main__':
    main()