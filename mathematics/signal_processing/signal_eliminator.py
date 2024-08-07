
"""
ref: https://www.sciencedirect.com/science/article/pii/S1746809420304328
"""

import numpy as np
import scipy.signal as signal
# import matplotlib.pyplot as plt


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


# 移動平均フィルター
def moving_average_filter(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


# 差分信号の計算
def calculate_difference_signal(data):
    return np.diff(data)


# Signal Elimination
def signal_elimination(
        original_signal, low_cutoff: float, high_cutoff: float, fs: int,
        window_size: int = 10, zero_phase: bool = True):
    # Step 1: 低周波ノイズの除去
    low_freq_noise = lowpass_filter(original_signal, low_cutoff, fs, order=5, zero_phase=zero_phase)
    signal_without_low_freq_noise = original_signal - low_freq_noise

    # Step 2: 高周波ノイズの除去
    high_freq_noise = highpass_filter(signal_without_low_freq_noise, high_cutoff, fs, order=5, zero_phase=zero_phase)
    signal_without_high_and_low_freq_noise = signal_without_low_freq_noise - high_freq_noise

    # Step 3: 移動平均フィルターの適用
    smoothed_signal = moving_average_filter(signal_without_high_and_low_freq_noise, window_size)
    avg_subtracted = signal_without_high_and_low_freq_noise - smoothed_signal

    # Step 4: 差分信号の計算
    difference_signal = calculate_difference_signal(avg_subtracted)

    # Ste; 5: Polarization
    polarized = np.abs(difference_signal) * difference_signal

    return smoothed_signal, avg_subtracted, difference_signal, polarized

