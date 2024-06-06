
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import signal

n = 5
# すべての i < j を満たすインデックスペアを取得
i_indices, j_indices = np.triu_indices(n, k=1)
print(i_indices)
print(j_indices)

time_max = 2.0
freqs = [2., 5., 20.]
amps = [2., .5, 0.7]
fs = 100
cutoff_freq = 3.0

def generate_wave(time_max: float, fs: int, freq_list: list[float], amp_list: list[float]) -> tuple[NDArray, NDArray]:
    number = int(fs * time_max)
    time_array = np.arange(number) / number * time_max
    x = np.zeros(number)
    for freq, amp in zip(freq_list, amp_list):
        x += amp * np.sin(time_array * 2 * np.pi * freq)
    return time_array, x


def envelope_detection(audio_signal: NDArray, fs: int, cutoff_freq: float):
    # 絶対値を取る
    abs_signal = np.abs(audio_signal)
    # ローパスフィルターの設計
    b, a = signal.butter(4, cutoff_freq / (0.5 * fs), btype='low')
    # フィルターを適用
    envelope = signal.filtfilt(b, a, abs_signal)
    return envelope


time_array, x = generate_wave(time_max, fs, freqs, amps)
_, y = generate_wave(time_max / 4., fs, [1.0], [2.0])
x[len(x) // 4:len(x) // 4 + len(y)] += y
# x += 1.5 * time_array

filtered = envelope_detection(x, fs, cutoff_freq)

plt.plot(time_array, x)
plt.plot(time_array, filtered)
plt.show()
