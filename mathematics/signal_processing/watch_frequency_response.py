
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray


def calculate_frequency_response(
        filter_a: NDArray, filter_b: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    frequency, response = signal.freqz(filter_b, filter_a, fs=fs)
    return frequency, response


def butter_lowpass(lowcut: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a


name_list = ['5', '10', '15']

filter_a_list = [
    [1.],
    [1.],
    [1.]
]

filter_b_list = [
    [1/5] * 5,
    [1/10] * 10,
    [1/15] * 15
]

# sampling frequency
fs = 500

low_b, low_a = butter_lowpass(10., fs, order=4)
filter_a_list.append(low_a)
filter_b_list.append(low_b)
name_list.append('butter 10')

res_freq_list = []
response_list = []

for a, b in zip(filter_a_list, filter_b_list):
    res_freq, response = calculate_frequency_response(np.array(a), np.array(b), fs)
    res_freq_list.append(res_freq)
    response_list.append(response)

# 振幅応答のプロット
plt.figure()
plt.subplot(2, 1, 1)
for freq, response, name in zip(res_freq_list, response_list, name_list):
    plt.plot(freq, 20 * np.log10(abs(response)), label=name)
# plt.xlim(0, 50)
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.xscale('log')
plt.title('Frequency Response (Magnitude)')

# 位相応答のプロット
plt.subplot(2, 1, 2)
for freq, response, name in zip(res_freq_list, response_list, name_list):
    plt.plot(freq, np.unwrap(np.angle(response)), label=name)
# plt.xlim(0, 50)
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.xscale('log')
plt.title('Frequency Response (Phase)')

plt.show()
