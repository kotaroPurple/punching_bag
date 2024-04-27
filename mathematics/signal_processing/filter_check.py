
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray


def calculate_frequency_response(
        filter_a: NDArray, filter_b: NDArray, fs: float) -> tuple[NDArray, NDArray]:
    frequency, response = signal.freqz(filter_b, filter_a, fs=fs)
    return frequency, response


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def precise_butter_bandpass(
        samplerate, fp: list[float], fs: list[float], gpass: float, gstop:float) \
        -> tuple[NDArray, NDArray]:
    fn = samplerate / 2
    wp = np.array(fp) / fn
    ws = np.array(fs) / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "band")
    return b, a


fs = 500.
filter_b, filter_a = butter_bandpass(3, 10., fs, order=7)
filter_b2, filter_a2 = precise_butter_bandpass(fs, [3, 10], [1, 20], 1, 50)

res_freq_list = []
response_list = []
for a, b in zip((filter_a, filter_a2), (filter_b, filter_b2)):
    res_freq, response = calculate_frequency_response(a, b, fs)
    res_freq_list.append(res_freq)
    response_list.append(response)

print(f'{filter_a=}')
print(f'{filter_b=}')
print(f'{filter_a2=}')
print(f'{filter_b2=}')

# 振幅応答のプロット
plt.figure()
plt.subplot(2, 1, 1)
for freq, response in zip(res_freq_list, response_list):
    plt.plot(freq, 20 * np.log10(abs(response)))
plt.xlim(0, 50)
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response (Magnitude)')

# 位相応答のプロット
plt.subplot(2, 1, 2)
for freq, response in zip(res_freq_list, response_list):
    plt.plot(freq, np.unwrap(np.angle(response)))
plt.xlim(0, 50)
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Phase (radians)')
plt.title('Frequency Response (Phase)')

plt.show()
