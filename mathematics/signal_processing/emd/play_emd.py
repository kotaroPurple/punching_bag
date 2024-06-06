
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import signal
from scipy.interpolate import CubicSpline


def generate_data(
        fs: int, times: float, freqs: list[float], amps: list[float], noise_amp: float) -> NDArray:
    # time
    number = int(fs * times)
    t = np.arange(number) / fs
    # main
    waves = np.zeros(number)
    for freq, amp in zip(freqs, amps):
        waves += amp * np.sin(2 * np.pi * freq * t)
    # add noise
    waves += 2 * noise_amp * (np.random.random(number) - 0.5)
    return waves


def find_extrema(data: NDArray) -> tuple[NDArray, NDArray]:
    # same_size = 5
    # tmp_data = np.concatenate(([data[0]] * same_size, data, [data[-1]] * same_size))
    first_is_maxima = data[0] >= data[1]
    last_is_maxima = data[-1] >= data[-2]
    maxima, _ = signal.find_peaks(data)
    minima, _ = signal.find_peaks(-data)
    if first_is_maxima:
        maxima = np.concatenate(([0], maxima))
    else:
        minima = np.concatenate(([0], minima))
    if last_is_maxima:
        maxima = np.concatenate((maxima, [len(data) - 1]))
    else:
        minima = np.concatenate((minima, [len(data) - 1]))
    # maxima -= same_size
    # minima -= same_size
    # maxima = maxima[maxima < len(data)]
    # minima = minima[minima < len(data)]
    return maxima, minima


def interpolate_by_cubic_spline(x: NDArray, y: NDArray, source_x: NDArray) -> NDArray:
    csl = CubicSpline(x, y)
    return csl(source_x)


def generate_imfs(data: NDArray, energy_threshold: float) -> tuple[list[NDArray], NDArray]:
    # init
    h_n_k: list[list[NDArray]] = []
    h_n_k.append([])
    h_n_k[0].append(data.copy())
    # main
    _n, _k = 0, 0
    x_data = np.arange(len(data))
    c_n_data = []
    trend_data = None
    while True:
        while True:
            # find maxima, minima lines
            current_h_n_k = h_n_k[_n][_k]
            _maxima, _minima = find_extrema(current_h_n_k)
            _maxima_value = current_h_n_k[_maxima]
            _minima_value = current_h_n_k[_minima]
            maxima_line = interpolate_by_cubic_spline(_maxima, _maxima_value, x_data)
            minima_line = interpolate_by_cubic_spline(_minima, _minima_value, x_data)
            # trend
            mu_line = 0.5 * (maxima_line + minima_line)
            # h_n_k[n][k+1]
            next_k_line = current_h_n_k - mu_line
            h_n_k[_n].append(next_k_line)
            # plt.plot(x_data, maxima_line)
            # plt.plot(x_data, minima_line)
            # plt.show()
            # check "is not trend"
            is_not_tend = _check_not_trend(current_h_n_k, next_k_line)
            if is_not_tend:
                c_n_data.append(next_k_line)
                break
            else:
                _k += 1
        # next
        next_h_n_k = h_n_k[_n][0] - c_n_data[-1]
        h_n_k.append([next_h_n_k])
        _k = 0
        _n += 1
        energy = (next_h_n_k**2).sum()
        # force quit
        if energy < energy_threshold:
            trend_data = next_h_n_k
            break
        # find maxima, minima lines
        current_h_n_k = h_n_k[_n][_k]
        _maxima, _minima = find_extrema(next_h_n_k)
        _maxima_value = current_h_n_k[_maxima]
        _minima_value = current_h_n_k[_minima]
        # force quit
        force_quit = (len(_maxima) <= 5) or (len(_minima) <= 5)
        if force_quit:
            trend_data = next_h_n_k
            break
    return c_n_data, trend_data


def _check_not_trend(data: NDArray, next_data: NDArray, sd_threshold: float = 0.3) -> bool:
    # 信号中の極の数が零交差数と同じであるか 1 つしか違わないこと
    # 信号中の極大値を結ぶエンベロープと極小値を結ぶエンベロープの平均値が任意の点におい て 0 であること
    # condition 1
    maxima, minima = find_extrema(next_data)
    sign_count = np.where(next_data * np.roll(next_data, shift=1) < 0, 1, 0)[1:]
    n_zero_cross = sign_count.sum()
    flag1 = abs((len(maxima) + len(minima)) - n_zero_cross) <= 1
    # condition 2
    sd_data = ((next_data - data)**2) / (next_data**2)
    sd = sd_data.sum()
    flag2 = sd <= sd_threshold
    # plt.plot(data)
    # plt.plot(next_data)
    # plt.show()
    # result
    return flag1 and flag2


data = generate_data(1000, 1., [5, 50], [0.5, 0.5], 0.1)
c_n_data, trend = generate_imfs(data, 2.0)

# plot original
plt.plot(data, c='gray')

# plot emd
for i, c_n in enumerate(c_n_data):
    plt.plot(c_n, label=f'{i}')
plt.plot(trend, c='black')
plt.legend()
plt.show()
