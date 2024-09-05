
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED
print(ALPHA)


# LIGHT_OF_SPEED = 3.e8
# SENSOR_FREQUENCY = 24.e9
# COEFF_ALPHA = 2 * SENSOR_FREQUENCY / LIGHT_OF_SPEED
SAMPLE_RATE = 1000  # [Hz]


def generate_waves(
        fs: int, time_range: float, d_list: list[float], freq_list: list[float],
        delta_list: list[float]) -> tuple[NDArray, NDArray, NDArray]:
    # make waves
    length = int(fs * time_range)
    times = np.arange(length) / fs
    # phase
    phase = np.zeros_like(times)
    for d, f, delta in zip(d_list, freq_list, delta_list):
        phase += ALPHA * d * np.sin(2 * np.pi * f * times - delta)
    # waves
    i_wave = np.cos(phase)
    q_wave = np.sin(phase)
    iq_wave = i_wave + 1j * q_wave
    return times, iq_wave, phase


def main():
    # prepare
    fs = 500
    time_range = 60.
    # amplitudes, frequency list, delta phase list
    # mix
    d_list = [200.e-6, 5.e-3]
    f_list = [1., 0.3]
    delta_list = [np.pi/6, np.pi/6]
    # # heart
    # d_list = [200.e-6]
    # f_list = [1.]
    # delta_list = [np.pi/6]
    # # respiration
    # d_list = [5.e-3]
    # f_list = [0.3]
    # delta_list = [np.pi/6]
    # waves
    times, iq_wave, phase = generate_waves(fs, time_range, d_list, f_list, delta_list)
    # fft
    fft_window = 2**(int(np.log2(len(iq_wave))))
    wave_with_window = iq_wave[:fft_window] * np.hanning(fft_window)
    fft_result = np.fft.fft(wave_with_window, fft_window)
    fft_freq = np.fft.fftfreq(fft_window, 1/fs)
    _fft_amp = np.abs(fft_result) / len(fft_result)
    fft_amp = _fft_amp[:len(_fft_amp)//2]
    fft_amp[1:] += _fft_amp[-1:-len(_fft_amp)//2:-1]
    # fft_amp[0] = 0.
    fft_freq = fft_freq[:len(fft_freq)//2]

    plt.plot(fft_freq, fft_amp)
    plt.xlim(0., 5.)
    plt.show()


if __name__ == '__main__':
    main()
