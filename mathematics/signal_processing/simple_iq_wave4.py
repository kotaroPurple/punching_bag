
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


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


def predict_displacement_from_iq_wave(iq_data: NDArray, initial_phase: float) -> NDArray:
    # prepare
    coeff = (LIGHT_OF_SPEED / SENSOR_FREQUENCY) / (4. * np.pi)
    i_data, q_data = np.real(iq_data), np.imag(iq_data)
    # calculate displacement
    i_n_minus_1_times_q_n = i_data[:-1] * q_data[1:]
    i_n_times_q_n_minus_1 = i_data[1:] * q_data[:-1]
    i_q_amp = np.sqrt((i_data**2) + (q_data**2))
    tmp = (i_n_minus_1_times_q_n - i_n_times_q_n_minus_1) / (i_q_amp[:-1] * i_q_amp[1:])
    displacements = np.r_[coeff * initial_phase, coeff * (initial_phase + np.cumsum(tmp))]
    return displacements


def main():
    # prepare
    fs = 500
    time_range = 70.
    # amplitudes, frequency list, delta phase list
    # mix
    d_list = [200.e-6, 4.e-3]
    f_list = [1., 0.3]
    delta_list = [np.pi/6, np.pi/6]
    # # mix
    # d_list = [1.e-3, 5.e-3]
    # f_list = [0.5, 0.3]
    # delta_list = [0., 0.]
    # # heart
    # d_list = [200.e-6]
    # f_list = [1.]
    # delta_list = [np.pi/6]
    # # respiration
    # d_list = [5.e-3]
    # f_list = [0.3]
    # delta_list = [0.]

    # waves
    times, iq_wave, phase = generate_waves(fs, time_range, d_list, f_list, delta_list)

    # predict displacement
    expected_displacement = (LIGHT_OF_SPEED / SENSOR_FREQUENCY) / (4. * np.pi) * phase
    predicted_displacement = predict_displacement_from_iq_wave(iq_wave, phase[0])

    plt.subplot(211)
    plt.plot(times, np.real(iq_wave), alpha=0.5)
    plt.plot(times, np.imag(iq_wave), alpha=0.5)

    plt.subplot(212)
    plt.plot(expected_displacement, alpha=0.5, label='expected')
    plt.plot(predicted_displacement, alpha=0.5, label='predicted')
    plt.show()


if __name__ == '__main__':
    main()
