
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


LIGHT_SPEED = 3e8
SENSOR_FREQ = 24e9
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQ / LIGHT_SPEED


def generate_iq_from_displacement(
        displacements: NDArray[np.floating], wave_number: float,
        amp: float = 1.0, theta: float = 0.0) -> NDArray[np.complex128]:
    # generate
    phase = 2 * wave_number * displacements + theta
    iq_data: NDArray[np.complex128] = amp * np.exp(1.j * phase)  # type: ignore
    return iq_data


def generate_displacement(
        times: NDArray[np.floating], f_list: list[float], theta_list: list[float], amp_list: list[float]) \
            -> NDArray[np.floating]:
    # generate
    displacements = np.zeros_like(times)
    for amp, f, theta in zip(amp_list, f_list, theta_list):
        displacements += amp * np.sin(2 * np.pi * f * times + theta)
    return displacements


def detrend_by_highpass(
        data: NDArray, cutoff: float, fs: int, zero_phase: bool = False) -> tuple[NDArray, NDArray]:
    # design high-pass filter
    _cutoff = 2 * cutoff / fs
    b, a = signal.butter(N=4, Wn=_cutoff, btype='high', analog=False)  # type: ignore

    # apply filter
    if zero_phase:
        filtered_data = signal.filtfilt(b, a, data)
    else:
        filtered_data = signal.lfilter(b, a, data)

    # calculate trend
    trend = data - filtered_data
    return filtered_data, trend  # type: ignore


def predict_parameters(
        base_iq: NDArray[np.complexfloating], comp_iq: NDArray[np.complexfloating]) \
            -> tuple[float, float]:
    # predict angle
    inner = np.vdot(comp_iq, base_iq)  # comp.conj * base
    best_angle = float(-1. * np.angle(inner, deg=False))
    # predict amplitude
    amp = np.real(np.exp(1.j * best_angle) * inner) / (np.vdot(base_iq, base_iq).real)
    return amp, best_angle


def main():
    # basic
    fs = 100
    times = np.arange(0, 10, 1 / fs)
    comp_amp = 10.0
    delta_theta = np.pi / 3

    # displacement
    diplacements = generate_displacement(
        times, [1., 2.], [0, np.pi / 4], amp_list=[0.000_1, 0.000_05])

    # base iq
    iq_base = generate_iq_from_displacement(diplacements, WAVE_NUMBER, amp=1.0, theta=np.deg2rad(0.))
    iq_base_cut, iq_base_trend = detrend_by_highpass(iq_base - np.mean(iq_base), cutoff=0.1, fs=fs, zero_phase=False)

    # comp iq
    iq_comp = generate_iq_from_displacement(diplacements, WAVE_NUMBER, amp=comp_amp, theta=delta_theta)
    iq_comp += 0.05 * (np.random.randn(len(iq_comp)) + 1.j * np.random.randn(len(iq_comp)))
    iq_comp_cut, iq_comp_trend = detrend_by_highpass(iq_comp - np.mean(iq_comp), cutoff=0.1, fs=fs, zero_phase=False)

    # predict
    amp, angle = predict_parameters(iq_base_cut, iq_comp_cut)
    print(f"Amplitude: {amp:.3f}, Angle [deg]: {np.rad2deg(angle):.2f}")

    # reconstruct
    reconstructed = amp * np.exp(1.j * angle) * iq_base_cut

    plt.plot(times, reconstructed.real, alpha=0.5)
    plt.plot(times, reconstructed.imag, alpha=0.5)
    plt.plot(times, iq_comp_cut.real, alpha=0.5)
    plt.plot(times, iq_comp_cut.imag, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
