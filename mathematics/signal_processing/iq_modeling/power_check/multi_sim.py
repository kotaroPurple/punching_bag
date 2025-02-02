
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.special import jn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


LIGHT_SPEED = 3.e8
SENSOR_FREQ = 24.e9
WAVE_NUMBER = 2 * np.pi / (LIGHT_SPEED / SENSOR_FREQ)


def generate_iq_wave(
        times: NDArray, init_phase: float, delayed_phase: float,
        displacement: float, omega: float, wave_number: float) -> NDArray[np.complex128]:
    # init
    init_wave = np.exp(1.j * init_phase)
    # phase
    phase = 2 * wave_number * displacement * np.sin(omega * times - delayed_phase)
    return init_wave * np.exp(1.j * phase)


def generate_iq_wave_from_multi_objects(
        times: NDArray, init_phases: list[float], delayed_phases: list[float],
        displacements: list[float], omegas: list[float], wave_number: float) \
            -> NDArray[np.complex128]:
    # main
    iq_wave = None
    for init_phase, delayed_phase, displacement, omega in zip(
            init_phases, delayed_phases, displacements, omegas):
        tmp_wave = generate_iq_wave(
            times, init_phase, delayed_phase, displacement, omega, wave_number)
        if iq_wave is None:
            iq_wave = tmp_wave
        else:
            iq_wave = iq_wave * tmp_wave
    # error
    if iq_wave is None:
        raise ValueError()
    return iq_wave


def generate_two_frequencies_at_n(
        times: NDArray, init_phases: list[float], delayed_phases: list[float], \
        displacements: list[float], omegas: list[float], order_n: int, order_m_max: int, \
        wave_number: float, only_side: bool, ignore_dc: bool = True,
        pos_order_m: bool = False, neg_order_m: bool = False) -> NDArray:
    # only_side: 正 or 負の周波数のみ, order_n の符号に従う
    def my_sign(value: int|float) -> int:
        if value == 0:
            return 1
        else:
            return np.sign(value)
    # constant phase term at order_n
    alphas = [2 * wave_number * d for d in displacements]
    wave_n = jn(order_n, alphas[0]) * np.exp(1j * order_n * (omegas[0] * times - delayed_phases[0]))
    wave_n *= np.exp(1j * init_phases[0])
    # order ms
    wave_ms = np.zeros_like(wave_n)
    wave_ms *= np.exp(1j * init_phases[1])
    max_order_m = order_m_max if pos_order_m else 0
    min_order_m = -order_m_max if neg_order_m else 0
    for m in range(min_order_m, max_order_m + 1):
        if only_side and (my_sign(order_n) * (order_n * omegas[0] + m * omegas[1]) < 0):
            continue
        if ignore_dc and (np.allclose(order_n * omegas[0] + m * omegas[1], 0.)):
            continue
        tmp_wave = jn(m, alphas[1]) * np.exp(1j * m * (omegas[1] * times - delayed_phases[1]))
        wave_ms += tmp_wave
    # output
    result = wave_n * wave_ms
    return result


def generate_time(start_: float, end_: float, fs: int) -> NDArray:
    time_step = 1 / fs
    times = np.arange(start_, end_, time_step)
    return times


def extract_frequency_info(data: NDArray, fs: int) -> tuple[NDArray, NDArray]:
    fft_value = np.fft.fftshift(np.fft.fft(data))
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(fft_value), d=1/fs))
    fft_abs = np.abs(fft_value) / len(data)
    return fft_abs, fft_freq


def main():
    # setting
    init_phases = [0.0 * (2 * np.pi), 0.25 * (2 * np.pi)]  # 物体までの距離依存 (同物体であれば同じ数値のはず)
    delayed_phases = [0., 0.0 * (2 * np.pi)]  # それぞれの位相ズレ
    displacements = [0.000_02, 0.000_2]  # 振幅 [m]
    _frequencies = [1., 0.2]  # [Hz]
    omegas = [2 * np.pi * f for f in _frequencies]
    # time
    start_time = 0.
    end_time = 60.
    fs = 100
    times = generate_time(start_time, end_time, fs)
    # all iq wave
    all_iq_wave = generate_iq_wave_from_multi_objects(times, init_phases, delayed_phases, displacements, omegas, WAVE_NUMBER)
    all_iq_wave -= all_iq_wave.mean()

    # iq wave
    only_side_freq = True
    order_ns = [2, 3]
    minus_order_ns = [-v for v in order_ns]
    order_m_max = 1
    iq_wave = np.zeros(times.shape, dtype=np.complex128)
    for _order_n in order_ns:
        tmp_wave = generate_two_frequencies_at_n(
            times, init_phases, delayed_phases, displacements, omegas, _order_n, order_m_max,
            WAVE_NUMBER, only_side_freq, pos_order_m=True, neg_order_m=False)
        iq_wave += tmp_wave

    minus_f_iq_wave = np.zeros(times.shape, dtype=np.complex128)
    for _order_n in order_ns:
        tmp_wave = generate_two_frequencies_at_n(
            times, init_phases, delayed_phases, displacements, omegas, _order_n, order_m_max,
            WAVE_NUMBER, only_side_freq, pos_order_m=False, neg_order_m=True)
        minus_f_iq_wave += tmp_wave

    # minus_f_iq_wave = np.zeros(times.shape, dtype=np.complex128)
    # for _order_n in minus_order_ns:
    #     tmp_wave = generate_two_frequencies_at_n(
    #         times, init_phases, delayed_phases, displacements, omegas, _order_n, order_m_max,
    #         WAVE_NUMBER, only_side_freq, pos_order_m=True)
    #     minus_f_iq_wave += tmp_wave

    # fft
    all_fft_abs, all_fft_freq = extract_frequency_info(all_iq_wave, fs)
    sub_fft_abs, _ = extract_frequency_info(iq_wave, fs)
    show_freq_max = 6.

    # power
    double_side_iq = (iq_wave + minus_f_iq_wave) / 2
    iq_wave_power = iq_wave * iq_wave.conj()
    minus_f_iq_power = minus_f_iq_wave * minus_f_iq_wave.conj()
    ds_iq_power = double_side_iq * double_side_iq.conj()

    # cut wave
    plot_proces_time = 20.
    plot_start_time = 0.
    plot_end_time = plot_start_time + plot_proces_time
    plot_start_index = int(fs * plot_start_time)
    plot_end_index = int(fs * plot_end_time)

    cut_times = times[plot_start_index:plot_end_index]
    cut_all_iq_wave = all_iq_wave[plot_start_index:plot_end_index]
    cut_iq_wave = iq_wave[plot_start_index:plot_end_index]
    cut_minus_f_iq_wave = minus_f_iq_wave[plot_start_index:plot_end_index]
    cut_double_side_iq = double_side_iq[plot_start_index:plot_end_index]
    cut_iq_wave_power = iq_wave_power[plot_start_index:plot_end_index]
    cut_minus_f_iq_power = minus_f_iq_power[plot_start_index:plot_end_index]
    cut_ds_iq_power = ds_iq_power[plot_start_index:plot_end_index]

    # plot
    fig = plt.figure(figsize=(12, 6))
    gs= GridSpec(3, 2, figure=fig)

    # # fft
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_fft_freq, all_fft_abs, alpha=0.5, label='all')
    ax1.plot(all_fft_freq, sub_fft_abs, alpha=0.5, label='sub')
    ax1.set_title('fft')
    ax1.legend()
    ax1.set_xlim(-show_freq_max, show_freq_max)
    ax1.set_yscale('log')

    # # wave: pos, neg, pos + neg
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(cut_times, cut_double_side_iq.real, alpha=0.5, c='blue', label='P,N I')
    ax01.plot(cut_times, cut_double_side_iq.imag, alpha=0.5, c='skyblue', label='P,N Q')
    ax01.plot(cut_times, cut_iq_wave.real, alpha=0.5, c='red', label='I')
    ax01.plot(cut_times, cut_iq_wave.imag, alpha=0.5, c='orange', label='Q')
    ax01.set_title('IQ waves')
    ax01.legend()

    # # wave: all vs plus freq
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(cut_times, cut_all_iq_wave.real, alpha=0.5, c='blue', label='All I')
    ax2.plot(cut_times, cut_all_iq_wave.imag, alpha=0.5, c='skyblue', label='All Q')
    ax2.plot(cut_times, cut_iq_wave.real, alpha=0.5, c='red', label='I')
    ax2.plot(cut_times, cut_iq_wave.imag, alpha=0.5, c='orange', label='Q')
    ax2.set_title('IQ waves')
    ax2.legend()

    # # wave: plus freq vs minus freq
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(cut_times, cut_iq_wave.real, alpha=0.5, c='red', label='I')
    ax3.plot(cut_times, cut_iq_wave.imag, alpha=0.5, c='orange', label='Q')
    ax3.plot(cut_times, cut_minus_f_iq_wave.real, alpha=0.5, c='green', label='minus f I')
    ax3.plot(cut_times, cut_minus_f_iq_wave.imag, alpha=0.5, c='lightgreen', label='minus f Q')
    ax3.set_title('IQ waves: positive vs negative frequency')
    ax3.legend()

    # # wave: plus freq vs minus freq
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(cut_times, cut_iq_wave_power, alpha=0.5, c='red', label='positive power')
    ax4.plot(cut_times, cut_minus_f_iq_power, alpha=0.5, c='green', label='netative power')
    # ax4.plot(cut_times, cut_ds_iq_power, alpha=0.5, c='black', label='pos + neg power')
    ax4.set_title('IQ Power: positive vs negative frequency')
    # ax4.set_ylim(0, 1.1 * max(cut_iq_wave_power.max(), cut_minus_f_iq_power.max(), cut_ds_iq_power.max()))
    ax4.legend()

    # # show
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
