
# want
# - 複数物体の IQ 波形を作る
# - 正の周波数を求める (フィルタ, FFT)
# - (正の周波数を求める from 理論)
# - パワーと位相の分布
# - それぞれの周期があるか

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.special import jn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from extended_hilbert_transform import reconstruct_phase_from_analytic_signal


LIGHT_SPEED = 3.e8
SENSOR_FREQ = 24.e9
WAVE_NUMBER = 2 * np.pi / (LIGHT_SPEED / SENSOR_FREQ)


def signed_coherence_from_fft(x: NDArray, fs: int):
    N = len(x)
    if N % 2 != 0:
        raise ValueError("data length must be even")

    # FFT を計算
    X = np.fft.fft(x * np.hanning(N))
    # 周波数軸の計算
    freqs = np.fft.fftfreq(N, d=1/fs)

    # 正の周波数に対応するインデックス：k = 1, 2, ..., N//2 - 1
    pos_indices = np.arange(1, N//2)
    # 対応する負の周波数はインデックス N - k
    neg_indices = N - pos_indices

    # 正の成分と対応する負の成分
    X_pos = X[pos_indices]
    X_neg = X[neg_indices]
    abs_x_pos = np.abs(X_pos)
    abs_x_neg = np.abs(X_neg)

    # ピークのみ求める
    valid_index1 = np.r_[(abs_x_pos[:-1] > abs_x_pos[1:]), False]
    valid_index2 = np.r_[False, (abs_x_pos[1:] > abs_x_pos[:-1])]
    valid_index = valid_index1 * valid_index2
    # plt.plot(freqs[pos_indices], abs_x_pos)
    # plt.scatter(freqs[pos_indices][valid_index], abs_x_pos[valid_index], c='red', s=10)
    # plt.xlim(0, 5.)
    # plt.show()

    # 複素正規化クロススペクトル
    C_complex = (X_pos[valid_index] * np.conjugate(X_neg[valid_index])) / (abs_x_pos[valid_index] * abs_x_neg[valid_index])

    # signed coherence はその実部
    C_real = np.real(C_complex)
    # 位相差 (radians)
    phase = np.angle(C_complex)

    # 対応する周波数軸: f = k * fs / N
    f = freqs[pos_indices][valid_index]

    return f, C_real, phase


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


def extract_frequency_info(data: NDArray, fs: int) -> tuple[NDArray, NDArray]:
    fft_value = np.fft.fftshift(np.fft.fft(data))
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(fft_value), d=1/fs))
    fft_abs = np.abs(fft_value) / len(data)
    return fft_abs, fft_freq


def generate_time(start_: float, end_: float, fs: int) -> NDArray:
    time_step = 1 / fs
    times = np.arange(start_, end_, time_step)
    return times


def remove_negative_frequency(data: NDArray) -> NDArray:
    fft_value = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(fft_value))
    negative_indices = fft_freq <= 0.
    fft_value[negative_indices] = 0.
    reconstructed = np.fft.ifft(fft_value)
    return reconstructed


def extract_specified_frequency(data: NDArray, ranges: tuple[float, float], fs: int) -> NDArray:
    fft_value = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(fft_value), d=1/fs)
    indices = (fft_freq < ranges[0]) + (fft_freq > ranges[1])
    fft_value[indices] = 0.
    reconstructed = np.fft.ifft(fft_value)
    return reconstructed


def generate_phase_from_iq(data: NDArray, movemean_size: int = 0) -> tuple[NDArray, NDArray]:
    phase_array = np.unwrap(np.arctan2(data.imag, data.real))
    if movemean_size > 1:
        phase_array = np.convolve(phase_array, np.full(movemean_size, 1/movemean_size), mode='valid')
        half_n = (len(data) - len(phase_array)) // 2
        half_n_right = half_n if movemean_size % 2 == 1 else half_n + 1
        phase_array = np.r_[[phase_array[0]] * half_n, phase_array, [phase_array[-1]] * half_n_right]
    diff_phase = phase_array[1:] - phase_array[:-1]
    diff_phase = np.r_[diff_phase[0], diff_phase]
    return phase_array, diff_phase


def apply_highpass_filter(
        data: NDArray, cutoff_freq: float, fs: int, order: int = 4, zero_phase: bool = False) -> tuple[NDArray, NDArray]:
    # 正規化カットオフ周波数 (ナイキスト周波数で正規化)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    if not 0 < normal_cutoff < 1:
        raise ValueError("カットオフ周波数は 0 < cutoff_freq < Nyquist frequency である必要があります。")

    # SOS 形式のハイパスバターワースフィルタを設計
    sos = signal.butter(N=order, Wn=normal_cutoff, btype='high', analog=False, output='sos')

    # データ拡張
    input_data = np.r_[data[::-1], data, data[::-1]]

    # フィルタリングの種類に応じて適用
    if zero_phase:
        filtered_signal = signal.sosfiltfilt(sos, input_data)
    else:
        # 因果的フィルタリング（リアルタイム処理向き）
        filtered_signal = signal.sosfilt(sos, input_data)
        if not isinstance(filtered_signal, np.ndarray):
            raise ValueError()
    return filtered_signal[len(data):2*len(data)], sos


def main():
    # objects
    init_phases = [0., 0.]  # 物体までの距離依存 (同物体であれば同じ数値のはず)
    delayed_phases = [0., 0.25 * (2 * np.pi)]  # それぞれの位相ズレ
    displacements = [0.001_0, 0.000_1]  # 振幅 [m]
    _frequencies = [0.2, 1.]  # [Hz]
    omegas = [2 * np.pi * f for f in _frequencies]
    # iq wave
    start_time = 0.
    end_time = 20.
    fs = 100
    times = generate_time(start_time, end_time, fs)
    iq_wave = generate_iq_wave_from_multi_objects(
        times, init_phases, delayed_phases, displacements, omegas, WAVE_NUMBER)

    noise_iq = 0.005 * np.random.randn(len(iq_wave)) + 1.j * 0.005 * np.random.randn(len(iq_wave))
    iq_wave = iq_wave + 0 * noise_iq
    iq_wave_minus_mean = iq_wave - iq_wave.mean()
    # iq_wave_minus_mean, _ = apply_highpass_filter(iq_wave, 0.1, fs, 4, zero_phase=False)

    # extract positive frequencies
    # iq_positive_frequency = remove_negative_frequency(iq_wave_minus_mean)
    iq_positive_frequency = extract_specified_frequency(iq_wave_minus_mean, (2., 2.8), fs)
    positive_amp = np.abs(iq_positive_frequency)
    positive_phase, positive_diff_phase = generate_phase_from_iq(iq_positive_frequency, 5)

    modified_phase, modified_outliers = reconstruct_phase_from_analytic_signal(iq_positive_frequency, tau=1/fs)
    modified_phase_modulation = np.r_[0., np.diff(modified_phase)]
    print(modified_outliers)

    # frequency powers
    fft_abs, fft_freq = extract_frequency_info(iq_positive_frequency, fs)
    min_freq = -5.
    max_freq = 5.

    # plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(times, iq_wave.real, alpha=0.5)
    plt.plot(times, iq_wave.imag, alpha=0.5)
    plt.plot(times, iq_positive_frequency.real, alpha=0.5)
    plt.plot(times, iq_positive_frequency.imag, alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(fft_freq, fft_abs)
    plt.xlim(min_freq, max_freq)
    # plt.yscale('log')

    ax1 = plt.subplot(2, 2, 3)
    ax2 = ax1.twinx()
    ax1.plot(times, positive_amp, c='C0', alpha=0.5, label='amp')
    ax2.plot(times, positive_diff_phase, c='C1', alpha=0.5, label='diff phase')
    ax2.plot(times, modified_phase_modulation, c='C2', alpha=0.5, label='extended diff phase')
    ax1.legend()
    ax2.legend()

    plt.subplot(2, 2, 4)
    plt.plot(times, iq_positive_frequency.real, alpha=0.5)
    plt.plot(times, iq_positive_frequency.imag, alpha=0.5)

    plt.show()


def coherence_trial():
    # objects
    init_phases = [0., 0.]  # 物体までの距離依存 (同物体であれば同じ数値のはず)
    delayed_phases = [0., 1. * (2 * np.pi)]  # それぞれの位相ズレ
    displacements = [0.001_0, 0.000_1]  # 振幅 [m]
    _frequencies = [0.1, 1.]  # [Hz]
    omegas = [2 * np.pi * f for f in _frequencies]
    # iq wave
    start_time = 0.
    end_time = 30.
    fs = 100
    times = generate_time(start_time, end_time, fs)
    iq_wave = generate_iq_wave_from_multi_objects(
        times, init_phases, delayed_phases, displacements, omegas, WAVE_NUMBER)

    # coherence
    coherence_freq, corehences, phase = signed_coherence_from_fft(iq_wave, fs)

    # fft
    fft_abs, fft_freq = extract_frequency_info(iq_wave, fs)
    fft_positive_freq = fft_freq[fft_freq >= 0]
    fft_positive_abs = fft_abs[fft_freq >= 0]

    _, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    ax.plot(fft_positive_freq, fft_positive_abs, alpha=0.5)
    ax2.scatter(coherence_freq, corehences, color='pink', s=10)
    ax.set_xlim(-0.5, 10.)
    ax2.set_xlim(-0.5, 10.)
    ax.set_yscale('log')
    plt.show()


if __name__ == '__main__':
    # main()
    coherence_trial()
