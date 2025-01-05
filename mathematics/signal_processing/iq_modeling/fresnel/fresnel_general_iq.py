
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad
from scipy.special import j0
from scipy import signal
from numpy.typing import NDArray



@dataclass
class FftInfo:
    abs_: NDArray
    abs_freq: NDArray
    all_abs: NDArray
    all_freq: NDArray


def apply_highpass_filter(
        data: NDArray, cutoff_freq: float, fs: int, order: int = 4, zero_phase: bool = False) -> tuple[NDArray, NDArray]:
    # 正規化カットオフ周波数 (ナイキスト周波数で正規化)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    if not 0 < normal_cutoff < 1:
        raise ValueError("カットオフ周波数は 0 < cutoff_freq < Nyquist frequency である必要があります。")

    # SOS 形式のハイパスバターワースフィルタを設計
    sos = signal.butter(N=order, Wn=normal_cutoff, btype='high', analog=False, output='sos')

    # フィルタリングの種類に応じて適用
    if zero_phase:
        filtered_signal = signal.sosfiltfilt(sos, data)
    else:
        # 因果的フィルタリング（リアルタイム処理向き）
        filtered_signal = signal.sosfilt(sos, data)
    return filtered_signal, sos


def make_z_simple_data(
        object_freq: float, object_amp: float, amp_offset: float, angle_amp: float, \
        process_time: float, fs: int) -> tuple[NDArray, NDArray]:
    # zプラス方向に位置し、法線ベクトルは zマイナス向き
    # 基準の z 位置から単振動を行う. 角度も x 軸方向に単振動する
    # time
    times = np.arange(int(process_time * fs)) / fs
    # center position
    zs = amp_offset + object_amp * np.sin(2. * np.pi * object_freq * times)
    centers = np.c_[np.zeros_like(zs), np.zeros_like(zs), zs]
    # rotation matrix
    _theta = angle_amp * np.sin(2. * np.pi * object_freq * times)
    _cos = np.cos(_theta)
    _sin = np.sin(_theta)
    vs = np.c_[np.zeros_like(_cos), np.full_like(_cos, -1.), np.zeros_like(_cos)]
    us = np.c_[_cos, np.zeros_like(_cos), -_sin]
    ns = np.c_[-_sin, np.zeros_like(_cos), -_cos]
    # transformation matrices
    t_mats = np.zeros((len(centers), 4, 4))
    t_mats[:, :-1, -1] = centers
    for i, value in enumerate([us, vs, ns]):
        t_mats[:, :-1, i] = value
    t_mats[:, -1, -1] = 1.
    return t_mats, times


def make_y_simple_data(
        object_freq: float, z_value: float, object_amp: float, amp_offset: float, angle_amp: float, \
        process_time: float, fs: int) -> tuple[NDArray, NDArray]:
    # zプラス方向に位置し、法線ベクトルは zマイナス向き
    # 基準の z 位置から単振動を行う. 角度も x 軸方向に単振動する
    # time
    times = np.arange(int(process_time * fs)) / fs
    # center position
    # マイナス x 方向から振動する
    xs = -amp_offset + object_amp * np.sin(2. * np.pi * object_freq * times)
    centers = np.c_[xs, np.zeros_like(xs), np.full_like(xs, z_value)]
    # rotation matrix
    _theta = angle_amp * np.sin(2. * np.pi * object_freq * times)
    _cos = np.cos(_theta)
    _sin = np.sin(_theta)
    us = np.c_[-_cos, np.zeros_like(_cos), -_sin]
    vs = np.c_[np.zeros_like(_cos), np.ones_like(_cos), np.zeros_like(_cos)]
    ns = np.c_[_sin, np.zeros_like(_cos), _cos]
    # transformation matrices
    t_mats = np.zeros((len(centers), 4, 4))
    t_mats[:, :-1, -1] = centers
    for i, value in enumerate([us, vs, ns]):
        t_mats[:, :-1, i] = value
    t_mats[:, -1, -1] = 1.
    return t_mats, times


def calculate_kappa_from_transformation(
        wave_number_vector: NDArray, t_mats: NDArray, normal_vec_index: int) -> NDArray:
    # normal vector
    normal_vectors = t_mats[:, :-1, normal_vec_index]
    # kappa = sqrt(wave_vec^2 - (wave_vec.normal_vec)^2)
    kappa = np.sqrt(
        np.dot(wave_number_vector, wave_number_vector) \
            - (normal_vectors @ wave_number_vector)**2)
    return kappa


def calculate_rn_from_transformation(
        positions: NDArray, t_mats: NDArray, normal_vec_index: int) -> NDArray:
    # normal vector
    normal_vectors = t_mats[:, :-1, normal_vec_index]
    # rn = sqrt(1 - (R.n)^2 / |R|^2)
    inners2 = np.sum(positions * normal_vectors, axis=1)**2
    length2 = np.sum(positions**2, axis=1)
    result = np.sqrt(1. - (inners2 / length2))
    return result


def _calculate_phase_term(wave_number_vector: NDArray, positions: NDArray) -> NDArray:
    # wave_vector: (3,)
    # positions: (N,3)
    # exp(i.(k.r + |k|.|r|))
    k_dot_r = positions @ wave_number_vector
    wave_number = np.linalg.norm(wave_number_vector)
    lengths = np.linalg.norm(positions, axis=1)
    return np.exp(1.j * (k_dot_r + wave_number * lengths))


def calculate_phase_term(wave_number_vector: NDArray, positions: NDArray) -> NDArray:
    # wave_vector: (3,)
    # positions: (N,3)
    # exp(i.(2.k.r + |k|.|r|))
    wave_number = np.linalg.norm(wave_number_vector)
    lengths = np.linalg.norm(positions, axis=1)
    return np.exp(2.j * wave_number * lengths)


def calculate_integration_term(
        wave_number: float, rns: NDArray, positions: NDArray, object_radius: float) -> NDArray:
    def integrate_func(x, rn, k, length):
        return x * j0(2. * k * rn * x) * np.exp(1.j * k / length * (x**2))

    results = [
        quad(integrate_func, 0., object_radius,
            args=(rn, wave_number, np.linalg.norm(position)), complex_func=True, limit=200)[0] \
                for position, rn in zip(positions, rns)]
    results = np.array(results)

    # 小さくなるので値を補正
    # int r dr で正規化
    coeff = 2 / (object_radius**2)
    return coeff * results


def main1():
    # plane select
    use_moving_z = True
    # parameters
    process_time = 20.  # [sec]
    fs = 100  # [Hz]
    object_freq = 0.5  # [Hz]
    object_amp = 0.001  # [m]
    object_base = 0.5  # [m]
    angle_amp = np.deg2rad(1.)  # [rad]
    object_radius = 0.1  # [m]
    # wave number vector
    light_speed = 3.e8
    sensor_freq = 24.e9
    wave_number = 2. * np.pi / (light_speed / sensor_freq)
    wave_number_vector = np.array([0., 0., wave_number])
    # object model
    if use_moving_z:
        normal_vec_index = 2
        t_mats, times = make_z_simple_data(object_freq, object_amp, object_base, angle_amp, process_time, fs)
    else:
        x_base_position = 0.1  # will be minus
        normal_vec_index = 0
        t_mats, times = make_y_simple_data(object_freq, object_base, object_amp, x_base_position, angle_amp, process_time, fs)
    # predict iq waves
    positions = t_mats[:, :-1, -1]
    # kappas = calculate_kappa_from_transformation(wave_number_vector, t_mats, normal_vec_index=2)
    rns = calculate_rn_from_transformation(positions, t_mats, normal_vec_index=normal_vec_index)
    phase_term = calculate_phase_term(wave_number_vector, positions)
    integration_term = calculate_integration_term(wave_number, rns, positions, object_radius)
    iq_wave = phase_term * integration_term

    # high pass filter
    cutoff = 0.1  # [Hz]
    filtered_iq, sos = apply_highpass_filter(iq_wave, cutoff, fs, order=4, zero_phase=False)
    filtered_phase, _ = apply_highpass_filter(phase_term, cutoff, fs, order=4, zero_phase=False)

    # cut for plot
    cut_index = 5
    n_period = 3
    sub_times = times[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]
    sub_phase_term = phase_term[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]
    sub_integration_term = integration_term[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]
    sub_iq_wave = iq_wave[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]
    sub_filtered_iq = filtered_iq[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]
    sub_filtered_phase = filtered_phase[
        cut_index * (int(fs / object_freq + 0.5)):(cut_index + n_period) * int(fs / object_freq + 0.5)]

    # plot
    fig = plt.figure(figsize=(12, 6))
    gs= GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sub_times, sub_phase_term.real)
    ax1.plot(sub_times, sub_phase_term.imag)
    ax1.set_ylim(-1, 1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(sub_times, sub_integration_term.real)
    ax2.plot(sub_times, sub_integration_term.imag)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(sub_times, sub_iq_wave.real)
    ax3.plot(sub_times, sub_iq_wave.imag)

    ax4 = fig.add_subplot(gs[0:2, 1])
    ax4.plot(sub_phase_term.real, sub_phase_term.imag, label='phase', alpha=0.5)
    ax4.plot(sub_iq_wave.real, sub_iq_wave.imag, label='iq', alpha=0.5)
    ax4.plot(sub_filtered_phase.real, sub_filtered_phase.imag, label='filt. phase', alpha=0.5)
    ax4.plot(sub_filtered_iq.real, sub_filtered_iq.imag, label='filt. iq', alpha=0.5)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.legend()

    # filter response
    response_freq, filter_response = signal.sosfreqz(sos, worN=2000, fs=fs)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5_phase = ax5.twinx()
    ax5.plot(response_freq, 20 * np.log10(abs(filter_response)), c='C0', label='Amp.', alpha=0.5)
    ax5_phase.plot(response_freq, np.rad2deg(np.unwrap(np.angle(filter_response))), c='C1', label='Phase [deg]', alpha=0.5)
    ax5_phase.axvline(cutoff, 0, 1, c='gray', alpha=0.3)
    ax5_phase.axhline(-360, 0, 1, c='gray', linestyle='--', alpha=0.5)
    ax5.set_xlim(-1, 5)
    ax5.set_xlabel('Freq [Hz]')
    ax5.set_ylabel('[dB]')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main1()
