
import time
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from data_generator import make_simple_data, make_simple_data2
from coordinate_transformation import (
    reflect_vector, reflect_point, transform_by_matrix, rotate_by_matrix,
    inverse_transformation_matrix)
from trial_wave_fourier import (
    WaveWithRange, fourier_plane_wave_on_circle,
    inverse_fourier_transform, inverse_fourier_transform_in_target,
    predict_bessel_x_range, predict_sinc_x_range)


class AnimationOption:
    WAVE = auto()
    K_Z0 = auto()
    K_Z1 = auto()


# global
is_running = True


def _on_close(event) -> None:
    global is_running
    _ = event
    print('Window Closed')
    is_running = False


def _prepare_coordinates_for_plot(
        rot_mat: NDArray, translation: NDArray, length: float = 1.0) -> tuple[NDArray, NDArray]:
    xyz = length * rot_mat
    ends = translation[:, None] + xyz
    return translation, ends


def angular_spectrum_trial():
    # animation
    show_animation = True
    animation_option = AnimationOption.K_Z1
    # shared
    ignore_imaginary = False
    process_time = 1.
    fs = 50
    light_speed = 3.e8
    sensor_freq = 24.e9
    sensor_position = np.array([0., 0., 0.])
    fft_number = 1023
    # object data
    object_radius = 0.01  # [m]
    object_freq = 1.  # [Hz]
    object_amp = 0.001  # [m]
    object_base_z = 0.1  # [m]
    object_max_angle = np.deg2rad(5.)  # [rad]
    range_k = predict_bessel_x_range(object_radius, n_waves=30)

    dx = 1 / (2 * range_k)
    # object_info = make_simple_data(object_freq, object_amp, object_base_z, process_time, fs)
    object_info = make_simple_data2(
        object_freq, object_amp, object_base_z, object_max_angle, process_time, fs)
    # wave number vector
    wave_number = 2 * np.pi / (light_speed / sensor_freq)
    wave_number_vector = np.array([0., 0., wave_number])

    # calculate wave
    if animation_option == AnimationOption.WAVE:
        range_xy = 2 * np.pi * dx * fft_number / 2.
    else:
        range_xy = range_k
    fig, ax = plt.subplots()
    plt_im = ax.imshow(
        np.zeros((fft_number, fft_number)), vmin=0., vmax=1.,
        extent=(-range_xy, range_xy, -range_xy, range_xy),
        origin='lower')
    plt.colorbar(plt_im)

    if animation_option == AnimationOption.WAVE:
        origin_scatter = ax.scatter([0.], [0.], color='black', s=10)

    # ウィンドウ閉鎖イベントのハンドラーを登録
    fig.canvas.mpl_connect('close_event', _on_close)

    values = []

    for object_t_mat in object_info:
        if is_running is False:
            break

        # prepare
        inv_t_mat = inverse_transformation_matrix(object_t_mat)
        rotated_wave_vector = rotate_by_matrix(inv_t_mat, wave_number_vector)
        rotated_origin = transform_by_matrix(inv_t_mat, sensor_position)
        reflected_wave_vector = reflect_vector(rotated_wave_vector, np.array([0., 0., 1.]))
        reflected_origin = reflect_point(rotated_origin, np.array([0., 0., 1.]), np.zeros(3))

        # frourier transform
        k1, k2 = reflected_wave_vector[:2]
        init_value = np.exp(-1.j * np.inner(reflected_wave_vector, reflected_origin))
        fft_info = fourier_plane_wave_on_circle(
            init_value, object_radius, k1, k2, range_k, fft_number)
        # propagate
        kz_squared = wave_number**2 - (fft_info.xs**2 + fft_info.ys**2)
        _abs_kz = np.abs(kz_squared)
        kz = np.where(kz_squared >= 0., np.sqrt(_abs_kz), 1.j * np.sqrt(_abs_kz))
        z = rotated_origin[2]
        term_k = np.exp(1.j * kz * z)
        if ignore_imaginary:
            term_k[kz_squared < 0] = complex(0.)
        fft_info2 = WaveWithRange(fft_info.value * term_k, fft_info.xs, fft_info.ys)
        # angular spectrum
        predicted_wave = inverse_fourier_transform_in_target(
            fft_info2, np.array([rotated_origin[0]]), np.array([rotated_origin[1]]))
        values.append(predicted_wave.value[0, 0])
        # animation
        if show_animation:
            all_predicted = inverse_fourier_transform(fft_info2, dx, apply_shift=True)
            if animation_option == AnimationOption.WAVE:
                _abs = np.abs(all_predicted.value)
            elif animation_option == AnimationOption.K_Z0:
                _abs = np.abs(fft_info.value)
                plt_im.set_extent((k1-range_k, k1+range_k, k2-range_k, k2+range_k))
            else:
                _abs = np.abs(fft_info2.value)
                plt_im.set_extent((k1-range_k, k1+range_k, k2-range_k, k2+range_k))
            plt_im.set_data(_abs / np.max(_abs))
            print(_abs.sum())
            if animation_option == AnimationOption.WAVE:
                origin_scatter.set_offsets(np.c_[[rotated_origin[0]], [rotated_origin[1]]])
            plt.pause(0.01)
    plt.close(fig)

    # max
    print(np.max(np.abs(values)))

    # max to 1
    values = np.array(values)
    values /= np.max(np.abs(values))

    # debug
    _theta = np.linspace(0., 2*np.pi, 100)
    _cos = np.cos(_theta)
    _sin = np.sin(_theta)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(values.real)
    plt.plot(values.imag)
    plt.plot(np.abs(values), color='gray', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(values.real, values.imag)
    plt.plot(_cos, _sin, color='gray', alpha=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.axis('equal')
    plt.show()


def trial_coordinate_conversion():
    def get_rot(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, :-1]

    def get_trans(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, -1]

    # shared
    process_time = 2.
    fs = 100
    light_speed = 3.e8
    sensor_freq = 24.e9
    sensor_position = np.array([0., 0., 0.])
    # object data
    # object_radius = 0.1  # [m]
    object_freq = 1.  # [Hz]
    object_amp = 0.1  # [m]
    object_base_z = 0.5  # [m]
    object_max_angle = np.deg2rad(10.)  # [rad]
    t_mats = make_simple_data2(
        object_freq, object_amp, object_base_z, object_max_angle, process_time, fs)
    # t_mats = make_simple_data(object_freq, object_amp, object_base_z, process_time, fs)
    # wave number vector
    wave_number = 2 * np.pi / (light_speed / sensor_freq)
    wave_number_vector = np.array([0., 0., wave_number])
    wave_number_direction = wave_number_vector / np.linalg.norm(wave_number_vector)

    # preapre plot
    axis_length = 0.1
    plt.ion()  # インタラクティブモードをオン
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # グラフの設定
    axes[0].set_xlabel('Z')
    axes[0].set_ylabel('X')
    axes[0].grid(True)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-0.5, 1.0)
    axes[0].set_ylim(-0.25, 0.25)

    axes[1].set_xlabel('new X')
    axes[1].set_ylabel('new Z')
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-0.25, 0.25)
    axes[1].set_ylim(-0.8, 0.8)

    # ウィンドウ閉鎖イベントのハンドラーを登録
    fig.canvas.mpl_connect('close_event', _on_close)

    # 座標系の描画
    axes[0].arrow(0, 0, axis_length, 0, head_width=0.02, head_length=0.03, fc='r', ec='r', label='x')  # z
    axes[0].arrow(0, 0, 0, axis_length, head_width=0.02, head_length=0.03, fc='b', ec='b', label='y')  # x
    axes[1].arrow(0, 0, axis_length, 0, head_width=0.02, head_length=0.03, fc='b', ec='b', label='x')  # new x
    axes[1].arrow(0, 0, 0, axis_length, head_width=0.02, head_length=0.03, fc='r', ec='r', label='y')  # new z

    # 描画する座標系の初期化
    current_R = get_rot(0, t_mats)
    current_T = get_trans(0, t_mats)
    starts, ends = _prepare_coordinates_for_plot(current_R, current_T, length=axis_length)

    # 座標系の軸をプロット
    coord_z, = axes[0].plot([starts[2], ends[2, 2]], [starts[0], ends[0, 2]], color='r', label='x')
    coord_x, = axes[0].plot([starts[2], ends[2, 0]], [starts[0], ends[0, 0]], color='b', label='y')

    arrow_wave, = axes[1].plot([0], [0.], color='pink')
    arrow_reflected_wave, = axes[1].plot([0], [0.], color='pink')

    # 原点を表示
    origin_scatter = axes[1].scatter([-1.], [-1.], c='black')
    reflected_scatter = axes[1].scatter([-1.], [-1.], c='gray')

    for i, object_t_mat in enumerate(t_mats):
        if is_running is False:
            break

        # prepare
        inv_t_mat = inverse_transformation_matrix(object_t_mat)
        rotated_wave_vector = rotate_by_matrix(inv_t_mat, wave_number_vector)
        rotated_origin = transform_by_matrix(inv_t_mat, sensor_position)
        reflected_wave_vector = reflect_vector(rotated_wave_vector, np.array([0., 0., 1.]))
        reflected_origin = reflect_point(rotated_origin, np.array([0., 0., 1.]), np.zeros(3))

        # 現在の回転行列と位置ベクトルを取得
        current_R = get_rot(i, t_mats)
        current_T = get_trans(i, t_mats)

        # 座標系の新しい軸を計算
        starts, ends = _prepare_coordinates_for_plot(current_R, current_T, length=axis_length)

        # 横軸と縦軸のデータを更新
        coord_z.set_data([starts[2], ends[2, 2]], [starts[0], ends[0, 2]])
        coord_x.set_data([starts[2], ends[2, 0]], [starts[0], ends[0, 0]])

        # 新座標系で元の原点を表示
        origin_scatter.set_offsets(np.c_[[rotated_origin[0]], [rotated_origin[2]]])
        reflected_scatter.set_offsets(np.c_[[reflected_origin[0]], [reflected_origin[2]]])

        # 波数ベクトル表示
        rotated_wave_direction = rotated_wave_vector / np.linalg.norm(rotated_wave_vector)
        starts = [rotated_origin[0], rotated_origin[2]]
        ends = [starts[0] + axis_length * rotated_wave_direction[0], starts[1] + axis_length * rotated_wave_direction[2]]
        arrow_wave.set_data([starts[0], ends[0]], [starts[1], ends[1]])

        rotated_wave_direction = reflected_wave_vector / np.linalg.norm(reflected_wave_vector)
        starts = [reflected_origin[0], reflected_origin[2]]
        ends = [starts[0] + axis_length * rotated_wave_direction[0], starts[1] + axis_length * rotated_wave_direction[2]]
        arrow_reflected_wave.set_data([starts[0], ends[0]], [starts[1], ends[1]])

        # プロットを再描画
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 一定時間待機（アニメーション速度の調整）
        time.sleep(0.05)  # 50ミリ秒

    plt.show()


def make_plane_wave_at_z0(
        wave_number: float, wave_number_direction: NDArray, width: float, height: float,
        number: int) -> tuple[NDArray, NDArray, NDArray]:
    # define dx, dy and meshgrid
    xs = np.linspace(-width/2, width/2, number)
    ys = np.linspace(-height/2, height/2, number)
    xx, yy = np.meshgrid(xs, ys)
    # plane wave at z=0
    wave_0 = np.exp(1.j * wave_number \
                    * (xx * wave_number_direction[0] + yy * wave_number_direction[1]))
    return wave_0, xx, yy


def calculate_angluar_spectrum(
        wave_0: NDArray, dx: float, dy: float, wave_number: float, z_: float) \
            -> tuple[NDArray, NDArray]:
    # prepare
    nx, ny = wave_0.shape
    freq_x = np.fft.fftfreq(nx, d=dx)
    freq_y = np.fft.fftfreq(ny, d=dy)
    fxx, fyy = np.meshgrid(freq_x, freq_y)
    # wave number vector
    kxx = 2 * np.pi * fxx
    kyy = 2 * np.pi * fyy

    # calcuate wave number at z
    kz_squared = wave_number**2 - (kxx**2 + kyy**2)
    _abs_kz = np.abs(kz_squared)
    kz = np.where(kz_squared >= 0., np.sqrt(_abs_kz), 1.j * np.sqrt(_abs_kz))
    term_k = np.exp(1.j * kz * z_)

    # fourier transform
    window_x = np.hanning(nx)
    window_y = np.hanning(ny)
    window_term = np.outer(window_y, window_x)
    wave_fft = np.fft.fft2(wave_0 * window_term)

    # propagation
    wave_z_fft = wave_fft * term_k

    # inverse fourier transform
    wave_z = np.fft.ifft2(wave_z_fft)
    return wave_fft, wave_z


def main():
    # wave parameter
    light_speed = 3.e8
    freq = 24.e9
    wave_number = 2 * np.pi / (light_speed / freq)
    wave_direction = np.array([0., 0., 1.])
    wave_direction = wave_direction / np.linalg.norm(wave_direction)
    # angluar spectrum parameter
    sample_width = 0.2  # [m]
    sample_height = 0.2  # [m]
    sample_number_in_one_direction = 2048

    # plane wave at z=0
    wave_0, mesh_x, mesh_y = make_plane_wave_at_z0(
        wave_number, wave_direction, sample_width, sample_height, sample_number_in_one_direction)

    # remove outside
    outside = (mesh_x > 0.025) + (mesh_x < -0.025) + (mesh_y > 0.025) + (mesh_y < -0.025)
    wave_0[outside] = 0.

    # angular sp
    # rctrum
    z_value = 0.5
    dx = sample_width / sample_number_in_one_direction
    dy = sample_height / sample_number_in_one_direction
    wave_fft, wave_z = calculate_angluar_spectrum(wave_0, dx, dy, wave_number, z_value)

    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(wave_0), extent=(mesh_x.min()*1000, mesh_x.max()*1000, mesh_y.min()*1000, mesh_y.max()*1000))
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(wave_z), extent=(mesh_x.min()*1000, mesh_x.max()*1000, mesh_y.min()*1000, mesh_y.max()*1000))
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar()

    shifted_fft_real = np.log(np.abs(np.fft.fftshift(wave_fft).real))
    shifted_fft_imag = np.log(np.abs(np.fft.fftshift(wave_fft).imag))

    plt.subplot(2, 2, 3)
    plt.imshow(shifted_fft_real)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(shifted_fft_imag)
    plt.colorbar()

    plt.show()



if __name__ == '__main__':
    angular_spectrum_trial()
    # trial_coordinate_conversion()
