
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

LIGHT_OF_SPEED = 3.e8
SENSOR_FREQUENCY = 24.e9
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED

# model
stick_length = [0.1, 0.1]
init_angle = [-np.pi/2, np.pi/2]
stick_length = [0.1, 0.05, 0.05, 0.1]
init_angle = [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2]

number = 10
_length1 = [0.001 * (i+1) for i in range(number)]
stick_length = _length1 + _length1
init_angle = [-np.pi/2 for i in range(number)] + [np.pi/2 for i in range(number)]


# アニメーションの設定
fig1, axes = plt.subplots(ncols=2, nrows=1)
ax1, ax2 = axes


# 最初のフレームを描画する関数
def init_plot():
    ax1.clear()
    ax2.clear()

    n_objects = len(stick_length)

    # ax1
    ax1_objects = [ax1.plot([], [])[0] for _ in range(n_objects)]

    # ax2
    ax2_scatter = [ax2.scatter([], [], label=f'IQ {i}', s=10, c=f'C{i}') for i in range(n_objects)]
    ax2_plot, = ax2.plot([], [], label='Sum IQ Wave', linestyle='--', c='gray')
    ax2_objects = ax2_scatter + [ax2_plot]

    # 軸の設定
    ax1.set_title('Positions over Time')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-1., 1.)
    ax1.set_ylim(-0.1, 1.5)

    ax2.set_title('IQ Waves')
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_xlim(-4., 4.)
    ax2.set_ylim(-4., 4.)
    ax2.legend()

    result = ax1_objects + ax2_objects
    print(len(result), type(result[0]))
    return result


# アニメーションの更新関数
def update(frame, positions, iq_waves, sum_iq_wave, plt_objects_):
    from_ = max(frame - 100, 0)
    to_ = frame

    n_plot = len(plt_objects_)
    ax1_objects = plt_objects_[:n_plot//2]
    ax2_objects = plt_objects_[n_plot//2:]

    # 1つ目のプロット
    for ax1_plot, trajectory in zip(ax1_objects, positions):
        ax1_plot.set_data(trajectory[from_:to_, 0], trajectory[from_:to_, 1])

    # 2つ目のプロット
    for ax2_scatter, iq in zip(ax2_objects[:-1], iq_waves):
        ax2_scatter.set_offsets((iq[frame].real, iq[frame].imag))
    ax2_plot = ax2_objects[-1]
    ax2_plot.set_data(sum_iq_wave[from_:to_].real, sum_iq_wave[from_:to_].imag)

    return ax1_objects + ax2_objects


def generate_wave(
        distance_to_first: float, velocity_array: NDArray, direction_array: NDArray, times: NDArray,
        init_angle: float, amplitude: float = 1,) -> NDArray:
    # accumulate v.direction (N, dim) -> (N)
    inners = np.sum(velocity_array * direction_array, axis=1)  # (N)
    pre_angles = np.r_[[0.], ALPHA * cumulative_trapezoid(inners, times)]
    delta_angle = ALPHA * distance_to_first
    angles = init_angle + pre_angles + delta_angle
    # wave
    wave = amplitude * np.exp(1.j * angles)
    return wave


def main():
    # define model
    sensor_position = np.zeros(2)
    stick_center = np.array([0., 1.])
    total_time = 10.
    step_time = 0.0005
    times = np.arange(0., total_time, step_time)
    # # angular velocity
    swing_period = 3.
    angle_range = [0., np.pi]
    angles = np.zeros_like(times)
    angular_velocity = np.zeros_like(times)
    coeff = (angle_range[1] - angle_range[0]) / (swing_period / 2.)
    for i, t in enumerate(times):
        diff_t = np.fmod(t, swing_period)
        if diff_t < swing_period / 2.:
            angles[i] = coeff * diff_t
            angular_velocity[i] = coeff
        else:
            angles[i] = angle_range[1] - coeff * (diff_t - swing_period/2.)
            angular_velocity[i] = -coeff
    # stick
    _positions = []
    _velocities = []
    for _l, angle, _w in zip(stick_length, init_angle, angular_velocity):
        phi = angles + angle
        tmp = stick_center + _l * np.c_[np.cos(phi), np.sin(phi)]
        tmp_v = _l * _w * np.c_[-np.sin(phi), np.cos(phi)]
        _positions.append(tmp)
        _velocities.append(tmp_v)
    positions = np.array(_positions)
    velocities = np.array(_velocities)
    # direction
    _directions = []
    for pos in positions:
        diff = pos - sensor_position
        _directions.append(diff / np.linalg.norm(diff, axis=1)[:, None])
    directions = np.array(_directions)
    # waves
    _iq_waves = []
    n_objects = len(positions)
    for position, velocity, direction in zip(positions, velocities, directions):
        sensor_to_first_position = position[0] - sensor_position
        length = np.linalg.norm(sensor_to_first_position)
        iq = generate_wave(length, velocity, direction, times, 0., 1/n_objects)
        _iq_waves.append(iq)
    iq_waves = np.array(_iq_waves)
    sum_iq_wave = np.sum(iq_waves, axis=0)

    # plt.figure()
    # # plt.plot(sum_iq_wave[:].real, sum_iq_wave[:].imag)
    # plt.plot(times, np.abs(sum_iq_wave))
    # plt.show()

    # アニメーションの生成
    plt_objects = init_plot()
    _ = FuncAnimation(
        fig1, update, frames=len(times), init_func=lambda: plt_objects, interval=0.02,
        fargs=(positions, iq_waves, sum_iq_wave, plt_objects), blit=False, repeat=True)

    plt.show()


if __name__ == '__main__':
    main()
