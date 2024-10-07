
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


LIGHT_OF_SPEED = 3.e8
SENSOR_FREQUENCY = 24.e9
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED


# アニメーションの設定
fig1, axes = plt.subplots(ncols=2, nrows=1)
ax1, ax2 = axes

# 最初のフレームを描画する関数
def init():
    ax1.clear()
    ax2.clear()
    return []


# アニメーションの更新関数
def update(frame, positions, iq_waves, sum_iq_wave):
    from_ = max(frame - 100, 0)
    to_ = frame

    # 1つ目のプロット
    ax1.clear()
    ax1.plot(positions[0, from_:to_, 0], positions[0, from_:to_, 1])
    ax1.plot(positions[1, from_:to_, 0], positions[1, from_:to_, 1])
    ax1.plot()
    ax1.set_title('Positions over Time')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-1., 1.)
    ax1.set_ylim(-0.1, 1.5)

    # 2つ目のプロット
    ax2.clear()
    ax2.scatter(np.real(iq_waves[0, frame]), np.imag(iq_waves[0, frame]), label='IQ Wave 1', s=10)
    ax2.scatter(np.real(iq_waves[1, frame]), np.imag(iq_waves[1, frame]), label='IQ Wave 2', s=10)
    ax2.plot(
        sum_iq_wave[from_:to_].real, sum_iq_wave[from_:to_].imag,
        label='Sum IQ Wave', linestyle='--', c='C2')
    ax2.set_title('IQ Waves')
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.legend()
    ax2.set_xlim(-2., 2.)
    ax2.set_ylim(-2., 2.)
    return []


def generate_wave(
        velocity_array: NDArray, direction_array: NDArray, times: NDArray,
        init_angle: float, amplitude: float = 1,) -> NDArray:
    # accumulate v.direction (N, dim) -> (N)
    inners = np.sum(velocity_array * direction_array, axis=1)  # (N)
    pre_angles = np.r_[[0.], ALPHA * cumulative_trapezoid(inners, times)]
    angles = init_angle + pre_angles
    # wave
    wave = amplitude * np.exp(1.j * angles)
    return wave


def main():
    # define model
    sensor_position = np.zeros(2)
    stick_center = np.array([0.5, 1.])
    total_time = 10
    step_time = 0.002
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
    length = [0.1, 0.1]
    init_angle = [-np.pi/2, np.pi/2]
    _positions = []
    _velocities = []
    for _l, angle, _w in zip(length, init_angle, angular_velocity):
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
    for velocity, direction in zip(velocities, directions):
        iq = generate_wave(velocity, direction, times, 0., 1.)
        _iq_waves.append(iq)
    iq_waves = np.array(_iq_waves)
    sum_iq_wave = np.sum(iq_waves, axis=0)

    # アニメーションの生成
    _ = FuncAnimation(fig1, update, frames=len(times), init_func=init,
                    fargs=(positions, iq_waves, sum_iq_wave), blit=False, repeat=True)

    plt.show()


if __name__ == '__main__':
    main()
