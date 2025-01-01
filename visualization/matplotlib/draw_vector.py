
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.typing import NDArray

# global
is_running = True


def on_close(event) -> None:
    global is_running
    _ = event
    print('Window Closed')
    is_running = False


def generate_simple_transformations(
        base_z: float, amp: float, angle_amp: float, freq: float,
        process_time: float, fs: int) -> NDArray:
    # times
    times = np.arange(int(process_time * fs)) / fs
    # centers
    zs = base_z + amp * np.sin(2 * np.pi * freq * times)
    # rotation matrix
    _theta = angle_amp * np.sin(2 * np.pi * freq * times)
    _cos = np.cos(_theta)
    _sin = np.sin(_theta)
    vs = np.repeat(np.array([[0., -1., 0.]]), len(times), axis=0)
    us = np.c_[_cos, np.zeros_like(_cos), -_sin]
    ns = np.c_[-_sin, np.zeros_like(_cos), -_cos]
    # transformation matrix
    t_mats = np.zeros((len(times), 4, 4))
    for i, values in enumerate((us, vs, ns)):
        t_mats[:, :-1, i] = values
    t_mats[:, 2, -1] = zs
    t_mats[:, -1, -1] = 1.
    return t_mats


def prepare_coordinates_for_plot(
        rot_mat: NDArray, translation: NDArray, length: float = 1.0) -> tuple[NDArray, NDArray]:
    xyz = length * rot_mat
    ends = translation[:, None] + xyz
    return translation, ends


if __name__ == '__main__':
    def get_rot(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, :-1]

    def get_trans(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, -1]

    # prepare data
    t_mats = generate_simple_transformations(
        base_z=1.0, amp=0.05, angle_amp=np.deg2rad(10), freq=1., process_time=1., fs=100)

    # preapre plot
    axis_length = 0.3
    plt.ion()  # インタラクティブモードをオン
    fig, ax = plt.subplots(figsize=(8, 8))

    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)

    # ウィンドウ閉鎖イベントのハンドラーを登録
    fig.canvas.mpl_connect('close_event', on_close)

    # 元の座標系の描画（オプション）
    ax.arrow(0, 0, 0.5, 0, head_width=0.05, head_length=0.1, fc='r', ec='r', label='x')
    ax.arrow(0, 0, 0, 0.5, head_width=0.05, head_length=0.1, fc='b', ec='b', label='y')

    # 描画する座標系の初期化
    current_R = get_rot(0, t_mats)
    current_T = get_trans(0, t_mats)
    starts, ends = prepare_coordinates_for_plot(current_R, current_T, length=axis_length)

    # 座標系の軸をプロット
    coord_x, = ax.plot([starts[0], ends[0, 0]], [starts[2], ends[2, 0]], color='r', label='x')
    coord_y, = ax.plot([starts[0], ends[0, 2]], [starts[2], ends[2, 2]], color='g', label='y')

    # 描画の更新ループ
    for i in range(len(t_mats)):
        if is_running is False:
            break

        # 現在の回転行列と位置ベクトルを取得
        current_R = get_rot(i, t_mats)
        current_T = get_trans(i, t_mats)

        # 座標系の新しい軸を計算
        starts, ends = prepare_coordinates_for_plot(current_R, current_T, length=axis_length)

        # X軸とY軸のデータを更新
        coord_x.set_data([starts[0], ends[0, 0]], [starts[2], ends[2, 0]])
        coord_y.set_data([starts[0], ends[0, 2]], [starts[2], ends[2, 2]])

        # プロットを再描画
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 一定時間待機（アニメーション速度の調整）
        time.sleep(0.05)  # 50ミリ秒

    plt.show()
