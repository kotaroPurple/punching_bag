
import numpy as np
from numpy.typing import NDArray

# global
is_running = True


def on_close(event) -> None:
    global is_running
    _ = event
    print('Window Closed')
    is_running = False


def make_simple_data(
        object_freq: float, object_amp: float, amp_offset: float, \
        process_time: float, fs: int) -> NDArray:
    # time
    times = np.arange(int(process_time * fs)) / fs
    # center position
    zs = amp_offset + object_amp * np.sin(2. * np.pi * object_freq * times)
    centers = np.c_[np.zeros_like(zs), np.zeros_like(zs), zs]
    # rotation matrix
    rot_mat = np.array([[1., 0., 0], [0., -1., 0.], [0., 0., -1.]]).T
    # transformation matrices
    t_mats = np.zeros((len(centers), 4, 4))
    t_mats[:, :-1, -1] = centers
    t_mats[:, :-1, :-1] = rot_mat
    t_mats[:, -1, -1] = 1.
    return t_mats


def make_simple_data2(
        object_freq: float, object_amp: float, amp_offset: float, angle_amp: float, \
        process_time: float, fs: int) -> NDArray:
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
    return t_mats


def prepare_coordinates_for_plot(
        rot_mat: NDArray, translation: NDArray, length: float = 1.0) -> tuple[NDArray, NDArray]:
    xyz = length * rot_mat
    ends = translation[:, None] + xyz
    return translation, ends


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    def get_rot(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, :-1]

    def get_trans(index: int, mat: NDArray) -> NDArray:
        return mat[index, :-1, -1]

    freq = 1.
    t_mats = make_simple_data(freq, 0.5, 0.1, 2., 100)
    t_mats = make_simple_data2(freq, 0.5, 0.1, np.deg2rad(10.), 2., 100)

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
