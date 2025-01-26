
"""
ComboBox で波形を選択し表示する
"""


import TkEasyGUI as eg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk


# Interactive Matplotlib の表示
# 参考: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib_Embedded_Toolbar.py
# (上のリンクがある Q&A: https://stackoverflow.com/questions/64403707/interactive-matplotlib-plot-in-pysimplegui)


WINDOW_NAME = 'Plot'
TEXTBOX_KEY = '-output-'
COMBO_KEY = '-combobox-'
BUTTON_KEY = '-button-'
CANVAS_KEY = '-canvas-'
CONTROL_CANVAS_KEY = '-control_canvas-'


# matplotlib toolbar 用
class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


def prepare_window(title: str) -> tuple[eg.Window, FigureCanvasTkAgg, Line2D]:
    # ウィジェット準備
    # 各ウィジェットを list に入れ layout に追加する: layout = [[...], [...]] のようにする
    layout = []

    # 表示用テキストボックス
    text_box = eg.Input(
        text='Select Plot Type',  # 最初の文字
        key=TEXTBOX_KEY,  # アクセス用
        background_color='white',  # 背景色
        color='black',  # 文字色
        readonly_background_color='white',
        readonly=True)  # read only

    # インクリメントの値
    combo = eg.Combo(
            values = ['-', 'sin', 'cos', 'sin2', 'cos2'],
            default_value='-',
            key=COMBO_KEY,  # アクセス用
            enable_events=True,  # アクションがあれば実行する引数
            readonly=True)

    # ボタンを押すとカウントアップする
    button = eg.Button(
        button_text='Plot',
        key=BUTTON_KEY)

    # 表示 canvas
    control_canvas = eg.Canvas(key=CONTROL_CANVAS_KEY)
    canvas = eg.Canvas(
        key=CANVAS_KEY,
        size=(400, 300))

    layout.append([text_box])
    layout.append([combo])
    layout.append([button])
    layout.append([control_canvas])
    layout.append([canvas])

    # ウィンドウ作成
    window = eg.Window(title, layout)

    # toolbar & plot を加える
    fig, ax = plt.subplots(figsize=(5, 4))
    line, = ax.plot([], [])
    ax.set_xlabel('time [sec]')
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-1.5, 1.5)

    canvas = FigureCanvasTkAgg(fig, master=window[CANVAS_KEY].TKCanvas)
    canvas.draw()
    toolbar = Toolbar(canvas, window[CONTROL_CANVAS_KEY].TKCanvas)
    toolbar.update()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    return window, canvas, line


def plot_wave(window: eg.Window, plot_option: str, canvas: FigureCanvasTkAgg, plt_line: Line2D) -> None:
    # 時間作成
    fs = 100
    process_time = 2.
    times = np.arange(int(fs * process_time)) / fs
    # plot 準備
    match plot_option:
        case 'sin':
            y_data = np.sin(2 * np.pi * 1 * times)
            text = 'y = sin(x)'
        case 'cos':
            y_data = np.cos(2 * np.pi * 1 * times)
            text = 'y = cos(x)'
        case 'sin2':
            y_data = np.sin(2 * np.pi * 2 * times)
            text = 'y = sin(2x)'
        case 'cos2':
            y_data = np.cos(2 * np.pi * 2 * times)
            text = 'y = cos(2x)'
        case _:
            y_data = np.zeros_like(times)
            text = 'y = 0'

    # 表示
    plt_line.set_data(times, y_data)
    canvas.draw()

    # テキストボックスに何を選択したか表示
    window[TEXTBOX_KEY].update(text)


def main() -> None:
    window, canvas, plt_line = prepare_window(title=WINDOW_NAME)

    # window があるときは無限ループ
    while window.is_running():
        # 全イベントを取得
        for event, _ in window.event_iter():
            # 閉じるボタンで終了: is_running() が False になる
            if event == eg.WIN_CLOSED:
                window.close()

            # ボタンを押したら選んだものを plot する
            if event == BUTTON_KEY:
                # plot 種類を取得し描画
                plot_option = window[COMBO_KEY].get()
                plot_wave(window, plot_option, canvas, plt_line)


if __name__ == '__main__':
    main()
