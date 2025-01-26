
"""
FileBrowse で指定した csv ファイルをグラフ表示する
https://github.com/kujirahand/tkeasygui-python/blob/main/tests/filebrowse_test.py
"""


import TkEasyGUI as eg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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


def prepare_window(title: str) -> tuple[eg.Window, FigureCanvasTkAgg, Axes, Line2D]:
    # ウィジェット準備
    # 各ウィジェットを list に入れ layout に追加する: layout = [[...], [...]] のようにする
    layout = []

    # 表示用テキストボックス
    text_box = eg.Input(
        text='Input a File',  # 最初の文字
        key=TEXTBOX_KEY,  # アクセス用
        size=(40, 1),
        background_color='white',  # 背景色
        color='black',  # 文字色
        readonly_background_color='white',)

    # FileBrowser: eg.Input と同じ list に入れると指定ファイルが Input に反映される
    file_browser = eg.FileBrowse(
        button_text='browse',  # 検索ボタンの文字列
        file_types=(('csv file', '*.csv'),))

    # インクリメントの値
    combo = eg.Combo(
            values = ['Column1', 'Column2', 'Column3', 'Column4'],
            default_value='Column1',
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

    layout.append([text_box, file_browser])
    layout.append([combo])
    layout.append([button])
    layout.append([control_canvas])
    layout.append([canvas])

    # ウィンドウ作成
    window = eg.Window(title, layout)

    # toolbar & plot を加える
    fig, ax = plt.subplots(figsize=(5, 4))
    line, = ax.plot([], [])
    ax.set_xlabel('index')

    canvas = FigureCanvasTkAgg(fig, master=window[CANVAS_KEY].TKCanvas)
    canvas.draw()
    toolbar = Toolbar(canvas, window[CONTROL_CANVAS_KEY].TKCanvas)
    toolbar.update()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    return window, canvas, ax, line


def plot_data(
        filepath: str, column_index: int, canvas: FigureCanvasTkAgg,
        ax: Axes, plt_line: Line2D) -> None:
    # ファイルを読み込む
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        if (data.ndim != 2) or (data.shape[1] != 4):
            eg.popup_warning('data shape is not (N,4)', title='Warning')
            return
    # ファイルがなければポップアップ表示
    except FileNotFoundError:
        eg.popup_warning(f'not found: {filepath}', title='Warning')
        return
    # 表示データ選択
    plot_data = data[:, column_index]
    plt_line.set_data(np.arange(len(plot_data)), plot_data)
    # 表示範囲を自動調整
    ax.relim()
    ax.autoscale_view()
    # canvas 再描画
    canvas.draw()


def main() -> None:
    window, canvas, plt_ax, plt_line = prepare_window(title=WINDOW_NAME)

    # window があるときは無限ループ
    while window.is_running():
        # 全イベントを取得
        for event, _ in window.event_iter():
            # 閉じるボタンで終了: is_running() が False になる
            if event == eg.WIN_CLOSED:
                window.close()

            # ボタンを押したら選んだものを plot する
            if event == BUTTON_KEY:
                # ファイルパス
                filepath = window[TEXTBOX_KEY].get()
                # カラムを選択し、ファイルにある情報を表示する
                column_select = window[COMBO_KEY].get()
                column_index = int(column_select.split('Column')[1]) - 1  # 0 始まり
                plot_data(filepath,  column_index, canvas, plt_ax, plt_line)


if __name__ == '__main__':
    main()
