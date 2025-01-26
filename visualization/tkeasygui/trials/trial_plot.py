
import TkEasyGUI as eg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

# 参考: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib_Embedded_Toolbar.py
# (上のQA: https://stackoverflow.com/questions/64403707/interactive-matplotlib-plot-in-pysimplegui)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


def main():
    layout = []
    layout.append([eg.Canvas(key='-controls_cv-')])
    layout.append([eg.Canvas(key='-CANVAS-')])

    with eg.Window("Matplotlib 埋め込み例", layout) as window:
        # Matplotlib のフィギュアを作成
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
        ax.set_title('embedded plot')

        # FigureCanvasTkAgg を作成
        canvas = FigureCanvasTkAgg(fig, master=window['-CANVAS-'].TKCanvas)
        canvas.draw()
        toolbar = Toolbar(canvas, window['-controls_cv-'].TKCanvas)
        toolbar.update()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        while window.is_running():
            for event in window.event_iter():
                # イベント処理
                if event == "閉じる":
                    window.close()
            # その他の処理
            window.refresh()


if __name__ == "__main__":
    main()
