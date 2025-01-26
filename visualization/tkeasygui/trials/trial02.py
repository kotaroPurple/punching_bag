
"""
Button を押すと表示する値が増える with 増える値選択
"""


import TkEasyGUI as eg


WINDOW_NAME = 'CountUp'
TEXTBOX_KEY = '-output-'
COMBO_KEY = '-combobox-'
BUTTON_KEY = '-button-'


def prepare_window(title: str) -> eg.Window:
    # ウィジェット準備
    # 各ウィジェットを list に入れ layout に追加する: layout = [[...], [...]] のようにする
    layout = []  #

    # 表示用テキストボックス
    text_box = eg.Input(
        text='',  # 最初の文字
        key=TEXTBOX_KEY,  # アクセス用
        background_color='white',  # 背景色
        color='black',  # 文字色
        readonly_background_color='white',
        readonly=True)  # read only

    # インクリメントの値
    combo = eg.Combo(
            values = ['1', '2', '10', '20'],
            default_value='1',
            key=COMBO_KEY,  # アクセス用
            enable_events=True,  # アクションがあれば実行する引数
            readonly=True)

    # ボタンを押すとカウントアップする
    button = eg.Button(
        button_text='Count Up',
        key=BUTTON_KEY)

    layout.append([text_box])
    layout.append([combo])
    layout.append([button])

    # ウィンドウ作成
    window = eg.Window(title, layout)
    return window


def main() -> None:
    window = prepare_window(title=WINDOW_NAME)

    # window があるときは無限ループ
    value = 0
    window[TEXTBOX_KEY].update(value)

    while window.is_running():
        # 全イベントを取得
        for event, values in window.event_iter():
            # 閉じるボタンで終了: is_running() が False になる
            if event == eg.WIN_CLOSED:
                window.close()

            # Count Up ボタンが押されていれば、値を増やしテキストボックスを更新する
            if event == BUTTON_KEY:
                # Combo Box から増やす量を取得
                increment = int(window[COMBO_KEY].get())
                value += increment
                window[TEXTBOX_KEY].update(value)


if __name__ == '__main__':
    main()
