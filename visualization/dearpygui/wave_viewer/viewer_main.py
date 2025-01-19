
import numpy as np
from scipy.signal import butter, lfilter
import dearpygui.dearpygui as dpg
import threading
import time


# フィルタの適用関数
def butter_filter(data, cutoff, fs, btype='low'):
    nyq = 0.5 * fs
    if isinstance(cutoff, float):
        normal_cutoff = cutoff / nyq
    else:
        normal_cutoff = [freq / nyq for freq in cutoff]
    # バターワースフィルタの設計
    b, a = butter(5, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, data)
    return y


# データの生成（サンプルとしてサイン波＋ノイズ）
def generate_data(duration=5, fs=1000):
    t = np.linspace(0, duration, int(fs*duration))
    freq = 5  # 5Hzの信号
    signal = np.sin(2 * np.pi * freq * t)
    noise = 0.5 * np.random.normal(size=t.shape)
    data = signal + noise
    return t, data


global_time, _ = generate_data(10, 1000)

# 再生制御用フラグ
is_playing = False


# 再生スレッド
def play_waveform():
    global is_playing
    while is_playing:
        # 現在のX軸の表示範囲を取得
        viewport = dpg.get_viewport_client_width()
        current_x_range = dpg.get_item_user_data("wave_plot")
        if current_x_range is None:
            current_x_range = [0, 5]  # 初期範囲
        start, end = current_x_range
        # 範囲をシフト
        shift = 0.05  # シフト量
        new_start = start + shift
        new_end = end + shift
        # dpg.set_plot_xlimits("wave_plot", min=new_start, max=new_end)
        dpg.set_axis_limits('x_axis', new_start, new_end)
        # ユーザーデータを更新
        dpg.set_item_user_data("wave_plot", [new_start, new_end])
        time.sleep(0.05)


# 再生ボタンのコールバック
def start_play():
    global is_playing, play_thread
    if not is_playing:
        is_playing = True
        play_thread = threading.Thread(target=play_waveform)
        play_thread.start()


def stop_play():
    global is_playing
    is_playing = False


# フィルタの適用とプロット更新
def apply_filter(sender, app_data, user_data):
    global filtered_data
    filter_type = dpg.get_value("filter_type")
    cutoff = dpg.get_value("cutoff_freq")
    low_cutoff = dpg.get_value("low_cutoff")
    high_cutoff = dpg.get_value("high_cutoff")
    if filter_type == "Low-Pass":
        filtered = butter_filter(original_data, cutoff, fs, btype='low')
    elif filter_type == "Band-Pass":
        # バンドパスの場合、cutoff_freqをリストで指定
        filtered = butter_filter(original_data, [low_cutoff, high_cutoff], fs, btype='band')
    filtered_data = filtered
    # 更新されたフィルタ後のデータをプロット
    dpg.set_value("filtered_series", [global_time, filtered_data])


# メイン処理
if __name__ == "__main__":
    # データ生成
    fs = 1000  # サンプリング周波数
    duration = 10  # 秒
    t, original_data = generate_data(duration, fs)
    filtered_data = original_data.copy()

    # DearPyGui のセットアップ
    dpg.create_context()

    with dpg.window(label="波形表示アプリ", width=800, height=600):
        # プロットの作成
        with dpg.plot(label="Waveform", height=400, width=780, tag="wave_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time [sec]", tag="x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Amp", tag="y_axis")
            # 受信波のプロット
            dpg.add_line_series(t.tolist(), original_data.tolist(), label="Original", parent="y_axis", tag="original_series")
            # フィルタ後の波形のプロット
            dpg.add_line_series(t.tolist(), filtered_data.tolist(), label="Filtered", parent="y_axis", tag="filtered_series")
            # ユーザーデータとしてX軸の範囲を保存
            dpg.set_item_user_data("wave_plot", [0, 5])  # 初期表示範囲

        # 再生・停止ボタン
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", callback=start_play)
            dpg.add_button(label="Stop", callback=stop_play)

        # フィルタ設定
        with dpg.collapsing_header(label="Filter Setting"):
            # フィルタタイプ選択
            dpg.add_combo(["Low-Pass", "Band-Pass"], default_value="Low-Pass", label="Filter Type", tag="filter_type")
            # カットオフ周波数入力
            dpg.add_input_float(label="CutOff [Hz]", default_value=10.0, tag="cutoff_freq")
            # バンドパス用の追加入力（表示/非表示を制御）
            with dpg.group(tag="bandpass_inputs", show=False):
                dpg.add_input_float(label="Low Cut [Hz]", default_value=5.0, tag="low_cutoff")
                dpg.add_input_float(label="High Cut [Hz]", default_value=15.0, tag="high_cutoff")
            # フィルタ適用ボタン
            dpg.add_button(label="Apply Filter", callback=apply_filter)

    # フィルタタイプ変更時のコールバック
    def filter_type_changed(sender, app_data):
        if app_data == "Band-Pass":
            dpg.configure_item("bandpass_inputs", show=True)
            dpg.configure_item("cutoff_freq", show=False)
        else:
            dpg.configure_item("bandpass_inputs", show=False)
            dpg.configure_item("cutoff_freq", show=True)

    dpg.set_item_callback("filter_type", filter_type_changed)

    dpg.create_viewport(title='DearPyGui 波形表示アプリ', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # dpg.set_primary_window("波形表示アプリ", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
