
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import stft

# Streamlitのページ設定
st.set_page_config(page_title="時系列データの可視化", layout="wide")

# タイトル
st.title("Frequency Viewer")

# サイドバーでパラメータを設定
st.sidebar.header("Settings")

# サンプルデータの生成
def generate_signal(freqs, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t)
    # ノイズを追加
    signal += 0.5 * np.random.randn(len(t))
    return t, signal


# STFTの計算
def compute_stft(signal, sampling_rate, nperseg=256):
    f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=nperseg, noverlap=3*nperseg//4)
    return f, t, np.abs(Zxx)


# データのパラメータ設定
sampling_rate = st.sidebar.slider("サンプリングレート (Hz)", 100, 2000, 1000)
duration = st.sidebar.slider("信号の持続時間 (秒)", 1, 10, 5)
_freqs = st.sidebar.text_input("周波数 (カンマ区切り)", "50,150,300")
freqs = [float(f.strip()) for f in _freqs.split(",") if f.strip().isdigit()]

f_max = st.sidebar.slider("最大周波数", 0.1, 500., value=100.)
stft_window = st.sidebar.slider("STFT Window", 256, 4096, 512, step=256)

st.sidebar.divider()

graph_width_ratio = st.sidebar.slider("横幅割合", 1, 9, 8, step=1)
graph_height_ratio = st.sidebar.slider("縦幅割合", 1, 8, 8, step=1)

# 信号の生成
t, signal = generate_signal(freqs, sampling_rate, duration)

# STFTの計算
f, t_stft, Zxx = compute_stft(signal, sampling_rate, stft_window)
# Zxx /= stft_window

# フーリエ変換して周波数スペクトルを計算
fft_vals = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
# 正の周波数のみ
pos_mask = fft_freq >= 0
fft_freq = fft_freq[pos_mask]
fft_vals = np.abs(fft_vals[pos_mask])
fft_vals /= len(signal)

# レイアウトの設定
# Seabornのスタイルを適用
sns.set_style("whitegrid")

_width_ratio = [10 - graph_width_ratio, graph_width_ratio]
_height_ratio = [(10 - graph_height_ratio)//2, (10 - graph_height_ratio)//2, graph_height_ratio]

fig = plt.figure(constrained_layout=True, figsize=(15, 10))
gs = fig.add_gridspec(3, 2, width_ratios=_width_ratio, height_ratios=_height_ratio)

# 上部に元の波形
ax_waveform = fig.add_subplot(gs[0, 1])
sns.lineplot(x=t, y=signal, ax=ax_waveform)
ax_waveform.set_title("Wave")
# ax_waveform.set_xlabel("time [sec]")
ax_waveform.set_ylabel("amp.")
ax_waveform.set_ylim(-3, 3)

ax_waveform = fig.add_subplot(gs[1, 1], sharex=ax_waveform)
sns.lineplot(x=t, y=signal, ax=ax_waveform)
ax_waveform.set_title("Wave")
# ax_waveform.set_xlabel("time [sec]")
ax_waveform.set_ylabel("amp.")
ax_waveform.set_ylim(-3, 3)

# 中央にSTFT
ax_stft = fig.add_subplot(gs[2, 1], sharex=ax_waveform)
# 軸のメモリを設定
pcm = ax_stft.pcolormesh(t_stft, f, Zxx, cmap='viridis')
ax_stft.set_title("STFT")
ax_stft.set_xlabel("time [sec]")
ax_stft.set_ylim(0, f_max)
# ax_stft.set_ylabel("freq. [Hz]")
fig.colorbar(pcm, ax=ax_stft, label='amp')

# 左側に周波数強度（90度回転）
ax_freq = fig.add_subplot(gs[2, 0], sharey=ax_stft)

# FFT強度を水平棒グラフ（barh）でプロット
ax_freq.plot(fft_vals, fft_freq, color='skyblue')
ax_freq.set_title("Frequency Amp.")
ax_freq.set_xlabel("amp.")
ax_freq.set_ylabel("freq. [Hz]")
# ax_freq.set_ylim(0, fft_freq.max())

# 強度が右から左に増加するようにx軸を反転
ax_freq.invert_xaxis()

# 不要なサブプロットの削除
# gs[0,0] は空白にする
ax_empty = fig.add_subplot(gs[0,0])
ax_empty.axis('off')
ax_empty = fig.add_subplot(gs[1,0])
ax_empty.axis('off')

# Streamlitにプロットを表示
st.pyplot(fig)
