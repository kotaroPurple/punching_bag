
import numpy as np
import matplotlib.pyplot as plt

# サンプリング周波数と全体のデュレーション（秒）を設定
fs = 16000            # 16 kHz
duration = 1.0        # 1秒
N = int(fs * duration)

# 1. フィルタ応答 a(n) の生成（例：50msの指数関数減衰）
t_a = np.arange(0, 0.05, 1/fs)   # 50ms分のサンプル
a = np.exp(-t_a * 50)            # 減衰の速さを調整（ここでは50を乗じる）

# 2. 励起 v(n) の生成（例：100Hzのパルス列）
f0 = 100                       # 基本周波数 100Hz
period = int(fs / f0)          # パルス間隔（サンプル数）
v = np.zeros(N)
v[::period] = 1                # period ごとに1を配置

# 3. 畳み込みによる信号 x(n) = a(n) * v(n) の生成
# 'full' モードで畳み込みし、元の長さに切り出す
x = np.convolve(v, a, mode='full')
x = x[:N]

# 2. 信号に窓関数を適用（今回はハミング窓を使用）
N = len(x)
window = np.hamming(N)
x_windowed = x * window

# 3. FFTを計算して振幅スペクトルを求める
X = np.fft.fft(x_windowed)
freq_x = np.fft.fftfreq(len(X), d=1/fs)
magX = np.abs(X)
print(freq_x.shape, X.shape)

# 4. ログスペクトルを計算
# 0除算を防ぐために小さな値を加えます
log_magX = np.log(magX + np.finfo(float).eps)

# 5. ケプストラムの計算（逆 FFT を実施）
cepstrum = np.fft.ifft(log_magX).real

# 6. プロット
# 横軸はケフレンシー（サンプル単位を秒に変換）
que_frency = np.arange(N) / fs
half_n = len(que_frency) // 2

# plot
_, axes = plt.subplots(3, 2, figsize=(12, 6))


# # 4. 信号のプロット
# plt.figure(figsize=(12, 8))

axes[0, 0].plot(t_a, a, 'b-')
axes[0, 0].set_xlabel("Time [sec]")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Impulse Response a(n)")

t_v = np.arange(N) / fs
axes[1, 0].stem(t_v, v, linefmt='g-', markerfmt='go', basefmt=" ")
axes[1, 0].set_xlabel("Time [sec]")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].set_title("Excitation v(n) (Pulse Train)")

t_x = np.arange(N) / fs
axes[2, 0].plot(t_x, x, 'r-')
axes[2, 0].set_xlabel("Time [sec]")
axes[2, 0].set_ylabel("Amplitude")
axes[2, 0].set_title("Convolved Signal x(n) = a(n) * v(n)")

axes[0, 1].plot(freq_x[:half_n], magX[:half_n], color='red')
axes[0, 1].set_xlabel("freq [Hzn]")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_title("ss")
axes[0, 1].grid(True)

axes[1, 1].plot(freq_x[:half_n], log_magX[:half_n], color='red')
axes[1, 1].set_xlabel("freq [Hzn]")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].set_title("ss")
axes[1, 1].grid(True)

axes[2, 1].plot(que_frency[:half_n], cepstrum[:half_n], color='red')
axes[2, 1].set_xlabel("Quefrency [sec]")
axes[2, 1].set_ylabel("Amplitude")
axes[2, 1].set_title("Real Cepstrum")
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()
