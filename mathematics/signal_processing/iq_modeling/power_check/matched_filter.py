
import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
a = 0.5
omega = 1.0  # 角周波数
t = np.linspace(0, 2*np.pi, 1000)  # サンプル時間
x = np.exp(1j * a * np.sin(omega * t))  # 複素信号
x = x * np.exp(1j * 0.1 * (2 * np.pi))

# wt = π/2 に対応する t の値
t_center = (np.pi / 2) / omega

# t_center 付近のみ抽出するための窓幅を設定（例: window_width = 0.2）
window_width = 0.1

# t_center 付近のデータだけを抽出（窓幅の半分以内の範囲）
mask = np.abs(t - t_center) < (window_width / 2)
t_template = t[mask]       # 短い時間軸
s_template = x[mask]       # 対応する複素信号の部分

# マッチドフィルタの設計：抽出したテンプレートの時間反転・複素共役
h = np.conjugate(s_template[::-1])

# マッチドフィルタによる畳み込み
# h は短いデータサイズですが、畳み込みにより信号全体に対する検出が行えます
y = np.convolve(x, h, mode='same')

# 結果のプロット
plt.figure(figsize=(12, 8))

# 元の複素信号のプロット
plt.subplot(4, 1, 1)
plt.plot(t, np.real(x), label='Real')
plt.plot(t, np.imag(x), label='Imag')
plt.title("x(t)")
plt.xlabel("t")
plt.ylabel("Amp")
plt.legend()

# マッチドフィルタ h(t) のプロット
plt.subplot(4, 1, 2)
plt.plot(t_template, np.real(h), label='h(t) Real')
plt.plot(t_template, np.imag(h), label='h(t) Imag')
plt.title("h(t)")
plt.xlabel("t")
plt.ylabel("Amp")
plt.legend()

# 畳み込み結果の大きさ |y(t)| のプロット
plt.subplot(4, 1, 3)
plt.plot(t, np.abs(y))
plt.title("|y(t)|")
plt.xlabel("t")
plt.ylabel("Amp")

plt.subplot(4, 1, 4)
plt.plot(t, np.unwrap(np.angle(y)))
plt.title("|y(t)|")
plt.xlabel("t")
plt.ylabel("Amp")

plt.tight_layout()
plt.show()
