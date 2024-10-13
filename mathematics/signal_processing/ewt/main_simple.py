
import numpy as np
import matplotlib.pyplot as plt

from calc_ewt import calculate_complex_ewt


# サンプリングレートと時間軸の設定
sampling_rate = 1000  # Hz
t = np.linspace(0, 1, sampling_rate, endpoint=False)

# 複素信号の作成（50Hzと120Hzの成分）
signal = np.exp(1j * 2 * np.pi * 50 * t) + 0.5 * np.exp(1j * 2 * np.pi * 120 * t)

# 複素EWTの適用
modes = calculate_complex_ewt(signal, sampling_rate=sampling_rate)

# 結果のプロット
plt.figure(figsize=(12, 8))
plt.subplot(len(modes) + 1, 1, 1)
plt.plot(t, signal.real, label='Real Part')
plt.plot(t, signal.imag, label='Imaginary Part')
plt.title('Original Complex Signal')
plt.legend()

for i, mode in enumerate(modes):
    plt.subplot(len(modes) + 1, 1, i + 2)
    plt.plot(t, mode.real, label='Real Part')
    plt.plot(t, mode.imag, label='Imaginary Part')
    plt.title(f'Mode {i + 1}')
    plt.legend()

# plt.tight_layout()
plt.show()
