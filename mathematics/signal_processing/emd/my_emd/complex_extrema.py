
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def calculate_envelope(z):
    # 複素数列の振幅と位相を計算
    r = np.abs(z)
    theta = np.angle(z)

    # 中間角度と符号の計算
    B = np.zeros_like(theta)
    E = np.zeros_like(theta)
    E[0] = 1  # 初期値
    B[0] = theta[0]

    for m in range(1, len(theta)):
        E[m] = np.sign(np.cos(theta[m] - B[m - 1]))
        B[m] = theta[m] + (np.pi / 2) * (1 - E[m])

    # 前進差分の計算
    M = np.zeros(len(z) - 1)
    for m in range(len(M)):
        M[m] = E[m + 1] * r[m + 1] - E[m] * r[m]

    # 極大類似点と極小類似点の抽出
    maxima = []
    minima = []
    for m in range(1, len(M)):
        if M[m] * M[m - 1] < 0:  # 符号が変化
            if M[m] - M[m - 1] < 0:
                maxima.append((m, z[m]))
            elif M[m] - M[m - 1] > 0:
                minima.append((m, z[m]))

    # 包絡線の計算
    maxima_indices, maxima_values = zip(*maxima)
    minima_indices, minima_values = zip(*minima)

    # スプライン補間
    upper_real = CubicSpline(maxima_indices, np.real(maxima_values))(np.arange(len(z)))
    upper_imag = CubicSpline(maxima_indices, np.imag(maxima_values))(np.arange(len(z)))
    lower_real = CubicSpline(minima_indices, np.real(minima_values))(np.arange(len(z)))
    lower_imag = CubicSpline(minima_indices, np.imag(minima_values))(np.arange(len(z)))

    upper_envelope = upper_real + 1j * upper_imag
    lower_envelope = lower_real + 1j * lower_imag

    return upper_envelope, lower_envelope

# テスト用の複素数列
np.random.seed(0)
number = 20
t = np.linspace(0, 10, number)
z = (np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t) + 0.001 * np.random.randn(number)) + \
    1j * (np.cos(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t) + 0.001 * np.random.randn(number))


# 複素数列の生成: 円周上に sin カーブを描く
t = np.linspace(0, 2 * np.pi, 500)
r = 1 + 0.5 * np.sin(5 * t)  # 振幅は sin カーブに従う
z = r * (np.cos(t) + 1j * np.sin(t))  # 複素平面上の円周
# z += 2.0 + 1.j*0

# 包絡線を計算
upper_envelope, lower_envelope = calculate_envelope(z)

# プロット
plt.figure(figsize=(12, 6))
plt.plot(np.real(z), np.imag(z), label="Original Complex Series", color="blue", alpha=0.7)
plt.plot(np.real(upper_envelope), np.imag(upper_envelope), label="Upper Envelope", color="red")
plt.plot(np.real(lower_envelope), np.imag(lower_envelope), label="Lower Envelope", color="green")
plt.legend()
plt.title("Complex Series and Envelopes")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid()
plt.show()
