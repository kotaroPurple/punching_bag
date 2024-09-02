
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

SENSOR_FREQUENCY = 24.e9
LIGHT_OF_SPEED = 3.e8
ALPHA = 4 * np.pi * SENSOR_FREQUENCY / LIGHT_OF_SPEED
print(ALPHA)

d_list = [100.e-6, 500.e-6, 1.e-3, 5.e-3, 10.e-3, 50.e-3, 100.e-3]

# x の範囲
x = np.linspace(0, 20, 500)

# ベッセル関数の次数
jn_dim = 20
n_values = [i+1 for i in range(jn_dim)]

# 周波数
f_obj = 1.0  # [Hz]
f_list = [_n * f_obj for _n in n_values]

# ベッセル関数
bessel_list = []
for d in d_list:
    jn_list = []
    for n, freq in zip(n_values, f_list):
        value = jn(n, ALPHA * d)
        jn_list.append((freq, value))
    bessel_list.append((d, jn_list))

# plot
one_besseld = bessel_list[0][1]
fs, values = zip(*one_besseld)

# plt.figure(figsize=(10, 6))
# for one_bessel in bessel_list:
#     d, data = one_bessel
#     fs, values = zip(*data)
#     plt.plot(fs, np.abs(values), label=f'{d}')
# plt.legend()
# plt.show()

# x の範囲
x = np.linspace(0, 20, 500)

# ベッセル関数の次数
n_values = [i for i in range(1, 11)]

# プロット
plt.figure(figsize=(10, 6))
for n in n_values:
    _jn = jn(n, x)
    plt.plot(x/ALPHA, _jn, label=f'J_{n}(x)')

plt.vlines(d_list, 0, 1.0, color='black', alpha=0.7)
plt.xlim(0, x.max()/ALPHA)
plt.ylim(-0.6, 0.6)
plt.legend()
plt.show()
