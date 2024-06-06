
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv

# iv(v, z) -> Iv(z)
# exp(q.sin(px)) のフーリエ変換はベッセル関数を使って示せる
#   X(ω) = ∑ I_k(q).F(exp(jkpt))
#        = ∑ I_k(q).2πδ(ω−kp)
# パワースペクトルは
#   P(ω) = (2π)^2 . ∑∣I_k(q)∣^2 .δ(ω−kp)
# ω = np (n : int) で 0 でない値
#   P(np) = (2π)^2 . ∣I_n(q)∣^2

number = 10
p = 1.
# omegas = [2 * np.pi * (n + 1) * p for n in range(number)]

qs = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
for q in qs:
    values = np.array([[n * p, iv(n, q)] for n in range(number)])
    plt.scatter(values[:, 0], values[:, 1], label=f'{q=:.1f}', alpha=0.5)
plt.legend()
plt.show()

# # q の値を設定
# q = 0.5

# # k の範囲を定義
# k_values = np.arange(0, 10, 1)

# # I_k(q) の値を計算
# I_k_values = iv(k_values, q)

# # プロット
# plt.figure(figsize=(10, 6))
# plt.stem(k_values, I_k_values, basefmt=" ", use_line_collection=True)
# plt.title(f'Modified Bessel Functions of the First Kind I_k({q})')
# plt.xlabel('k')
# plt.ylabel(f'I_k({q})')
# plt.grid(True)
# plt.show()

