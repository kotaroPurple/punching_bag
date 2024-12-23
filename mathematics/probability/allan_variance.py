
import numpy as np
import matplotlib.pyplot as plt

# def allan_variance(x, max_tau):
#     N = len(x)
#     tau_values = []
#     allan_variances = []

#     # tau を 1 から max_tau までの値に設定
#     for tau in range(1, max_tau + 1):
#         m = N // tau  # スロット数
#         if m <= 1:  # 十分なデータポイントがない場合はスキップ
#             continue

#         # 各スロットの平均を計算
#         averages = [np.mean(x[i * tau: (i + 1) * tau]) for i in range(m)]

#         # 隣接スロット間の変動を計算
#         delta_x = np.diff(averages)

#         # アラン分散を計算
#         allan_var = 0.5 * np.mean(delta_x**2)

#         tau_values.append(tau)
#         allan_variances.append(allan_var)

#     return np.array(tau_values), np.array(allan_variances)


def allan_variance(x, max_tau_exp=10):
    N = len(x)
    tau_values = []
    allan_variances = []

    # 2の累乗でスケールを取る
    tau_values = [2**i for i in range(1, max_tau_exp+1)]
    taus = []

    for tau in tau_values:
        m = N // tau  # スロット数
        if m <= 1:
            continue

        # 各スロットの平均を計算
        averages = [np.mean(x[i * tau: (i + 1) * tau]) for i in range(m)]

        # 隣接スロット間の変動を計算
        delta_x = np.diff(averages)

        # アラン分散を計算
        allan_var = 0.5 * np.mean(delta_x**2)

        taus.append(tau)
        allan_variances.append(allan_var)

    return np.array(taus), np.array(allan_variances)


def allan_variance_optimized(x, max_tau_exp=10):
    N = len(x)
    tau_values = []
    allan_variances = []

    # 2の累乗でスケールを取る
    tau_values = [2**i for i in range(1, max_tau_exp + 1)]
    taus = []

    # 和を保存するための累積和
    previous_sums = np.cumsum(x)

    # 和を再利用して計算
    for tau in tau_values:
        m = N // tau  # スロット数
        if m <= 1:
            continue

        # start_idx と end_idx を配列として生成
        start_idx = np.arange(0, N - tau + 1, tau)
        end_idx = np.minimum(start_idx + tau, N - 1)
        # 各スロットの和を計算
        slot_sums = previous_sums[end_idx - 1]  # 終了インデックスで累積和を取得
        slot_sums[1:] -= previous_sums[start_idx[1:] - 1]  # 開始インデックスを引く（スライディングウィンドウの効果）
        # 各スロットの平均を求める
        slot_counts = end_idx - start_idx  # スロットごとのサンプル数
        averages = slot_sums / slot_counts
        # 隣接スロット間の変動を計算
        delta_x = np.diff(averages)
        # アラン分散を計算
        allan_var = 0.5 * np.mean(delta_x**2)
        allan_variances.append(allan_var)
        taus.append(tau)

    return np.array(taus), np.array(allan_variances)


# 例: 信号列 x の生成（ホワイトノイズ + ランダムウォーク）
N = 1000
white_noise = np.random.normal(0, 1, N)  # ホワイトノイズ
random_walk = np.cumsum(np.random.normal(0, 1, N))  # ランダムウォーク

# ホワイトノイズのアラン分散を計算
# tau_values, allan_variances_white = allan_variance(white_noise, 10)
tau_values, allan_variances_white = allan_variance_optimized(white_noise, 10)

# ランダムウォークのアラン分散を計算
# tau_values, allan_variances_rw = allan_variance(random_walk, 10)
tau_values, allan_variances_rw = allan_variance_optimized(random_walk, 10)

# アラン分散をプロット
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(white_noise)
plt.plot(random_walk)

plt.subplot(212)
plt.plot(tau_values, allan_variances_white, label='White Noise', color='b')
plt.plot(tau_values, allan_variances_rw, label='Random Walk', color='r')
plt.xlabel('Tau')
plt.ylabel('Allan Variance')
plt.title('Allan Variance for White Noise and Random Walk')
plt.legend()
plt.grid(True)
plt.show()
