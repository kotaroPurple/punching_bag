
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def main1():
    # 時系列データの生成
    np.random.seed(0)
    t = np.arange(0, 100)
    x = np.sin(2 * np.pi * t / 30) + 0.1 * np.random.randn(100)

    # 1. 埋め込み
    L = 20  # ウィンドウ長
    K = len(x) - L + 1
    X = np.array([x[i:i+L] for i in range(K)]).T

    # 2. SVD
    U, Sigma, VT = svd(X)
    # d = np.diag(Sigma)

    # 3. 再構成
    # 最初の成分のみを使用
    X1 = Sigma[0] * np.outer(U[:, 0], VT[0, :])
    # X1 = d[0] * np.dot(U[:, 0], VT[0, :])

    # 4. 元の時系列データに戻す
    x1 = np.zeros(len(x))
    for i in range(L):
        x1[i:i+K] += X1[i, :]
    x1 /= L

    # プロット
    plt.plot(t, x, label='Original Data')
    plt.plot(t, x1, label='Trend (SSA)')
    plt.legend()
    plt.show()


def main2():
    # 1. データの生成
    # 時系列データに周期成分を含むデータを生成
    np.random.seed(0)
    t = np.arange(0, 100)
    x = np.sin(2 * np.pi * t / 30) + np.sin(2 * np.pi * t / 15) + 0.1 * np.random.randn(100)

    # 2. 埋め込み
    L = 30  # ウィンドウ長を適切に設定
    K = len(x) - L + 1
    X = np.array([x[i:i+K] for i in range(L)])

    # 3. 特異値分解（SVD）
    U, Sigma, VT = svd(X)

    # 4. 成分の再構成と選択
    # 最初のいくつかの成分を選択して再構成
    num_components = 4  # ここでは最初の4つの成分を使用
    X_reconstructed = np.zeros_like(X)

    for i in range(num_components):
        X_reconstructed += Sigma[i] * np.outer(U[:, i], VT[i, :])

    # 5. 元の時系列データに戻す
    x_reconstructed = np.zeros(len(x))
    weight = np.zeros(len(x))

    for i in range(L):
        x_reconstructed[i:i+K] += X_reconstructed[i, :]
        weight[i:i+K] += 1

    x_reconstructed /= weight

    # プロット
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, label='Original Data')
    plt.plot(t, x_reconstructed, label='Reconstructed Data (First 4 Components)')
    plt.title('SSA - Periodic Component Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main2()
