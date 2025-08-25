
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Reproducibility
np.random.seed(0)

# ----- Synthetic data -----
n_samples = 200
# X1: large variance, NOT related to y
X1 = 5 * np.random.randn(n_samples, 1)
# X2: small variance, strongly related to y
X2 = 0.5 * np.random.randn(n_samples, 1)
# y depends mainly on X2
y = 3 * X2[:, 0] + 0.1 * np.random.randn(n_samples)

X = np.hstack([X1, X2])

# ----- Train/test split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- PCR -----
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_pca, y_train)
y_pred_pcr = lr.predict(X_test_pca)

# ----- PLS -----
pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)
y_pred_pls = pls.predict(X_test).ravel()

# ----- Print metrics -----
print("Explained variance ratio (first PC):", pca.explained_variance_ratio_[0])
print("PCR MSE:", mean_squared_error(y_test, y_pred_pcr))
print("PLS MSE:", mean_squared_error(y_test, y_pred_pls))

# ----- PLOTS: each chart its own figure (no subplots) -----

# ----- Combined plots in subplots -----
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
fig.suptitle('PCR vs PLS Analysis', fontsize=16, y=1.02)

# Original data plots
axes[0,0].scatter(X1[:, 0], y, alpha=0.7)
axes[0,0].set_xlabel("X1 (large variance, irrelevant)")
axes[0,0].set_ylabel("y")
axes[0,0].set_title("Original data: X1 vs y")

axes[0,1].scatter(X2[:, 0], y, alpha=0.7)
axes[0,1].set_xlabel("X2 (small variance, relevant)")
axes[0,1].set_ylabel("y")
axes[0,1].set_title("Original data: X2 vs y")

# Diagnostic plots
t_pcr_train = X_train_pca[:, 0]
axes[1,0].scatter(t_pcr_train, y_train, alpha=0.7)
axes[1,0].set_xlabel("PC1 score (from X only)")
axes[1,0].set_ylabel("y (train)")
axes[1,0].set_title("PCR perspective: PC1 vs y (train)")

t_pls_train = pls.x_scores_[:, 0]
axes[1,1].scatter(t_pls_train, y_train, alpha=0.7)
axes[1,1].set_xlabel("PLS score t1 (from X and y)")
axes[1,1].set_ylabel("y (train)")
axes[1,1].set_title("PLS perspective: t1 vs y (train)")

# Prediction plots
axes[2,0].scatter(y_test, y_pred_pcr, alpha=0.7)
axes[2,0].set_xlabel("True y (test)")
axes[2,0].set_ylabel("Predicted y (PCR)")
axes[2,0].set_title("PCR: Predicted vs True (test)")

axes[2,1].scatter(y_test, y_pred_pls, alpha=0.7)
axes[2,1].set_xlabel("True y (test)")
axes[2,1].set_ylabel("Predicted y (PLS)")
axes[2,1].set_title("PLS: Predicted vs True (test)")

plt.tight_layout()
plt.show()

# # Original data: X1 vs y (large variance, weak relation)
# plt.figure()
# plt.scatter(X1[:, 0], y, alpha=0.7)
# plt.xlabel("X1 (large variance, irrelevant)")
# plt.ylabel("y")
# plt.title("Original data: X1 vs y")
# plt.tight_layout()
# plt.show()

# # Original data: X2 vs y (small variance, strong relation)
# plt.figure()
# plt.scatter(X2[:, 0], y, alpha=0.7)
# plt.xlabel("X2 (small variance, relevant)")
# plt.ylabel("y")
# plt.title("Original data: X2 vs y")
# plt.tight_layout()
# plt.show()

# # Diagnostic: PC1 score vs y (PCR builds on X-only variance)
# t_pcr_train = X_train_pca[:, 0]
# plt.figure()
# plt.scatter(t_pcr_train, y_train, alpha=0.7)
# plt.xlabel("PC1 score (from X only)")
# plt.ylabel("y (train)")
# plt.title("PCR perspective: PC1 vs y (train)")
# plt.tight_layout()
# plt.show()

# # Diagnostic: PLS latent score vs y (PLS optimizes correlation with y)
# t_pls_train = pls.x_scores_[:, 0]
# plt.figure()
# plt.scatter(t_pls_train, y_train, alpha=0.7)
# plt.xlabel("PLS score t1 (from X and y)")
# plt.ylabel("y (train)")
# plt.title("PLS perspective: t1 vs y (train)")
# plt.tight_layout()
# plt.show()

# # Predictions: PCR
# plt.figure()
# plt.scatter(y_test, y_pred_pcr, alpha=0.7)
# plt.xlabel("True y (test)")
# plt.ylabel("Predicted y (PCR)")
# plt.title("PCR: Predicted vs True (test)")
# plt.tight_layout()
# plt.show()

# # Predictions: PLS
# plt.figure()
# plt.scatter(y_test, y_pred_pls, alpha=0.7)
# plt.xlabel("True y (test)")
# plt.ylabel("Predicted y (PLS)")
# plt.title("PLS: Predicted vs True (test)")
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # 再現性のため乱数固定
# np.random.seed(0)

# # データ生成
# n_samples = 200
# # X1: 大きな分散だがyに関係しない
# X1 = 5 * np.random.randn(n_samples, 1)
# # X2: 小さい分散だがyに強く効く
# X2 = 0.5 * np.random.randn(n_samples, 1)
# # 目的変数 y は X2 に依存
# y = 3 * X2[:, 0] + 0.1 * np.random.randn(n_samples)

# X = np.hstack([X1, X2])

# # 訓練テスト分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # --- PCR ---
# pca = PCA(n_components=1)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# lr = LinearRegression()
# lr.fit(X_train_pca, y_train)
# y_pred_pcr = lr.predict(X_test_pca)

# # --- PLS ---
# pls = PLSRegression(n_components=1)
# pls.fit(X_train, y_train)
# y_pred_pls = pls.predict(X_test)

# # 評価
# print("PCR MSE:", mean_squared_error(y_test, y_pred_pcr))
# print("PLS MSE:", mean_squared_error(y_test, y_pred_pls))

# # 可視化
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.scatter(y_test, y_pred_pcr)
# plt.xlabel("True y")
# plt.ylabel("Predicted y")
# plt.title("PCR")

# plt.subplot(1,2,2)
# plt.scatter(y_test, y_pred_pls)
# plt.xlabel("True y")
# plt.ylabel("Predicted y")
# plt.title("PLS")

# plt.tight_layout()
# plt.show()
