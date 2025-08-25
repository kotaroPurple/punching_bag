
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
