
# 参考: https://github.com/ghmagazine/ml_interpret_book/blob/main/ch3/ch3_Permutation_Feature_Importance.ipynb

# %%
import warnings
from dataclasses import dataclass
from typing import Any  # 型ヒント用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
# sns.set(**get_visualization_setting())
warnings.simplefilter("ignore")  # warningsを非表示に

# %%
# シミュレーションデータも訓練データとテストデータを分けたいので


def generate_simulation_data(N, beta, mu, Sigma):
    """線形のシミュレーションデータを生成し、訓練データとテストデータに分割する

    Args:
        N: インスタンスの数
        beta: 各特徴量の傾き
        mu: 各特徴量は多変量正規分布から生成される。その平均。
        Sigma: 各特徴量は多変量正規分布から生成される。その分散共分散行列。
    """

    # 多変量正規分布からデータを生成する
    X = np.random.multivariate_normal(mu, Sigma, N)

    # ノイズは平均0標準偏差0.1(分散は0.01)で決め打ち
    epsilon = np.random.normal(0, 0.1, N)

    # 特徴量とノイズの線形和で目的変数を作成
    y = X @ beta + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの設定
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
beta = np.array([0, 1, 2])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data(N, beta, mu, Sigma)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
def plot_scatter(X, y, var_names):
    """目的変数と特徴量の散布図を作成"""

    # 特徴量の数だけ散布図を作成
    J = X.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=J, figsize=(4 * J, 4))

    for d, ax in enumerate(axes):
        sns.scatterplot(x=X[:, d], y=y, alpha=0.3, ax=ax)
        ax.set(
            xlabel=var_names[d],
            ylabel="Y",
            xlim=(X.min() * 1.1, X.max() * 1.1)
        )

    fig.show()


# 可視化
var_names = [f"X{j}" for j in range(J)]
plot_scatter(X_train, y_train, var_names)

# %%
def plot_bar(variables, values, title=None, xlabel=None, ylabel=None):
    """回帰係数の大きさを確認する棒グラフを作成"""

    fig, ax = plt.subplots()
    ax.barh(variables, values)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, None))
    fig.suptitle(title)
    fig.show()


# 線形回帰モデルの学習
lm = LinearRegression().fit(X_train, y_train)

# 回帰係数の可視化
plot_bar(var_names, lm.coef_, "coeff values", "coeff")

# %%
@dataclass
class PermutationFeatureImportance:
    """Permutation Feature Importance (PFI)

    Args:
        estimator: 全特徴量を用いた学習済みモデル
        X: 特徴量
        y: 目的変数
        var_names: 特徴量の名前
    """

    estimator: Any
    X: np.ndarray
    y: np.ndarray
    var_names: list[str]

    def __post_init__(self) -> None:
        # シャッフルなしの場合の予測精度
        # mean_squared_error()はsquared=TrueならMSE、squared=FalseならRMSE
        self.baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )

    def _permutation_metrics(self, idx_to_permute: int) -> float:
        """ある特徴量の値をシャッフルしたときの予測精度を求める

        Args:
            idx_to_permute: シャッフルする特徴量のインデックス
        """

        # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
        X_permuted = self.X.copy()

        # 特徴量の値をシャッフルして予測
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )
        y_pred = self.estimator.predict(X_permuted)

        return mean_squared_error(self.y, y_pred, squared=False)

    def permutation_feature_importance(self, n_shuffle: int = 10) -> None:
        """PFIを求める

        Args:
            n_shuffle: シャッフルの回数。多いほど値が安定する。デフォルトは10回
        """

        J = self.X.shape[1]  # 特徴量の数

        # J個の特徴量に対してPFIを求めたい
        # R回シャッフルを繰り返して平均をとることで値を安定させている
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(j) for r in range(n_shuffle)]
            )
            for j in range(J)
        ]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、
        # 差(difference)と比率(ratio)の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )

    def plot(self, importance_type: str = "difference") -> None:
        """PFIを可視化

        Args:
            importance_type: PFIを差(difference)と比率(ratio)のどちらで計算するか
        """

        fig, ax = plt.subplots()
        ax.barh(
            self.feature_importance["var_name"],
            self.feature_importance[importance_type],
            label=f"baseline: {self.baseline:.2f}",
        )
        ax.set(xlabel=importance_type, ylabel=None)
        ax.invert_yaxis() # 重要度が高い順に並び替える
        ax.legend(loc="lower right")
        fig.suptitle(f"Importance({importance_type})")
        fig.show()

# %%
# Random Forestの予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
print(f"R2: {r2_score(y_test, rf.predict(X_test)):.2f}")

# PFIを計算して可視化
# PFIのインスタンスの作成
pfi = PermutationFeatureImportance(rf, X_test, y_test, var_names)

# PFIを計算
pfi.permutation_feature_importance()

# PFIを可視化
pfi.plot(importance_type="difference")
