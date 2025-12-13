# shap_demo.py
# 目的:
# 1) 簡単な合成データで分類モデルを学習
# 2) SHAPで「個別の説明」「全体の重要度」「特徴量値と寄与の関係」を可視化

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1) データ作成（非線形 + 交互作用 + ノイズ + ダミー特徴を混ぜる） ---
def make_toy_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)

    # 交互作用: x1*x2、非線形: sin(x3)、しきい値的: (x4 > 0.8)
    # これらが "真のルール"（でも観測にはノイズが乗る）
    score = (
        1.3 * x1
        + 1.0 * x1 * x2
        + 1.2 * np.sin(2.2 * x3)
        + 1.5 * (x4 > 0.8).astype(float)
        - 0.7 * x2
        + 0.3 * rng.normal(0, 1, n)  # ノイズ
    )

    # ロジスティックで確率化して二値ラベル化
    p = 1 / (1 + np.exp(-score))
    y = (rng.random(n) < p).astype(int)

    # “役に立たない”ダミー特徴も混ぜて、SHAPで落ちる様子を見せる
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)

    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "noise1": noise1,
            "noise2": noise2,
        }
    )
    return X, y


# --- 2) 学習 ---
X, y = make_toy_data(n=3000, seed=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = HistGradientBoostingClassifier(
    max_depth=3,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, (proba > 0.5).astype(int)))


# --- 3) SHAP計算 ---
# まず shap を入れてない場合:
#   pip install shap

# shap.Explainer はモデルに応じて良い Explainer を推定してくれる（Tree系なら速い）
# ここでは分類なので、出力は基本「確率」になることが多い
explainer = shap.Explainer(model, X_train)

# SHAP値（テスト全体）
shap_values = explainer(X_test)

# --- 4) 見方①: 全体（重要度ランキング + 分布） ---
# beeswarm: 特徴量ごとの SHAP分布（符号も分かる）
# beeswarm：全体傾向（どの特徴が、正/負どっちに効きやすいか）
plt.figure()
# plt.title("each feature")
shap.plots.beeswarm(shap_values, show=False)
# plt.tight_layout()
# plt.show()

# bar: mean(|SHAP|) で重要度（符号は消えるが“効いてる強さ”が分かる）
# bar（mean|SHAP|）：重要度ランキング（強さ）
plt.figure()
# plt.title("feature importance")
shap.plots.bar(shap_values, show=False)
# plt.tight_layout()
# plt.show()


# --- 5) 見方②: 個別（このサンプルがなぜそう判定されたか） ---
idx = 0  # 見たいサンプル番号
print("Sample idx:", idx)
print("x =", X_test.iloc[idx].to_dict())
print("pred_proba =", float(model.predict_proba(X_test.iloc[[idx]])[:, 1]))

# waterfall: ベースラインからどの特徴がどれだけ押し上げ/押し下げしたか
# waterfall：1サンプルの説明（ベースラインからの押し上げ/押し下げ）
plt.figure()
# plt.title("baseline")
shap.plots.waterfall(shap_values[idx], show=False)
# plt.tight_layout()
# plt.show()


# --- 6) 見方③: 依存（特徴量値と寄与の関係 + 交互作用の匂い） ---
# scatter（dependence）：非線形・しきい値・交互作用の気配

# たとえば x1 は単独でも効くが、x2 と掛け算で効くように設計したので
# dependence で “相互作用っぽい形” が見えるはず
# plt.figure()
# plt.title("interaction x1 & x2")
shap.plots.scatter(shap_values[:, "x1"], color=shap_values[:, "x2"], show=False)
# plt.tight_layout()
# plt.show()

# 非線形 (sin) にした x3 も “波っぽい” 寄与が見えるはず
# plt.figure()
# plt.title("nonlinear x3")
shap.plots.scatter(shap_values[:, "x3"], show=False)
# plt.tight_layout()
# plt.show()

# しきい値にした x4 も “段差” が見えるはず
# plt.figure()
# plt.title("threshold x4")
shap.plots.scatter(shap_values[:, "x4"], show=False)
# plt.tight_layout()

plt.show()
