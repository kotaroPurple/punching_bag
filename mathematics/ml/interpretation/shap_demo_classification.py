# shap_demo.py
# 目的:
# 1) 簡単な合成データで多クラス分類モデルを学習
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

    # クラスごとに異なる関係（交互作用/非線形/しきい値）を持たせる
    logit0 = (
        1.2 * x1
        - 0.8 * x2
        + 1.0 * np.sin(2.0 * x3)
        + 0.9 * (x4 > 0.7).astype(float)
        + 0.3 * rng.normal(0, 1, n)
    )
    logit1 = (
        -0.9 * x1
        + 1.3 * x2
        + 0.8 * x1 * x2
        - 0.7 * np.cos(1.7 * x3)
        + 0.6 * (x4 < -0.4).astype(float)
        + 0.3 * rng.normal(0, 1, n)
    )
    logit2 = (
        0.5 * x1
        - 0.4 * x2
        - 1.1 * np.sin(2.4 * x3)
        + 1.0 * (x4 > 0.2).astype(float)
        - 0.6 * x1 * x2
        + 0.3 * rng.normal(0, 1, n)
    )

    logits = np.vstack([logit0, logit1, logit2]).T
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / prob.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in prob])

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
X, y = make_toy_data(n=1000, seed=42)
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

proba = model.predict_proba(X_test)
print("ROC-AUC (ovr):", roc_auc_score(y_test, proba, multi_class="ovr"))
print(classification_report(y_test, model.predict(X_test)))


# --- 3) SHAP計算 ---
# まず shap を入れてない場合:
#   pip install shap

# shap.Explainer はモデルに応じて良い Explainer を推定してくれる（Tree系なら速い）
# HistGradientBoostingClassifier は callable ではないので predict_proba を渡す
explainer = shap.Explainer(model.predict_proba, X_train)

# SHAP値（テスト全体）
shap_values = explainer(X_test)
class_idx = 1  # 表示したいクラス
class_names = [f"class_{i}" for i in range(proba.shape[1])]
shap_values_class = shap_values[:, :, class_idx]

# 全クラス統合の貢献度（mean|SHAP|、クラス平均）を数値で確認
all_class_importance = pd.Series(
    np.mean(np.abs(shap_values.values), axis=(0, 2)),
    index=shap_values.feature_names,
).sort_values(ascending=False)
print("All-classes mean(|SHAP|):")
print(all_class_importance.to_string())

# --- 4) 見方①: 全体（重要度ランキング + 分布） ---
# beeswarm: 特徴量ごとの SHAP分布（符号も分かる）
# beeswarm：全体傾向（どの特徴が、正/負どっちに効きやすいか）
plt.figure()
# plt.title("each feature")
shap.plots.beeswarm(shap_values_class, show=False)
plt.title(f"beeswarm (class={class_names[class_idx]})")
# plt.tight_layout()
# plt.show()

# bar: mean(|SHAP|) で重要度（符号は消えるが“効いてる強さ”が分かる）
# bar（mean|SHAP|）：重要度ランキング（強さ）
plt.figure()
# plt.title("feature importance")
shap.plots.bar(shap_values_class, show=False)
plt.title(f"feature importance (class={class_names[class_idx]})")
# plt.tight_layout()
# plt.show()


# --- 5) 見方②: 個別（このサンプルがなぜそう判定されたか） ---
idx = 0  # 見たいサンプル番号
print("Sample idx:", idx)
print("x =", X_test.iloc[idx].to_dict())
print("pred_proba =", model.predict_proba(X_test.iloc[[idx]])[0].tolist())
print("pred_class =", class_names[int(model.predict(X_test.iloc[[idx]])[0])])

# waterfall: ベースラインからどの特徴がどれだけ押し上げ/押し下げしたか
# waterfall：1サンプルの説明（ベースラインからの押し上げ/押し下げ）
plt.figure()
# plt.title("baseline")
shap.plots.waterfall(shap_values_class[idx], show=False)
plt.title(f"waterfall (class={class_names[class_idx]})")
# plt.tight_layout()
# plt.show()


# --- 6) 見方③: 依存（特徴量値と寄与の関係 + 交互作用の匂い） ---
# scatter（dependence）：非線形・しきい値・交互作用の気配

# たとえば x1 は単独でも効くが、x2 と掛け算で効くように設計したので
# dependence で “相互作用っぽい形” が見えるはず
# plt.figure()
# plt.title("interaction x1 & x2")
shap.plots.scatter(
    shap_values_class[:, "x1"],
    color=shap_values_class[:, "x2"],
    show=False,
)
plt.title(f"dependence x1 (color=x2, class={class_names[class_idx]})")
# plt.tight_layout()
# plt.show()

# 非線形 (sin) にした x3 も “波っぽい” 寄与が見えるはず
# plt.figure()
# plt.title("nonlinear x3")
shap.plots.scatter(shap_values_class[:, "x3"], show=False)
plt.title(f"dependence x3 (class={class_names[class_idx]})")
# plt.tight_layout()
# plt.show()

# しきい値にした x4 も “段差” が見えるはず
# plt.figure()
# plt.title("threshold x4")
shap.plots.scatter(shap_values_class[:, "x4"], show=False)
plt.title(f"dependence x4 (class={class_names[class_idx]})")
# plt.tight_layout()

plt.show()
