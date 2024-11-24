
# https://toukei-lab.com/shap#google_vignette

# %%
import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split


# %% データの取得
data = fetch_california_housing()
X_df = pd.DataFrame(data.data, columns=data.feature_names)
y_df = pd.Series(data.target)
X_df.describe()

# %% モデルのトレーニング
model = DecisionTreeRegressor(random_state=42)
model.fit(X_df, y_df)

# SHAPの初期化
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)

# SHAPの可視化
shap.summary_plot(shap_values, X_df)

# %%
shap.summary_plot(shap_values, X_df, plot_type="bar")
