# 荷重センサ特徴量可視化ツール

## 概要
荷重センサの時系列データから機械学習用の特徴量を計算し、可視化するGUIアプリケーションです。

## 機能
- CSV/Excelファイルの読み込み
- 20種類以上の特徴量自動計算
  - 統計的特徴量（平均、標準偏差、最大値、最小値など）
  - 周波数領域特徴量（スペクトル重心、スペクトルエネルギーなど）
  - 時間領域特徴量（移動統計、変化点検出など）
- 時系列データのプロット
- 特徴量分布の可視化
- 特徴量相関行列の表示
- 計算結果のCSV保存

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python load_sensor_gui.py
```

### 基本的な使い方
1. "ファイル選択"ボタンでCSV/Excelファイルを選択
2. "データ読み込み"でファイルを読み込み
3. データ列を選択し、ウィンドウサイズを設定
4. "特徴量計算"で特徴量を計算
5. 各種プロットボタンで可視化
6. "特徴量保存"で結果をCSV出力

## データ形式
- CSV形式またはExcel形式
- 1列目に時系列の荷重データ
- ヘッダー行必須

## 計算される特徴量
- **統計的特徴量**: mean, std, max, min, range, rms, peak_to_peak, crest_factor, skewness, kurtosis
- **周波数領域**: spectral_centroid, spectral_energy
- **時間領域**: mean_of_rolling_mean, std_of_rolling_mean, mean_of_rolling_std, std_of_rolling_std
- **変化点**: mean_diff, std_diff, max_diff, zero_crossing_rate


## DearPyGUI プロット機能の実装ガイド

### 基本的なプロット作成手順

#### 1. プロットウィンドウの作成
```python
with dpg.plot(label="Plot", height=350, width=480, tag="main_plot"):
    dpg.add_plot_legend()
    dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="x_axis")
    dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="y_axis")
```

#### 2. データ系列の追加
```python
# 線グラフ
dpg.add_line_series(x_values, y_values, label="Series Name", parent="y_axis")

# 棒グラフ
dpg.add_bar_series(x_values, y_values, label="Bar Series", parent="y_axis")

# 散布図
dpg.add_scatter_series(x_values, y_values, label="Scatter", parent="y_axis")
```

#### 3. 軸の設定と調整
```python
# 軸ラベルの設定
dpg.set_item_label("x_axis", "Time (Sample)")
dpg.set_item_label("y_axis", "Value")

# カスティックを設定
dpg.set_axis_ticks("x_axis", tuple(zip(x_values, labels)))

# 軸を自動フィット
dpg.fit_axis_data("x_axis")
dpg.fit_axis_data("y_axis")
```

### 重要なノウハウと注意点

#### 1. プロット系列のクリア
```python
def clear_plot(self) -> None:
    # 既存の系列を削除
    children = dpg.get_item_children("y_axis", 1)
    if children:
        for child in children:
            dpg.delete_item(child)
```

#### 2. 複数Y軸の実装
```python
# 第二Y軸の追加
dpg.add_plot_axis(dpg.mvYAxis, label="Feature Values", tag="y_axis_2")

# 各軸に系列を追加
dpg.add_line_series(x_values, sensor_data, parent="y_axis")
dpg.add_line_series(x_values, feature_data, parent="y_axis_2")
```

#### 3. エラーハンドリング
```python
try:
    dpg.delete_item("y_axis_2")
except SystemError:
    pass  # アイテムが存在しない場合のエラーを無視
```

#### 4. matplotlibからの移行のポイント

**避けるべき問題:**
- matplotlibのスレッド警告: `matplotlib.use('Agg')`が必要
- テクスチャ変換の複雑さ: バッファ処理でIndexError発生のリスク

**DearPyGUIの利点:**
- ネイティブGUI統合: matplotlibより軽量で高速
- リアルタイム更新: プロットの動的な変更が容易
- メモリ効率: テクスチャ変換が不要

#### 5. データ準備のベストプラクティス
```python
# numpy配列をリストに変換（DearPyGUI要求）
x_values = list(range(len(data)))
y_values = list(data.values)

# 時系列特徴量の時間軸調整
window_size = dpg.get_value("window_size")
feature_x_values = list(range(window_size//2, window_size//2 + len(feature_values)))
```

#### 6. パフォーマンス最適化
- 大量データの場合は表示する系列数を制限（例: 最大4つの特徴量）
- プロット更新前に必ず既存系列をクリア
- 軸の自動フィットを適切に使用してユーザビリティを向上

### トラブルシューティング

**よくある問題と解決方法:**
1. **SystemError**: アイテムが存在しない → try-catch で処理
2. **系列が表示されない**: parent指定を確認、軸タグの一致を確認
3. **メモリリーク**: プロット更新時に古い系列を適切に削除
4. **座標系の不一致**: データ型をlistに統一、範囲を確認