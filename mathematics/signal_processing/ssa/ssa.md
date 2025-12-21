# SSA (Singular Spectrum Analysis) アルゴリズム概要

この文書は `ssa.py` の実装に対応する SSA の処理手順を簡潔に説明します。

## 目的

一次元系列 $x = (x_0, x_1, \dots, x_{n-1})$ を、低次元の構造成分（トレンド、周期成分、ノイズなど）に分解し、必要な成分のみを再構成します。

## 手順

### 1. 埋め込み（Hankel 行列の構成）

窓長を $L$ とし、列数 $k$ を次で定義します。

$k = n - L + 1$

Hankel 行列 $X$ を次のように構成します。

$X = [x_0, x_1, \dots, x_{k-1}]$

各列は長さ $L$ の部分系列です。

$X_{:,j} = (x_j, x_{j+1}, \dots, x_{j+L-1})^T$

この結果、$X$ の形状は $L \times k$ になります。

### 2. 特異値分解（SVD）

Hankel 行列に SVD を適用します。

$X = U \Sigma V^T$

ここで $U$ は左特異ベクトル、$\Sigma$ は特異値、$V^T$ は右特異ベクトルです。

### 3. 成分の選択と行列再構成

任意の成分インデックス集合 $I$ を選び、対応する成分のみで部分行列 $X_I$ を再構成します。

単一成分 $i$ の場合:

$X_i = \sigma_i \cdot u_i v_i^T$

複数成分 $I$ の場合:

$X_I = \sum_{i \in I} \sigma_i \cdot u_i v_i^T$

### 4. 対角平均化（系列の復元）

再構成行列 $X_I$ の反対角方向（$i+j$ が一定）に平均を取り、元系列と同じ長さ $n$ の系列 $y$ を復元します。

$y_{p} = \frac{1}{w_p} \sum_{i+j=p} (X_I)_{i,j}$

ここで $w_p$ は反対角に含まれる要素数です。

## 実装対応

- Hankel 行列構成: `SSA._hankel`
- SVD: `SSA.fit`
- 成分再構成: `SSA.reconstruct`
- 対角平均化: `SSA._diagonal_averaging`

## 使い方の流れ

1. `SSA(window_length)` でモデル作成
2. `fit(series)` で分解
3. `reconstruct(indices)` で選択成分から系列を復元
