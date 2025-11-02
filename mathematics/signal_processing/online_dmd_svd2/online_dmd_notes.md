# オンライン DMD（インクリメンタル SVD）

本資料は `online_dmd.py` に実装されている Online Dynamic Mode Decomposition (DMD) アルゴリズムの概要をまとめたものである。逐次到着するスナップショット列 $x_k \in \mathbb{C}^n$ を用いて、線形発展作用素 $A$ の近似を $x_{k+1} \approx A x_k$ の形で逐次推定することを目的とする。

## 重み付きスナップショット行列

指数的忘却を導入するため、スナップショットには $\lambda \in (0, 1]$ の重みを掛ける。バッチ $X = [x_0, x_1, \ldots, x_{m-1}]$ に対して重み付き行列は
$
X_w = X \operatorname{diag}\left(\lambda^{m-1}, \lambda^{m-2}, \ldots, 1\right)
$
となる。遅らせたブロック $X_w^{(x)} = X_w[:, :-1]$ のインクリメンタル SVD を保持し
$
X_w^{(x)} \approx U \Sigma V^\ast
$
と分解する。ここで $U \in \mathbb{C}^{n \times r}$、対角行列 $\Sigma \in \mathbb{R}^{r \times r}$、正規直交行列 $V \in \mathbb{C}^{(m-1) \times r}$ であり、ランク $r$ は状況に応じて $r \le r_{\max}$ まで変動する。

## ストリーミング更新

新しいスナップショット $x_{\text{new}}$ が到着した際、まず直前の列 $x_{\text{prev}}$ を現在の基底に射影する。
$
p = U^\ast x_{\text{prev}}, \qquad r = x_{\text{prev}} - U p, \qquad \rho = \lVert r \rVert_2.
$
残差ノルムが
$
\rho > \tau_{\text{add}} \max\left(\lVert x_{\text{prev}} \rVert_2, 10^{-12}\right)
$
を満たし、かつ $r < r_{\max}$ のとき、正規化残差 $q = r / \rho$ を付加してランクを 1 増加させる。

SVD の更新は、小さな行列 $K$ の SVD を解くことで実現する。
$
K =
\begin{cases}
\begin{bmatrix}
\operatorname{diag}(\Sigma) & p \\
0 & \rho
\end{bmatrix} & \text{(ランク増加)}, \\
\left[\operatorname{diag}(\Sigma) \;\; p \right] & \text{(ランク維持)}.
\end{cases}
$
ここで $K = U_t \Sigma_t V_t^\ast$ とすると、更新後の基底はランク増加の場合 $\tilde{U} = [U \; q]$ を用いて $U \leftarrow \tilde{U} U_t$、ランク維持の場合は $U \leftarrow U U_t$ となり、特異値は $\Sigma \leftarrow \Sigma_t$ に置き換える。

## 交差項の更新

$V$ を保持する代わりに、圧縮された交差項
$
H = U^\ast Y_w V
$
を管理する。ここで $Y_w = X_w[:, 1:]$ は 1 列先行させたブロックである。SVD 更新後は、$H$ を
$
H \leftarrow U_t^\ast \tilde{H} V_t^\ast
$
で更新する。$\tilde{H}$ は前の $H$ と新たな射影 $z = U^\ast x_{\text{new}}$ から構成される。忘却係数は新しい列を挿入する前に $\Sigma$ と $H$ の両方へ $\lambda$ を乗じる。

## ランク切り詰め

数値安定性を保つため、次のどちらかを満たす特異値インデックス $i$ が現れた時点で切り詰めを行う。
$
\frac{\Sigma_i}{\Sigma_{\max}} < \tau_{\text{rel}}, \qquad \frac{\sum_{j=1}^i \Sigma_j^2}{\sum_{j=1}^r \Sigma_j^2} > \tau_{\text{energy}}.
$
切り詰め後、$U$、$\Sigma$、$H$ は残ったランクに合わせて縮退させる。

## 低次元作用素とスペクトル解析

低次元の線形作用素は
$
\tilde{A} = H \Sigma^{-1}
$
で構成される。$\tilde{A}$ の固有値・固有ベクトル $(\lambda_i, w_i)$ は原系のスペクトルを近似し、全状態の DMD モードは
$
\phi_i = U w_i
$
で再構成できる。初期状態 $x_0$ に対してモード振幅 $a$ は
$
\min_a \lVert U W a - x_0 \rVert_2
$
を解く最小二乗問題となる。さらに、成長率および周波数は
$
\alpha_i = \frac{\log \lambda_i}{\Delta t}, \qquad f_i = \frac{\operatorname{Im}(\log \lambda_i)}{2 \pi \Delta t}
$
から得られる。

以上が `OnlineDMD` クラスの処理手順であり、実装の拡張や他環境への移植を行う際の参考となる。
