# Lomb-Scargle 理論メモ

## 1. なぜ Lomb-Scargle で周波数解析できるか

Lomb-Scargle periodogram は、各周波数 $f$（角周波数 $\omega = 2\pi f$）ごとに

$y(t_i) \approx A\cos(\omega t_i) + B\sin(\omega t_i)$

という正弦・余弦モデルを最小二乗で当てはめ、その「当てはまりの良さ」を周波数ごとに評価します。

要するに、

- その周波数成分がデータ中に強ければ残差が大きく減る
- 弱ければ残差はあまり減らない

ため、周波数軸上でピークを見れば卓越周波数が分かります。

通常のFFTは等間隔サンプリングを前提としますが、Lomb-Scargle は $t_i$ が不等間隔でも直接最小二乗を解くため、欠損区間や不規則観測でも使えます。

---

古典的な（平均除去後の）Lomb-Scargle の1つの形は

$P(\omega) = \frac{1}{2\sigma_y^2}\left[\frac{\left(\sum_i y_i\cos\omega(t_i-\tau)\right)^2}{\sum_i \cos^2\omega(t_i-\tau)} + \frac{\left(\sum_i y_i\sin\omega(t_i-\tau)\right)^2}{\sum_i \sin^2\omega(t_i-\tau)}\right]$

です。ここで $\tau$ は

$\tan(2\omega\tau) = \frac{\sum_i \sin(2\omega t_i)}{\sum_i \cos(2\omega t_i)}$

を満たす位相シフトで、正弦項と余弦項の直交性を整える役割があります。

この式は「各周波数で正弦波基底に射影して得られる分散説明量」に相当し、ピーク位置が周波数推定値になります。

## 2. 計算量の基本

周波数点数を $M$、観測点数を $N$ とすると、素直な実装は各周波数で全サンプル和を計算するため概ね

$O(NM)$

です。NumPy の自作版や SciPy の通常形はこのタイプ（ただし SciPy はC実装で定数倍が小さい）です。

## 3. Astropy の fast 法がなぜ速いか

Astropy の `method='fast'` / `method='fastchi2'` は、等間隔の周波数グリッドを利用して高速化します。

直感的には、必要な和

$\sum_i y_i e^{j\omega t_i},\; \sum_i e^{j\omega t_i},\; \sum_i y_i\cos(2\omega t_i),\; \sum_i y_i\sin(2\omega t_i)$

のような「多周波数での三角和」を、

- 近傍格子への補間（extirpolation / gridding）
- 畳み込み
- FFT

でまとめて計算し、1周波数ずつ愚直に回すのを避けます。

その結果、条件が揃うと計算量はおおむね

$O(N\log N)$（実装や設定次第で $O((N+M)\log M)$ とみなせる形）

になります。

### 重要な前提

- 周波数グリッドが等間隔（`freq = f_0 + k\Delta f`）
- `method='auto'` で fast 条件を満たすと fast が選ばれる
- 条件を外すと `scipy`/`slow` 系にフォールバックし、実質 $O(NM)$ に戻る

## 4. 「FFTと同じか？」への整理

- **同じなのは計算量クラスが近づく点**（$\log$ を含む高速化）
- **同一問題ではない**
  - FFT: 等間隔時系列の離散フーリエ変換
  - Lomb-Scargle: 不等間隔サンプルに対する周波数ごとの回帰（最小二乗）

fast Lomb-Scargle は「不等間隔データ向けのLS問題をFFT系テクニックで加速する方法」であり、FFTそのものではありません。

## 5. 実務上の見方

- ピーク**位置**比較: 手法間で一致しやすい（周波数推定）
- パワー**値**比較: 正規化定義の違いで一致しないことがある
- 手法間比較の可視化では、`power / max(power)` のような共通正規化が有効

以上を踏まえると、あなたのスクリプトで見えた

- ピーク周波数は一致
- 生パワー値は実装ごとに異なる
- Astropy fast が特定条件で非常に速い

という挙動は理論と整合的です。
