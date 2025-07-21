# 強度変化が基本周波数になるか

現在、ドップラーセンサを使い心拍 (と呼吸) 推定を実施している。心拍数を検出する手法として強度の時系列変化が心拍数の基本周波数になることを想定している。これを理論的に解明し、どのような周波数を選ぶかなどを検討する。

## 基礎0: CWセンサと移動物体

CWドップラーセンサは連続波（Continuous Wave）を送信し、物体からの反射波の周波数シフトを検出します。物体が速度 v で動いている場合、ドップラー周波数シフト $f_d$ は次式で表されます：

$$f_d = \frac{2v}{\lambda} = \frac{2vf}{c}$$

ここで、$\lambda$ は波長、$f$ は送信周波数、$c$ は光速です。心臓の動きのような単振動の場合、速度は $v(t) = d_0 \omega \cos(\omega t)$ となり、ドップラー周波数シフトは時間とともに変化します。

## 基礎1: 単振動で動く物体の IQ 波形

IQ 波形はベッセル関数を使い表現できる。基本周波数とその逓倍成分が現れる。

$k$ を CWドップラーセンサの電波の波数とする (24 GHz の場合, $k \simeq 500$), $d(t)$ を変位とし $d(t) = d_0 \sin(\omega t)$ の単振動とする。ミキシング後の IQ 波形は次式で表される。

$$
\begin{align*}
s(t) &= \exp(2ki\ d(t)) \\
  &= \exp(2ki\ d_0 \sin(\omega t))
\end{align*}
$$

指数関数の肩に sin 関数があり、これはベッセル関数を使い級数展開できる。これはフーリエ級数展開となる。各周波数の振幅はベッセル関数の値で決まり、正と負の周波数の対応は $J_{-n}=(-1)^n J_n \ \ (n \geq 1)$ となる。

$$
\begin{align*}
  s(t) &= \sum_{n=-\infty}^{\infty} J_n (2k d_0) \exp(in\omega t) \\
  &= \sum_{n=-\infty}^{\infty} \hat{J}_n \exp(in\omega t) \\
\hat{J}_n &= J_n(2k d_0)
\end{align*}
$$

## DC を除いた単振動物体の IQ 波形の強度変化

強度が定数とcos(α sin(wt)) の和になる。cos部分は引数が微小な時 cos(2wt) で振動し、一般的には偶数次数のベッセル関数とcosの級数になる。

$s(t)$ から直流成分 (DC) を除いた信号の強度 $(\|\cdot\|^2)$ がどういった振る舞いをするか求める。直流成分は明らかに $\hat{J}_0$ であり、 $\| s(t) - \hat{J}_0\|^2$ を求めることになる。

$$
\begin{align*}
  \|s^{\prime}(t)\|^2 &= \| s(t) - \hat{J}_0\| ^2 \\
  &= (\overline{s(t) - \hat{J}_0}) (s(t) - \hat{J}_0) \\
  &= \|s(t)\|^2 + \hat{J}_0^2 - \overline{s(t)} \hat{J}_0 - s(t) \hat{J}_0 \\
  &= 1 + \hat{J}_0^2 - 2\hat{J}_0 \cos(2k d_0 \sin(\omega t))
\end{align*}
$$

1, 2項目は定数であり、3項目の $\cos(2kd_0 \sin(\omega t))$ が強度変化を表す。ベッセル関数の公式を使い説明する。

$$
\begin{align*}
  \cos(a\sin\theta) &= \Re (\cos(a\sin\theta) + i\sin(a\sin\theta)) \\
  &= \Re (\exp(ia\sin\theta)) \\
  &= \Re (\sum_{n=-\infty}^{\infty} J_n (a) \exp(in\theta)) \\
  &= \sum_{n=-\infty}^{\infty} J_n (a) \cos(n\theta) \\
  &= J_0(a) + \sum_{n=1}^{\infty} J_n(a) \cos(n\theta) + \sum_{m=1}^{\infty} J_{-m}(a) \cos(-m\theta) \\
  &= J_0(a) + \sum_{n=1}^{\infty} (1 + (-1)^n) J_n(a) \cos(n\theta) \ \ (J_{-n} = (-1)^n J_n) \\
  &= J_0(a) + \sum_{n=1}^{\infty} 2J_{2n}(a) \cos(2n\theta)
\end{align*}
$$

これを使うと、角周波数 $\omega$ を使った級数で書ける。

$$
\begin{align*}
  \|s^{\prime}\|^2 &= 1 + \hat{J}_0^2 - 2\hat{J}_0 \left(\hat{J}_0 + \sum_{n=1}^{\infty} 2\hat{J}_{2n} \cos(2n\omega t))\right) \\
  &= 1 - \hat{J}_0^2 - 4\hat{J}_0 \sum_{n=1}^{\infty} \hat{J}_{2n} \cos(2n\omega t)
\end{align*}
$$

よって求めたい $\|s^{\prime}\|^2$ は 定数と角周波数 $\omega$ の偶数倍音 $(2\omega, 4\omega, \cdots)$ で構成されることがわかる。周波数ごとの強度は $-4\hat{J}_0 \hat{J}_{2n}$ である。

どの周波数が強いかは $\hat{J}_n = J_n(2kd_0) \ (n\geq 1)$ で決まる。感覚的に言えば $d_0$ が十分小さければ $n=1$ の時が最大であり、角周波数 $2\omega$ にピークが現れる。

## DC と基本周波数の逓倍までを除いた IQ 波形の強度変化

前節では直流成分 (DC成分) を除いた。すなわち $n = 0$ の項を除いた。本節では一般に $|n| \leq N$ の項を除いた時の IQ 波形強度について議論する。

$s(t)$ から $|n| \leq N$ の項を除いた信号 $s_N'(t)$ を考えます：

$$
\begin{align*}
s_N'(t) &= s(t) - \sum_{n=-N}^{N} \hat{J}_n \exp(in\omega t) \\
\|s_N'(t)\|^2 &= \left\| s(t) - \sum_{n=-N}^{N} \hat{J}_n \exp(in\omega t) \right\|^2 \\
&= \|s(t)\|^2 + \left\|\sum_{n=-N}^{N} \hat{J}_n \exp(in\omega t)\right\|^2 - 2\Re\left\{s(t)\overline{\sum_{n=-N}^{N} \hat{J}_n \exp(in\omega t)}\right\} \\
\end{align*}
$$

二重和の部分を周波数成分ごとに整理します。$k = n-m$ と置くと：

$$
\begin{align*}
&= 1 + \sum_{n=-N}^{N}\sum_{m=-N}^{N} \hat{J}_n\hat{J}_m \exp(ik\omega t) - 2\Re\left\{\sum_{n=-N}^{N} \hat{J}_n \sum_{m=-\infty}^{\infty} \hat{J}_m \exp(-ik\omega t)\right\} \\
\end{align*}
$$

複素指数関数を実数表現に変換すると：

$$
\begin{align*}
&= 1 + \sum_{k=-2N}^{2N} \left( \sum_{n=\max(-N,k-N)}^{\min(N,k+N)} \hat{J}_n\hat{J}_{n-k} \right) \cos(k\omega t) - 2\sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(-N,k-N)}^{\min(N,k+N)} \hat{J}_n\hat{J}_{n-k} \right) \cos(k\omega t)
\end{align*}
$$

この式から、基本周波数 $\omega$ の整数倍の周波数成分が現れることがわかります。各角周波数 $k\omega$ の成分の振幅は、$\sum_{n=\max(-N,k-N)}^{\min(N,k+N)} \hat{J}_n\hat{J}_{n-k}$ で決まります。$N$ が大きくなるほど、より高次の周波数成分が強度変化に寄与します。

## 片側周波数の強度変化

$ s(t) = \sum_{n=-\infty}^{\infty} J_n (2k d_0) \exp(in\omega t)$ において $n \geq 0$ をとし正の周波数 (片側周波数) を持つ $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ を考える。これの強度は次のようになる。

$$
\begin{align*}
\|s_+(t)\|^2 &= s_+(t)\overline{s_+(t)} \\
&= \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} \hat{J}_n\hat{J}_m \exp(in\omega t)\exp(-im\omega t) \\
&= \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
\end{align*}
$$

ここで、$k = n-m$ と置くと周波数成分ごとに整理できます：

$$
\begin{align*}
\|s_+(t)\|^2 &= \sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(0,k)}^{\infty} \hat{J}_n\hat{J}_{n-k} \right) \exp(ik\omega t)
\end{align*}
$$

ここで、$k=0$ の項は実数であり、$k > 0$ と $k < 0$ の項は共役なので、実数表現に変換できます：

$$
\begin{align*}
\|s_+(t)\|^2 &= \sum_{n=0}^{\infty} \hat{J}_n^2 + \sum_{k=1}^{\infty} \left( \sum_{n=k}^{\infty} \hat{J}_n\hat{J}_{n-k} \exp(ik\omega t) + \sum_{n=0}^{\infty} \hat{J}_n\hat{J}_{n+k} \exp(-ik\omega t) \right) \\
\end{align*}
$$

$\exp(ik\omega t) + \exp(-ik\omega t) = 2\cos(k\omega t)$ を使ってさらに変形すると：

$$
\begin{align*}
\|s_+(t)\|^2 &= \sum_{n=0}^{\infty} \hat{J}_n^2 + 2\sum_{k=1}^{\infty} \left( \sum_{n=k}^{\infty} \hat{J}_n\hat{J}_{n-k} \right) \cos(k\omega t)
\end{align*}
$$

具体的なベッセル関数を代入すると：

$$
\begin{align*}
\|s_+(t)\|^2 &= \sum_{n=0}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=k}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

この式から、片側周波数の強度は以下の周波数成分を含むことがわかります：
- 定数項：$\sum_{n=0}^{\infty} J_n^2(2kd_0)$
- 角周波数 $k\omega$ の成分 ($k = 1, 2, 3, ...$)：$2\left(\sum_{n=k}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)\right)\cos(k\omega t)$

この式から、片側周波数の強度は基本周波数 $\omega$ の整数倍の周波数成分を含むことがわかります。特に、$k=1$ の項は基本周波数そのものに対応します。

## DC までを除いた IQ 波形の片側周波数の強度変化

前節の $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ から DC $(n=0)$ を取り除き強度を考える。

$$
\begin{align*}
s_+^{DC}(t) &= \sum_{n=1}^{\infty} \hat{J}_n \exp(in\omega t)
\end{align*}
$$

同様に、DC成分を除いた場合の強度を計算します：

$$
\begin{align*}
\|s_+^{DC}(t)\|^2 &= s_+^{DC}(t)\overline{s_+^{DC}(t)} \\
&= \sum_{n=1}^{\infty}\sum_{m=1}^{\infty} \hat{J}_n\hat{J}_m \exp(in\omega t)\exp(-im\omega t) \\
&= \sum_{n=1}^{\infty}\sum_{m=1}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
\end{align*}
$$

ここで、$k = n-m$ と置くと周波数成分ごとに整理できます：

$$
\begin{align*}
\|s_+^{DC}(t)\|^2 &= \sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(1,k+1)}^{\infty} \hat{J}_n\hat{J}_{n-k} \right) \exp(ik\omega t)
\end{align*}
$$

$k=0$ の項は実数であり、$k > 0$ と $k < 0$ の項は共役なので、実数表現に変換できます：

$$
\begin{align*}
\|s_+^{DC}(t)\|^2 &= \sum_{n=1}^{\infty} \hat{J}_n^2 + \sum_{k=1}^{\infty} \left( \sum_{n=\max(1,k+1)}^{\infty} \hat{J}_n\hat{J}_{n-k} \exp(ik\omega t) + \sum_{n=1}^{\infty} \hat{J}_n\hat{J}_{n+k} \exp(-ik\omega t) \right) \\
\end{align*}
$$

$\exp(ik\omega t) + \exp(-ik\omega t) = 2\cos(k\omega t)$ を使ってさらに変形すると：

$$
\begin{align*}
\|s_+^{DC}(t)\|^2 &= \sum_{n=1}^{\infty} \hat{J}_n^2 + 2\sum_{k=1}^{\infty}\left(\sum_{n=k+1}^{\infty} \hat{J}_n\hat{J}_{n-k}\right)\cos(k\omega t) \\
&= \sum_{n=1}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=k+1}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

この式から、DC成分を除いた場合も、角周波数 $k\omega$ ($k = 1, 2, 3, ...$) の成分が現れます。各周波数成分の振幅は $2\sum_{n=k+1}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)$ で決まります。特に $k=1$ の項は基本周波数そのものに対応します。

## DC と基本周波数の逓倍までを除いた IQ 波形の片側周波数の強度変化

同様にして $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ から $ 0 \leq n < M$ を取り除き強度を考える。

$$
\begin{align*}
s_+^M(t) &= \sum_{n=M}^{\infty} \hat{J}_n \exp(in\omega t)
\end{align*}
$$

同様に、$0 \leq n < M$ の項を除いた場合の強度を周波数成分ごとに整理します。$k = n-m$ と置くと：

$$
\begin{align*}
\|s_+^M(t)\|^2 &= \sum_{n=M}^{\infty}\sum_{m=M}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
&= \sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(M,k+M)}^{\infty} \hat{J}_n\hat{J}_{n-k} \right) \exp(ik\omega t)
\end{align*}
$$

実数表現に変換すると：

$$
\begin{align*}
\|s_+^M(t)\|^2 &= \sum_{n=M}^{\infty} \hat{J}_n^2 + 2\sum_{k=1}^{\infty}\left(\sum_{n=\max(M,k+M)}^{\infty} \hat{J}_n\hat{J}_{n-k}\right)\cos(k\omega t) \\
&= \sum_{n=M}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=\max(M,k+M)}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

この式から、$M$ が大きくなるほど、より高次の周波数成分のみが強度変化に寄与することがわかります。各角周波数 $k\omega$ ($k = 1, 2, 3, ...$) の成分の振幅は、$\sum_{n=\max(M,k+M)}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0)$ で決まります。

$M$ が大きくなるほど、より高次の周波数成分のみが強度変化に寄与することになります。これにより、基本周波数 $\omega$ の整数倍の周波数成分のうち、特定の成分を強調することが可能になります。

## 心拍数推定への応用

以上の理論解析から、CWドップラーセンサを用いた心拍数推定において重要なポイントは以下の通りです：

1. 単振動する物体（心臓）からのIQ波形の強度変化は、基本周波数 $\omega$ の偶数倍（特に $2\omega$）に強いピークを持つ
2. 振幅 $d_0$ が小さい場合（心臓の動きは微小）、$2\omega$ の成分が最も強くなる
3. 心拍数を推定するには、強度変化のスペクトルから最も強い周波数成分を検出し、それを2で割ることで基本周波数（心拍数）を得る

実際の応用では、信号処理によってDC成分や不要な周波数成分を除去し、心拍数に対応する周波数成分を強調することが重要です。