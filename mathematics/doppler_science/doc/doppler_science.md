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

前節では直流成分 (DC成分) を除いた。すなわち $n = 0$ の項を除いた。本節では一般に $n \ (\geq 1)$ (正負両方) の項を除いた時の IQ 波形強度について議論する。

## 片側周波数の強度変化

$ s(t) = \sum_{n=-\infty}^{\infty} J_n (2k d_0) \exp(in\omega t)$ において $n \geq 0$ をとし正の周波数 (片側周波数) を持つ $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ を考える。これの強度は次のようになる。

$$
\begin{align*}
\|s_+(t)\|^2 &= s_+(t)\overline{s_+(t)} \\
&= \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
&= \sum_{n=0}^{\infty} \hat{J}_n^2 + 2\sum_{n=0}^{\infty}\sum_{m=n+1}^{\infty} \hat{J}_n\hat{J}_m \cos((m-n)\omega t) \\
&= \sum_{n=0}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=0}^{\infty} J_n(2kd_0)J_{n+k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

## DC までを除いた IQ 波形の片側周波数の強度変化

前節の $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ から DC $(n=0)$ を取り除き強度を考える。

$$
\begin{align*}
s_+^{DC}(t) &= \sum_{n=1}^{\infty} \hat{J}_n \exp(in\omega t) \\
\|s_+^{DC}(t)\|^2 &= \sum_{n=1}^{\infty}\sum_{m=1}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
&= \sum_{n=1}^{\infty} \hat{J}_n^2 + 2\sum_{n=1}^{\infty}\sum_{m=n+1}^{\infty} \hat{J}_n\hat{J}_m \cos((m-n)\omega t) \\
&= \sum_{n=1}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=1}^{\infty} J_n(2kd_0)J_{n+k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

## DC と基本周波数の逓倍までを除いた IQ 波形の片側周波数の強度変化

同様にして $s_{+}(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$ から $ 0 \leq n < M$ を取り除き強度を考える。

$$
\begin{align*}
s_+^M(t) &= \sum_{n=M}^{\infty} \hat{J}_n \exp(in\omega t) \\
\|s_+^M(t)\|^2 &= \sum_{n=M}^{\infty}\sum_{m=M}^{\infty} \hat{J}_n\hat{J}_m \exp(i(n-m)\omega t) \\
&= \sum_{n=M}^{\infty} \hat{J}_n^2 + 2\sum_{n=M}^{\infty}\sum_{m=n+1}^{\infty} \hat{J}_n\hat{J}_m \cos((m-n)\omega t) \\
&= \sum_{n=M}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=M}^{\infty} J_n(2kd_0)J_{n+k}(2kd_0)\right)\cos(k\omega t)
\end{align*}
$$

# 心拍と呼吸の動きがあるときの強度変化

人間の場合心拍と呼吸による動きが現れる。このときの変位を $d(t) = d_0 \sin(\omega_0 t) + d_1 \sin(\omega_1 t - \delta)$ とする。一般に $d_0 $ は $d_1$ よりも小さく、おおよそ 1桁程度異なる。 $d_0$ が 100 [μm] 程度であり、 $d_1$ は 2 [mm] 程度である。このとき心拍数と呼吸数の周期を求めるため、 IQ 信号自体の性質、その強度の性質を理解したい。

## 複合変位に対するIQ波形の展開

複合変位に対するIQ波形は次のように表される：

$$s(t) = \exp(2ki \cdot d(t)) = \exp(2ki \cdot [d_0 \sin(\omega_0 t) + d_1 \sin(\omega_1 t - \delta)])$$

この式は直接ベッセル関数で展開することが難しいため、以下のように分解して考える：

$$s(t) = \exp(2ki \cdot d_0 \sin(\omega_0 t)) \cdot \exp(2ki \cdot d_1 \sin(\omega_1 t - \delta))$$

それぞれの項をベッセル関数で展開すると：

$$\exp(2ki \cdot d_0 \sin(\omega_0 t)) = \sum_{n=-\infty}^{\infty} J_n(2kd_0) \exp(in\omega_0 t)$$

$$\exp(2ki \cdot d_1 \sin(\omega_1 t - \delta)) = \sum_{m=-\infty}^{\infty} J_m(2kd_1) \exp(im(\omega_1 t - \delta))$$

これらの積は畳み込みになる：

$$s(t) = \sum_{n=-\infty}^{\infty} \sum_{m=-\infty}^{\infty} J_n(2kd_0) J_m(2kd_1) e^{-im\delta} \exp(i(n\omega_0 + m\omega_1)t)$$

## 振幅の大きさの影響

$d_0 < d_1$（心拍の振幅は呼吸の振幅より小さい）という条件から、以下の近似が可能である：

1. $2kd_0$ は比較的小さい値となり、低次のベッセル関数項が主に寄与する
   - $J_0(2kd_0) \approx 1 - (kd_0)^2$
   - $J_{\pm 1}(2kd_0) \approx \pm kd_0$
   - $J_{\pm 2}(2kd_0) \approx (kd_0)^2/2$

2. $2kd_1$ はやや大きい値となり、複数のベッセル関数項が寄与する
   - 24GHzのセンサで $d_1 = 2$ [mm] の場合、$2kd_1 \approx 2$

この条件下では、IQ波形は以下のように近似できる：

$$s(t) \approx J_0(2kd_0) \sum_{m=-\infty}^{\infty} J_m(2kd_1) e^{-im\delta} \exp(im\omega_1 t) + J_1(2kd_0) \sum_{m=-\infty}^{\infty} J_m(2kd_1) e^{-im\delta} \exp(i(\omega_0 + m\omega_1)t) + J_{-1}(2kd_0) \sum_{m=-\infty}^{\infty} J_m(2kd_1) e^{-im\delta} \exp(i(-\omega_0 + m\omega_1)t)$$

## 強度変化の解析

IQ波形の強度 $|s(t)|^2$ を考えると、複雑な周波数成分が現れる：

1. **呼吸成分**: $\omega_1$ とその倍音
2. **心拍成分**: $\omega_0$ とその倍音
3. **相互変調成分**: $n\omega_0 \pm m\omega_1$（$n$, $m$は整数）

特に重要な相互変調成分は：
- $\omega_0 \pm \omega_1$（心拍と呼吸の和と差）
- $2\omega_0 \pm \omega_1$（心拍の2倍と呼吸の和と差）

## 振幅比の影響

$d_1/d_0 \approx 20$（呼吸の振幅が心拍の振幅より約20倍）という条件から：

1. **両成分の寄与**: 呼吸成分は心拍成分より強いが、心拍成分も十分に検出可能

2. **相互変調**: 心拍成分 $\omega_0$ と呼吸成分 $\omega_1$ の相互変調により、サイドバンド $\omega_0 \pm \omega_1$ が生成される

3. **周波数分離**: 心拍（0.8-1.5Hz）と呼吸（0.2-0.5Hz）の周波数帯域は比較的分離しやすい

## 信号処理への示唆

この理論解析から、心拍と呼吸を分離するための信号処理手法として以下が考えられる：

1. **バンドパスフィルタリング**:
   - 呼吸: 0.1-0.5Hz帯域
   - 心拍: 0.8-2.0Hz帯域（または2倍の1.6-4.0Hz帯域）

2. **適応フィルタリング**:
   - 呼吸成分を推定し、その影響を除去することで心拍成分を抽出

3. **時間-周波数解析**:
   - 短時間フーリエ変換やウェーブレット変換を用いて、時間とともに変化する周波数成分を追跡

4. **位相情報の活用**:
   - IQ信号の位相情報を利用して、振幅変化だけでなく位相変化も考慮した解析

## 簡略化した計算のための近似

実際の計算では、ベッセル関数の性質を利用して無限和を有限和に近似できる。

1. **心拍成分のみの場合**:
   $$\|s^{\prime}\|^2 \approx 2(kd_0)^2 - 2(kd_0)^2 \cos(2\omega_0 t)$$

2. **呼吸成分のみの場合**:
   $$\|s^{\prime}\|^2 \approx 1 - J_0^2(2kd_1) - 4J_0(2kd_1) \sum_{n=1}^{N} J_{2n}(2kd_1) \cos(2n\omega_1 t)$$
   ここで $N = 3$ または $4$ で十分な精度が得られる。

3. **心拍と呼吸の複合成分の場合**:
   振幅比が約20倍程度であれば、両方の成分を考慮した計算が必要となる。相互変調成分も含めて、以下のように近似できる：
   
   $$\|s(t)\|^2 \approx 1 + 2(kd_0)^2\cos(2\omega_0 t) + 2J_0(2kd_0)J_1(2kd_1)\cos(\omega_1 t) + 2J_1(2kd_0)J_1(2kd_1)\cos((\omega_0 + \omega_1)t) + 2J_1(2kd_0)J_1(2kd_1)\cos((\omega_0 - \omega_1)t) + \ldots$$