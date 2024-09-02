
# ドップラーセンサ数学

参考

- http://www.wannyan.net/scidog/spfunc/ch03.pdf (ベッセル関数)
- [https://www.takatafound.or.jp/support/articles/pdf/160615_03.pdf](https://www.takatafound.or.jp/support/articles/pdf/160615_03.pdf) (ドップラーセンサ基礎)

センサ方向に対し単振動を行う物体の IQ 波形について。

物体は振幅 $d_{\text{obj}}$ , 周波数 $f_{\text{obj}}$ で動くため以下で動く。($\delta$ : 位相ずれ, 物体までの位置から求まる)

$$
d= d_\text{obj}\sin(2\pi f_{\text{obj}}t - \delta)
$$

ドップラセンサーの送信波、受信波を求める。受信波の周波数 $f_r$ は時間的に変化するため (単振動で動く場合速度が変化する)、受信波の位相を $\theta_r (t)$ とする。

$$
\begin{align*}T(t) &= A\sin(2\pi f_0 t)\\ R(t) &= A'\sin(\theta_r (t))\\ \theta_r(t) &= \int_0^t 2\pi f_r(t) dt\end{align*}
$$

ドップラー効果により $f_r$ は以下のように近似できる。 $c$ は大気中の光速である。

$$
\begin{align*}f_r &= \frac{c+v}{c-v} f_0\\ &= \frac{1+v/c}{1-v/c}f_0 \\ &\simeq (1+v/c)(1+v/c)f_0\\ &\simeq (1+\frac{2v}{c})f_0\end{align*}
$$

ドップラーセンサは送信波と受信波をミキシングするため、I波 $I(t)$  は次になる。

$$
\begin{align*}I(t) &= AA'\sin(2\pi f_0 t) \sin(\theta_r(t))\\ &=-\frac{AA'}{2} \left(\cos(2\pi f_0 t + \theta_r(t)) - \cos(2\pi f_0 t - \theta_r(t)) \right)\\ & \end{align*}
$$

上記の第1項は $f_0$ 近くの大きい周波数で動作し、センサ側のバンドパスやローパスフィルタにより除去される。そのため、第2項が実際に取得されるセンサの値である。

$$
\begin{align*}I(t) &= \frac{AA'}{2} \cos(2\pi f_0 t-\theta_r(t)) \\ &= \frac{AA'}{2} \cos\left(\int_0^t 2\pi f_0 dt - \int_0^t 2\pi (1+\frac{2v}{c})f_0 dt \right)\\ &= \frac{AA'}{2}\cos\left(\int_0^t 2\pi\frac{2v(t)}{c} f_0\right)\\ &= \frac{AA'}{2} \cos\left(\frac{4\pi f_0}{c}\int_0^t v(t) dt \right)\end{align*}
$$

速度の積分は位置になるため、上記の積分は単振動の場合 $d_\text{obj}\sin(2\pi f_{\text{obj}}t - \delta)$ になる。以降簡単のため、 $\frac{AA’}{2}=B$, $\frac{4\pi f_0}{c} = \alpha$ とする。

$$
I(t) = B\cos\left(\alpha d_\text{obj} \sin(2\pi f_\text{obj}t - \delta) \right)
$$

Q波はI波を位相 $\pi/2$ 遅らせたものなので以下になる。またIQ波を複素数で表すと以下になる。

$$
\begin{align*}Q(t) &= B\sin\left(\alpha d_\text{obj} \sin(2\pi f_\text{obj}t - \delta) \right)\\ s(t) &= I(t) + iQ(t)\\ &= B\exp\left(i\alpha d_\text{obj} \sin(2\pi f_\text{obj}t- \delta) \right) \end{align*}
$$

$\alpha$ はセンサの周波数, 円周率, 光速で表せるため定数である。センサの周波数 $f_0 = 24 \text{[GHz]}$ とすると、 $\alpha \simeq 1000$ となる。 $s(t)$ の位相は $-\alpha d_\text{obj} \sim \alpha d_\text{obj}$ で変化し、その周波数は物体の単振動の周波数 $f_\text{obj}$ である。

IQ波はsin, cos の中に sin がある形である。これはベッセル関数を使うことで sin の和に変換できる。すなわちフーリエ級数変換の形になる。ベッセル関数 $J_n(x)$ の母関数について、

$$
\exp(\frac{x}{2}(t-\frac{1}{t})) = \sum_{n=-\infty}^{\infty}J_n(x) t^n
$$

 $t=e^{i\theta}$ を代入すると指数関数の中に $\sin$ 関数を持つものが得られる。これは単身動物体を観測した IQ 波と同じ形を持つ。

$$
\exp(ix\sin \theta) = \sum_{n=-\infty}^{\infty}J_n(x) e^{in\theta}
$$

$\theta=2\pi f_\text{obj} t - \delta$, $x=\alpha d_\text{obj}$ とすると、IQ 波形 $s(t)$ は次のように書ける。

$$
s(t) = B \sum_{n=-\infty}^{\infty}J_n(\alpha d_\text{obj}) \exp(i 2\pi n f_\text{obj} t) \exp(-in\delta)
$$

離散的な周波数 $n f_\text{obj}$ で分解されており、その振幅はベッセル関数を使った $J_n(\alpha d_\text{obj})$ になる。 $\exp(-in\delta)$ は位相ずれであり、絶対値は 1 になる。
