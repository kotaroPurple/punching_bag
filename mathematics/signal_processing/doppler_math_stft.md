# 単振動物体の STFT の時間変化

単振動物体から得られる IQ波形に対して STFT を行う。角速度 (周波数) において、その強度が単振動の角速度で現れることを示す。一般的な定式化を行なったのちに、具体例として心拍と呼吸由来の皮膚変動での STFT の強度が変動の周波数で変化することを示す。

## STFT

STFT を一般的な形で記述する。

$$
\text{STFT}(t,\omega) = \int_{-\infty}^{\infty} x(\tau)w(\tau-t)\exp(-i\omega t) d\tau
$$

$w(t)$ は窓関数であり、 $0\sim T$ の範囲で 1 , それ以外を 0 とする。  

$$
\begin{align*}\text{STFT}(t,\omega) &= \int_{t}^{t+T} x(\tau)\exp(-i\omega t) d\tau \\ &= \int_{0}^{T} x(t+p) \exp(-i\omega (t+p)) dp \\ &= \exp(-i\omega t) \int_{0}^{T} x(t+p) \exp(-i\omega p) dp \end{align*}
$$

## 単振動物体 IQ 波形の STFT

周波数 $f_{\text{obj}}$, 振幅 $d_{\text{obj}}$ で単振動をする物体の IQ 波形は次になる。

$$
s(t) = B \sum_{n=-\infty}^{\infty}J_n(\alpha d_\text{obj}) \exp(i 2\pi n f_\text{obj} t) \exp(-in\delta)
$$

振幅は振動数に関して関係ないため $B=1$ とする。表記を簡単にするため次のように物体の振動数と振幅を使ったものを再定義する。 $\hat{J_n} = J_n(\alpha d_{\text{obj}})$, $\hat{\omega} = 2\pi f_{\text{obj}}$ 。

またベッセル関数の値 $\hat{J_n}$ は実数であり、複素共役と等しい。

STFT の式に $s(t)$ を代入する。

$$
\begin{align*}\text{STFT}(t,\omega) &= \exp(-i\omega t) \int_{0}^{T} dp\sum_{n=-\infty}^{\infty} \hat{J_n} \exp(i\hat{\omega}n(t+p))\exp(-in\delta) \exp(-i\omega p) \\ &=  \sum_{n=-\infty}^{\infty} \hat{J_n} \exp(-i\omega t + i\hat{\omega}nt -in\delta) \int_{0}^{T} dp\exp(ip (n\hat{\omega} - \omega)) \\ &= \sum_{n=-\infty}^{\infty} \hat{J_n} \exp(i(\hat{n\omega}-\omega)t) \exp(-in\delta) \cdot g^n_{\omega} \end{align*}
$$

上式の積分は時刻 $t$ によらない値になり、 $n, \omega$ によることから定数列 $g^n_\omega$ と定義する。この値は簡単な計算で求まる。 $n\hat{\omega} \neq \omega$ の時、

$$
\begin{align*}g^n_\omega &= \int_{0}^{T} dp\exp(ip (n\hat{\omega} - \omega)) \\ &= \left[\frac{-i}{n\hat{\omega}-\omega}(\exp(ip(n\hat{\omega}-\omega)) \right]_0^T \\ &= \frac{-i}{n\hat{\omega}-\omega} (\exp(iT(n\hat{\omega}-\omega) - 1)\end{align*}
$$

$n\hat{\omega} = \omega$ の時は明らかに、 $g^n_\omega = T$ となる。

## STFT の強度

STFT の強度は自身の値とその複素共役の積になる。強度を $P(t,\omega)$ とする。

$$
\begin{align*}P(t,\omega) &= \text{STFT}(t,\omega) \overline{\text{STFT}(t,\omega)} \\ &= \left(\sum_{n=-\infty}^{\infty} \hat{J_n} \exp(i(\hat{n\omega}-\omega)t) \exp(-in\delta) \cdot g^n_{\omega} \right)\left(\overline{\sum_{m=-\infty}^{\infty} \hat{J_m} \exp(i(\hat{m\omega}-\omega)t) \exp(-im\delta) \cdot g^m_{\omega}}\right) \\ &= \left(\sum_{n=-\infty}^{\infty} \hat{J_n} \exp(i(\hat{n\omega}-\omega)t) \exp(-in\delta) g^n_{\omega} \right)\left(\sum_{m=-\infty}^{\infty} \hat{J_m} \exp(-i(\hat{m\omega}-\omega)t) \exp(im\delta) \cdot \overline{g^m_{\omega}}\right) \\ &= \sum_{n=-\infty}^{\infty}\sum_{m=-\infty}^{\infty} \hat{J_n} \hat{J_m} \exp(i(n-m)\hat{\omega}t) \exp(-i(n-m)\delta)g^n_\omega \overline{g^m_{\omega}}\ \end{align*}
$$

上式から $P(t,\omega)$ は単振動物体の角速度 $\hat{\omega}$ の定数倍の振動を足し合わせたものであることが分かる。　$\exp(-i(n-m)\delta)$ は位相ずれであり、振幅は次式になる。

$$
\hat{J_n} \hat{J_m} g^n_\omega \overline{g^m_{\omega}}
$$

## 角速度ごとの強度

1以上の整数 $l=n-m$ とし、角速度 $l\hat{\omega}$ での振幅を求める。位相ずれも考慮するため、周期関数と位相も含める。

$$
\begin{align*}C_l &= \sum_{n=-\infty}^{\infty} \hat{J_n} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}} \exp(-il\delta) \exp(il\hat{\omega}t)\end{align*}
$$

角速度ごとの強度を求めるためには $-l\hat{\omega}$ となる角速度の振幅が必要である。 $-l=n-m$ ($l$ : 1以上の整数) とすると、

$$
\begin{align*}C_{-l} &= \sum_{m=-\infty}^{\infty} \hat{J_{m-l}} \hat{J_{m}} g^{m-l}_{\omega} \overline{g^{m}_{\omega}} \exp(il\delta) \exp(-il\hat{\omega}t) \\ &= \sum_{n=-\infty}^{\infty} \hat{J_{n-l}} \hat{J_{n}} g^{n-l}_{\omega} \overline{g^{n}_{\omega}} \exp(il\delta) \exp(-il\hat{\omega}t) \\ &= \sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} \overline{g^{n}_{\omega}} g^{n-l}_{\omega}\exp(il\delta) \exp(-il\hat{\omega}t) \\ &= \sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} \overline{g^n_{\omega} \overline{g^{n-l}_{\omega}} \exp(-il\delta) \exp(il\hat{\omega}t)}\end{align*}
$$

$C_l, C_{-l}$ の和が角速度 $l\hat{\omega}$ を持つ振幅強度になる。

$$
\begin{align*}C_l + C_{-l} &= \sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} \left(g^n_{\omega} \overline{g^{n-l}_{\omega}} \exp(il(\hat{\omega}t-\delta)) + \overline{g^n_{\omega} \overline{g^{n-l}_{\omega}}} \exp(-il(\hat{\omega}t-\delta))\right) \\ &= \left(\sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}} \right)\exp(il(\hat{\omega}t-\delta)) + \left(\sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} \overline{g^n_{\omega} \overline{g^{n-l}_{\omega}}} \right)\exp(-il(\hat{\omega}t-\delta)) \\ &= \left(\sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}} \right)\exp(il(\hat{\omega}t-\delta)) + \overline{\left(\sum_{n=-\infty}^{\infty} \hat{J_{n}} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}} \right)}\exp(-il(\hat{\omega}t-\delta))\end{align*}
$$

$\sum$ の演算部分は単なる複素数の値になるため、 $X = \sum \hat{J_{n}} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}}$ と定義する。偏角を $\theta_X$ とすると $X=|X| \exp(i\theta_X)$ と書ける。上式を $X$ を使って書くと以下になる。

$$
\begin{align*}C_l + C_{-l} &= |X|\exp(i\theta_X)\exp(il(\hat{\omega}t-\delta)) + |X|\exp(-i\theta_X)\exp(-il(\hat{\omega}t-\delta))\\ &= |X| \exp(il(\hat{\omega}t-\delta) + i\theta_X) + |X| \exp(-il(\hat{\omega}t-\delta) - i\theta_X) \\ &= 2|X| \cos(il(\hat{\omega}t-\delta) + i\theta_X) \end{align*}
$$

角速度ごとの強度を比較するためには、1以上の整数 $l$ ごとの $|X|$ を比較すればよい。ただし、直流のみ異なる。以下に $l=0$ を代入する。

$$
\begin{align*}C_l &= \sum_{n=-\infty}^{\infty} \hat{J_n} \hat{J_{n-l}} g^n_{\omega} \overline{g^{n-l}_{\omega}} \exp(-il\delta) \exp(il\hat{\omega}t) \\ C_0 &= \sum_{n=-\infty}^{\infty} \hat{J_n}^2 \|g^n_{\omega}\|^2  \end{align*}
$$

## 具体例: 心拍・呼吸由来の皮膚変動

書籍「[ワイヤレス人体センシング](https://www.ohmsha.co.jp/book/9784274229978/)」によると、心拍による皮膚変動は数 100 [um] 程度で (P21 図1.8)、呼吸では数 [mm] 程度 (P13 図1.4) で単振動のように簡単な動きをする。

心拍の周波数を 1 [Hz]、振幅を 200 [um] とする。STFTをとる範囲を 0.5 秒、周波数を 1, 2, 3, 4, 5, 6 [Hz] にし、物体周波数のどの定数倍の周波数が支配的かを計算した。
下図が結果である。横軸が周波数 (物体周波数の倍音) であり、縦軸が強度である。
物体周波数の 1倍で STFT の振幅が変化することが分かる。2倍以上の値は 1倍のものより十分に小さい。また、STFT の周波数を変えても同様である。


呼吸の周波数を 0.1 [Hz]、振幅を 2 [mm] とする。STFTをとる範囲を 5 秒、周波数を 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 [Hz] にし、物体周波数のどの定数倍の周波数が支配的かを計算した。心拍と同様に、物体周波数の 1倍の周波数で振幅が変化することが分かる。
