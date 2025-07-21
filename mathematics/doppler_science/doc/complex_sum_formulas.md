# 複素指数関数の和と積の公式集

## 基本公式

### 複素指数関数の定義

$$e^{ix} = \cos(x) + i\sin(x)$$

### オイラーの公式

$$\cos(x) = \frac{e^{ix} + e^{-ix}}{2}$$

$$\sin(x) = \frac{e^{ix} - e^{-ix}}{2i}$$

## $J_n J_m e^{i(n-m)\omega t}$ 型の和の公式

### 一般形

$n, m \geq 0$ のとき、以下の和を考える：

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n-m)\omega t}$$

これは以下のように変形できる：

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n-m)\omega t} = \sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(0,k)}^{\infty} J_n(z_1)J_{n-k}(z_2) \right) e^{ik\omega t}$$

ここで $k = n - m$ と置いた。

### 特殊ケース: $z_1 = z_2 = z$ の場合

$z_1 = z_2 = z$ のとき、以下の関係が成り立つ：

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t} = \left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2$$

これは実数値関数となり、以下のように展開できる：

$$\sum_{n=0}^{\infty} J_n^2(z) + 2\sum_{k=1}^{\infty} \left( \sum_{n=k}^{\infty} J_n(z)J_{n-k}(z) \right) \cos(k\omega t)$$

### 特殊ケース: $k = 1$ の場合

$k = 1$ (つまり $n - m = 1$) のとき：

$$\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z)e^{i\omega t} + \sum_{m=1}^{\infty} J_{m-1}(z)J_m(z)e^{-i\omega t} = 2\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z)\cos(\omega t)$$

ベッセル関数の性質から、この和は以下のように簡略化できる：

$$2\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z)\cos(\omega t) = \frac{z}{2}\cos(\omega t)$$

### 特殊ケース: $k = 2$ の場合

$k = 2$ (つまり $n - m = 2$) のとき：

$$\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z)e^{2i\omega t} + \sum_{m=2}^{\infty} J_{m-2}(z)J_m(z)e^{-2i\omega t} = 2\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z)\cos(2\omega t)$$

これは以下のように簡略化できる：

$$2\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z)\cos(2\omega t) = \frac{z^2}{8}\cos(2\omega t)$$

## 片側和の公式

### 片側和の一般形

$$\sum_{n=0}^{\infty} J_n(z)e^{in\omega t} = e^{iz\sin(\omega t)}$$

### 片側和の強度

$$\left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2 = \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t}$$

これは以下のように実数表現できる：

$$\sum_{n=0}^{\infty} J_n^2(z) + 2\sum_{k=1}^{\infty} \left( \sum_{n=k}^{\infty} J_n(z)J_{n-k}(z) \right) \cos(k\omega t)$$

## 複合振動の公式

### 二つの周波数成分を持つ場合

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t}$$

この強度は：

$$\left| \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t} \right|^2$$

これは以下の周波数成分を含む：
- $k\omega_1$ ($k$ は整数)
- $l\omega_2$ ($l$ は整数)
- $k\omega_1 \pm l\omega_2$ ($k, l$ は整数)

### 小さな振幅の場合の近似

$z_2 \ll z_1$ のとき、以下の近似が有効：

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t} \approx \sum_{n=0}^{\infty} J_n(z_1)e^{in\omega_1 t} \cdot \left( J_0(z_2) + J_1(z_2)e^{i\omega_2 t} - J_1(z_2)e^{-i\omega_2 t} \right)$$

これは以下のように展開できる：

$$\sum_{n=0}^{\infty} J_n(z_1)J_0(z_2)e^{in\omega_1 t} + \sum_{n=0}^{\infty} J_n(z_1)J_1(z_2)e^{i(n\omega_1 + \omega_2)t} - \sum_{n=0}^{\infty} J_n(z_1)J_1(z_2)e^{i(n\omega_1 - \omega_2)t}$$

## 実用的な計算例

### 例1: 単振動の強度変化

単振動 $d(t) = d_0 \sin(\omega t)$ に対するIQ波形の強度：

$$|s(t)|^2 = \left| \sum_{n=-\infty}^{\infty} J_n(2kd_0)e^{in\omega t} \right|^2 = 1$$

DC成分を除いた強度：

$$|s(t) - J_0(2kd_0)|^2 = 1 + J_0^2(2kd_0) - 2J_0(2kd_0)\cos(2kd_0\sin(\omega t))$$

これは以下のように展開できる：

$$1 - J_0^2(2kd_0) - 4J_0(2kd_0)\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)$$

### 例2: 摂動を含む振動の強度変化

$d(t) = d_0 \sin(\omega_0 t) + d_p \sin(n\omega_0 t + \phi)$ に対するIQ波形：

$$s(t) = \sum_{m=-\infty}^{\infty}\sum_{l=-\infty}^{\infty} J_m(2kd_0)J_l(2kd_p)e^{il\phi}e^{i(m\omega_0 + ln\omega_0)t}$$

$d_p \ll d_0$ のとき、主要な周波数成分は：
- $m\omega_0$ (基本振動の倍音)
- $(m \pm n)\omega_0$ (相互変調成分)