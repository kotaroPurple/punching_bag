# ベッセル関数と複素指数関数の公式集

## 基本公式

### ベッセル関数の定義

ベッセル関数 $J_n(z)$ は以下の級数で定義される：

$$J_n(z) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!(m+n)!} \left(\frac{z}{2}\right)^{2m+n}$$

### 複素指数関数とベッセル関数の関係

$$e^{iz\sin\theta} = \sum_{n=-\infty}^{\infty} J_n(z) e^{in\theta}$$

## 積和の変換公式

### 複素指数関数の積

$$e^{in\omega t} \cdot e^{im\omega t} = e^{i(n+m)\omega t}$$

### ベッセル関数の積和と三角関数

以下の公式は、$J_n J_m \exp(i(n-m)\omega t)$ の形式の項を三角関数で表現する際に有用である。

#### 1. $n = m$ の場合

$$J_n(z) J_n(z) = J_n^2(z)$$

これは実数であり、時間依存性がない定数項となる。

#### 2. $n \neq m$ の場合

$$J_n(z) J_m(z) e^{i(n-m)\omega t} + J_m(z) J_n(z) e^{i(m-n)\omega t} = 2 J_n(z) J_m(z) \cos((n-m)\omega t)$$

### 特殊な場合の公式

#### 1. $n = m+1$ の場合

$$J_{m+1}(z) J_m(z) e^{i\omega t} + J_m(z) J_{m+1}(z) e^{-i\omega t} = 2 J_{m+1}(z) J_m(z) \cos(\omega t)$$

#### 2. $n = m+2$ の場合

$$J_{m+2}(z) J_m(z) e^{2i\omega t} + J_m(z) J_{m+2}(z) e^{-2i\omega t} = 2 J_{m+2}(z) J_m(z) \cos(2\omega t)$$

## 総和公式

### 1. ベッセル関数の二乗和

$$\sum_{n=-\infty}^{\infty} J_n^2(z) = 1$$

### 2. 隣接するベッセル関数の積和

$$\sum_{n=-\infty}^{\infty} J_n(z) J_{n+1}(z) = 0$$

### 3. 一般的な積和公式

$$\sum_{n=-\infty}^{\infty} J_n(z) J_{n+k}(z) = 0 \quad (k \neq 0)$$

## 複合振動の展開

### 二つの正弦波の合成

$$e^{iz_1\sin\theta_1 + iz_2\sin\theta_2} = \sum_{n=-\infty}^{\infty} \sum_{m=-\infty}^{\infty} J_n(z_1) J_m(z_2) e^{in\theta_1 + im\theta_2}$$

### 強度計算のための公式

IQ波形 $s(t) = \sum_{n} a_n e^{in\omega t}$ の強度は以下のように計算できる：

$$|s(t)|^2 = s(t) \overline{s(t)} = \sum_{n} \sum_{m} a_n \overline{a_m} e^{i(n-m)\omega t}$$

これは以下のように実数部で表現できる：

$$|s(t)|^2 = \sum_{n} |a_n|^2 + 2\sum_{n < m} |a_n||a_m|\cos((n-m)\omega t + \phi_{nm})$$

ここで $\phi_{nm}$ は $a_n$ と $a_m$ の位相差である。

## 小さな引数に対する近似

$z$ が小さい場合のベッセル関数の近似：

$$J_0(z) \approx 1 - \frac{z^2}{4}$$

$$J_1(z) \approx \frac{z}{2}$$

$$J_n(z) \approx \frac{1}{n!}\left(\frac{z}{2}\right)^n \quad (n \geq 2)$$

## 実用的な計算例

### 例1: 単振動の強度変化

単振動 $d(t) = d_0 \sin(\omega t)$ に対するIQ波形の強度：

$$|s(t)|^2 = 1 - J_0^2(2kd_0) - 4J_0(2kd_0) \sum_{n=1}^{\infty} J_{2n}(2kd_0) \cos(2n\omega t)$$

$d_0$ が小さい場合の近似：

$$|s(t)|^2 \approx 2(kd_0)^2 - 2(kd_0)^2 \cos(2\omega t)$$

### 例2: 摂動を含む振動の強度変化

$d(t) = d_0 \sin(\omega t) + d_p \sin(n\omega t + \phi)$ に対する主要な周波数成分：

- $2\omega$: 基本振動の2倍周波数
- $n\omega$: 摂動の周波数
- $(n \pm 2)\omega$: 相互変調成分

$d_p \ll d_0$ の場合、$(n \pm 2)\omega$ の振幅は約 $2kd_0 \cdot kd_p$ に比例する。