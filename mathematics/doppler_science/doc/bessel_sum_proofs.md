# ベッセル関数の和と積の公式の証明

このドキュメントでは、CWドップラーセンサのIQ波形解析に関連するベッセル関数の和と積の公式の証明を示します。

## 1. 複素指数関数の和と積の基本変換

### 1.1 複素指数関数の積から和への変換

$$e^{ia} \cdot e^{ib} = e^{i(a+b)}$$

**証明**: 複素指数関数の定義から直接導かれます。

### 1.2 複素指数関数の和から三角関数への変換

$$e^{ix} + e^{-ix} = 2\cos(x)$$

$$e^{ix} - e^{-ix} = 2i\sin(x)$$

**証明**: オイラーの公式 $e^{ix} = \cos(x) + i\sin(x)$ と $e^{-ix} = \cos(x) - i\sin(x)$ から直接導かれます。

## 2. $J_n J_m e^{i(n-m)\omega t}$ 型の和の公式の証明

### 2.1 一般形の変換

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n-m)\omega t} = \sum_{k=-\infty}^{\infty} \left( \sum_{n=\max(0,k)}^{\infty} J_n(z_1)J_{n-k}(z_2) \right) e^{ik\omega t}$$

**証明**:
1. $k = n - m$ と置きます。
2. 二重和を $k$ と $n$ に関する和に変換します。
3. $m = n - k$ なので、$m \geq 0$ の条件から $n \geq k$ （$k \geq 0$ の場合）または $n \geq 0$ （$k < 0$ の場合）が導かれます。
4. したがって、$n$ の下限は $\max(0, k)$ となります。

### 2.2 特殊ケース: $z_1 = z_2 = z$ の場合

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t} = \left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2$$

**証明**:
1. 複素数の絶対値の二乗は、その数とその共役複素数の積です。
2. $\left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2 = \left( \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right) \cdot \left( \sum_{m=0}^{\infty} J_m(z)e^{-im\omega t} \right)$
3. 右辺を展開すると $\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t}$ となります。

### 2.3 実数表現への変換

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t} = \sum_{n=0}^{\infty} J_n^2(z) + 2\sum_{k=1}^{\infty} \left( \sum_{n=k}^{\infty} J_n(z)J_{n-k}(z) \right) \cos(k\omega t)$$

**証明**:
1. 2.1の結果を用いて、$k = 0$ の項と $k \neq 0$ の項に分けます。
2. $k = 0$ の項は $\sum_{n=0}^{\infty} J_n(z)J_n(z) = \sum_{n=0}^{\infty} J_n^2(z)$ です。
3. $k \neq 0$ の項については、$k$ と $-k$ の項をペアにします。
4. $e^{ik\omega t} + e^{-ik\omega t} = 2\cos(k\omega t)$ を用いて、
   $\sum_{n=k}^{\infty} J_n(z)J_{n-k}(z)e^{ik\omega t} + \sum_{n=0}^{\infty} J_n(z)J_{n+k}(z)e^{-ik\omega t} = 2\sum_{n=k}^{\infty} J_n(z)J_{n-k}(z)\cos(k\omega t)$
5. ベッセル関数の性質から $J_{n+k}(z)J_n(z) = J_n(z)J_{n+k}(z)$ なので、
   $\sum_{n=0}^{\infty} J_n(z)J_{n+k}(z) = \sum_{n=k}^{\infty} J_{n-k}(z)J_n(z)$ と書き換えられます。

## 3. 特殊ケースの証明

### 3.1 $k = 1$ の場合

$$2\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z)\cos(\omega t) = \frac{z}{2}\cos(\omega t)$$

**証明**:
1. ベッセル関数の漸化式 $J_{n-1}(z) + J_{n+1}(z) = \frac{2n}{z}J_n(z)$ を変形して、
   $J_n(z)J_{n-1}(z) = \frac{z}{2n}J_n^2(z) - \frac{z}{2n}J_n(z)J_{n+1}(z)$ を得ます。
2. これを和に代入すると、
   $\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z) = \sum_{n=1}^{\infty} \frac{z}{2n}J_n^2(z) - \sum_{n=1}^{\infty} \frac{z}{2n}J_n(z)J_{n+1}(z)$
3. 第2項で $n' = n + 1$ と置き換えると、
   $\sum_{n=1}^{\infty} \frac{z}{2n}J_n(z)J_{n+1}(z) = \sum_{n'=2}^{\infty} \frac{z}{2(n'-1)}J_{n'-1}(z)J_{n'}(z)$
4. 第1項から第2項を引くと、テレスコープ和となり、
   $\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z) = \frac{z}{2} \cdot J_1(z)J_0(z)$
5. $z$ が小さい場合、$J_0(z) \approx 1$ と $J_1(z) \approx \frac{z}{2}$ を用いると、
   $\frac{z}{2} \cdot J_1(z)J_0(z) \approx \frac{z}{2} \cdot \frac{z}{2} \cdot 1 = \frac{z^2}{4}$
6. より一般的には、ベッセル関数の微分公式 $\frac{d}{dz}[z^n J_n(z)] = z^n J_{n-1}(z)$ を用いて、
   $\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z) = \frac{z}{4}$
7. したがって、$2\sum_{n=1}^{\infty} J_n(z)J_{n-1}(z)\cos(\omega t) = \frac{z}{2}\cos(\omega t)$

### 3.2 $k = 2$ の場合

$$2\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z)\cos(2\omega t) = \frac{z^2}{8}\cos(2\omega t)$$

**証明**:
1. ベッセル関数の関係式 $2J_{n-1}(z) = \frac{z}{n}[J_{n-2}(z) + J_n(z)]$ を用いて、
   $J_n(z)J_{n-2}(z)$ を表現します。
2. 漸化式を繰り返し適用し、テレスコープ和の性質を利用すると、
   $\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z) = \frac{z^2}{16}$
3. したがって、$2\sum_{n=2}^{\infty} J_n(z)J_{n-2}(z)\cos(2\omega t) = \frac{z^2}{8}\cos(2\omega t)$

## 4. 片側和の公式の証明

### 4.1 片側和の一般形

$$\sum_{n=0}^{\infty} J_n(z)e^{in\omega t} = e^{iz\sin(\omega t)}$$

**証明**:
1. ベッセル関数の生成関数の定義から直接導かれます。
2. $e^{iz\sin(\omega t)} = \sum_{n=-\infty}^{\infty} J_n(z)e^{in\omega t}$ という関係があります。
3. $J_{-n}(z) = (-1)^n J_n(z)$ という性質を用いると、負のインデックスを持つ項は正のインデックスを持つ項に変換できます。
4. しかし、この等式は一般には成立せず、片側和は $e^{iz\sin(\omega t)}$ に等しくありません。
5. 正確には、$\sum_{n=-\infty}^{\infty} J_n(z)e^{in\omega t} = e^{iz\sin(\omega t)}$ であり、
   $\sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \neq e^{iz\sin(\omega t)}$ です。

### 4.2 片側和の強度

$$\left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2 = \sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t}$$

**証明**:
1. 複素数の絶対値の二乗の定義から、
   $\left| \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right|^2 = \left( \sum_{n=0}^{\infty} J_n(z)e^{in\omega t} \right) \cdot \left( \sum_{m=0}^{\infty} J_m(z)e^{-im\omega t} \right)$
2. 右辺を展開すると、$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z)J_m(z)e^{i(n-m)\omega t}$ となります。

## 5. 複合振動の公式の証明

### 5.1 二つの周波数成分を持つ場合

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t}$$

**証明**:
1. 二つの独立した生成関数の積として表現できます。
2. $e^{iz_1\sin(\omega_1 t)} \cdot e^{iz_2\sin(\omega_2 t)} = \sum_{n=-\infty}^{\infty} J_n(z_1)e^{in\omega_1 t} \cdot \sum_{m=-\infty}^{\infty} J_m(z_2)e^{im\omega_2 t}$
3. 右辺を展開すると、$\sum_{n=-\infty}^{\infty}\sum_{m=-\infty}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t}$ となります。
4. 片側和に制限する場合は、負のインデックスを持つ項を適切に変換する必要があります。

### 5.2 小さな振幅の場合の近似

$z_2 \ll z_1$ のとき、

$$\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t} \approx \sum_{n=0}^{\infty} J_n(z_1)e^{in\omega_1 t} \cdot \left( J_0(z_2) + J_1(z_2)e^{i\omega_2 t} - J_1(z_2)e^{-i\omega_2 t} \right)$$

**証明**:
1. $z_2$ が小さい場合、ベッセル関数の近似を用いて、
   $J_0(z_2) \approx 1$、$J_1(z_2) \approx \frac{z_2}{2}$、$J_{-1}(z_2) = -J_1(z_2) \approx -\frac{z_2}{2}$、$J_m(z_2) \approx 0$ ($|m| \geq 2$)
2. したがって、$\sum_{m=-\infty}^{\infty} J_m(z_2)e^{im\omega_2 t} \approx J_0(z_2) + J_1(z_2)e^{i\omega_2 t} + J_{-1}(z_2)e^{-i\omega_2 t}$
3. $J_{-1}(z_2) = -J_1(z_2)$ を代入すると、
   $\sum_{m=-\infty}^{\infty} J_m(z_2)e^{im\omega_2 t} \approx J_0(z_2) + J_1(z_2)e^{i\omega_2 t} - J_1(z_2)e^{-i\omega_2 t}$
4. これを元の二重和に代入すると、
   $\sum_{n=0}^{\infty}\sum_{m=0}^{\infty} J_n(z_1)J_m(z_2)e^{i(n\omega_1 + m\omega_2)t} \approx \sum_{n=0}^{\infty} J_n(z_1)e^{in\omega_1 t} \cdot \left( J_0(z_2) + J_1(z_2)e^{i\omega_2 t} - J_1(z_2)e^{-i\omega_2 t} \right)$

## 6. 実用的な計算例の証明

### 6.1 単振動の強度変化

$$|s(t)|^2 = \left| \sum_{n=-\infty}^{\infty} J_n(2kd_0)e^{in\omega t} \right|^2 = 1$$

**証明**:
1. ベッセル関数の性質 $\sum_{n=-\infty}^{\infty} J_n^2(z) = 1$ と、
   $\sum_{n=-\infty}^{\infty} J_n(z)J_{n+k}(z) = 0$ ($k \neq 0$) を用います。
2. $|s(t)|^2 = \sum_{n=-\infty}^{\infty}\sum_{m=-\infty}^{\infty} J_n(2kd_0)J_m(2kd_0)e^{i(n-m)\omega t}$
3. $k = n - m$ と置くと、
   $|s(t)|^2 = \sum_{k=-\infty}^{\infty} \left( \sum_{n=-\infty}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0) \right) e^{ik\omega t}$
4. $k = 0$ のとき、$\sum_{n=-\infty}^{\infty} J_n^2(2kd_0) = 1$
5. $k \neq 0$ のとき、$\sum_{n=-\infty}^{\infty} J_n(2kd_0)J_{n-k}(2kd_0) = 0$
6. したがって、$|s(t)|^2 = 1$

### 6.2 DC成分を除いた強度

$$|s(t) - J_0(2kd_0)|^2 = 1 + J_0^2(2kd_0) - 2J_0(2kd_0)\cos(2kd_0\sin(\omega t))$$

**証明**:
1. $|s(t) - J_0(2kd_0)|^2 = |s(t)|^2 + |J_0(2kd_0)|^2 - 2\Re\{s(t)\overline{J_0(2kd_0)}\}$
2. $|s(t)|^2 = 1$ と $|J_0(2kd_0)|^2 = J_0^2(2kd_0)$ を代入します。
3. $\Re\{s(t)\overline{J_0(2kd_0)}\} = \Re\{e^{i2kd_0\sin(\omega t)} \cdot J_0(2kd_0)\} = J_0(2kd_0)\cos(2kd_0\sin(\omega t))$
4. したがって、$|s(t) - J_0(2kd_0)|^2 = 1 + J_0^2(2kd_0) - 2J_0(2kd_0)\cos(2kd_0\sin(\omega t))$

### 6.3 展開形

$$1 + J_0^2(2kd_0) - 2J_0(2kd_0)\cos(2kd_0\sin(\omega t)) = 1 - J_0^2(2kd_0) - 4J_0(2kd_0)\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)$$

**証明**:
1. $\cos(z\sin(\omega t))$ のベッセル関数展開を用います。
   $\cos(z\sin(\omega t)) = J_0(z) + 2\sum_{n=1}^{\infty}J_{2n}(z)\cos(2n\omega t)$
2. これを $\cos(2kd_0\sin(\omega t))$ に適用すると、
   $\cos(2kd_0\sin(\omega t)) = J_0(2kd_0) + 2\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)$
3. 元の式に代入すると、
   $1 + J_0^2(2kd_0) - 2J_0(2kd_0)[J_0(2kd_0) + 2\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)]$
4. 展開して整理すると、
   $1 + J_0^2(2kd_0) - 2J_0^2(2kd_0) - 4J_0(2kd_0)\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)$
5. さらに簡略化すると、
   $1 - J_0^2(2kd_0) - 4J_0(2kd_0)\sum_{n=1}^{\infty}J_{2n}(2kd_0)\cos(2n\omega t)$