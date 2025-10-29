
import numpy as np
from dataclasses import dataclass


@dataclass
class SlidingWindowCov:
    r"""
    Fixed-size sliding window statistics for vector-valued signals.

    Computes **moving mean vectors** and **moving covariance matrices**
    on streaming data of shape :math:`x_t \in \mathbb{R}^d`.

    This class keeps the following sufficient statistics
    within the sliding window of width :math:`w`:

    .. math::

        S_t = \sum_{i=t-w+1}^{t} x_i \in \mathbb{R}^d \\
        G_t = \sum_{i=t-w+1}^{t} x_i x_i^\top \in \mathbb{R}^{d\times d}

    Then the **moving mean** :math:`\mu_t` and
    **moving covariance** :math:`\Sigma_t` are given by:

    .. math::

        \mu_t = \frac{S_t}{w}, \qquad
        \Sigma_t = \frac{G_t}{w} - \mu_t \mu_t^\top

    For unbiased covariance (:math:`\mathrm{ddof}=1`):

    .. math::

        \Sigma_t
        = \frac{G_t - \frac{S_t S_t^\top}{w}}{w-1}

    With each new sample :math:`x_t` entering the window,
    and the oldest sample :math:`x_{t-w}` leaving, the statistics
    can be updated in **:math:`O(1)` time per sample**:

    .. math::

        S_t = S_{t-1} + x_t - x_{t-w} \\
        G_t = G_{t-1} + x_t x_t^\top - x_{t-w} x_{t-w}^\top

    Thus this class provides efficient streaming computation of
    both **vector means** and **covariance matrices**.

    Attributes:
        w (int):
            Window size :math:`w`.
        d (int):
            Dimension :math:`d` of each vector sample.
        unbiased (bool):
            If True, use unbiased covariance (:math:`w-1` denominator).
        use_warmup_count (bool):
            If True, denominators use current count until the
            window becomes full.

    Notes:
        - :meth:`push_batch` is fully vectorized (no Python loops over samples).
        - Supports input batch larger than the window width.

    See Also:
        mean, cov, push, push_batch
    """
    w: int
    d: int
    unbiased: bool = False
    use_warmup_count: bool = False

    def __post_init__(self):
        if self.w <= 0 or self.d <= 0:
            raise ValueError("w>0 and d>0 required")
        self.buf = np.zeros((self.w, self.d), dtype=np.float64)
        self.idx = 0
        self.count = 0
        self.S = np.zeros(self.d, dtype=np.float64)
        self.G = np.zeros((self.d, self.d), dtype=np.float64)

    def _history(self):
        # 古→新の順に既存窓を取り出す
        if self.count == 0:
            return np.empty((0, self.d), dtype=np.float64)
        if self.count < self.w:
            return self.buf[:self.count].copy()
        return np.concatenate([self.buf[self.idx:], self.buf[:self.idx]])

    def push_batch(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        # (N) -> (N,1) の二次元配列にする
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # (N,d) check
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must be (N,{self.d})")

        # --- 履歴 + 新データを結合 ---
        hist = self._history()  # hist: (H,d)
        seq = np.vstack((hist, X))  # (L,d)
        L = seq.shape[0]

        # --- 一次和: 累積和で区間和を作成 ---
        cs1 = np.cumsum(seq, axis=0)  # (L,d)

        S_all = cs1.copy()
        if L > self.w:
            S_all[self.w:] = cs1[self.w:] - cs1[:-self.w]

        # --- 二次和: 3D累積で区間外積和を構築 ---
        seq_outer = seq[:, :, None] * seq[:, None, :]  # (L,d,d)
        cs2 = np.cumsum(seq_outer, axis=0)  # (L,d,d)

        G_all = cs2.copy()
        if L > self.w:
            G_all[self.w:] = cs2[self.w:] - cs2[:-self.w]

        # --- 新データ分を切り出し (窓の末尾位置 H..H+N-1) ---
        idxs = np.arange(hist.shape[0], L)
        S = S_all[idxs]  # (N,d)
        G = G_all[idxs]  # (N,d,d)

        # 分母（ウォームアップ対応）
        win = np.minimum(idxs + 1, self.w).astype(np.float64)  # (N,)
        if self.use_warmup_count:
            denom = win
        else:
            denom = np.full(X.shape[0], float(self.w), dtype=np.float64)  # (N,)

        # 平均
        denom_mu = denom.reshape(-1,1)  # (N,1)
        mu = S / denom_mu  # (N,d)

        # 共分散
        mu_outer = mu[:, :, None] * mu[:, None, :]  # (N,d,d)
        if self.unbiased:
            denom_cov = np.maximum(1.0, denom-1.0).reshape(-1,1,1)  # (N,1,1)
            cov = (G - S[:, :, None] * S[:, None, :] / denom_mu[:, :, None]) / denom_cov
        else:
            denom_cov = denom.reshape(-1,1,1)
            cov = G / denom_cov - mu_outer

        # 数値安定化
        cov = 0.5 * (cov + cov.transpose(0, 2, 1))  # (N,d,d)
        diag = np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0)  # (N,d)
        cov[:, range(self.d), range(self.d)] = diag

        # --- 内部状態更新 (直近 w 点を保持) ---
        keep = min(self.w, L)
        last = seq[-keep:]
        self.buf[:] = 0.0
        self.buf[:keep] = last
        self.count = keep
        self.idx = keep % self.w
        self.S = S[-1]
        self.G = G[-1]

        ready = (win == self.w)
        return mu, cov, ready


def _reference_moments(X, w, unbiased=False, use_warmup_count=False):
    """
    NumPyのみで移動平均ベクトルと移動共分散（母/不偏）を計算する基準実装。
    出力は SlidingWindowCov.push_batch と同じ長さ N で、各時点に対して
    ウォームアップも含めて値を出します。

    X: (N, d)
    w: int
    unbiased: bool  -> ddof = 1
    use_warmup_count: bool -> 分母は min(t+1, w) / それ以外は常に w
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    mu = np.zeros((N, d), dtype=np.float64)
    cov = np.zeros((N, d, d), dtype=np.float64)
    ready = np.zeros(N, dtype=bool)

    # 逐次だが NumPy だけで書く（テスト用途なので十分高速）
    S = np.zeros(d, dtype=np.float64)
    G = np.zeros((d, d), dtype=np.float64)
    buf = np.zeros((w, d), dtype=np.float64)
    count = 0
    idx = 0

    for t in range(N):
        x_new = X[t]
        x_old = buf[idx].copy()
        buf[idx] = x_new.copy()
        idx = (idx + 1) % w

        S += x_new - x_old
        G += np.outer(x_new, x_new) - np.outer(x_old, x_old)
        if count < w:
            count += 1

        # 分母
        win = count if use_warmup_count else w
        win = max(win, 1)

        mu_t = S / win
        if unbiased:
            ddof = max(1, win - 1)
            cov_t = (G - np.outer(S, S) / win) / ddof
        else:
            cov_t = G / win - np.outer(mu_t, mu_t)

        # 数値安定（実装側と同等に）
        cov_t = 0.5 * (cov_t + cov_t.T)
        cov_t[np.diag_indices_from(cov_t)] = np.maximum(cov_t.diagonal(), 0.0)

        mu[t] = mu_t
        cov[t] = cov_t
        ready[t] = (count == w)

    return mu, cov, ready


def _reference_moments_basic(X, w, unbiased=False, use_warmup_count=False):
    """
    NumPyのみで移動平均ベクトルと移動共分散（母/不偏）を計算する基準実装。
    出力は SlidingWindowCov.push_batch と同じ長さ N で、各時点に対して
    ウォームアップも含めて値を出します。

    X: (N, d)
    w: int
    unbiased: bool  -> ddof = 1
    use_warmup_count: bool -> 分母は min(t+1, w) / それ以外は常に w
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, d = X.shape
    mu = np.zeros((N, d), dtype=np.float64)
    cov = np.zeros((N, d, d), dtype=np.float64)
    ready = np.zeros(N, dtype=bool)

    for t in range(N):
        start = max(0, t - w + 1)
        x_win = X[start:t+1]
        count = x_win.shape[0]

        win = count if use_warmup_count else w
        win = max(win, 1)

        S = x_win.sum(axis=0, dtype=np.float64)
        mu_t = S / win
        G = x_win.T @ x_win

        if unbiased:
            denom_cov = max(1, win - 1)
            cov_t = (G - np.outer(S, S) / win) / denom_cov
        else:
            cov_t = G / win - np.outer(mu_t, mu_t)

        mu[t] = mu_t
        cov[t] = cov_t

        win = count if use_warmup_count else w
        ready[t] = (count == w)

    return mu, cov, ready


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    N, w, d = 513, 32, 3
    unbiased = True
    use_warmup_count = True
    X = rng.normal(size=(N, d))

    for _unbiased, _use_warmup in [(False, False), (False, True), (True, False), (True, True)]:
        print(f"unbiased={_unbiased}, use_warmup_count={_use_warmup}")
        sw = SlidingWindowCov(w=w, d=d, unbiased=unbiased, use_warmup_count=use_warmup_count)
        mu, cov, ready = sw.push_batch(X)

        # mu_ref, cov_ref, ready_ref = _reference_moments(
        mu_ref, cov_ref, ready_ref = _reference_moments_basic(
            X, w=w, unbiased=unbiased, use_warmup_count=use_warmup_count
        )
        print('mu', np.allclose(mu, mu_ref))
        print('cov', np.allclose(cov, cov_ref))
        print('ready', np.array_equal(ready, ready_ref))
        print()
