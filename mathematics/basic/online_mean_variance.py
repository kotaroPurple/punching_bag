
import numpy as np
from dataclasses import dataclass


@dataclass
class SlidingWindowStats:
    w: int
    unbiased: bool = False          # True: 不偏分散 (ddof=1). 窓が埋まる前は分母保護
    use_warmup_count: bool = False  # True: ウォームアップ中(窓未満)は分母=count

    def __post_init__(self):
        if self.w <= 0:
            raise ValueError("window size w must be positive")
        self.buf = np.zeros(self.w, dtype=np.float64)  # リングバッファ
        self.idx = 0               # 次に書き込む位置
        self.count = 0             # 窓内に実際入っている要素数 (<= w)
        self.S = 0.0               # 現在窓の一次和
        self.Q = 0.0               # 現在窓の二次和

    # --- 便利: 既存履歴を時系列順（古→新）に取り出す ---
    def _history(self) -> np.ndarray:
        if self.count == 0:
            return np.empty(0, dtype=np.float64)
        if self.count < self.w:
            # buf[0:count] に時系列順で入っている前提にしておく（後述の更新で保証）
            return self.buf[:self.count].copy()
        # full の場合は buf[idx:] → buf[:idx] の順が古→新
        return np.concatenate([self.buf[self.idx:], self.buf[:self.idx]]).copy()

    # --- 単発 push（後方互換） ---
    def push(self, x: float) -> tuple[float, float, bool]:
        means, vars_, ready = self.push_batch(np.asarray([x], dtype=np.float64))
        return float(means[-1]), float(vars_[-1]), bool(ready[-1])

    # --- 配列一括 push: 連続サンプルをまとめて入れる ---
    def push_batch(self, x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        x_arr: 1D配列（floatに変換）を想定。長さNの連続データをまとめて投入。
        戻り値:
            means: 形状 (N,)
            vars_: 形状 (N,)
            ready: 形状 (N,) の bool（True: 窓が埋まった）
        """
        x = np.asanyarray(x_arr, dtype=np.float64).ravel()
        N = x.size
        if N == 0:
            # 空なら何も変えずに空配列を返す
            return (np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=bool))

        # 直前の履歴（時系列順）
        hist = self._history()             # 長さ H = self.count
        H = hist.size

        # 履歴 + 新データをつなげた列に対して prefix sums を一度だけ計算
        seq = np.concatenate([hist, x])    # 長さ L = H + N
        L = seq.size
        cs1 = np.cumsum(seq)               # 一次和の累積
        cs2 = np.cumsum(seq * seq)         # 二次和の累積

        # 可変窓（ウォームアップ中は win=j+1、以降は win=w）
        # 末尾基準の区間和 S_all[j] = sum_{j-win+1..j} seq
        # ベクトル化: まず S_all = cs1 をコピーし、j>=w について差分に置換
        S_all = cs1.copy()
        Q_all = cs2.copy()
        w = self.w
        if L > w:
            S_all[w:] = cs1[w:] - cs1[:-w]
            Q_all[w:] = cs2[w:] - cs2[:-w]
        # j < w の区間和は cs1[j] / cs2[j]（開始は 0）

        # 新しく入れた N 点に対応する出力は、j = H..H+N-1
        idxs = np.arange(H, L)
        # 各ステップの有効な窓長（ウォームアップ対応）
        win = np.minimum(idxs + 1, w)          # 形状 (N,)

        S = S_all[idxs]
        Q = Q_all[idxs]

        if self.use_warmup_count:
            denom = win.astype(np.float64)
        else:
            denom = np.full(N, float(w), dtype=np.float64)

        means = S / denom

        if self.unbiased:
            # 不偏分散: (Q - S^2/denom) / max(1, denom-1)
            d = np.maximum(1.0, denom - 1.0)
            vars_ = (Q - (S*S)/denom) / d
        else:
            vars_ = Q/denom - (S/denom)*(S/denom)

        # 数値誤差の負をクリップ
        np.maximum(vars_, 0.0, out=vars_)

        # ready: 窓が埋まったか
        ready = (win == w)

        # ---- 内部状態の更新（次回のためのリングバッファ再構成）----
        # 直近の「保持すべき」系列（最大 w 件）を取り出し、buf を時系列順で詰め直す
        keep = min(w, L)
        last = seq[-keep:]                      # 直近 keep サンプル（古→新）
        self.buf[:] = 0.0
        self.buf[:keep] = last
        self.count = keep
        self.idx = keep % w                     # 次の書き込み位置

        # 現在窓の S,Q（最後のステップの値）を保存
        self.S = float(S[-1])
        self.Q = float(Q[-1])

        return means, vars_, ready

    def mean(self) -> float:
        denom = self.count if self.use_warmup_count else self.w
        return self.S / denom

    def var(self) -> float:
        denom = self.count if self.use_warmup_count else self.w
        if self.unbiased:
            d = max(1, denom - 1)
            v = (self.Q - (self.S*self.S)/denom) / d
        else:
            v = self.Q/denom - (self.S/denom)**2
        return max(0.0, v)


if __name__ == '__main__':
    sw = SlidingWindowStats(w=5, unbiased=False, use_warmup_count=True)
    m1, v1, r1 = sw.push_batch([1, 2, 3])         # 窓未満 → 分母が 1,2,3（オプション）
    m2, v2, r2 = sw.push_batch([4, 5, 6, 7, 8])   # 窓を越えてもOK（内部でまとめて転がす）

    print(m1, v1, r1)  # 長さ3
    print(m2, v2, r2)  # 長さ5
    # sw.mean(), sw.var() は「直近の窓」の統計を返す
    print(sw.mean(), sw.var())
