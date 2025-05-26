
import numpy as np

def incremental_svd(U, S, Vt, x, *, tol=1e-10, rmax=None, forgetting=None):
    """
    Incrementally updates the thin SVD  U Σ Vᵀ  when one new column `x`
    is appended on the right.

    Parameters
    ----------
    U : (m, r) ndarray or None
        Current left singular vectors (orthonormal columns).
        Pass None on the very first call.
    S : (r,) ndarray or None
        Current singular values in descending order.
    Vt : (r, k) ndarray or None
        Current right singular vectors (transposed).
    x : (m,) ndarray
        New column to append to X = [X_k | x].
    tol : float, default 1e-10
        Numerical tolerance below which the residual ‖x−U Uᵀx‖₂
        is considered zero (no rank increase).
    rmax : int or None, optional
        Maximum rank to keep (truncates smallest σ’s after each update).
    forgetting : float in (0,1] or None, optional
        If given, multiplies previous singular values by √(forgetting)
        before the update (exponential forgetting).

    Returns
    -------
    U_new : (m, r′) ndarray
    S_new : (r′,) ndarray
    Vt_new : (r′, k+1) ndarray
        Updated thin SVD of the enlarged matrix.
    """
    x = np.asarray(x).reshape(-1)           # ensure 1-D
    # m = x.size

    # ──────────────────────────── 1. 初回だけフル SVD ───────────────────────────
    if U is None or U.size == 0:
        rho = np.linalg.norm(x)
        if rho < tol:
            raise ValueError("Initial vector has near-zero norm.")
        return (x / rho).reshape(-1, 1), np.array([rho]), np.array([[1.0]])

    r = S.size
    k = Vt.shape[1]

    # ──────────────────────────── 2. （任意）忘却係数 ──────────────────────────
    if forgetting is not None:
        S = np.sqrt(forgetting) * S

    # ──────────────────────────── 3. 新列の分解：射影 + 残差 ────────────────────
    p   = U.T @ x                 # 射影係数 (r,)
    r_v = x - U @ p               # 残差ベクトル
    rho = np.linalg.norm(r_v)     # 残差ノルム

    if rho < tol:                 # ── ランクは増えない：既存空間に収まった ──
        # ブロック行列を明示的に組まなくても「x を付け足した全行列」の
        # フル SVD を一度取り直せば確実に整合が取れる（r が大きくない
        # 場合はこちらの方がシンプル）
        X_full = np.hstack([U * S, x[:, None]]) @ np.vstack([Vt, np.zeros((1, k))])
        X_full[:, -1] = x                       # ← 最後の列を書き換え
        U_new, S_new, Vt_new = np.linalg.svd(X_full, full_matrices=False)
        if rmax is not None and S_new.size > rmax:
            U_new, S_new, Vt_new = U_new[:, :rmax], S_new[:rmax], Vt_new[:rmax]
        return U_new, S_new, Vt_new

    # ──────────────────────────── 4. 残差を正規化し j を作る ────────────────────
    j = r_v / rho                  # (m,)

    # ──────────────────────────── 5. “小行列 K” を構築 ──────────────────────────
    K = np.zeros((r + 1, r + 1))
    K[:r, :r] = np.diag(S)         # 既存 Σ_k
    K[:r,  r] = p                  # 追加列の既存空間成分
    K[r,   r] = rho                # 真に新しい成分

    # ──────────────────────────── 6. K を小規模 SVD ─────────────────────────────
    Uk, Sk, VkT = np.linalg.svd(K, full_matrices=False)  # 全て (r+1)×(r+1)

    # ──────────────────────────── 7. 大きい基底に埋め込む ──────────────────────
    U_aug  = np.hstack([U, j[:, None]])     # (m, r+1)
    U_new  = U_aug @ Uk                     # m × (r+1)

    # 右基底ブロック W = [[V_k 0]; [0 1]]ᵀ  (サイズ (r+1)×(k+1))
    W      = np.zeros((r + 1, k + 1))
    W[:r, :k] = Vt
    W[-1,  -1] = 1.0
    Vt_new = VkT @ W                         # (r+1) × (k+1)

    # ──────────────────────────── 8. 任意：ランク上限で切り詰め ───────────────
    if rmax is not None and Sk.size > rmax:
        idx = np.argsort(-Sk)[:rmax]         # 大きい方 rmax 本
        U_new, Sk, Vt_new = U_new[:, idx], Sk[idx], Vt_new[idx]

    # ──────────────────────────── 9. σ を降順に並べ替え（安全策） ─────────────
    order = np.argsort(-Sk)
    return U_new[:, order], Sk[order], Vt_new[order]


def sample_A(m=2, n_each=5):
    """[1,0] を n_each 本 ➜ [0,1] を n_each 本"""
    part1 = np.tile(np.array([1, 0])[:, None], n_each)
    part2 = np.tile(np.array([0, 1])[:, None], n_each)
    return np.hstack([part1, part2])          # shape (2, 2*n_each)


def sample_B(m=5):
    """標準基底 e1..em を列に並べる (m×m)"""
    return np.eye(m)


if __name__ == "__main__":
    # m, n = 15, 10  # 50 次元に 30 列をストリームで足す想定
    # rng = np.random.default_rng(0)
    # X_ref = rng.standard_normal((m, n))
    X_ref = sample_B()
    m, n = X_ref.shape

    U = S = Vt = None                 # 初期化
    for i in range(n):
        # x = rng.standard_normal(m)    # 新しい列ベクトル
        x = X_ref[:, i]
        U, S, Vt = incremental_svd(U, S, Vt, x, rmax=5)
        print(S, U.shape, Vt.shape)

    # X を再構成して誤差を確認（参考）
    rng = np.random.default_rng(0)
    X_inc = U @ np.diag(S) @ Vt
    # X_ref = np.column_stack(rng.standard_normal((m, n)))  # ← 実際の列が欲しければ保存しておく
    print(X_inc)
    print()
    print(X_ref)
