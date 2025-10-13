import numpy as np
from numpy.typing import NDArray


class OnlineDMD:
    """Online DMD with Incremental SVD (V 非保持版, NumPy only)

    仕様:
        - ランク増加/維持を残差ノルム rho の相対閾値で判定
        - ランク上限 r_max（超えたら上位特異値でトランケーション）
        - 忘却係数 lambda_ を Σ と H に同時適用（weighted online DMD と整合）
        - 交差項 H は rank-1 更新:  H <- λH + (U^T x_next) v_last^T
            ※ v_last は "小 SVD の右特異行列の最後の列"（np.linalg.svd の Vt の最後の列を使う）

    example:
        odmd = OnlineDMD(n_dim, r_max=10, lambda_=1.0, tau_add=1e-2)
        odmd.initialize(X)  # X.shape == (n, m)
        for x in stream: odmd.update(x)   # x.shape == (n,)
        Atilde = odmd.A_tilde()
        evals, modes = odmd.eigs()
    """

    def __init__(
            self, n_dim: int, r_max: int = 10, lambda_: float = 1.0, tau_add: float = 1e-2) -> None:
        self.n = int(n_dim)
        self.r_max = int(r_max)
        self.lambda_ = float(lambda_)
        self.tau_add = float(tau_add)

        # 保持するのは U, S, C のみ（V は保持しない）
        self.U = np.empty(0)  # (n, r)
        self.S = np.empty(0)  # (r,)   singular values
        self.H = np.empty(0)  # (r, r) = U^T X' V への低ランク交差項

        # ストリーミング用バッファ
        self._x_prev = np.empty(0)  # 次に取り込む列

    def initialize(self, X: NDArray):
        """
        バッチ初期化。X.shape = (n, m)
        X = U Σ V^T の SVD を計算し、self.U, self.S, self.H を初期化。
        m > 1 の場合は X' = X[:, 1:] を使って self.H = U.T @ X' @ V[1:, :] を初期化。
        _x_prev は X の最後の列に設定。
        """
        X = np.asarray(X, dtype=float)
        n, m = X.shape
        if n != self.n:
            raise ValueError(f"Input X has incompatible dimension {n}, expected {self.n}")

        # U, S, Vt = np.linalg.svd(X, full_matrices=True)
        U, S, Vt = np.linalg.svd(X[:, :-1], full_matrices=True)
        r = min(self.r_max, S.size)
        self.U = U[:, :r]
        self.S = S[:r]
        V = Vt.T[:, :r]

        if m > 1:
            # 入力X は Y=AX の Y,X を共に含むため 1 列目を落として整列させる
            X_prime = X[:, 1:]
            # V_trim = V[:-1, :]
            # V_trim = V[1:, :]
            # self.H = self.U.T @ X_prime @ V_trim
            self.H = self.U.T @ X_prime @ V
        else:
            self.H = np.zeros((r, r))

        self._x_prev = X[:, -1].copy()

    # ---------------------- 更新 ----------------------
    def update(self, x_new: NDArray):
        """
        新しい観測 x_new (shape=(n,)) を投入。
        直前の x_prev を SVD に取り込み → 小 SVD の Vt から v_last を取り出し → x_new で C を rank-1 更新。
        initialize が呼ばれていることを前提とする。
        """
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        assert x_new.shape[0] == self.n
        if self.U.size == 0:
            raise RuntimeError("OnlineDMD must be initialized with initialize() before update().")

        # ここから毎回: x_prev を取り込み、x_new で C を更新
        x_prev = self._x_prev
        U, S = self.U, self.S
        r = S.shape[0]

        # 既存空間への射影と残差
        p = U.T @ x_prev  # (r,)
        r_vec = x_prev - U @ p  # (n,)
        rho = np.linalg.norm(r_vec)

        # ランク増加の判定（相対閾値 & 上限）
        add_rank = (rho > self.tau_add * max(np.linalg.norm(x_prev), 1e-12)) and (r < self.r_max)

        # self.H *= np.sqrt(self.lambda_)
        # S *= np.sqrt(self.lambda_)
        self.H *= self.lambda_
        S *= self.lambda_

        if add_rank:
            # --- ランク +1: 正方 (r+1)x(r+1) の broken-arrow SVD ---
            # r_hat = (r_vec / rho).reshape(-1, 1)  # (n,1)
            K = np.block([
                [np.diag(S), p.reshape(-1, 1)],
                [np.zeros((1, r)), np.array([[rho]])]
            ])  # (r+1, r+1)

            Ut_r, St_r, Vt_r = np.linalg.svd(K, full_matrices=True)  # K = Ut * diag(St) * Vt
            U_aug = np.column_stack((U, rho))
            # U の拡張→小回転
            # U_ext = np.hstack([U, r_hat])  # (n, r+1)
            # U_new_full = U_ext @ Ut  # (n, r+1)
            # svals = St.copy()  # (r+1,)

            # update H
            z_old = U.T @ x_new  # (r,)
            z_rho = np.dot(rho, x_new)  # scalar
            H_aug = np.block([
                [self.H, z_old[:, None]],
                [np.zeros((1, self.H.shape[1])), np.array([[z_rho]])]
            ])
            U_new = U_aug @ Ut_r  # (n, r)
            S_new = St_r


            # # ---- v_last の取り出し（V 非保持）----
            # # v_last^T = e_{r+1}^T * (Vtilde) * Pi  ≡ （Vt の 最後の列）を列選択 Pi で間引いたもの
            # # 現段階では選抜前なので、まずは v_last_full = Vt[:, -1]
            # v_last_full = Vt[:, -1].copy()  # shape (r+1,)

            # # 上位 r_max へトランケーション
            # r_new = min(self.r_max, svals.shape[0])
            # idx = np.argsort(svals)[::-1][:r_new]  # 上位 r_new の列を選ぶ
            # U_new = U_new_full[:, idx]  # (n, r_new)
            # S_new = svals[idx]  # (r_new,)
            # v_last = v_last_full[idx]  # (r_new,)

        else:
            # --- ランク維持: 薄い (r x (r+1)) の SVD ---
            Kthin = np.hstack([np.diag(S), p.reshape(-1, 1)])  # (r, r+1)
            Ut_r, St_r, Vt_r = np.linalg.svd(Kthin, full_matrices=False)  # Vt_r: (r, r+1)
            U_new = U @ Ut_r  # (n, r)
            S_new = St_r  # (r,)

            z_old = U.T @ x_new  # (r,)
            H_aug = np.column_stack([self.H, z_old[:, None]])  # (r, r+1)

            # # v_last^T = e_{r+1}^T * Vthin で、Vthin = Vt_r^T → v_last = Vt_r[:, -1]
            # v_last = Vt_r[:, -1].copy()  # (r,)  ← これが直接使える

        # # ---- 忘却（weighted の等価表現）----
        # if self.lambda_ != 1.0:
        #     S_new = self.lambda_ * S_new

        # H
        H_new = Ut_r.T @ H_aug @ Vt_r.T

        # # ---- 交差項 H の rank-1 更新（V 非保持）----
        # # H のサイズを新ランクに揃えつつ減衰
        # H_new = self._resize_and_decay_H(target_r=S_new.shape[0], decay=self.lambda_)
        # alpha_next = (U_new.T @ x_new).ravel()  # (r_new,)
        # # v_last は上で (r_new,) にしておいたので、そのまま外積
        # H_new = H_new + np.outer(alpha_next, v_last)

        # 保存
        self.U, self.S, self.H = U_new, S_new, H_new
        self._x_prev = x_new.copy()

    # ---------------------- 出力 ----------------------
    def A_tilde(self) -> None|NDArray:
        """低ランク小行列 Ã = C Σ^{-1} （shape = (r, r)）"""
        if self.U.size == 0:
            return None
        Sinv = 1.0 / np.maximum(self.S, 1e-15)
        return self.H @ np.diag(Sinv)

    def eigs(self):
        """(固有値, DMD モード)。モード = U @ W（Ã W = W Λ）。"""
        A = self.A_tilde()
        if A is None:
            return None, None
        evals, W = np.linalg.eig(A)
        modes = self.U @ W
        return evals, modes

    # ---------------------- Helper ----------------------
    def _resize_and_decay_H(self, target_r: int, decay: float):
        """C を λ で減衰し target_r に合わせてゼロ詰め/切り出し。"""
        if self.H.size == 0:
            return np.zeros((target_r, target_r))
        H = decay * self.H
        r_old = H.shape[0]
        if target_r == r_old:
            return H
        Z = np.zeros((target_r, target_r))
        r_min = min(target_r, r_old)
        Z[:r_min, :r_min] = H[:r_min, :r_min]
        return Z

    def _get_eigendecomposition(self) -> tuple[NDArray, NDArray]:
        """Get cached eigendecomposition of A_tilde."""
        A_tilde = self.A_tilde()
        if A_tilde is not None:
            self._eigvals, self._eigvecs = np.linalg.eig(A_tilde)
            return self._eigvals, self._eigvecs  # type: ignore
        else:
            raise ValueError()

    def get_mode_amplitudes(self, x_init: NDArray|None = None) -> NDArray:
        _, modes = self.eigs()
        if x_init is None:
            x_init = np.array(self._x_prev)
        if modes is None:
            raise ValueError("No modes available")
        return np.linalg.lstsq(modes, x_init, rcond=None)[0]

    def get_mode_frequencies(self, dt: float = 1.0) -> NDArray:
        """Get frequencies of DMD modes.

        Args:
            dt: Time step size

        Returns:
            Frequencies in Hz (if dt is in seconds)
        """
        eigvals = self.eigs()[0]
        if eigvals is None:
            raise ValueError("No modes available")
        return np.imag(np.log(eigvals)) / (2 * np.pi * dt)

    def get_mode_growth_rates(self, dt: float = 1.0) -> NDArray:
        """Get growth rates of DMD modes.

        Args:
            dt: Time step size

        Returns:
            Growth rates (positive = growing, negative = decaying)
        """
        eigvals = self.eigs()[0]
        if eigvals is None:
            raise ValueError("No modes available")
        return np.real(np.log(eigvals)) / dt
