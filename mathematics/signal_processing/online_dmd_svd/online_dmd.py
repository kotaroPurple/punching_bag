
import numpy as np
from numpy.typing import NDArray


class OnlineDMD:
    """Incremental-SVD based Online DMD with forgetting factor.

    Maintains:
        U : (n, r)   - left singular vectors (orthonormal basis of X)
        S : (r,)     - singular values (diag of Σ)
        H : (r, r)   - U^T Y V  (for A_tilde = H Σ^{-1})
    Optional:
        store_x0 : first snapshot to compute amplitudes b for reconstruction

    Parameters
    ----------
    n : int
        State dimension (length of each snapshot x,y).
    r_max : int, default 50
        Max retained rank.
    lam : float in (0,1], default 1.0
        Forgetting factor. lam=1 → no forgetting. Apply every update.
    tol_resid : float, default 1e-10
        Threshold on residual norm γ to decide rank growth.
    tol_sv : float, default 1e-12
        Singular values below this are truncated after each update.

    Notes
    -----
    - This class does *projected* DMD modes: Phi = U W where A_tilde = H Σ^{-1}.
    - If you need exact modes ψ = (1/λ) Y V Σ^{-1} w, maintain G = Y V and update it
        the same way as H (right-multiply by V̂, augment with y_new), or keep a short buffer.
    """

    def __init__(
            self, n, r_max: int = 50, lam: float = 1.0, tol_resid: float = 1e-10,
            tol_sv: float = 1e-12) -> None:
        self.n = n
        self.r_max = r_max
        self.lam = lam
        self.tol_resid = tol_resid
        self.tol_sv = tol_sv

        # Online state (None until initialized)
        self.U = np.zeros((0, 0))          # (n, r)
        self.S = np.zeros(0)          # (r,)
        self.H = np.zeros((0, 0))          # (r, r)

        self.k = 0             # number of processed pairs
        self.x0 = None         # first snapshot for amplitude computation

        # Cache for eigen decomposition (invalidate on update)
        self._eigvals = None
        self._W = None

    # ---------- Public API ----------

    def warm_start(self, X0: NDArray, Y0: NDArray, rank: int|None = None) -> None:
        """
        Initial batch to start the online process.
        X0, Y0: (n, m0) arrays. Y0 should be shifted snapshots (like DMD).
        rank: truncate rank after initial SVD (auto if None).
        """
        self.k = X0.shape[1]
        self.x0 = X0[:, 0].copy()

        U, S, Vt = np.linalg.svd(X0, full_matrices=False)
        if rank is None:
            rank = min(self.r_max, (S > self.tol_sv).sum())
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]

        self.U = U
        self.S = S
        self.H = U.T @ Y0 @ Vt.T  # (r, r)

        self._invalidate_cache()

    def update(self, x_new, y_new) -> None:
        """
        Consume one new snapshot pair (x_k, y_k).
        Applies forgetting, incremental SVD, and updates H.
        """
        x_new = np.asarray(x_new).reshape(-1)
        y_new = np.asarray(y_new).reshape(-1)
        assert x_new.size == self.n and y_new.size == self.n

        if self.k == 0:
            # no warm start: initialize with the first pair as rank-1
            self.x0 = x_new.copy()
            self._init_from_first_pair(x_new, y_new)
            self.k += 1
            return

        # 0) Forgetting factor
        if self.lam < 1.0:
            sqrt_l = np.sqrt(self.lam)
            self.S *= sqrt_l
            self.H *= self.lam  # because both sides get sqrt(l)

        # 1) Incremental SVD update of X using new x_new
        U_old = self.U  # keep for H-update projections

        p = self.U.T @ x_new                    # (r,)
        r_vec = x_new - self.U @ p
        gamma = np.linalg.norm(r_vec)

        if gamma > self.tol_resid and self.U.shape[1] < self.r_max:
            # Rank up
            q = r_vec / gamma
            # Build K
            K = np.block([
                [np.diag(self.S), p[:, None]],
                [np.zeros((1, self.S.size)), np.array([[gamma]])]])

            # SVD of small (r+1) x (r+1)
            Uh, Sh, Vh = np.linalg.svd(K, full_matrices=True)

            # Update U, S
            U_aug = np.column_stack([self.U, q])        # (n, r+1)
            self.U = U_aug @ Uh                         # (n, r+1)
            self.S = Sh                                  # (r+1,)

            # H-update
            z_old = U_old.T @ y_new                      # (r,)
            z_q = np.dot(q, y_new)                       # scalar
            H_aug = np.block(
                [[self.H, z_old[:, None]], [np.zeros((1, self.H.shape[1])), np.array([[z_q]])]])
            self.H = Uh.T @ H_aug @ Vh                   # (r+1, r+1)

        else:
            # # Rank stays (r)
            # # K is r x r (last row/col is zeros)
            # K = np.block(
            #     [[np.diag(self.S), p[:, None]],
            #     [np.zeros((1, self.S.size)), np.array([[0.0]])]])
            # # Remove the last row/col to keep square (r x r)
            # K = K[:-1, :-1]  # because γ≈0, we skip augmentation
            # Uh, Sh, Vh = np.linalg.svd(K, full_matrices=True)

            # K: r x (r+1)
            K = np.column_stack([np.diag(self.S), p])   # (r, r+1)
            Uh, Sh, Vh = np.linalg.svd(K, full_matrices=False)
            V_small = Vh.T
            # Uh: (r, r), Sh: (r,), Vh: (r+1, r)

            self.U = self.U @ Uh
            self.S = Sh

            # H-update
            # z_old = U_old.T @ y_new
            # H_aug = np.block([[self.H, z_old[:, None]]])  # (r, r+1)
            # # Right side: extend with zero col to match Vh size if needed
            # self.H = Uh.T @ H_aug @ Vh
            z_old = U_old.T @ y_new         # (r,)
            H_aug = np.column_stack([self.H, z_old[:, None]])     # (r, r+1)
            self.H = Uh.T @ H_aug @ V_small      # (r, r)

        # Rank truncation (if too small singular values)
        self._truncate()

        self.k += 1
        self._invalidate_cache()

    def eig(self) -> tuple[NDArray, NDArray]:
        """Return eigenvalues λ and eigenvectors W of A_tilde (small system)."""
        if self._eigvals is None:
            A_tilde = self.H @ np.diag(1.0 / self.S)
            eigvals, W = np.linalg.eig(A_tilde)
            self._eigvals = eigvals
            self._W = W
        return self._eigvals, self._W  # type: ignore

    def modes(self) -> NDArray:
        """
        Return projected DMD modes Phi = U W.
        """
        _, W = self.eig()
        return self.U @ W

    def amplitudes(self, x_init: NDArray|None = None) -> NDArray:
        """
        Compute modal amplitudes b solving Phi b = x_init (least squares).
        By default use the first snapshot x0.
        """
        if x_init is None:
            x_init = self.x0
        Phi = self.modes()
        b = np.linalg.lstsq(Phi, x_init, rcond=None)[0]  # type: ignore
        return b

    def reconstruct(self, steps, x_init=None) -> NDArray:
        """
        Reconstruct snapshots for 0..steps-1 using current modes/eigs.
        Discrete-time evolution: x_j ≈ Phi * diag(λ^j) * b
        """
        lam, _ = self.eig()
        Phi = self.modes()
        b = self.amplitudes(x_init)
        # powers of eigenvalues
        j = np.arange(steps)
        Lambda_pow = lam[None, :] ** j[:, None]  # (steps, r)
        X_rec = (Phi @ (b[:, None] * Lambda_pow.T)).T  # sloppy broadcast
        # More clearly:
        # X_rec = Phi @ (b * Lambda_pow[j].T) for each j
        # We'll vectorize properly:
        # X_rec = (Phi @ (b * Lambda_pow.T)).T  # (steps, n)
        return X_rec.T

    # ---------- Internal helpers ----------

    def _init_from_first_pair(self, x, y) -> None:
        # trivial 1-column SVD
        normx = np.linalg.norm(x)
        if normx < 1e-15:
            raise ValueError("First vector is near zero.")
        U = x[:, None] / normx
        S = np.array([normx])
        # V = np.array([[1.0]])  # not stored

        H = U.T @ y[:, None] * 1.0  # since V=1

        self.U = U
        self.S = S
        self.H = H

    def _truncate(self) -> None:
        # Drop tiny singular values, keep at most r_max
        keep = (self.S > self.tol_sv)
        if keep.sum() > self.r_max:
            # keep biggest r_max
            idx = np.argsort(-self.S)[:self.r_max]
            keep = np.zeros_like(keep, dtype=bool)
            keep[idx] = True

        if keep.sum() < self.S.size:
            idx = np.where(keep)[0]
            self.U = self.U[:, idx]
            self.S = self.S[idx]
            self.H = self.H[np.ix_(idx, idx)]

    def _invalidate_cache(self) -> None:
        self._eigvals = None
        self._W = None


if __name__ == '__main__':
    n = 100
    odmd = OnlineDMD(n, r_max=30, lam=0.99)

    # 初期バッチ
    X0 = np.random.randn(n, 20)
    Y0 = np.random.randn(n, 20)
    odmd.warm_start(X0, Y0)

    # ストリーム到着
    for k in range(30):
        xk = np.random.randn(n)
        yk = np.random.randn(n)
        odmd.update(xk, yk)

    # モード・固有値
    lam, W = odmd.eig()
    Phi = odmd.modes()

    # 初期値から 50 ステップ再構成
    Xrec = odmd.reconstruct(steps=50)
