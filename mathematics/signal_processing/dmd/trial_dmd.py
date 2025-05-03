
import numpy as np
import matplotlib.pyplot as plt


def dmd(signal, fs=1.0, L=128, r=10):
    """
    Dynamic Mode Decomposition (time-delay Hankel embedding) in pure NumPy.

    Parameters
    ----------
    signal : (N,) ndarray
        Real or complex time-series.
    fs     : float, optional
        Sampling frequency [Hz]. Default 1.0.
    L      : int, optional
        Embedding dimension (Hankel window length). Default 128.
    r      : int, optional
        Truncated SVD rank (≲ L). Default 10.

    Returns
    -------
    eigvals   : (r,) ndarray
        Discrete-time eigenvalues μ_k.
    lambdas   : (r,) ndarray
        Continuous-time eigenvalues λ_k = ln(μ_k)/Δt.
    modes     : (L, r) ndarray
        DMD modes Φ (columns).
    amplitudes: (r,) ndarray
        Initial mode amplitudes b_k.
    recon     : (N-L,) ndarray
        Reconstructed signal (first row of Φ e^{λt}).
    """
    signal = np.asarray(signal)
    N = signal.size
    if N < L + 1:
        raise ValueError("signal length must exceed L + 1")

    # ----- Hankel embedding --------------------------------------------------
    K = N - L + 1                       # number of columns
    H = np.vstack([signal[i:i + K] for i in range(L)])
    X, Xp = H[:, :-1], H[:, 1:]         # snapshots (shift by one step)

    # ----- truncated SVD -----------------------------------------------------
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    U_r, S_r, V_r = U[:, :r], np.diag(s[:r]), Vh.conj().T[:, :r]

    # ----- reduced operator --------------------------------------------------
    A_tilde = U_r.conj().T @ Xp @ V_r @ np.linalg.inv(S_r)

    # ----- eigen-decomposition ----------------------------------------------
    eigvals, W = np.linalg.eig(A_tilde)                 # μ_k
    modes = Xp @ V_r @ np.linalg.inv(S_r) @ W           # Φ

    # ----- continuous-time eigenvalues λ_k -----------------------------------
    dt = 1.0 / fs
    lambdas = np.log(eigvals) / dt

    # ----- initial amplitudes & reconstruction ------------------------------
    amplitudes = np.linalg.lstsq(modes, H[:, 0], rcond=None)[0]
    t_idx = np.arange(X.shape[1])
    time_dynamics = amplitudes[:, None] * np.exp(lambdas[:, None] * dt * t_idx)
    X_dmd = modes @ time_dynamics
    recon = X_dmd[0, :]  # reconstructed signal

    return eigvals, lambdas, modes, amplitudes, recon


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 1_000                                            # Hz
    t = np.arange(0, 2.0, 1/fs)                           # 2-s trace
    sig = (np.exp(1j*2*np.pi*1.2*t) +                     # 1.2 Hz component
           0.3*np.exp(1j*2*np.pi*10*t) +                  # 10 Hz component
           0.05*(np.random.randn(t.size) + 1j*np.random.randn(t.size)))

    eigvals, lambdas, modes, b, recon = dmd(sig, fs=fs, L=128, r=10)

    print(t.shape, sig.shape)
    print()
    print(f'{eigvals=}')
    print(f'{lambdas=}')
    print(f'{type(modes)}, {modes.shape}')
    print(type(b), type(recon), b.shape, recon.shape)


    # eigvals (discrete-time) and lambdas (continuous) now contain
    # the Doppler frequencies and growth/decay rates.
    plt.plot(t, sig)
    plt.plot(t[:len(recon)], recon)
    plt.show()
