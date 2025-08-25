"""Utility functions for SVD Online DMD."""

import numpy as np
from numpy.typing import NDArray


def make_hankel_matrix(data: NDArray, window_size: int) -> NDArray:
    """Create Hankel matrix from 1D signal.

    Args:
        data: 1D signal array
        window_size: Number of rows in Hankel matrix

    Returns:
        Hankel matrix of shape (window_size, len(data) - window_size + 1)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(data, len(data) - window_size + 1)


def flatten_hankel_matrix(hankel_mat: NDArray) -> NDArray:
    """Convert Hankel matrix back to 1D signal.

    Args:
        hankel_mat: Hankel matrix

    Returns:
        1D signal array
    """
    n_rows, n_cols = hankel_mat.shape
    row_indices = np.arange(n_rows)[:, None]
    col_indices = np.arange(n_cols)[None, :]
    indices = (row_indices + col_indices).ravel()
    sums = np.bincount(indices, weights=hankel_mat.ravel().real, minlength=n_rows + n_cols - 1)
    counts = np.bincount(indices, minlength=n_rows + n_cols - 1)
    return sums / counts


def apply_svd(matrix: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Apply SVD decomposition.

    Args:
        matrix: Input matrix

    Returns:
        U, S, Vh matrices from SVD
    """
    return np.linalg.svd(matrix, full_matrices=False)


def truncate_svd(U: NDArray, S: NDArray, Vh: NDArray, max_rank: int, tol: float = 1e-12) -> tuple[NDArray, NDArray, NDArray]:
    """Truncate SVD based on rank and tolerance.

    Args:
        U, S, Vh: SVD components
        max_rank: Maximum rank to retain
        tol: Tolerance for small singular values

    Returns:
        Truncated U, S, Vh
    """
    keep = (S > tol)
    if keep.sum() > max_rank:
        idx = np.argsort(-S)[:max_rank]
        keep = np.zeros_like(keep, dtype=bool)
        keep[idx] = True

    if keep.sum() < S.size:
        idx = np.where(keep)[0]
        U = U[:, idx]
        S = S[idx]
        Vh = Vh[idx, :]

    return U, S, Vh
