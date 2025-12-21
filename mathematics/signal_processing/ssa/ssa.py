
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt


class SSA:
    """Singular Spectrum Analysis (SSA) model.

    Args:
        window_length: Window length for Hankel embedding.
    """

    def __init__(self, window_length: int) -> None:
        self.window_length = int(window_length)
        self.u: npt.NDArray[np.floating] | None = None
        self.s: npt.NDArray[np.floating] | None = None
        self.vt: npt.NDArray[np.floating] | None = None
        self._hankel_matrix: npt.NDArray[np.floating] | None = None

    @staticmethod
    def _hankel(
        series: npt.ArrayLike,
        window_length: int,
    ) -> npt.NDArray[np.floating]:
        """Build a Hankel matrix from a 1D series.

        Args:
            series: Input 1D series.
            window_length: Window length for embedding.

        Returns:
            Hankel matrix with shape (window_length, column_count).
        """
        series = np.asarray(series)
        series_length = len(series)
        column_count = series_length - window_length + 1
        if window_length <= 0 or column_count <= 0:
            raise ValueError("window_length must satisfy 1 <= window_length <= len(series)")
        return np.column_stack([series[i:i + window_length] for i in range(column_count)])

    @staticmethod
    def _diagonal_averaging(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Reconstruct a series by diagonal averaging.

        Args:
            matrix: Matrix to average along anti-diagonals.

        Returns:
            Reconstructed 1D series.
        """
        row_count, column_count = matrix.shape
        series_length = row_count + column_count - 1
        values = np.zeros(series_length)
        weights = np.zeros(series_length)
        for row_index in range(row_count):
            for column_index in range(column_count):
                values[row_index + column_index] += matrix[row_index, column_index]
                weights[row_index + column_index] += 1
        return values / weights

    def fit(self, series: npt.ArrayLike) -> "SSA":
        """Fit the SSA model to a 1D series.

        Args:
            series: Input 1D series.

        Returns:
            Fitted SSA instance.
        """
        self._hankel_matrix = self._hankel(series, self.window_length)
        self.u, self.s, self.vt = np.linalg.svd(self._hankel_matrix, full_matrices=False)
        return self

    def reconstruct(self, indices: int | Sequence[int] | npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
        """Reconstruct a component or components from the SSA decomposition.

        Args:
            indices: Singular component index or indices.

        Returns:
            Reconstructed 1D series from selected components.
        """
        if self.u is None or self.s is None or self.vt is None:
            raise ValueError("Call fit(x) before reconstruct().")
        if np.isscalar(indices):
            component_index = int(indices)
            reconstructed_matrix = self.s[component_index] * np.outer(
                self.u[:, component_index],
                self.vt[component_index],
            )
        else:
            component_indices = np.asarray(indices, dtype=int)
            reconstructed_matrix = (
                self.u[:, component_indices] * self.s[component_indices]
            ) @ self.vt[component_indices]
        return self._diagonal_averaging(reconstructed_matrix)


def ssa(series: npt.ArrayLike, window_length: int) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Compute SSA decomposition for a 1D series.

    Args:
        series: Input 1D series.
        window_length: Window length for Hankel embedding.

    Returns:
        Tuple of (u, s, vt) from the SVD.
    """
    model = SSA(window_length).fit(series)
    return model.u, model.s, model.vt


def reconstruct(
    u: npt.NDArray[np.floating],
    s: npt.NDArray[np.floating],
    vt: npt.NDArray[np.floating],
    indices: int | Sequence[int] | npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    """Reconstruct a series from SSA components.

    Args:
        u: Left singular vectors.
        s: Singular values.
        vt: Right singular vectors (transposed).
        indices: Singular component index or indices.

    Returns:
        Reconstructed 1D series.
    """
    if np.isscalar(indices):
        component_index = int(indices)
        reconstructed_matrix = s[component_index] * np.outer(
            u[:, component_index],
            vt[component_index],
        )
    else:
        component_indices = np.asarray(indices, dtype=int)
        reconstructed_matrix = (u[:, component_indices] * s[component_indices]) @ vt[
            component_indices
        ]
    return SSA._diagonal_averaging(reconstructed_matrix)
