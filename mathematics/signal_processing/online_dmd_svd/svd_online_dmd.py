"""SVD-based Online DMD for 1D signals."""

import numpy as np
from numpy.typing import NDArray
from utils import make_hankel_matrix, apply_svd, truncate_svd, flatten_hankel_matrix


class SvdOnlineDmd:
    """SVD-based Online Dynamic Mode Decomposition for 1D signals.

    Uses Hankel matrix embedding to process 1D time series data.
    Maintains incremental SVD for efficient online updates.
    """

    def __init__(self, window_size: int, max_rank: int = 50, forgetting_factor: float = 1.0, tol: float = 1e-12):
        """Initialize SVD Online DMD.

        Args:
            window_size: Size of sliding window for Hankel matrix
            max_rank: Maximum rank to retain
            forgetting_factor: Forgetting factor (0 < λ <= 1)
            tol: Tolerance for singular value truncation
        """
        self.window_size = window_size
        self.max_rank = max_rank
        self.forgetting_factor = forgetting_factor
        self.tol = tol

        # Internal state
        self.U = np.zeros((0, 0))  # Left singular vectors
        self.S = np.zeros(0)       # Singular values
        self.H = np.zeros((0, 0))  # Projected dynamics matrix
        self.initialized = False
        self.data_buffer = []      # Buffer for recent data

        # Cache for eigendecomposition
        self._eigvals = None
        self._eigvecs = None

    def initialize(self, initial_data: NDArray) -> None:
        """Initialize with initial data batch.

        Args:
            initial_data: 1D array of initial time series data
        """
        if len(initial_data) < self.window_size + 1:
            raise ValueError(f"Initial data length {len(initial_data)} must be >= window_size + 1 ({self.window_size + 1})")

        # Create Hankel matrices with proper dimensions
        hankel_mat = make_hankel_matrix(initial_data, self.window_size)
        X = hankel_mat[:, :-1]  # Input snapshots (window_size, n_snapshots-1)
        Y = hankel_mat[:, 1:]   # Output snapshots (window_size, n_snapshots-1)

        # Initial SVD
        U, S, Vh = apply_svd(X)
        U, S, Vh = truncate_svd(U, S, Vh, self.max_rank, self.tol)

        self.U = U  # Shape: (window_size, rank)
        self.S = S  # Shape: (rank,)
        self.H = U.T @ Y @ Vh.T  # Projected dynamics (rank, rank)

        # Store last window_size elements for updates
        self.data_buffer = list(initial_data[-self.window_size:])
        self.initialized = True
        self._invalidate_cache()

    def update(self, new_value: float) -> None:
        """Update with new data point.

        Args:
            new_value: New scalar value to add
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        # Only proceed if we have full window
        if len(self.data_buffer) != self.window_size:
            return

        # Update buffer - maintain exactly window_size elements
        self.data_buffer.append(new_value)

        # Create snapshot pair from current buffer
        # x_new is the current window, y_new is the shifted window
        x_new = np.array(self.data_buffer[:-1])  # First window_size-1 elements
        y_new = np.array(self.data_buffer[1:])   # Last window_size-1 elements

        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)

        # Apply forgetting factor
        if self.forgetting_factor < 1.0:
            sqrt_lam = np.sqrt(self.forgetting_factor)
            self.S *= sqrt_lam
            self.H *= sqrt_lam

        # Incremental SVD update
        self._incremental_svd_update(x_new, y_new)
        self._invalidate_cache()

    def get_modes(self) -> NDArray:
        """Get DMD modes.

        Returns:
            Projected DMD modes (U @ W)
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        _, W = self._get_eigendecomposition()
        return self.U @ W

    def get_eigenvalues(self) -> NDArray:
        """Get DMD eigenvalues.

        Returns:
            Complex eigenvalues
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        eigvals, _ = self._get_eigendecomposition()
        return eigvals

    def reconstruct(self, steps: int, x_init: NDArray|None = None) -> NDArray:
        """Reconstruct time series for given steps.

        Args:
            steps: Number of time steps to reconstruct
            x_init: Initial condition (uses last buffer if None)

        Returns:
            Reconstructed time series
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        if x_init is None:
            x_init = np.array(self.data_buffer)

        eigvals, W = self._get_eigendecomposition()
        Phi = self.U @ W

        # Compute amplitudes
        b = np.linalg.lstsq(Phi, x_init, rcond=None)[0]

        # Reconstruct
        j = np.arange(steps)
        Lambda_pow = eigvals[None, :] ** j[:, None]
        X_rec = (Phi @ (b[:, None] * Lambda_pow.T)).T

        return X_rec.T

    def decompose_by_modes(self, steps: int, x_init: NDArray|None = None) -> tuple[NDArray, list[NDArray]]:
        """Decompose signal into individual mode contributions.

        Args:
            steps: Number of time steps to reconstruct
            x_init: Initial condition (uses last buffer if None)

        Returns:
            Tuple of (total_reconstruction, list_of_mode_signals)
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        if x_init is None:
            x_init = np.array(self.data_buffer)

        eigvals, W = self._get_eigendecomposition()
        Phi = self.U @ W

        # Compute amplitudes
        b = np.linalg.lstsq(Phi, x_init, rcond=None)[0]

        # Reconstruct each mode separately
        j = np.arange(steps)
        mode_signals = []

        for i in range(len(eigvals)):
            phi_i = Phi[:, [i]]
            lambda_i = eigvals[i]
            b_i = b[i]

            # Time evolution for this mode
            time_evolution = b_i * (lambda_i ** j)
            mode_hankel = phi_i @ time_evolution[None, :]

            # Convert back to 1D signal
            mode_signal = flatten_hankel_matrix(mode_hankel)
            mode_signals.append(mode_signal[:steps])

        # Total reconstruction
        total_reconstruction = np.sum(mode_signals, axis=0)

        return total_reconstruction, mode_signals

    def get_mode_amplitudes(self, x_init: NDArray|None = None) -> NDArray:
        """Get modal amplitudes for given initial condition.

        Args:
            x_init: Initial condition (uses last buffer if None)

        Returns:
            Complex amplitudes for each mode
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        if x_init is None:
            x_init = np.array(self.data_buffer)

        _, W = self._get_eigendecomposition()
        Phi = self.U @ W

        return np.linalg.lstsq(Phi, x_init, rcond=None)[0]

    def get_mode_frequencies(self, dt: float = 1.0) -> NDArray:
        """Get frequencies of DMD modes.

        Args:
            dt: Time step size

        Returns:
            Frequencies in Hz (if dt is in seconds)
        """
        eigvals = self.get_eigenvalues()
        return np.imag(np.log(eigvals)) / (2 * np.pi * dt)

    def get_mode_growth_rates(self, dt: float = 1.0) -> NDArray:
        """Get growth rates of DMD modes.

        Args:
            dt: Time step size

        Returns:
            Growth rates (positive = growing, negative = decaying)
        """
        eigvals = self.get_eigenvalues()
        return np.real(np.log(eigvals)) / dt

    def get_dominant_modes(self, n_modes: int|None = None, energy_threshold: float = 0.95) -> tuple[NDArray, NDArray, NDArray]:
        """Get dominant modes based on energy content for 1D signals.

        Args:
            n_modes: Number of modes to return (None for auto)
            energy_threshold: Energy threshold for auto mode selection

        Returns:
            Tuple of (eigenvalues, modes, amplitudes) for dominant modes
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        eigvals = self.get_eigenvalues()
        modes = self.get_modes()
        amplitudes = self.get_mode_amplitudes()

        # Calculate mode energy (magnitude of eigenvalues * amplitudes)
        energies = np.abs(eigvals) * np.abs(amplitudes)

        if n_modes is None:
            # Auto-select based on energy threshold
            total_energy = np.sum(energies)
            cumulative_energy = np.cumsum(np.sort(energies)[::-1])
            _n_modes = np.searchsorted(cumulative_energy / total_energy, energy_threshold) + 1
        else:
            _n_modes = n_modes

        # Get indices of most energetic modes
        dominant_indices = np.argsort(energies)[::-1][:_n_modes]

        return eigvals[dominant_indices], modes[:, dominant_indices], amplitudes[dominant_indices]

    def reconstruct_backward(self, steps: int, x_init: NDArray|None = None) -> tuple[NDArray, list[NDArray]]:
        """Reconstruct past time series (backward in time).

        Args:
            steps: Number of time steps to reconstruct backward
            x_init: Initial condition (uses last buffer if None)

        Returns:
            Tuple of (total_reconstruction, list_of_mode_signals)
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        if x_init is None:
            x_init = np.array(self.data_buffer)

        eigvals, W = self._get_eigendecomposition()
        Phi = self.U @ W

        # Compute amplitudes
        b = np.linalg.lstsq(Phi, x_init, rcond=None)[0]

        # Reconstruct backward (negative time indices)
        mode_signals = []
        for i in range(len(eigvals)):
            phi_i = Phi[:, i]
            lambda_i = eigvals[i]
            b_i = b[i]

            # Backward time evolution: use lambda^(-k) for k = 1, 2, ..., steps
            # Only use stable eigenvalues (|lambda| close to 1)
            if abs(lambda_i) > 1e-12 and abs(abs(lambda_i) - 1.0) < 0.5:
                # time_indices = np.arange(1, steps + 1)
                time_indices = np.arange(steps)
                # Use conjugate for backward evolution to maintain stability
                # lambda_inv = np.conj(lambda_i) / (abs(lambda_i)**2)
                # time_evolution = b_i * (lambda_inv ** time_indices)
                time_evolution = b_i / (lambda_i ** time_indices[::-1])

                # Create Hankel matrix for this mode
                mode_hankel = phi_i[:, None] @ time_evolution[None, :]

                # Convert back to 1D signal
                mode_signal = flatten_hankel_matrix(mode_hankel)
                mode_signals.append(mode_signal[:steps])
            else:
                # For unstable or very small eigenvalues, return zeros
                mode_signals.append(np.zeros(steps, dtype=complex))

        # Total reconstruction
        total_reconstruction = np.sum(mode_signals, axis=0)

        return total_reconstruction, mode_signals

    def reconstruct_from_start(self, start_vec: NDArray, steps: int) -> tuple[NDArray, list[NDArray]]:
        """Reconstruct forward from a given starting vector.

        Args:
            start_vec: Starting vector (window_size elements)
            steps: Number of time steps to reconstruct forward

        Returns:
            Tuple of (total_reconstruction, list_of_mode_signals)
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize() first")

        eigvals, W = self._get_eigendecomposition()
        Phi = self.U @ W

        # Compute amplitudes from starting vector
        b = np.linalg.lstsq(Phi, start_vec, rcond=None)[0]

        # Reconstruct forward
        mode_signals = []
        for i in range(len(eigvals)):
            phi_i = Phi[:, i]
            lambda_i = eigvals[i]
            b_i = b[i]

            # Forward time evolution
            time_indices = np.arange(steps)
            time_evolution = b_i * (lambda_i ** time_indices)

            # Create Hankel matrix for this mode
            mode_hankel = phi_i[:, None] @ time_evolution[None, :]

            # Convert back to 1D signal
            mode_signal = flatten_hankel_matrix(mode_hankel)
            mode_signals.append(mode_signal[:steps])

        # Total reconstruction
        total_reconstruction = np.sum(mode_signals, axis=0)

        return total_reconstruction, mode_signals

    def _incremental_svd_update(self, x_new: NDArray, y_new: NDArray) -> None:
        """Perform incremental SVD update."""
        # Check if matrices are properly initialized
        if self.U.size == 0 or self.S.size == 0:
            return

        # Ensure x_new and y_new have correct dimensions
        if len(x_new) != self.U.shape[0] or len(y_new) != self.U.shape[0]:
            return

        # Project new snapshot
        p = self.U.T @ x_new
        r_vec = x_new - self.U @ p
        gamma = np.linalg.norm(r_vec)

        if gamma > self.tol and self.U.shape[1] < self.max_rank:
            # Rank increase
            q = r_vec / gamma
            K = np.block([
                [np.diag(self.S), p[:, None]],
                [np.zeros((1, self.S.size)), np.array([[gamma]])]
            ])

            Uh, Sh, Vh = np.linalg.svd(K, full_matrices=True)

            U_aug = np.column_stack([self.U, q])
            self.U = U_aug @ Uh
            self.S = Sh

            # Update H
            z_old = self.U[:, :-1].T @ y_new
            z_q = np.dot(q, y_new)
            H_aug = np.block([
                [self.H, z_old[:, None]],
                [np.zeros((1, self.H.shape[1])), np.array([[z_q]])]
            ])
            self.H = Uh.T @ H_aug @ Vh
        else:
            # Rank stays same
            K = np.column_stack([np.diag(self.S), p])
            Uh, Sh, Vh = np.linalg.svd(K, full_matrices=False)

            self.U = self.U @ Uh
            self.S = Sh

            # Update H
            z_old = (self.U @ Uh[:, :len(self.S)]).T @ y_new
            H_aug = np.column_stack([self.H, z_old[:, None]])
            self.H = Uh.T @ H_aug @ Vh.T

        # Truncate if needed
        # self.U, self.S, _ = truncate_svd(self.U, self.S, np.eye(self.U.shape[1]), self.max_rank, self.tol)
        # self.H = self.H[:self.U.shape[1], :self.U.shape[1]]

    def _get_eigendecomposition(self) -> tuple[NDArray, NDArray]:
        """Get cached eigendecomposition of A_tilde."""
        if self._eigvals is None:
            A_tilde = self.H @ np.diag(1.0 / self.S)
            self._eigvals, self._eigvecs = np.linalg.eig(A_tilde)
        return self._eigvals, self._eigvecs  # type: ignore

    def _invalidate_cache(self) -> None:
        """Invalidate eigendecomposition cache."""
        self._eigvals = None
        self._eigvecs = None
