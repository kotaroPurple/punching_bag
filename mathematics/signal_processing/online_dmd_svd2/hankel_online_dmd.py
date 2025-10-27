"""Hankel-aware Online DMD utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from online_dmd import OnlineDMD
from hankel import array_to_hankel_matrix, flatten_hankel_matrix
from signal_generator import SignalGenerator


class HankelOnlineDMD(OnlineDMD):
    """Online DMD wrapper that accepts 1-D streaming data and builds Hankel states."""

    def __init__(
            self,
            window_size: int,
            r_max: int = 10,
            lambda_: float = 1.0,
            tau_add: float = 1e-2,
            tau_rel: float = 1e-3,
            tau_energy: float = 0.99) -> None:
        super().__init__(
            n_dim=window_size,
            r_max=r_max,
            lambda_=lambda_,
            tau_add=tau_add,
            tau_rel=tau_rel,
            tau_energy=tau_energy)
        self.window_size = int(window_size)
        self._window_buffer = np.zeros(self.window_size, dtype=np.complex128)
        self._initialized_series = False

    def initialize_series(self, data: NDArray) -> None:
        """Initialize the DMD state from a 1-D time series."""
        series = np.asarray(data, dtype=np.complex128).reshape(-1)
        if series.ndim != 1:
            raise ValueError("Input series must be 1-D")
        if series.size < self.window_size + 1:
            raise ValueError("Series length must be at least window_size + 1 for Hankel DMD")
        hankel_mat = array_to_hankel_matrix(series, self.window_size)
        super().initialize(hankel_mat)
        self._window_buffer[...] = hankel_mat[:, -1]
        self._initialized_series = True

    def update_series(self, sample: float | complex) -> None:
        """Push a new scalar sample and update the internal DMD state."""
        if not self._initialized_series:
            raise RuntimeError("Call initialize_series() before streaming updates.")
        sample = np.complex128(sample)
        self._window_buffer[:-1] = self._window_buffer[1:]
        self._window_buffer[-1] = sample
        super().update(self._window_buffer.copy())

    def reconstruct_series(self, n_columns: int, x_init: NDArray | None = None) -> NDArray:
        """Reconstruct future Hankel columns and flatten them back to 1-D."""
        hankel_forecast = super().reconstruct_signal(x_init, n_columns)
        return flatten_hankel_matrix(hankel_forecast)

    def reconstruct_mode_series(
            self,
            n_columns: int,
            x_init: NDArray | None = None,
            backward: bool = False) -> NDArray:
        """Return per-mode contributions as 1-D signals."""
        mode_states = self.reconstruct_mode_signals(n_columns, x_init=x_init, backward=backward)
        if mode_states.size == 0:
            return np.empty((0, self.window_size + n_columns - 1), dtype=np.complex128)
        flattened = [flatten_hankel_matrix(mode_state) for mode_state in mode_states]
        return np.stack(flattened, axis=0)


def demo_hankel_online_dmd() -> None:
    """Simple demo mirroring try_online_dmd.py but using HankelOnlineDMD."""
    sample_rate = 100
    duration = 8.0
    total_samples = int(sample_rate * duration)

    generator = SignalGenerator(sample_rate)
    signal = np.array([generator.generate_time_varying(noise_level=0.02) for _ in range(total_samples)])

    model = HankelOnlineDMD(window_size=120, r_max=6, lambda_=0.995, tau_add=1e-3)
    init_len = model.window_size + 20
    model.initialize_series(signal[:init_len])

    freq_trace = []
    growth_trace = []
    amp_trace = []
    trace_time = []
    dt = 1.0 / sample_rate

    stride = 5
    for idx in range(init_len, total_samples, stride):
        for sample in signal[idx: idx + stride]:
            model.update_series(sample)
        try:
            frequencies = model.get_mode_frequencies(dt=dt)
            amplitudes = model.get_mode_amplitudes()
            growth = model.get_mode_growth_rates(dt=dt)
        except ValueError:
            continue
        if amplitudes.size == 0:
            continue
        dom = int(np.argmax(np.abs(amplitudes)))
        freq_trace.append(abs(frequencies[dom]))
        growth_trace.append(growth[dom])
        amp_trace.append(np.abs(amplitudes[dom]))
        trace_time.append(idx / sample_rate)

    print("=== HankelOnlineDMD Demo ===")
    print(f"Samples processed: {total_samples}, Window size: {model.window_size}")
    if freq_trace:
        print(f"Dominant frequency range: {min(freq_trace):.2f} - {max(freq_trace):.2f} Hz")
        print(f"Average dominant amplitude: {np.mean(amp_trace):.3f}")
        print(f"Average dominant growth rate: {np.mean(growth_trace):.3f}")
    else:
        print("Not enough data to compute dominant mode statistics.")

    forecast = model.reconstruct_series(model.window_size)
    print(f"Forecasted series length: {forecast.size}")
    print(f"First 5 reconstructed samples: {np.real(forecast[:5])}")


if __name__ == "__main__":
    demo_hankel_online_dmd()
