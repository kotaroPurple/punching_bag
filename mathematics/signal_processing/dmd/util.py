
import numpy as np
from numpy.typing import NDArray


def generate_data(mode: int) -> list[NDArray]:
    if mode == 0:
        return _make_data(123)
    elif mode == 1:
        fs = 100
        duration = 10.0  #  Signal length (seconds)
        times = np.linspace(0, duration, int(fs * duration), endpoint=False)
        data = generate_trend_chirp(times, trend_slope=0.05, f0=2.0, f1=3.0)
        return [data, times]
    else:
        return []


def _make_data(seed: int = 123) -> list[NDArray]:
    np.random.seed(seed)
    number = 400
    process_time = 1.0
    t_raw = np.linspace(0, process_time, number)
    trend = 10 * (t_raw - process_time) ** 2
    periodic1 = np.sin(10 * 2 * np.pi * t_raw) / np.exp(-2 * t_raw)
    periodic2 = np.sin(20 * 2 * np.pi * t_raw)
    noise = 1.5 * (np.random.rand(number) - 0.5)
    data = trend + periodic1 + periodic2 + noise
    return [data, trend, periodic1, periodic2, noise, t_raw]


def generate_trend_chirp(
        t: NDArray, trend_slope: float = 0.01, f0: float = 1.0, f1: float = 3.0) -> NDArray:
    k = (f1 - f0) / t[-1]  # Chirp rate (Hz/s)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)  # Integrate instantaneous frequency
    data = trend_slope * t + np.sin(phase)
    return data
