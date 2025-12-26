
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ssa import SSA


def generate_data() -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Generate sample displacement data."""
    # trend: parabola
    fs = 100.0  # [Hz]
    duration = 10.0  # [s]
    times = np.arange(0, duration, 1 / fs)
    trend = 0.2 * np.sin(2 * np.pi * 0.1 * times + np.pi / 2)
    # frequency components
    wave1 = 0.05 * np.sin(2 * np.pi * 1.0 * times)  # 1.0 Hz
    wave1 *= 1 + 0.2 * np.abs(times - duration / 2)  # amplitude modulation
    wave2 = 0.04 * np.sin(2 * np.pi * 2.0 * times + np.pi / 2)  # 2.0 Hz
    wave2 *= 1 - 0.2 * np.abs(times - duration / 2)  # amplitude modulation
    # noise
    noise = 0.005 * np.random.normal(size=len(times))
    result = trend + wave1 + wave2 + noise
    return times, result, [trend, wave1, wave2, noise]


def main() -> None:
    times, data, components = generate_data()
    # apply SSA
    window_length = 150
    ssa_model = SSA(window_length=window_length)
    ssa_model.fit(data)
    reconstructed_trend = ssa_model.reconstruct([0, 1, 2, 3])

    # reconstruction
    indices = [[0, 1], [2, 3], [4, 5]]
    reconstructions = [
        ssa_model.reconstruct(_indices) for _indices in indices
    ]

    # components
    u_mat, s, vt_mat = ssa_model.get_svd()

    print("Singular values:")
    print(s[:10])

    ## 累積寄与率のプロット
    cumulative_contribution = np.cumsum(s**2) / np.sum(s**2)
    print(cumulative_contribution[:10])

    u_vectors = u_mat[:, 0:6:1]
    plt.figure()
    for i in range(u_vectors.shape[1]):
        plt.plot(np.arange(len(u_vectors[:, i])) / 100, u_vectors[:, i], label=f"U Vector {i}")
    plt.show()

    v_vectors = vt_mat[0:6:1, :].T
    plt.figure()
    for i in range(v_vectors.shape[1]):
        plt.plot(np.arange(len(v_vectors[:, i])) / 100, v_vectors[:, i], label=f"V Vector {i}")
    plt.show()

    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(times, data, label="Original Data", alpha=0.6)
    plt.plot(times, reconstructed_trend, label="Reconstructed Trend (SSA)", color='orange', alpha=0.6)
    for i, recon in zip(indices, reconstructions):
        plt.plot(times, recon, label=f"Component {i}", linestyle='--', alpha=0.7)
    for i, comp in enumerate(components):
        if i >= 3:
            break
        plt.plot(times, comp, label=f"True Component {i}", c='gray', alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement")
    plt.title("SSA Trend Extraction Demo")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
