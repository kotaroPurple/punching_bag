
# ref: https://qiita.com/tanaka_soichiro/items/56c02ad53bc0babe62ab

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def make_data(seed: int = 123) -> list[NDArray]:
    np.random.seed(seed)
    number = 200
    process_time = 1.0
    t_raw = np.linspace(0, process_time, number)
    trend = 10 * (t_raw - process_time) ** 2
    periodic1 = np.sin(10 * 2 * np.pi * t_raw) / np.exp(-2 * t_raw)
    periodic2 = np.sin(20 * 2 * np.pi * t_raw)
    noise = 1.5 * (np.random.rand(number) - 0.5)
    data = trend + periodic1 + periodic2 + noise
    return [data, trend, periodic1, periodic2, noise, t_raw]


def make_hanakel_matrix(data: NDArray, window_size: int) -> NDArray:
    return sliding_window_view(data, window_size)


def flatten_hanakel_matrix(data: NDArray) -> NDArray:
    sub1 = data[:-1, 0]
    sub2 = data[-1, :]
    return np.concat((sub1, sub2))


def lower_svd(mat_u: NDArray, sigmas: NDArray, mat_vh: NDArray, rank: int) -> tuple[NDArray, ...]:
    return mat_u[:, :rank], sigmas[:rank], mat_vh[:rank, :]


def main():
    data, trend, periodic1, periodic2, noise, times = make_data()

    window_size = 70
    hanakel_mat = make_hanakel_matrix(data, len(data) - window_size)
    mat_x = hanakel_mat[:, :-1]
    mat_y = hanakel_mat[:, 1:]

    # SVD
    _U, _S, _Vh = np.linalg.svd(mat_x)
    sigma_sumsq = (_S ** 2).sum()
    sigma_array = (_S ** 2) / sigma_sumsq
    cumlative_contrib = (_S ** 2).cumsum() / sigma_sumsq * 100.

    # lower SVD
    low_rank = 5
    low_u, low_sigmas, low_vh = lower_svd(_U, _S, _Vh, rank=low_rank)

    # A
    low_v = low_vh.conj().T
    mat_a_tilda = low_u.conj().T @ mat_y @ low_v @ np.diag(1. / low_sigmas)
    eigens, eigen_vectors = np.linalg.eig(mat_a_tilda)

    # frequency
    dt = times[1] - times[0]
    freqs = np.angle(eigens) / (2 * np.pi *dt)
    print(freqs)

    phi = mat_y @ low_v @ np.diag(1. / low_sigmas) @ eigen_vectors
    vector_b = (phi.conj().T) @ mat_x[:, 0]
    psi = np.zeros((low_rank, mat_x.shape[1]), dtype=np.complex128)
    print(f'{phi.shape=}')
    for idx, mu_elem in enumerate(eigens):
        for _k in range(mat_x.shape[1]):
            psi[idx, _k] = np.exp(np.log(mu_elem) / dt * _k * dt) * vector_b[idx]

    # Compute DMD reconstruction for each mode
    x_t_list = []
    for idx, mu_elem in enumerate(eigens):
        x_t = []
        for t_ in times:
            x_t.append(phi[:, idx] * np.exp(np.log(mu_elem) / dt * t_ * dt) * vector_b[idx])
        x_t = np.array(x_t).T
        print(x_t.shape)
        x_t_list.append(x_t)

    # Convert Hankel matrix to time-series
    x_t_recon = []
    for x_t in x_t_list:
        # x_t_recon.append(flatten_hanakel_matrix(x_t))
        x_t_recon.append(x_t[0, :])
    x_t_recon = np.array(x_t_recon)
    reconstructed = x_t_recon.sum(axis=0)
    print(reconstructed.shape)

    # plt.scatter(eigens.real, eigens.imag)
    # plt.xlim(-1.3, 1.3)
    # plt.ylim(-1.3, 1.3)
    # plt.show()

    # plt.plot(cumlative_contrib)
    # plt.show()

    plt.plot(times, data)
    plt.plot(times, reconstructed)
    plt.show()


if __name__ == '__main__':
    main()
