
# ref: https://qiita.com/tanaka_soichiro/items/56c02ad53bc0babe62ab

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def make_data(seed: int = 123) -> list[NDArray]:
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


def make_hankel_matrix(data: NDArray, window_size: int) -> NDArray:
    return sliding_window_view(data, window_size)


def flatten_hankel_matrix(data: NDArray) -> NDArray:
    sub1 = data[:-1, 0]
    sub2 = data[-1, :]
    return np.concat((sub1, sub2))


def conjugate_transpose(data: NDArray) -> NDArray:
    return data.conj().T


def lower_svd(mat_u: NDArray, sigmas: NDArray, mat_vh: NDArray, rank: int) -> tuple[NDArray, ...]:
    return mat_u[:, :rank], sigmas[:rank], mat_vh[:rank, :]


def make_eigen_info(mat: NDArray) -> tuple[NDArray, NDArray]:
    # eigens are sorted (np.abs(eigens): descending)
    eigens, eigen_vectors = np.linalg.eig(mat)
    return eigens, eigen_vectors


def main():
    data, trend, periodic1, periodic2, noise, times = make_data()

    window_size = 70
    hankel_mat = make_hankel_matrix(data, len(data) - window_size)
    mat_x = hankel_mat[:, :-1]
    mat_y = hankel_mat[:, 1:]

    print()
    print(f'{hankel_mat.shape=}, {mat_x.shape=}, {mat_y.shape=}')

    # SVD
    _U, _S, _Vh = np.linalg.svd(mat_x)
    sigma_sumsq = (_S ** 2).sum()
    sigma_array = (_S ** 2) / sigma_sumsq
    cumlative_contrib = (_S ** 2).cumsum() / sigma_sumsq * 100.

    print()
    print('---- SVD (X) ----')
    print(f'{_U.shape=}, {_S.shape=}, {_Vh.shape=}')

    # lower SVD
    low_rank = 5
    low_u, low_sigmas, low_vh = lower_svd(_U, _S, _Vh, rank=low_rank)

    # A
    low_v = conjugate_transpose(low_vh)
    low_uh = conjugate_transpose(low_u)
    inv_low_sigma = np.diag(1. / low_sigmas)
    mat_a_tilda = low_uh @ mat_y @ low_v @ inv_low_sigma
    eigens, eigen_vectors = np.linalg.eig(mat_a_tilda)

    # predict A
    predicted_a = low_u @ mat_a_tilda @ low_uh  # (L,L)

    make_eigen_info(predicted_a)

    print()
    print('---- A_tilda ----')
    print(f'{mat_a_tilda.shape=}')
    print(f'{predicted_a.shape=}')

    # frequency
    dt = times[1] - times[0]
    freqs = np.angle(eigens) / (2 * np.pi *dt)
    print()
    print('---- frequency [Hz] ----')
    print(f'{freqs=}')

    phi = mat_y @ low_v @ inv_low_sigma @ eigen_vectors
    vector_b = np.linalg.solve(phi.T @ phi, phi.T @ (mat_x[:, 0]))

    print()
    print('---- Phi, vector b (amplitude) ----')
    print(f'{phi.shape=}')
    print(f'{vector_b.shape=}, {vector_b}')

    # psi = np.zeros((low_rank, mat_x.shape[1]), dtype=np.complex128)
    # for idx, mu_elem in enumerate(eigens):
    #     for _k in range(mat_x.shape[1]):
    #         psi[idx, _k] = np.exp(np.log(mu_elem) / dt * _k * dt) * vector_b[idx]

    # Compute DMD reconstruction for each mode
    x_t_list = []
    for idx, mu_elem in enumerate(eigens):
        x_t = []
        for t_ in times:
            x_t.append(phi[:, idx] * np.exp(np.log(mu_elem) / dt * t_) * vector_b[idx])
        x_t = np.array(x_t).T  # (L, T)
        x_t_list.append(x_t)

    # Convert Hankel matrix to time-series
    x_t_recon = []
    for x_t in x_t_list:
        x_t_recon.append(flatten_hankel_matrix(x_t))
        # x_t_recon.append(x_t[0, :])
    x_t_recon = np.array(x_t_recon)
    reconstructed = x_t_recon.sum(axis=0)

    _, ax = plt.subplots(figsize=(8, 4))
    for i, one_phi in enumerate(phi.T):
        ax.plot(one_phi.real, alpha=0.5, label=str(i))
    ax.legend()

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, data)
    ax.plot(times, reconstructed[:len(times)].real)

    for sub_wave in x_t_recon:
        ax.plot(times, sub_wave[:len(times)].real, alpha=0.5)
    plt.show()

    # A
    lowers = 5
    eigens, eigen_vectors = make_eigen_info(predicted_a)
    eigens = eigens[:lowers]
    eigen_vectors = eigen_vectors[:, :lowers]
    predicted_b = np.linalg.solve(eigen_vectors.T @ eigen_vectors, eigen_vectors.T @ mat_x[:, 0])

    x_t_list = []
    for idx, mu_elem in enumerate(eigens):
        x_t = []
        for t_ in times:
            x_t.append(eigen_vectors[:, idx] * np.exp(np.log(mu_elem) / dt * t_) * predicted_b[idx])
        x_t = np.array(x_t).T  # (L, T)
        x_t_list.append(x_t)

    # Convert Hankel matrix to time-series
    x_t_recon = []
    for x_t in x_t_list:
        x_t_recon.append(flatten_hankel_matrix(x_t))
        # x_t_recon.append(x_t[0, :])
    x_t_recon = np.array(x_t_recon)
    reconstructed = x_t_recon.sum(axis=0)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, data)
    ax.plot(times, reconstructed[:len(times)].real)

    for sub_wave in x_t_recon:
        ax.plot(times, sub_wave[:len(times)].real, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
