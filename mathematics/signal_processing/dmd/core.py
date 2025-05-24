
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def make_hankel_matrix(data: NDArray, n_rows: int) -> NDArray:
    """1次元データ列から Hankel Matrix を作る

    Args:
        data (NDArray): 1次元データ列
        n_rows (int): 行数

    Returns:
        NDArray: Hankel Matrix
    """
    return sliding_window_view(data, len(data) - n_rows + 1)


def flatten_hankel_matrix(hankel_mat: NDArray) -> NDArray:
    """Hankel Matrix を1次元信号に戻す

    Args:
        hankel_mat (NDArray): Hankel Matrix

    Returns:
        NDArray: 1次元信号
    """
    sub1 = hankel_mat[:-1, 0]
    sub2 = hankel_mat[-1, :]
    return np.concat((sub1, sub2))


def conjugate_transpose(mat_data: NDArray) -> NDArray:
    """随伴行列に変換する

    Args:
        mat_data (NDArray): 入力行列

    Returns:
        NDArray: 随伴行列
    """
    # 複素共役を取り, 転置する
    return mat_data.conj().T


def apply_svd(mat_data: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """SVD を行う。sigma 項は対角成分

    Args:
        mat_data (NDArray): 入力行列 (p,q)

    Returns:
        tuple[NDArray, NDArray, NDArray]: U (p,r), sigmas (r), Vh (r,q)
    """
    u_mat, sigmas, vh_mat = np.linalg.svd(mat_data, full_matrices=False)
    return u_mat, sigmas, vh_mat


def lower_svd(mat_u: NDArray, sigmas: NDArray, mat_vh: NDArray, rank: int) -> tuple[NDArray, NDArray, NDArray]:
    """SVD の結果に対して低ランク化する

    Args:
        mat_u (NDArray): U
        sigmas (NDArray): 対角成分
        mat_vh (NDArray): Vh
        rank (int): 低ランク基準

    Returns:
        tuple[NDArray, NDArray, NDArray]: U, sigmas, Vh
    """
    return mat_u[:, :rank], sigmas[:rank], mat_vh[:rank, :]


def apply_standard_dmd(
        mat_x: NDArray, mat_y: NDArray, svd_u: NDArray, svd_sigmas: NDArray, svd_vh: NDArray,
        low_rank: int) -> tuple[NDArray, ...]:
    """Dynamic Mode Decomposition を行う

    Args:
        mat_x (NDArray): X in (Y=AX)
        mat_y (NDArray): Y in (Y=AX)
        svd_u (NDArray): U in X=U.S.Vh
        svd_sigmas (NDArray): diagonal S in X=U.S.Vh
        svd_vh (NDArray): Vh in X=U.S.Vh
        low_rank (int): 低ランク基準

    Returns:
        tuple[NDArray, ...]: 行列Aの固有値, 行列Aの固有ベクトル, 各モードの振幅
    """
    # 低ランク化
    low_u, low_sigmas, low_vh = lower_svd(svd_u, svd_sigmas, svd_vh, low_rank)
    # A~
    low_uh = conjugate_transpose(low_u)
    low_v = conjugate_transpose(low_vh)
    inv_sigma = np.diag(1. / low_sigmas)
    a_tilda = low_uh @ mat_y @ low_v @ inv_sigma
    # eigen values, eigen vectors
    eigens, eigen_vectors = np.linalg.eig(a_tilda)
    # mode
    mat_phi = mat_y @ low_v @ inv_sigma @ eigen_vectors
    amps = np.linalg.solve(mat_phi.T @ mat_phi, mat_phi.T @ mat_x[:, 0])
    return eigens, mat_phi, amps


def predict_matrix_a_by_dmd(
        mat_y: NDArray, svd_u: NDArray, svd_sigmas: NDArray, svd_vh: NDArray,
        low_rank: int) -> NDArray:
    """Dynamic Mode Decomposition を行う

    Args:
        mat_y (NDArray): Y in (Y=AX)
        svd_u (NDArray): U in X=U.S.Vh
        svd_sigmas (NDArray): diagonal S in X=U.S.Vh
        svd_vh (NDArray): Vh in X=U.S.Vh
        low_rank (int): 低ランク基準

    Returns:
        NDArray: 行列A
    """
    # 低ランク化
    low_u, low_sigmas, low_vh = lower_svd(svd_u, svd_sigmas, svd_vh, low_rank)
    # A~
    low_uh = conjugate_transpose(low_u)
    low_v = conjugate_transpose(low_vh)
    inv_sigma = np.diag(1. / low_sigmas)
    a_tilda = low_uh @ mat_y @ low_v @ inv_sigma
    # matrix a
    mat_a = low_u @ a_tilda @ low_uh
    return mat_a


def reconstruct_by_dmd(
        mat_a_eigens: NDArray, mat_phi: NDArray, amps: NDArray, times: NDArray) -> tuple[NDArray, list[NDArray]]:
    """DMD結果から波形を再構成する

    Args:
        mat_a_eigens (NDArray): 行列Aの固有値
        mat_phi (NDArray): 行列Aの固有ベクトル
        amps (NDArray): 各モードの振幅
        times (NDArray): 時刻情報

    Returns:
        tuple[NDArray, list[NDArray]]: 再構成波形, 各波形の list
    """
    # delta time
    dt = times[1] - times[0]
    valid_times = times / dt
    size = len(times)
    # reconstruct wave
    wave_list = []
    for phi, mu, amp in zip(mat_phi.T, mat_a_eigens, amps):
        xs = phi[:, None] * amp * (np.exp(np.log(mu) * valid_times)[None, :])
        wave_list.append(flatten_hankel_matrix(xs)[:size])
    reconstructed = np.sum(np.array(wave_list), axis=0)
    return reconstructed, wave_list
