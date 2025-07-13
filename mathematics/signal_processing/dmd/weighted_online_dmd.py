
import numpy as np
from numpy.typing import NDArray

from core import (make_hankel_matrix, apply_svd, lower_svd, hankel_to_signal)


def _hermitian(mat_data: NDArray) -> NDArray:
    return np.conjugate(mat_data.T)


def _predict_p_and_q_matrix(mat_x: NDArray, mat_y: NDArray, threshold: float) -> tuple[NDArray, NDArray]:
    # Q = Y.(XT)
    mat_q = mat_y @ mat_x.T
    # P = (X.(XT))^(-1)
    # X.XT = (U.S.Vh).(VhT.ST.UT)
    #      = U.S.ST.UT
    # (X.XT)^(-1) = U.(S^(-2)).UT
    svd_u, svd_sigmas, svd_vh = apply_svd(mat_x)
    # 低ランク近似
    svd_u, svd_sigmas, _ = lower_svd(
        svd_u, svd_sigmas, svd_vh, rank=-1, threshold=threshold)
    mat_p = svd_u @ np.diag(1. / (svd_sigmas**2)) @ _hermitian(svd_u)
    return mat_p, mat_q


def _calculate_a_matrix(mat_p: NDArray, mat_q: NDArray) -> NDArray:
    return mat_q @ mat_p


def _valid_eigens(eigens: NDArray, threshold: float) -> int:
    # 累積寄与率から閾値以下のインデックスを取得
    values = eigens * eigens.conj()
    cumulated = np.cumsum(values) / values.sum()
    index = np.searchsorted(cumulated, threshold, side='left') + 1
    return int(index)


class WeightedOnlineDmd:
    def __init__(self, window_size: int, rho: float) -> None:
        self._window_size = window_size
        self._rho = rho
        if not (0 < rho <= 1):
            raise ValueError("rho must be 0 < rho <= 1")

    def set_initial_data(self, data_array: NDArray, low_rank_threshold: float) -> None:
        # A0, P0 を求める
        hankel_mat = make_hankel_matrix(data_array, self._window_size)
        mat_x = hankel_mat[:, :-1]
        mat_y = hankel_mat[:, 1:]
        # rho^(k-i) をかける
        rhos = np.array([self._rho ** i for i in range(mat_x.shape[1])])[::-1]
        mat_x = mat_x * rhos[None, :]
        mat_y = mat_y * rhos[None, :]
        #
        _mat_p, self._mat_q = _predict_p_and_q_matrix(
            mat_x, mat_y, low_rank_threshold)
        self._mat_a = _calculate_a_matrix(_mat_p, self._mat_q)
        self._mat_p = _mat_p / self._rho
        # keep the last col
        self._last_array = mat_y[:, -1]

    def update(self, new_data: float) -> None:
        # 過去データの最後を更新する
        vector_x = self._last_array
        vector_y = np.r_[self._last_array[1:], new_data]
        # 係数更新
        coeff = 1. / (1. + vector_x.T @ self._mat_p @ vector_x)
        self._mat_a = self._mat_a + coeff * np.outer((vector_y - self._mat_a @ vector_x), vector_x) @ self._mat_p
        self._mat_p = 1. / self._rho * (self._mat_p - coeff * self._mat_p @ np.outer(vector_x, vector_x) @ self._mat_p)
        # update
        self._last_array = vector_y

    def reconstruct(self, valid_number: int, time_index: int, threshold: float) -> list[NDArray]:
        eigens, phi_mat = np.linalg.eig(self._mat_a)
        bn = np.linalg.solve(_hermitian(phi_mat) @ phi_mat, _hermitian(phi_mat) @ self._last_array)
        wave_list = []
        # main
        if valid_number <= 0:
            valid_number = _valid_eigens(eigens, threshold)

        window_size = phi_mat.shape[0]
        for i in range(valid_number):
            phi_i = phi_mat[:, [i]]
            coeff = bn[i] * (1 / (eigens[i] ** np.arange(time_index - window_size + 1)[::-1]))
            hankel_mat = phi_i @ coeff[None, :]
            xs = hankel_to_signal(hankel_mat)
            wave_list.append(xs)
        return wave_list

    def reconstruct_from_start(
            self, start_vec: NDArray, valid_number: int, time_index: int, threshold: float) -> list[NDArray]:
        eigens, phi_mat = np.linalg.eig(self._mat_a)
        bn = np.linalg.solve(_hermitian(phi_mat) @ phi_mat, _hermitian(phi_mat) @ start_vec)
        wave_list = []
        # main
        if valid_number <= 0:
            valid_number = _valid_eigens(eigens, threshold)

        for i in range(valid_number):
            phi_i = phi_mat[:, [i]]
            coeff = bn[i] * (eigens[i] ** np.arange(time_index))
            hankel_mat = phi_i @ coeff[None, :]
            xs = hankel_to_signal(hankel_mat)
            wave_list.append(xs)
        return wave_list
