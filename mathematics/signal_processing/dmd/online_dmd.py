
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


def _calculate_next_q_matrix(mat_q: NDArray, vector_x: NDArray, vector_y: NDArray) -> NDArray:
    # Q_next = Q + y.xT
    return mat_q + np.outer(vector_y, vector_x)


def _calculate_next_p_matrix(mat_p: NDArray, vector_x: NDArray):
    # x: new array
    # P_next = P - c.P.x.xT.P
    # c = 1 / (1 + x.T.P.x)
    c = 1. / (1 + vector_x @ mat_p @ vector_x)
    next_p = mat_p - c * (mat_p @ np.outer(vector_x, vector_x) @ mat_p)
    return next_p


def _calculate_a_matrix(mat_p: NDArray, mat_q: NDArray) -> NDArray:
    return mat_q @ mat_p


class OnlineDmd:
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size

    def set_initial_data(self, data_array: NDArray, low_rank_threshold: float) -> None:
        hankel_mat = make_hankel_matrix(data_array, self._window_size)
        mat_x = hankel_mat[:, :-1]
        mat_y = hankel_mat[:, 1:]
        self._mat_p, self._mat_q = _predict_p_and_q_matrix(
            mat_x, mat_y, low_rank_threshold)
        self._mat_a = _calculate_a_matrix(self._mat_p, self._mat_q)
        # keep the last col
        self._last_array = mat_y[:, -1]

    def update(self, new_data: NDArray) -> None:
        # 過去データの最後を更新する
        vector_x = self._last_array
        vector_y = np.r_[self._last_array[1:], new_data]
        self._mat_q = _calculate_next_q_matrix(self._mat_q, vector_x, vector_y)
        self._mat_p = _calculate_next_p_matrix(self._mat_p, vector_x)
        self._mat_a = _calculate_a_matrix(self._mat_p, self._mat_q)
        # update
        self._last_array = vector_y

    def reconstruct(self, valid_number: int, time_index: int) -> list[NDArray]:
        eigens, phi_mat = np.linalg.eig(self._mat_a)
        bn = np.linalg.solve(_hermitian(phi_mat) @ phi_mat, _hermitian(phi_mat) @ self._last_array)
        wave_list = []
        # main
        for i in range(valid_number):
            phi_i = phi_mat[:, [i]]
            coeff = bn[i] * (1 / (eigens[i] ** np.arange(time_index)[::-1]))
            hankel_mat = phi_i @ coeff[None, :]
            xs = hankel_to_signal(hankel_mat)
            wave_list.append(xs)
        return wave_list

    def reconstruct_from_start(
            self, start_vec: NDArray, valid_number: int, time_index: int) -> list[NDArray]:
        eigens, phi_mat = np.linalg.eig(self._mat_a)
        bn = np.linalg.solve(_hermitian(phi_mat) @ phi_mat, _hermitian(phi_mat) @ start_vec)
        wave_list = []
        # main
        for i in range(valid_number):
            phi_i = phi_mat[:, [i]]
            coeff = bn[i] * (eigens[i] ** np.arange(time_index))
            hankel_mat = phi_i @ coeff[None, :]
            xs = hankel_to_signal(hankel_mat)
            wave_list.append(xs)
        return wave_list
