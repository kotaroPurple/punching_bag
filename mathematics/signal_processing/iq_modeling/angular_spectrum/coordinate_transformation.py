
import numpy as np
from numpy.typing import NDArray


def make_simple_transformation_mats() -> NDArray:
    center = np.array([1., 0., 2.])
    _rot_mat_t = np.array([[1., 0., 0], [0., -1., 0.], [0., 0., -1.]])
    t_mat = np.eye(4)
    t_mat[:3, :3] = _rot_mat_t.T
    t_mat[:3, 3] = center
    return t_mat[None, ...]


def inverse_transformation_matrix(t_mat: NDArray) -> NDArray:
    result_mat = np.eye(len(t_mat))
    rot_mat_t = t_mat[:-1, :-1].T
    trans = t_mat[:-1, -1]
    result_mat[:-1, :-1] = rot_mat_t
    result_mat[:-1, -1] = -rot_mat_t @ trans
    return result_mat


def transform_by_matrix(t_mat: NDArray, position: NDArray) -> NDArray:
    rot_mat = t_mat[:-1, :-1]
    trans = t_mat[:-1, -1]
    return rot_mat @ position + trans


def rotate_by_matrix(t_mat: NDArray, position: NDArray) -> NDArray:
    rot_mat = t_mat[:-1, :-1]
    return rot_mat @ position


def reflect_vector(data: NDArray, normal_vector: NDArray) -> NDArray:
    return data - 2 * np.inner(data, normal_vector) * normal_vector


def reflect_point(
        data: NDArray, normal_vector: NDArray, point_in_plane: NDArray = np.zeros(3)) -> NDArray:
    tmp = data - 2. * np.inner(data, normal_vector) * normal_vector
    tmp += 2. * np.inner(normal_vector, point_in_plane) * normal_vector
    return tmp


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # inverse transformation matrix
    t_mats = make_simple_transformation_mats()
    inv_t_mat = inverse_transformation_matrix(t_mats[0])
    # reflection
    normal_vector = np.array([0., 0., -1.])
    base_vector = np.array([1., 1., 2.])
    base_point = np.array([1., 0., -2.])
    reflected_vector = reflect_vector(base_vector, normal_vector)
    reflected_point = reflect_point(base_point, normal_vector)
    print(f'{base_vector=} to {reflected_vector=}')
    print(f'{base_point=} to {reflected_point=}')
