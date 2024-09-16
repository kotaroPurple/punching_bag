
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class FuncOption(Enum):
    cos_cos = auto()
    sin_cos = auto()
    sin_sin = auto()


def calculate_cos_cos(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = 0.5 * np.sin((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = angles - start_rad
    else:
        term2 = 0.5 * np.sin((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def calculate_sin_sin(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = -0.5 * np.sin((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = angles - start_rad
    else:
        term2 = 0.5 * np.sin((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def calculate_sin_cos(
        start_rad: float, end_rad: float, m_: int, n_: int, number: int = 100) \
            -> tuple[NDArray, NDArray]:
    # angle
    angles = np.linspace(start_rad, end_rad, number)
    # n+m
    term1 = -0.5 * np.cos((m_ + n_) * angles) / (m_ + n_)
    # n-m
    if m_ == n_:
        term2 = np.zeros_like(angles)
    else:
        term2 = -0.5 * np.cos((m_ - n_) * angles) / (m_ - n_)
    return angles, term1 + term2


def integrate_cos_cos(
        start_time: float, end_time: float, angular_vel: float, delta_time: float, m_: int, n_: int, number: int = 100) \
        -> tuple[NDArray, NDArray, NDArray]:
    # angle
    starts = np.linspace(start_time, end_time, number)
    ends = starts + delta_time
    # original wave
    wave_ = np.cos(n_ * angular_vel * starts) * np.cos(m_ * angular_vel * starts)
    # n+m
    term1 = 0.5 * (np.sin((m_ + n_) * angular_vel * ends) - np.sin((m_ + n_) * angular_vel * starts)) / ((m_ + n_) * angular_vel)  # NOQA
    # n-m
    if m_ == n_:
        term2 = 0.5 * (ends - starts)
    else:
        term2 = 0.5 * (np.sin((n_ - m_) * ends) - np.sin((n_ - m_) * starts)) / ((n_ - m_) * angular_vel)  # NOQA
    return starts, wave_, term1 + term2


def integrate_sin_cos(
        start_time: float, end_time: float, angular_vel: float, delta_time: float, m_: int, n_: int, number: int = 100) \
        -> tuple[NDArray, NDArray, NDArray]:
    # angle
    starts = np.linspace(start_time, end_time, number)
    ends = starts + delta_time
    # original wave
    wave_ = np.sin(n_ * angular_vel * starts) * np.cos(m_ * angular_vel * starts)
    # n+m
    term1 = -0.5 * (np.cos((m_ + n_) * angular_vel * ends) - np.cos((m_ + n_) * angular_vel * starts)) / ((m_ + n_) * angular_vel)  # NOQA
    # n-m
    if m_ == n_:
        term2 = np.zeros_like(starts)
    else:
        term2 = -0.5 * (np.cos((n_ - m_) * angular_vel * ends) - np.cos((n_ - m_) * angular_vel * starts)) / ((n_ - m_) * angular_vel)  # NOQA
    return starts, wave_, term1 + term2


def integrate_sin_sin(
        start_time: float, end_time: float, angular_vel: float, delta_time: float, m_: int, n_: int, number: int = 100) \
        -> tuple[NDArray, NDArray, NDArray]:
    # angle
    starts = np.linspace(start_time, end_time, number)
    ends = starts + delta_time
    # original wave
    wave_ = np.sin(n_ * angular_vel * starts) * np.sin(m_ * angular_vel * starts)
    # n+m
    term1 = -0.5 * (np.sin((m_ + n_) * angular_vel * ends) - np.sin((m_ + n_) * angular_vel * starts)) / ((m_ + n_) * angular_vel)  # NOQA
    # n-m
    if m_ == n_:
        term2 = 0.5 * (ends - starts)
    else:
        term2 = 0.5 * (np.sin((n_ - m_) * ends) - np.sin((n_ - m_) * starts)) / ((n_ - m_) * angular_vel)  # NOQA
    return starts, wave_, term1 + term2


def main_simple():
    start_rad = 0.
    end_rad = 2. * np.pi
    ms = [i for i in range(1, 6)]
    ns = [i for i in range(1, 6)]

    fig, axes = plt.subplots(nrows=len(ns), ncols=len(ms), figsize=(8, 8))
    for j, n_ in enumerate(ns):
        for i, m_ in enumerate(ms):
            angles, values = calculate_cos_cos(start_rad, end_rad, m_, n_, number=100)
            axes[j, i].plot(angles, values)
            axes[j, i].set_ylim(-end_rad, end_rad)
    plt.show()


def plot_integration(option: FuncOption, end_flag: bool) -> None:
    base_freq = 1.
    base_period = 1. / base_freq
    base_angluar_vel = 2 * np.pi * base_freq
    start_time = 0.
    end_time = 2. * base_period
    delta_time = base_period / 2.
    ms = [i for i in range(1, 6)]
    ns = [i for i in range(1, 6)]

    def get_index(j, i) -> int:
        return j * len(ms) + i

    match option:
        case FuncOption.cos_cos:
            func = integrate_cos_cos
            title = 'cos.cos'
        case FuncOption.sin_cos:
            func = integrate_sin_cos
            title = 'sin.cos'
        case FuncOption.sin_sin:
            func = integrate_sin_sin
            title = 'sin.sin'
        case _:
            raise NameError()

    _, axes = plt.subplots(nrows=len(ns), ncols=len(ms), figsize=(12, 8))
    axes = axes.ravel()
    for j, n_ in enumerate(ns):
        for i, m_ in enumerate(ms):
            angles, wave_, values = func(
                start_time, end_time, base_angluar_vel, delta_time, m_, n_, number=100)
            index = get_index(j, i)
            twin_x = axes[index].twinx()
            axes[index].plot(angles, values)
            twin_x.plot(angles, wave_, c='pink', alpha=0.8)
            if m_ == n_:
                axes[index].set_ylim(-1.0, 1.0)
            else:
                axes[index].set_ylim(-0.2, 0.2)
            twin_x.axes.yaxis.set_visible(False)
    plt.suptitle(title)
    plt.tight_layout()

    if end_flag:
        plt.show()


def main_integration():
    plot_integration(FuncOption.cos_cos, end_flag=False)
    plot_integration(FuncOption.sin_sin, end_flag=False)
    plot_integration(FuncOption.sin_cos, end_flag=True)


if __name__ == '__main__':
    # main_simple()
    main_integration()
