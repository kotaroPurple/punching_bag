
import numpy as np
from numpy.typing import NDArray


LIGHT_SPEED = 299_792_458.0  # [m/s]

# parameters
# plane: center, かたむき (rho) の時間変化
# 電波周波数 f0 (24 GHz), 波数ベクトル k (向き含む)

# 瞬間の反射波を計算 -> 時系列の反射波 -> ミキシング & IQ
# . exp(ik.rho) の面積分: フラウンホーファー近似
# . exp(ik.rho).exp(ik|rho|^2/(2.R0)) の面積分: フレネル近似
# . 共通: exp(-iwt).E0.exp(i(kR0+k.R0))/(4pi.R0)


def calculate_amplitude_term(
        times: NDArray, base_positions: NDArray, direction: NDArray, frequency: float,
        const_denominator: bool = True, coeff: float = 1.0) -> NDArray:
    # time factor
    time_wave = np.exp(1.j * 2 * np.pi * frequency * times)
    # prepare
    base_length = np.linalg.norm(base_positions, axis=1)
    # numerator
    _k = 2 * np.pi * frequency / LIGHT_SPEED
    p1 = _k * base_length
    p2 = _k * (base_positions @ direction)
    numerator = np.exp(1.j * (p1 + p2))
    # denominator
    if const_denominator:
        mean_base = np.mean(base_positions, axis=0)
        denominator = 4. * np.pi * np.linalg.norm(mean_base)
    else:
        denominator = 4. * np.pi * base_length
    # result
    result = coeff * time_wave * numerator / denominator
    return result


