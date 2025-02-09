
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def _calculate_phase_from_complex_signal(data: NDArray, use_unwrap: bool) -> NDArray:
    if use_unwrap:
        angles = np.unwrap(np.angle(data, deg=False))
        return angles
    else:
        # phase unwrap
        # 1. exp(iT(n+1)) / exp(iT(n)) = exp(i.(T(n+1) - T(n)))
        # 2. angle(exp(i.(T(n+1) - T(n)))) = T(n+1) - T(n) = dT(n)
        # 3. cumsum(dT(n))_N = T(0), T(1), ..., T(n)
        init_angle = np.angle(data[0], deg=False)
        phase_diff = np.angle(data[1:] / data[:-1])
        phase_h = init_angle + np.r_[0., np.cumsum(phase_diff)]
        return phase_h


def reconstruct_phase_from_analytic_signal(
        complex_data: NDArray, tau: float) -> tuple[NDArray, NDArray]:
    # 位相算出, 初期位相を 0 にする
    phases = _calculate_phase_from_complex_signal(complex_data, use_unwrap=False)
    init_phase = phases[0]
    phases = phases - init_phase

    # (前提) 線形に位相が単調増加 (減少) する
    # 区間の平均各周波数を求め, 位相のトレンドを求める
    number = len(phases)
    effective_angular_freq = (np.mean(phases[1:] - phases[:-1])) / tau
    times = np.arange(number) * tau
    effective_trend = effective_angular_freq * times

    _m = round(effective_angular_freq / (2 * np.pi) * number * tau)
    phase_modulation = phases - effective_trend
    fft_mod = np.fft.fft(phase_modulation)
    fft_mod_new = _reconstruct_phase_modulation(fft_mod, _m)
    phase_modulation_new = np.fft.ifft(fft_mod_new)

    phase_new = init_phase + effective_trend + phase_modulation_new.real
    phase_new, outliers = _phase_postprocess(phase_new)
    return phase_new, outliers


def _reconstruct_phase_modulation(ft_mod: NDArray, m: int):
    """
    Reconstruct the true phase-modulation from phase-modulation reconstructed via the HT.

    Parameters
    ----------
        ft_mod: ndarray with shape (N)
            Fourier transform of phase-modulation.
        m: int
            Index coresponding to the effective frenqency of the oscillatory signal.

    Returns
    -------
        ft_mod_new: ndarray with shape (N)
            Reconstructed Fourier transform of phase-modulation.
    """
    N = len(ft_mod)
    # ft_mod_new = (np.zeros(N) + 1j*np.zeros(N))
    ft_mod_new = np.zeros(N, dtype=np.complex128)
    Nh = int(N/2)
    for n in np.arange(2*m)+Nh-2*m+1:
        ft_mod_new[n] = 2*ft_mod[n]
    for n in np.arange(Nh - 3*m)[::-1] + m+1:
        ft_mod_new[n] = 2*ft_mod[n] + ft_mod_new[n+2*m]
    ft_mod_new[m] = 2*np.real(ft_mod[m]+ 0.5*ft_mod_new[3*m]) + 1j*np.imag(ft_mod[m]+ 0.5*ft_mod_new[3*m])
    for n in np.arange(m)[::-1]:
        ft_mod_new[n] = ft_mod[n] + 0.5 * np.conjugate(ft_mod_new[2*m - n]) + 0.5*ft_mod_new[n+2*m]
    for n in np.arange(Nh+1, N):
        ft_mod_new[n] = np.conjugate(ft_mod_new[N-n])

    return ft_mod_new


def _phase_postprocess(phase: NDArray) -> tuple[NDArray, NDArray]:
    """
    Detect outliers in the phase difference and replace the phase at that point with linear interpolation.

    Parameters
    ----------
        phase: ndarray with shape (N)
            Phase signal.

    Returns
    ----------
        phase_new: ndarray with shape (N)
            Postprocessed phase signal.
        outlier_list: ndarray
            List of outliers.

    """
    # K = np.size(phase, axis=0)
    # N = np.size(phase, axis=1) #Length of the data
    N = len(phase)

    diff = phase[1:] - phase[:-1]
    diff_mad = np.sum(np.abs(diff - np.median(diff))) / (N-1)
    outlier = np.abs(diff - np.median(diff)) > 3 * diff_mad
    # outlier_list = np.append(outlier_list.reshape(k-1, N-1), outlier.reshape(1, N-1), axis=0)
    phase_new_k = np.zeros(N)

    n = 0
    while n < N - 1:
        if outlier[n] is True:
            J = 1
            while ((n+J < N-2) and (outlier[n+J] is True)):
                J += 1
            for j in range(J):
                if ((n == 0) or (n + J - 1 == N - 1)):
                    phase_new_k[n+j] = phase[n+j]
                else:
                    phase_new_k[n+j] = (phase[n-1]* (J-j)+ phase[n+J]*(j+1)) / (J+1)
            n += J
        else:
            phase_new_k[n] = phase[n]
            n+=1
    # output
    phase_new_k[N-1] = phase[N-1]
    return phase_new_k, outlier
