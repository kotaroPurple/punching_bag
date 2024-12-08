
import numpy as np
from PyEMD import EMD
from numpy.typing import NDArray

from my_emd.util import extract_local_mean


def generate_noise(size: int, n_ensemble: int, sigma: float) -> NDArray:
    return np.random.normal(0., sigma, (n_ensemble, size))


def iceemdan(data: NDArray, max_imf: int, n_ensemble: int) -> NDArray:
    # prepare
    epsilon = 0.005
    init_sigma = 1.0
    number = data.size
    nbsym = 2
    emd = EMD()
    all_noises = generate_noise(number, n_ensemble, init_sigma)

    # step 1
    # x_i= x + b0. E1(w_i)
    # r1 = <M(x_i)>

    # decompose noise (list=10, (imfs, data length))
    all_noise_emd = [emd(noise, max_imf=-1) for noise in all_noises]
    mean_x = np.zeros_like(data)
    for i in range(n_ensemble):
        xi = data + epsilon * all_noise_emd[i][0]
        mean_xi = extract_local_mean(np.arange(number), xi, nbsym=nbsym)
        mean_x += mean_xi
    mean_x /= n_ensemble

    # step 2
    # d1 = x - r1
    r1 = mean_x
    d1 = data - r1

    # step 3
    # d2 = r1 - r2
    #    = r1 - <M(r1 + b1.E2(w_i))>

    # step 4, 5
    # rk = <M(rk-1 + bk-1.Ek(w_i))>
    # dk = rk-1 - rk

    return d1
