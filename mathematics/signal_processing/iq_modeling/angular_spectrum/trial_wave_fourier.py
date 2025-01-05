
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.typing import NDArray


@dataclass
class WaveWithRange:
    value: NDArray = field(default_factory=lambda: np.zeros(0))
    xs: NDArray = field(default_factory=lambda: np.zeros(0))
    ys: NDArray = field(default_factory=lambda: np.zeros(0))


def predict_bessel_x_range(radius: float, n_waves: int = 20) -> float:
    _range = 3. * n_waves
    return _range / radius


def predict_sinc_x_range(width: float, n_waves: int = 20) -> float:
    _range = 6. * n_waves
    return _range / width


def fourier_plane_wave_on_circle(
        init_value: complex, radius: float, k1: float, k2: float, range_k: float, number: int) \
            -> WaveWithRange:
    # kx, ky array
    _kx = np.linspace(-range_k, range_k, number) + k1
    _ky = np.linspace(-range_k, range_k, number) + k2
    kxx, kyy = np.meshgrid(_kx, _ky)
    # kappa = sqrt((k-kx)**2 + (k-ky)**2)
    kappa = np.sqrt((k1 - kxx)**2 + (k2 - kyy)**2)
    # transform
    # exp(init) * 2 * pi * r^2 * J1(kappa.r) / (kappa.r)
    result = np.empty(kappa.shape, dtype=np.complex128)
    valid_kappa = kappa != 0.
    result[valid_kappa] = init_value * 2. * np.pi * radius * j1(kappa[valid_kappa] * radius) / kappa[valid_kappa]
    result[~valid_kappa] = init_value * 2. * np.pi * radius * radius * 0.5  # (j1(ax)/ax ~ 1/2 (x=0))
    result_data = WaveWithRange(value=result, xs=kxx, ys=kyy)
    return result_data


def fourier_plane_wave_on_rectangle(
        init_value: complex, length_x: float, length_y: float, k1: float, k2: float, range_k: float, number: int) \
            -> WaveWithRange:
    # kx, ky array
    _kx = np.linspace(-range_k, range_k, number) + k1
    _ky = np.linspace(-range_k, range_k, number) + k2
    kxx, kyy = np.meshgrid(_kx, _ky)
    # x, y
    xs = (kxx - k1) * length_x / 2.
    ys = (kyy - k2) * length_y / 2.
    # transform
    # np.sinc(x) = sin(pi.x) / (pi.x)
    value = init_value * length_x * length_y * np.sinc(xs / np.pi) * np.sinc(ys / np.pi)
    result = WaveWithRange(value=value, xs=kxx, ys=kyy)
    return result


def inverse_fourier_transform(
        fft_data: WaveWithRange, dx: float, apply_shift: bool) -> WaveWithRange:
    # ifft
    value = np.fft.ifft2(fft_data.value) * fft_data.value.size
    if apply_shift:
        value = np.fft.fftshift(value)
    # range
    number_y, number_x = fft_data.value.shape
    half_y = 2. * np.pi * dx * number_y / 2.
    half_x = 2. * np.pi * dx * number_x / 2.
    xs = np.linspace(-half_x, half_x, number_x)
    ys = np.linspace(-half_y, half_y, number_y)
    xx, yy = np.meshgrid(xs, ys)
    result = WaveWithRange(value=value, xs=xx, ys=yy)
    return result


def inverse_fourier_transform_in_target(
        fft_data: WaveWithRange, x_array: NDArray, y_array: NDArray) -> WaveWithRange:
    # meshgrid
    xx, yy = np.meshgrid(x_array, y_array)
    # inverse fft at each point
    ifft_list = []
    for _x, _y in zip(xx.flatten(), yy.flatten()):
        phase = fft_data.xs * _x + fft_data.ys * _y
        ifft_value = np.sum(fft_data.value * np.exp(1.j * phase))
        ifft_list.append(ifft_value)
    # output
    value = np.array(ifft_list).reshape(xx.shape)
    return WaveWithRange(value=value, xs=xx, ys=yy)


if __name__ == '__main__':
    k1 = 50.
    k2 = 0.
    number = 511
    radius = 0.05

    range_k_circle = predict_bessel_x_range(radius)
    range_k_rectangle = predict_sinc_x_range(radius * 2.)

    circle_fft = fourier_plane_wave_on_circle(
        complex(1.0), radius, k1, k2, range_k_circle, number)
    rectangular_fft = fourier_plane_wave_on_rectangle(
        complex(1.0), 2 * radius, 2 * radius, k1, k2, range_k_rectangle, number)

    dx = 1 / (2 * range_k_circle)
    circle_ifft = inverse_fourier_transform(circle_fft, dx=dx, apply_shift=True)

    manual_xs = np.linspace(-2 * radius, 2 * radius, 10)
    manual_ys = np.linspace(-2 * radius, 2 * radius, 10)
    manual_ifft = inverse_fourier_transform_in_target(circle_fft, manual_xs, manual_ys)

    print(circle_fft.value.sum(), rectangular_fft.value.sum())

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.title('Circle')
    plt.imshow(
        circle_fft.value[::-1, :].real,
        extent=(circle_fft.xs.min(), circle_fft.xs.max(), circle_fft.ys.min(), circle_fft.ys.max()))
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title('Rectangular')
    plt.imshow(
        rectangular_fft.value[::-1, :].real,
        extent=(
            rectangular_fft.xs.min(), rectangular_fft.xs.max(),
            rectangular_fft.ys.min(), rectangular_fft.ys.max()))
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.title('Circle IFFT')
    plt.imshow(
        np.abs(circle_ifft.value[::-1, :]),
        extent=(
            circle_ifft.xs.min(), circle_ifft.xs.max(),
            circle_ifft.ys.min(), circle_ifft.ys.max()))
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.title('Manual Circle IFFT')
    plt.imshow(
        np.abs(manual_ifft.value[::-1, :]),
        extent=(
            manual_ifft.xs.min(), manual_ifft.xs.max(),
            manual_ifft.ys.min(), manual_ifft.ys.max()))
    plt.colorbar()

    y_max = max(circle_fft.value.real.max(), rectangular_fft.value.real.max())
    y_min = min(circle_fft.value.real.min(), rectangular_fft.value.real.min())

    plt.subplot(2, 3, 4)
    argmax_index = np.argmax(np.abs(circle_fft.value))
    max_y_index = argmax_index // circle_fft.value.shape[0]
    plt.title('Circle at max horizontal')
    plt.plot(circle_fft.xs[max_y_index, :], circle_fft.value[max_y_index, :].real)
    plt.ylim((y_min, y_max))

    plt.subplot(2, 3, 5)
    argmax_index = np.argmax(np.abs(rectangular_fft.value))
    max_y_index = argmax_index // rectangular_fft.value.shape[0]
    plt.title('Rectangular at max horizontal')
    plt.plot(rectangular_fft.xs[max_y_index, :], rectangular_fft.value[max_y_index, :].real)
    plt.ylim((y_min, y_max))

    plt.show()
