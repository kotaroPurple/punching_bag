
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

LIGHT_SPEED = 3e8
SENSOR_FREQUENCY = 24e9
WAVE_NUMBER = 2 * np.pi * SENSOR_FREQUENCY / LIGHT_SPEED


def generate_iq_wave(
        displacements: NDArray[np.floating], wave_number: float, amp: float = 1.) -> NDArray[np.complex128]:
    return amp * np.exp(2j * wave_number * displacements).astype(np.complex128)
