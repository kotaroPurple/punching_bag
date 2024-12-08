
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from my_emd.iceemdan import iceemdan


def _generate_sample_data() -> NDArray:
    t = np.linspace(0, 1, 200)

    def _sin(x, p):
        return np.sin(2*np.pi*x*t+p)

    data = 3 * _sin(18, 0.2)*(t-0.2)**2
    data += 5 * _sin(11, 2.7)
    data += 3 * _sin(14, 1.6)
    data += 1 * np.sin(4*2*np.pi*(t-0.8)**2)
    data += (t**2.1) -t
    return data


def main() -> None:
    data = _generate_sample_data()
    result = iceemdan(data, max_imf=1, n_ensemble=10)
    plt.plot(data)
    plt.plot(result)
    plt.show()


if __name__ == '__main__':
    main()
