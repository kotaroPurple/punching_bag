
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def make_noise(number: int, mu: float, sigma: float) -> NDArray:
    return np.random.normal(mu, sigma, number)

number = 10_000
mu = 0.
sigma = 300.

data = make_noise(number, mu, sigma)
int_data = data.astype(np.int64)

unique_data, unique_count = np.unique(int_data, return_counts=True)
predicted_mu = np.mean(int_data)
predicted_sigma = np.std(int_data)

plt.bar(unique_data, unique_count)
plt.vlines(predicted_mu, 0, np.max(unique_count), colors='black')
plt.vlines([predicted_mu - 3 * predicted_sigma, predicted_mu + 3 * predicted_sigma], 0, np.max(unique_count), colors='red')
plt.show()
