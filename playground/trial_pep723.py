#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "matplotlib",
# ]
# ///
#
# >>> uv run trial_pep723.py
# 参考: https://qiita.com/ShigemoriMasato/items/b254709391d2cbb1bbe6

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
