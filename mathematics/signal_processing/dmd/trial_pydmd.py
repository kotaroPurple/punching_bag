
import numpy as np
from pydmd import DMD
from pydmd.plotter import plot_summary


fs = 1_000                                            # Hz
t = np.arange(0, 2.0, 1/fs)                           # 2-s trace
sig = (np.exp(1j*2*np.pi*1.2*t) +                     # 1.2 Hz component
        0.3*np.exp(1j*2*np.pi*10*t) +                  # 10 Hz component
        0.05*(np.random.randn(t.size) + 1j*np.random.randn(t.size)))

L = 128
N = len(sig)
K = N - L + 1                       # number of columns
H = np.vstack([sig[i:i + K] for i in range(L)])
X, Xp = H[:, :-1], H[:, 1:]         # snapshots (shift by one step)

# Build an exact DMD model with 12 spatiotemporal modes.
dmd = DMD(svd_rank=12)

# Fit the DMD model.
# X = (n, m) numpy array of time-varying snapshot data.
dmd.fit(X)

# Plot a summary of the key spatiotemporal modes.
plot_summary(dmd)
