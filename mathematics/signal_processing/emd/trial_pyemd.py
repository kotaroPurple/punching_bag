
from PyEMD import EMD, EEMD, CEEMDAN
import numpy  as np
import pylab as plt


"""
EMD
"""

# Define signal
t = np.linspace(0, 1, 200)
s = np.cos(11*2*np.pi*t*t) + 6*t*t

# # サンプル信号の生成
# t = np.linspace(0, 1, 1000)
# # 例: 複数の周波数を持つ信号
# s = (np.sin(2 * np.pi * 5 * t) +
#           0.5 * np.sin(2 * np.pi * 20 * t) +
#           0.2 * np.sin(2 * np.pi * 50 * t) +
#           0.1 * np.random.randn(len(t)))  # ノイズを追加

# Execute EMD on signal
IMF = EMD().emd(s,t)
N = IMF.shape[0]+1

# Plot results
plt.figure(0)
plt.subplot(N,1,1)
plt.plot(t, s, 'r')
plt.title("Input signal: $S(t)=cos(22\\pi t^2) + 6t^2$")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
# plt.show()


"""
EEMD
"""
# Define signal
t = np.linspace(0, 1, 300)

def _sin(x, p):
    return np.sin(2*np.pi*x*t+p)

S = 3*_sin(18,0.2)*(t-0.2)**2
S += 5*_sin(11,2.7)
S += 3*_sin(14,1.6)
S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
S += t**2.1 -t
S += 0.5 * (np.random.random(len(S)) - 0.5)

# Assign EEMD to `eemd` variable
eemd = EEMD()

# Say we want detect extrema using parabolic method
emd = eemd.EMD
emd.extrema_detection="parabol"

# Execute EEMD on S
eIMFs = eemd.eemd(S, t)
nIMFs = eIMFs.shape[0]

# Plot results
plt.figure(1, figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(t, S, 'r')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(t, eIMFs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
# plt.show()


"""
CEEMDAN
"""
ceemdan = CEEMDAN(range_thr=0.001, total_power_thr=0.01)
c_imfs = ceemdan.ceemdan(S, t)
nIMFs = c_imfs.shape[0]

# Plot results
plt.figure(2, figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(t, S, 'r')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(t, c_imfs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()
