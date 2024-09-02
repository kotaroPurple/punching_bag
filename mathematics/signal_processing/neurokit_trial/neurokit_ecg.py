
"""
ref: https://neuropsychology.github.io/NeuroKit/examples/ecg_heartbeats/ecg_heartbeats.html

want:
- start time (when ECG starts)
- peak indicex
- sampling rate (peak index * fs + start_time = peak time in real world)
- settings (argments)
"""

# %%
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

# %%
# Simulate 30 seconds of ECG Signal (recorded at 250 samples / second)
ecg_signal = nk.ecg_simulate(duration=30, sampling_rate=250)  # NDArray
print(type(ecg_signal))
print()

# Automatically process the (raw) ECG signal
signals, info = nk.ecg_process(ecg_signal, sampling_rate=250)  # DataFrame, dict
print(signals.columns)
print(info.keys())

# Extract clean ECG and R-peaks location
rpeaks = info["ECG_R_Peaks"]
cleaned_ecg = signals["ECG_Clean"]

print(rpeaks)

# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks, cleaned_ecg)
plt.show()
