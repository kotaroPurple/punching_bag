
import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
fs = 1000  # サンプリング周波数
T = 1.0    # 収録時間（秒）
t = np.arange(0, T, 1/fs)  # 時間ベクトル

# 2つの音源信号（正弦波）を生成
f1 = 100  # 音源1の周波数
f2 = 80  # 音源2の周波数
signal1 = np.sin(2 * np.pi * f1 * t)
signal2 = np.sin(2 * np.pi * f2 * t)

# # 音源信号をプロット
# plt.figure(figsize=(10, 4))
# plt.plot(t, signal1, label='Signal 1 (100 Hz)')
# plt.plot(t, signal2, label='Signal 2 (200 Hz)')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.show()

# パラメータ設定
num_antennas = 4  # 集音機の数
d = 0.5  # 集音機間の距離（波長単位）
theta1 = -30 * np.pi / 180  # 音源1の到来方向（-30度）
theta2 = 45 * np.pi / 180  # 音源2の到来方向（45度）

# 各集音機での受信信号をシミュレート
signals = np.zeros((num_antennas, len(t)), dtype=complex)

for i in range(num_antennas):
    # 各集音機での信号は、到来方向に応じた位相シフトがある
    phase_shift1 = np.exp(-1j * 2 * np.pi * d * i * np.sin(theta1))
    phase_shift2 = np.exp(-1j * 2 * np.pi * d * i * np.sin(theta2))
    signals[i, :] = signal1 * phase_shift1 + signal2 * phase_shift2

# print(signals.shape)

# # 受信信号をプロット（アンテナ1）
# plt.figure(figsize=(10, 4))
# for i in range(num_antennas):
#     plt.plot(t, np.real(signals[i, :]), label='Antenna 1')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Received Signal at Antenna 1')
# plt.show()

def apply_music(signal_matrix, num_sources, scan_angles, num_antennas):
    R = np.dot(signal_matrix, signal_matrix.conj().T) / signal_matrix.shape[1]
    eigenvalues, eigenvectors = np.linalg.eig(R)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, num_sources:]
    P_music = np.zeros(len(scan_angles))
    for i, theta in enumerate(scan_angles):
        steering_vector = np.exp(-1j * 2 * np.pi * np.arange(num_antennas) * np.sin(theta))
        P_music[i] = 1 / np.abs(steering_vector.conj().T @ En @ En.conj().T @ steering_vector)
    return P_music

# 音源方向をスキャンする角度の範囲
scan_angles = np.linspace(-np.pi/2, np.pi/2, 180)  # -90度から90度までスキャン

# MUSIC法を適用
P_music = apply_music(signals, 2, scan_angles, num_antennas)

# MUSICスペクトルをプロット
plt.figure(figsize=(10, 4))
plt.plot(np.degrees(scan_angles), 10 * np.log10(P_music))
plt.xlabel('Angle (degrees)')
plt.ylabel('MUSIC Spectrum (dB)')
plt.title('MUSIC Spectrum')
plt.grid(True)
plt.show()
