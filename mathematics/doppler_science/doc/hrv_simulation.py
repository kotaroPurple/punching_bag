#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心拍変動（HRV）を考慮した心拍数推定の全周波数と片側周波数の比較シミュレーション
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # ベッセル関数
from scipy.fft import fft, fftfreq
from scipy import signal
import time

def simulate_iq_signal(d0, omega0, k, t):
    """単一周波数の心拍動きによるIQ信号をシミュレーション"""
    # 変位
    d = d0 * np.sin(omega0 * t)
    # IQ信号
    s = np.exp(2j * k * d)
    return s
    
def simulate_iq_signal_with_hrv(d0, f0, hrv_std, k, t, fs):
    """心拍変動（HRV）を考慮したIQ信号のシミュレーション"""
    # 心拍周波数の時間変動を生成
    # ガウスノイズをフィルタリングして滑らかな変動を生成
    hrv_noise = np.random.normal(0, hrv_std, len(t))
    # 低域通過フィルタで滑らかな変動にする
    b, a = signal.butter(2, 0.1, fs=fs)
    smooth_hrv = signal.filtfilt(b, a, hrv_noise)
    
    # 心拍周波数の時間変動
    f_t = f0 + smooth_hrv
    
    # 環周波数から位相を積分して得る
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # 変位
    d = d0 * np.sin(phase)
    
    # IQ信号
    s = np.exp(2j * k * d)
    return s, f_t

def simulate_complex_iq_signal(d0, d1, omega0, omega1, delta, k, t):
    """心拍と呼吸の複合動きによるIQ信号をシミュレーション"""
    # 変位
    d = d0 * np.sin(omega0 * t) + d1 * np.sin(omega1 * t - delta)
    # IQ信号
    s = np.exp(2j * k * d)
    return s

def simulate_complex_iq_signal_with_hrv(d0, d1, f0, f1, hrv_std, delta, k, t, fs):
    """心拍変動を含む心拍と呼吸の複合動きによるIQ信号をシミュレーション"""
    # 心拍周波数の時間変動を生成
    hrv_noise = np.random.normal(0, hrv_std, len(t))
    b, a = signal.butter(2, 0.1, fs=fs)
    smooth_hrv = signal.filtfilt(b, a, hrv_noise)
    
    # 心拍周波数の時間変動
    f_t = f0 + smooth_hrv
    
    # 環周波数から位相を積分して得る
    phase_heart = 2 * np.pi * np.cumsum(f_t) / fs
    
    # 呼吸の位相
    phase_resp = 2 * np.pi * f1 * t - delta
    
    # 変位
    d = d0 * np.sin(phase_heart) + d1 * np.sin(phase_resp)
    
    # IQ信号
    s = np.exp(2j * k * d)
    return s, f_t

def calculate_full_spectrum_intensity(s):
    """全周波数を使った強度変化"""
    # DC成分を除去
    dc = np.mean(s)
    s_prime = s - dc
    # 強度計算
    intensity = np.abs(s_prime)**2
    return intensity

def calculate_positive_spectrum_intensity(s):
    """片側周波数を使った強度変化"""
    # フーリエ変換
    s_fft = fft(s)
    N = len(s)
    # 負の周波数成分をゼロにする
    s_fft[N//2:] = 0
    # 逆フーリエ変換で片側周波数信号を得る
    s_pos = np.fft.ifft(s_fft)
    # 強度計算
    intensity = np.abs(s_pos)**2
    return intensity

def analyze_frequency(signal, fs):
    """周波数解析"""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    return xf[:N//2], 2.0/N * np.abs(yf[:N//2])

def calculate_snr_with_hrv(signal, fs, target_freq, bandwidth_factor=1.5):
    """心拍変動を考慮したSNR計算
    心拍変動がある場合、帯域幅を広げて計算する
    """
    # 帯域幅を心拍周波数に応じて設定
    # 心拍変動が大きいほど帯域幅を広げる
    bandwidth = target_freq * 0.2 * bandwidth_factor  # 心拍周波数の20%程度を基本帯域幅とする
    
    # FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    
    # 正の周波数のみ
    xf = xf[:N//2]
    yf = yf[:N//2]
    power = np.abs(yf)**2 / N
    
    # 目標周波数帯域のインデックス
    freq_min = target_freq - bandwidth/2
    freq_max = target_freq + bandwidth/2
    
    idx_min = np.argmin(np.abs(xf - freq_min))
    idx_max = np.argmin(np.abs(xf - freq_max))
    
    # 信号パワー（目標周波数帯域内）
    signal_power = np.sum(power[idx_min:idx_max+1])
    
    # 雑音パワー（目標周波数帯域外）
    noise_indices = np.ones(len(xf), dtype=bool)
    noise_indices[idx_min:idx_max+1] = False
    noise_power = np.sum(power[noise_indices])
    
    # SNR計算
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def plot_spectrogram(signal, fs, title):
    """スペクトログラム表示"""
    f, t, Sxx = signal.spectrogram(signal, fs, nperseg=256, noverlap=128)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.colorbar(label='Power [dB]')
    plt.title(title)
    plt.ylabel('周波数 [Hz]')
    plt.xlabel('時間 [s]')
    plt.ylim(0, 5)

def evaluate_hrv_impact(hrv_std_values, fs=100, duration=30):
    """心拍変動の大きさが全周波数と片側周波数の性能に与える影響を評価"""
    # パラメータ設定
    t = np.arange(0, duration, 1/fs)
    k = 500  # 波数 (24GHz)
    d0 = 100e-6  # 心拍による変位振幅 [m]
    f0 = 1.2  # 心拍周波数 [Hz]
    
    # 結果格納用
    snr_full_1x = []
    snr_pos_1x = []
    snr_full_2x = []
    snr_pos_2x = []
    
    # 各心拍変動の大きさでシミュレーション
    for hrv_std in hrv_std_values:
        # 心拍変動ありのIQ信号生成
        s_hrv, f_t = simulate_iq_signal_with_hrv(d0, f0, hrv_std, k, t, fs)
        
        # 全周波数と片側周波数の強度計算
        intensity_full = calculate_full_spectrum_intensity(s_hrv)
        intensity_pos = calculate_positive_spectrum_intensity(s_hrv)
        
        # SNR計算（帯域幅は心拍変動の大きさに応じて調整）
        bandwidth_factor = 1.0 + 5.0 * hrv_std / f0  # 心拍変動が大きいほど帯域幅を広げる
        
        # 心拍周波数でのSNR計算
        snr_full = calculate_snr_with_hrv(intensity_full, fs, f0, bandwidth_factor)
        snr_pos = calculate_snr_with_hrv(intensity_pos, fs, f0, bandwidth_factor)
        
        # 2倍周波数でのSNR計算
        snr_full_2 = calculate_snr_with_hrv(intensity_full, fs, 2*f0, bandwidth_factor)
        snr_pos_2 = calculate_snr_with_hrv(intensity_pos, fs, 2*f0, bandwidth_factor)
        
        # 結果を格納
        snr_full_1x.append(snr_full)
        snr_pos_1x.append(snr_pos)
        snr_full_2x.append(snr_full_2)
        snr_pos_2x.append(snr_pos_2)
    
    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(hrv_std_values, snr_full_1x, 'b-', label='全周波数 (1x周波数)')
    plt.plot(hrv_std_values, snr_pos_1x, 'r-', label='片側周波数 (1x周波数)')
    plt.plot(hrv_std_values, snr_full_2x, 'b--', label='全周波数 (2x周波数)')
    plt.plot(hrv_std_values, snr_pos_2x, 'r--', label='片側周波数 (2x周波数)')
    plt.xlabel('心拍変動の標準偏差 [Hz]')
    plt.ylabel('SNR [dB]')
    plt.title('心拍変動の大きさがSNRに与える影響')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hrv_impact_on_snr.png')
    
    return {
        'hrv_std_values': hrv_std_values,
        'snr_full_1x': snr_full_1x,
        'snr_pos_1x': snr_pos_1x,
        'snr_full_2x': snr_full_2x,
        'snr_pos_2x': snr_pos_2x
    }

def main():
    """メイン関数"""
    # 心拍変動の大きさの影響を評価
    hrv_std_values = np.linspace(0.01, 0.3, 10)  # 心拍変動の標準偏差 [Hz]
    results = evaluate_hrv_impact(hrv_std_values)
    
    # 結果の表示
    print("心拍変動の大きさとSNRの関係:")
    for i, hrv_std in enumerate(results['hrv_std_values']):
        print(f"HRV標準偏差: {hrv_std:.3f} Hz")
        print(f"  全周波数 (1x): {results['snr_full_1x'][i]:.2f} dB")
        print(f"  片側周波数 (1x): {results['snr_pos_1x'][i]:.2f} dB")
        print(f"  全周波数 (2x): {results['snr_full_2x'][i]:.2f} dB")
        print(f"  片側周波数 (2x): {results['snr_pos_2x'][i]:.2f} dB")
    
    plt.show()

if __name__ == "__main__":
    main()