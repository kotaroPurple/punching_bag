# 心拍数推定における全周波数と片側周波数の有効性評価

## 理論的背景

ドップラーセンサを用いた心拍数推定において、IQ信号の周波数成分の扱い方が重要となります。特に、全周波数（正負両方の周波数成分）と片側周波数（正の周波数成分のみ）のどちらが心拍数推定に有効かを評価する必要があります。また、実際の心拍には変動（摂動）が存在するため、その影響も考慮する必要があります。

理論解析から、以下の点が明らかになっています：

1. 全周波数の場合：
   - IQ波形 $s(t) = \sum_{n=-\infty}^{\infty} J_n (2k d_0) \exp(in\omega t)$
   - DCを除いた強度変化は偶数次高調波 $(2\omega, 4\omega, ...)$ を含む
   - 強度は $\|s^{\prime}\|^2 = 1 - \hat{J}_0^2 - 4\hat{J}_0 \sum_{n=1}^{\infty} \hat{J}_{2n} \cos(2n\omega t)$

2. 片側周波数の場合：
   - IQ波形 $s_+(t) = \sum_{n=0}^{\infty} J_n (2k d_0) \exp(in\omega t)$
   - 強度変化は全ての整数次高調波 $(\omega, 2\omega, 3\omega, ...)$ を含む
   - 強度は $\|s_+(t)\|^2 = \sum_{n=0}^{\infty} J_n^2(2kd_0) + 2\sum_{k=1}^{\infty}\left(\sum_{n=0}^{\infty} J_n(2kd_0)J_{n+k}(2kd_0)\right)\cos(k\omega t)$

3. 心拍と呼吸の複合成分の場合：
   - 相互変調成分 $n\omega_0 \pm m\omega_1$ が発生
   - 片側周波数では基本周波数 $\omega_0$ 成分が直接現れる可能性がある

## 評価方法

### 1. シミュレーションによる評価

#### 1.1 単一周波数（心拍のみ）のシミュレーション

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # ベッセル関数
from scipy.fft import fft, fftfreq
from scipy import signal

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

# パラメータ設定
fs = 100  # サンプリング周波数 [Hz]
duration = 30  # 計測時間 [s]
t = np.arange(0, duration, 1/fs)
k = 500  # 波数 (24GHz)
d0 = 100e-6  # 心拍による変位振幅 [m]
f0 = 1.2  # 心拍周波数 [Hz]
omega0 = 2 * np.pi * f0
hrv_std = 0.1  # 心拍変動の標準偏差 [Hz]

# 定常心拍IQ信号生成
s_constant = simulate_iq_signal(d0, omega0, k, t)

# 心拍変動を含むIQ信号生成
s_hrv, f_t = simulate_iq_signal_with_hrv(d0, f0, hrv_std, k, t, fs)

# 全周波数と片側周波数の強度計算
# 定常心拍

intensity_full_constant = calculate_full_spectrum_intensity(s_constant)
intensity_pos_constant = calculate_positive_spectrum_intensity(s_constant)

# 心拍変動あり
intensity_full_hrv = calculate_full_spectrum_intensity(s_hrv)
intensity_pos_hrv = calculate_positive_spectrum_intensity(s_hrv)

# FFTによる周波数解析
def analyze_frequency(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    return xf[:N//2], 2.0/N * np.abs(yf[:N//2])

# 周波数解析
# 定常心拍
freq_full_constant, amp_full_constant = analyze_frequency(intensity_full_constant, fs)
freq_pos_constant, amp_pos_constant = analyze_frequency(intensity_pos_constant, fs)

# 心拍変動あり
freq_full_hrv, amp_full_hrv = analyze_frequency(intensity_full_hrv, fs)
freq_pos_hrv, amp_pos_hrv = analyze_frequency(intensity_pos_hrv, fs)

# 結果の可視化

# 1. 定常心拍の場合
plt.figure(figsize=(12, 10))

# 時間領域波形
plt.subplot(2, 2, 1)
plt.plot(t, intensity_full_constant)
plt.title('全周波数強度変化 (定常心拍)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

plt.subplot(2, 2, 2)
plt.plot(t, intensity_pos_constant)
plt.title('片側周波数強度変化 (定常心拍)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

# 周波数領域
plt.subplot(2, 2, 3)
plt.plot(freq_full_constant, amp_full_constant)
plt.axvline(x=f0, color='r', linestyle='--', label=f'心拍周波数 {f0} Hz')
plt.axvline(x=2*f0, color='g', linestyle='--', label=f'2倍周波数 {2*f0} Hz')
plt.xlim(0, 5)
plt.title('全周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(freq_pos_constant, amp_pos_constant)
plt.axvline(x=f0, color='r', linestyle='--', label=f'心拍周波数 {f0} Hz')
plt.axvline(x=2*f0, color='g', linestyle='--', label=f'2倍周波数 {2*f0} Hz')
plt.xlim(0, 5)
plt.title('片側周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.tight_layout()
plt.savefig('heart_rate_constant_comparison.png')

# 2. 心拍変動ありの場合
plt.figure(figsize=(12, 10))

# 時間領域波形
plt.subplot(2, 2, 1)
plt.plot(t, intensity_full_hrv)
plt.title('全周波数強度変化 (心拍変動あり)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

plt.subplot(2, 2, 2)
plt.plot(t, intensity_pos_hrv)
plt.title('片側周波数強度変化 (心拍変動あり)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

# 周波数領域
plt.subplot(2, 2, 3)
plt.plot(freq_full_hrv, amp_full_hrv)
plt.axvline(x=f0, color='r', linestyle='--', label=f'平均心拍周波数 {f0} Hz')
plt.axvline(x=2*f0, color='g', linestyle='--', label=f'2倍周波数 {2*f0} Hz')
plt.xlim(0, 5)
plt.title('全周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(freq_pos_hrv, amp_pos_hrv)
plt.axvline(x=f0, color='r', linestyle='--', label=f'平均心拍周波数 {f0} Hz')
plt.axvline(x=2*f0, color='g', linestyle='--', label=f'2倍周波数 {2*f0} Hz')
plt.xlim(0, 5)
plt.title('片側周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.tight_layout()
plt.savefig('heart_rate_hrv_comparison.png')

# 3. 心拍変動の時間変化を表示
plt.figure(figsize=(10, 6))
plt.plot(t, f_t)
plt.axhline(y=f0, color='r', linestyle='--', label=f'平均心拍周波数 {f0} Hz')
plt.title('心拍周波数の時間変化')
plt.xlabel('時間 [s]')
plt.ylabel('心拍周波数 [Hz]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('heart_rate_variability.png')

# 4. 時間-周波数解析
def plot_spectrogram(signal, fs, title):
    f, t, Sxx = signal.spectrogram(signal, fs, nperseg=256, noverlap=128)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.colorbar(label='Power [dB]')
    plt.title(title)
    plt.ylabel('周波数 [Hz]')
    plt.xlabel('時間 [s]')
    plt.ylim(0, 5)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plot_spectrogram(intensity_full_constant, fs, '全周波数強度 (定常心拍)')

plt.subplot(2, 2, 2)
plot_spectrogram(intensity_pos_constant, fs, '片側周波数強度 (定常心拍)')

plt.subplot(2, 2, 3)
plot_spectrogram(intensity_full_hrv, fs, '全周波数強度 (心拍変動あり)')

plt.subplot(2, 2, 4)
plot_spectrogram(intensity_pos_hrv, fs, '片側周波数強度 (心拍変動あり)')

plt.tight_layout()
plt.savefig('heart_rate_spectrograms.png')
plt.show()
```

#### 1.2 複合周波数（心拍と呼吸）のシミュレーション

```python
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

# パラメータ設定
d1 = 2e-3  # 呼吸による変位振幅 [m]
f1 = 0.3  # 呼吸周波数 [Hz]
omega1 = 2 * np.pi * f1
delta = np.pi/4  # 位相差

# 定常心拍と呼吸の複合IQ信号生成
s_complex_constant = simulate_complex_iq_signal(d0, d1, omega0, omega1, delta, k, t)

# 心拍変動ありの複合IQ信号生成
s_complex_hrv, f_t_complex = simulate_complex_iq_signal_with_hrv(d0, d1, f0, f1, hrv_std, delta, k, t, fs)

# 全周波数と片側周波数の強度計算
# 定常心拍の場合
intensity_full_complex_constant = calculate_full_spectrum_intensity(s_complex_constant)
intensity_pos_complex_constant = calculate_positive_spectrum_intensity(s_complex_constant)

# 心拍変動ありの場合
intensity_full_complex_hrv = calculate_full_spectrum_intensity(s_complex_hrv)
intensity_pos_complex_hrv = calculate_positive_spectrum_intensity(s_complex_hrv)

# 周波数解析
# 定常心拍の場合
freq_full_complex_constant, amp_full_complex_constant = analyze_frequency(intensity_full_complex_constant, fs)
freq_pos_complex_constant, amp_pos_complex_constant = analyze_frequency(intensity_pos_complex_constant, fs)

# 心拍変動ありの場合
freq_full_complex_hrv, amp_full_complex_hrv = analyze_frequency(intensity_full_complex_hrv, fs)
freq_pos_complex_hrv, amp_pos_complex_hrv = analyze_frequency(intensity_pos_complex_hrv, fs)

# 結果の可視化
# 1. 定常心拍と呼吸の複合信号
plt.figure(figsize=(12, 10))

# 時間領域波形
plt.subplot(2, 2, 1)
plt.plot(t, intensity_full_complex_constant)
plt.title('全周波数強度変化 (定常心拍+呼吸)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

plt.subplot(2, 2, 2)
plt.plot(t, intensity_pos_complex_constant)
plt.title('片側周波数強度変化 (定常心拍+呼吸)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

# 周波数領域
plt.subplot(2, 2, 3)
plt.plot(freq_full_complex_constant, amp_full_complex_constant)
plt.axvline(x=f0, color='r', linestyle='--', label=f'心拍周波数 {f0} Hz')
plt.axvline(x=f1, color='b', linestyle='--', label=f'呼吸周波数 {f1} Hz')
plt.axvline(x=f0+f1, color='m', linestyle=':', label=f'和周波数 {f0+f1} Hz')
plt.axvline(x=f0-f1, color='c', linestyle=':', label=f'差周波数 {f0-f1} Hz')
plt.xlim(0, 5)
plt.title('全周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(freq_pos_complex_constant, amp_pos_complex_constant)
plt.axvline(x=f0, color='r', linestyle='--', label=f'心拍周波数 {f0} Hz')
plt.axvline(x=f1, color='b', linestyle='--', label=f'呼吸周波数 {f1} Hz')
plt.axvline(x=f0+f1, color='m', linestyle=':', label=f'和周波数 {f0+f1} Hz')
plt.axvline(x=f0-f1, color='c', linestyle=':', label=f'差周波数 {f0-f1} Hz')
plt.xlim(0, 5)
plt.title('片側周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.tight_layout()
plt.savefig('complex_constant_comparison.png')

# 2. 心拍変動ありと呼吸の複合信号
plt.figure(figsize=(12, 10))

# 時間領域波形
plt.subplot(2, 2, 1)
plt.plot(t, intensity_full_complex_hrv)
plt.title('全周波数強度変化 (心拍変動あり+呼吸)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

plt.subplot(2, 2, 2)
plt.plot(t, intensity_pos_complex_hrv)
plt.title('片側周波数強度変化 (心拍変動あり+呼吸)')
plt.xlabel('時間 [s]')
plt.ylabel('強度')

# 周波数領域
plt.subplot(2, 2, 3)
plt.plot(freq_full_complex_hrv, amp_full_complex_hrv)
plt.axvline(x=f0, color='r', linestyle='--', label=f'平均心拍周波数 {f0} Hz')
plt.axvline(x=f1, color='b', linestyle='--', label=f'呼吸周波数 {f1} Hz')
plt.axvline(x=f0+f1, color='m', linestyle=':', label=f'和周波数 {f0+f1} Hz')
plt.axvline(x=f0-f1, color='c', linestyle=':', label=f'差周波数 {f0-f1} Hz')
plt.xlim(0, 5)
plt.title('全周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(freq_pos_complex_hrv, amp_pos_complex_hrv)
plt.axvline(x=f0, color='r', linestyle='--', label=f'平均心拍周波数 {f0} Hz')
plt.axvline(x=f1, color='b', linestyle='--', label=f'呼吸周波数 {f1} Hz')
plt.axvline(x=f0+f1, color='m', linestyle=':', label=f'和周波数 {f0+f1} Hz')
plt.axvline(x=f0-f1, color='c', linestyle=':', label=f'差周波数 {f0-f1} Hz')
plt.xlim(0, 5)
plt.title('片側周波数強度変化 (周波数領域)')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅')
plt.legend()

plt.tight_layout()
plt.savefig('complex_hrv_comparison.png')

# 3. 時間-周波数解析の比較
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plot_spectrogram(intensity_full_complex_constant, fs, '全周波数強度 (定常心拍+呼吸)')

plt.subplot(2, 2, 2)
plot_spectrogram(intensity_pos_complex_constant, fs, '片側周波数強度 (定常心拍+呼吸)')

plt.subplot(2, 2, 3)
plot_spectrogram(intensity_full_complex_hrv, fs, '全周波数強度 (心拍変動あり+呼吸)')

plt.subplot(2, 2, 4)
plot_spectrogram(intensity_pos_complex_hrv, fs, '片側周波数強度 (心拍変動あり+呼吸)')

plt.tight_layout()
plt.savefig('complex_spectrograms.png')
plt.show()
```

### 2. 実データを用いた評価

#### 2.1 データ収集プロトコル

1. 被験者の安静時データ収集（5分間）
   - 被験者は椅子に座り、安静状態を保つ
   - 同時に参照用の心拍計（例：パルスオキシメータ）でデータ収集

2. 運動後データ収集（3分間）
   - 軽い運動（その場足踏み30秒）後に測定
   - 心拍数の変化を観察

#### 2.2 データ処理と評価

```python
def process_real_data(iq_data, fs):
    """実データの処理"""
    # 全周波数処理
    intensity_full = calculate_full_spectrum_intensity(iq_data)

    # 片側周波数処理
    intensity_pos = calculate_positive_spectrum_intensity(iq_data)

    # 時間-周波数解析（短時間フーリエ変換）
    def stft_analysis(signal, fs, window_size=5, overlap=0.5):
        """短時間フーリエ変換による時間-周波数解析"""
        from scipy import signal as sg

        window_samples = int(window_size * fs)
        overlap_samples = int(window_samples * overlap)

        f, t, Zxx = sg.stft(signal, fs=fs, window='hann',
                           nperseg=window_samples,
                           noverlap=overlap_samples)
        return f, t, np.abs(Zxx)

    # 時間-周波数解析
    f_full, t_full, Zxx_full = stft_analysis(intensity_full, fs)
    f_pos, t_pos, Zxx_pos = stft_analysis(intensity_pos, fs)

    # 心拍数推定（各時間窓での最大ピーク検出）
    def estimate_heart_rate(f, Zxx, hr_range=(0.8, 2.0)):
        """周波数スペクトルから心拍数を推定"""
        hr_min_idx = np.argmin(np.abs(f - hr_range[0]))
        hr_max_idx = np.argmin(np.abs(f - hr_range[1]))

        hr_estimates = []
        for i in range(Zxx.shape[1]):
            spectrum = Zxx[hr_min_idx:hr_max_idx, i]
            peak_idx = np.argmax(spectrum) + hr_min_idx
            hr_estimates.append(f[peak_idx] * 60)  # Hz to BPM

        return np.array(hr_estimates)

    # 心拍数推定
    hr_full = estimate_heart_rate(f_full, Zxx_full)
    hr_pos = estimate_heart_rate(f_pos, Zxx_pos)

    # 心拍変動性（HRV）の評価
    def calculate_hrv_metrics(hr_estimates, t):
        """心拍変動性の評価指標を計算"""
        # 心拍間隔（RR間隔）の計算（秒単位）
        rr_intervals = 60 / hr_estimates

        # 時間領域の指標
        sdnn = np.std(rr_intervals)  # RR間隔の標準偏差
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # 連続するRR間隔の差の二乗平均平方根

        # 周波数領域の指標を計算する場合は、ここに追加

        return {
            'SDNN': sdnn,
            'RMSSD': rmssd
        }

    # 心拍変動性の評価
    hrv_metrics_full = calculate_hrv_metrics(hr_full, t_full)
    hrv_metrics_pos = calculate_hrv_metrics(hr_pos, t_pos)

    return {
        'intensity_full': intensity_full,
        'intensity_pos': intensity_pos,
        'f_full': f_full,
        't_full': t_full,
        'Zxx_full': Zxx_full,
        'f_pos': f_pos,
        't_pos': t_pos,
        'Zxx_pos': Zxx_pos,
        'hr_full': hr_full,
        'hr_pos': hr_pos,
        'hrv_metrics_full': hrv_metrics_full,
        'hrv_metrics_pos': hrv_metrics_pos
    }

# 評価指標の計算
def calculate_metrics(estimated_hr, reference_hr):
    """心拍数推定の評価指標を計算"""
    # 平均絶対誤差
    mae = np.mean(np.abs(estimated_hr - reference_hr))

    # 二乗平均平方根誤差
    rmse = np.sqrt(np.mean((estimated_hr - reference_hr)**2))

    # 相関係数
    corr = np.corrcoef(estimated_hr, reference_hr)[0, 1]

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': corr
    }
```

### 3. SNR（信号対雑音比）評価

```python
def calculate_snr(signal, fs, target_freq, bandwidth=0.1):
    """信号対雑音比の計算"""
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

# 心拍変動を考慮したSNR計算
def calculate_snr_with_hrv(signal, fs, target_freq, bandwidth_factor=1.5):
    """心拍変動を考慮したSNR計算
    心拍変動がある場合、帯域幅を広げて計算する
    """
    # 帯域幅を心拍周波数に応じて設定
    # 心拍変動が大きいほど帯域幅を広げる
    bandwidth = target_freq * 0.2 * bandwidth_factor  # 心拍周波数の20%程度を基本帯域幅とする

    return calculate_snr(signal, fs, target_freq, bandwidth)

# 全周波数と片側周波数のSNR比較
def compare_snr(iq_data, fs, heart_rate_hz, hrv_factor=1.0):
    """全周波数と片側周波数のSNR比較"""
    # 全周波数処理
    intensity_full = calculate_full_spectrum_intensity(iq_data)

    # 片側周波数処理
    intensity_pos = calculate_positive_spectrum_intensity(iq_data)

    # 心拍周波数でのSNR計算
    snr_full = calculate_snr_with_hrv(intensity_full, fs, heart_rate_hz, hrv_factor)
    snr_pos = calculate_snr_with_hrv(intensity_pos, fs, heart_rate_hz, hrv_factor)

    # 2倍周波数でのSNR計算
    snr_full_2x = calculate_snr_with_hrv(intensity_full, fs, 2*heart_rate_hz, hrv_factor)
    snr_pos_2x = calculate_snr_with_hrv(intensity_pos, fs, 2*heart_rate_hz, hrv_factor)

    return {
        'SNR_full_1x': snr_full,
        'SNR_pos_1x': snr_pos,
        'SNR_full_2x': snr_full_2x,
        'SNR_pos_2x': snr_pos_2x
    }
```

## 評価指標

以下の指標を用いて全周波数と片側周波数の有効性を比較評価します：

1. **心拍数推定精度**
   - 平均絶対誤差 (MAE)
   - 二乗平均平方根誤差 (RMSE)
   - 参照値との相関係数

2. **信号対雑音比 (SNR)**
   - 心拍基本周波数におけるSNR
   - 心拍2倍周波数におけるSNR

3. **計算効率**
   - 処理時間
   - メモリ使用量

4. **周波数ピークの明瞭さ**
   - ピーク対平均比 (PAR)
   - ピーク対隣接比 (PAR-N)

## 実験計画

1. **シミュレーションデータによる評価**
   - 単一周波数（心拍のみ）
     - 定常心拍
     - 心拍変動あり（時間変化する心拍数）
   - 複合周波数（心拍と呼吸）
     - 定常心拍と呼吸
     - 心拍変動ありと呼吸
   - ノイズ耐性評価（SNRを変化させた場合）
   - 心拍変動の大きさによる影響評価（変動の標準偏差を変化）

2. **実データによる評価**
   - 安静時データ（心拍変動が小さい状態）
   - 運動後データ（心拍変動が大きい状態）
   - 異なる被験者間での比較
   - 心拍変動性（HRV）の違いによる比較

3. **パラメータ感度分析**
   - 心拍振幅の変化に対する感度
   - 呼吸振幅の変化に対する感度
   - 心拍・呼吸周波数の変化に対する感度
   - 心拍変動の大きさに対する感度
   - 心拍変動の時間スケール（速い変動か遅い変動か）に対する感度

## 期待される結果と考察

理論的には以下のような結果が予想されます：

1. **全周波数処理の場合**
   - 心拍の2倍周波数（$2\omega_0$）に強いピークが現れる
   - 偶数次高調波のみが現れるため、スペクトルがクリーンになる可能性がある
   - 呼吸との相互変調成分が少なくなる可能性がある
   - 心拍変動がある場合、$2\omega_0$付近にピークが広がり、検出精度が低下する可能性がある

2. **片側周波数処理の場合**
   - 心拍の基本周波数（$\omega_0$）に直接ピークが現れる可能性がある
   - 全ての整数次高調波が現れるため、スペクトルが複雑になる可能性がある
   - 呼吸との相互変調成分が多くなる可能性がある
   - 心拍変動がある場合、$\omega_0$付近にピークが広がるが、基本周波数を直接検出できるため、検出精度が高い可能性がある

3. **心拍変動の影響**
   - 心拍変動が小さい場合（安静時など）は、全周波数処理の方がスペクトルがクリーンで有利な可能性
   - 心拍変動が大きい場合（運動後やストレス時など）は、片側周波数処理の方が基本周波数を直接検出できるため有利な可能性
   - 心拍変動の時間スケール（速い変動か遅い変動か）によっても最適な手法が異なる可能性

最終的に、心拍数推定の精度、SNR、計算効率などの総合評価に基づいて、どちらの手法が優れているかを判断します。また、特定の条件下（例：安静時vs運動後、高SNRvs低SNR、心拍変動小vs心拍変動大）でどちらの手法が有利かも明らかにします。

心拍変動がある場合の検証結果に基づいて、実際の心拍数推定アルゴリズムにおいて、心拍変動の大きさに応じて動的に全周波数処理と片側周波数処理を切り替えるハイブリッドアプローチの有効性も考察します。