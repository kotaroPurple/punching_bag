
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.special import jn
from numpy.typing import NDArray


LIGHT_SPEED = 3.e8
SENSOR_FREQ = 24.e9
WAVE_NUMBER = 2 * np.pi / (LIGHT_SPEED / SENSOR_FREQ)


def extract_frequency_info(data: NDArray, fs: int) -> tuple[NDArray, NDArray]:
    fft_value = np.fft.fftshift(np.fft.fft(data * np.hanning(len(data))))
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(fft_value), d=1/fs))
    fft_abs = np.abs(fft_value) / len(data)
    return fft_abs, fft_freq


def generate_iq_wave(
        times: NDArray, init_phase: float, delayed_phase: float,
        displacement: float, omega: float, wave_number: float) -> NDArray[np.complex128]:
    # init
    init_wave = np.exp(1.j * init_phase)
    # phase
    phase = 2 * wave_number * displacement * np.sin(omega * times - delayed_phase)
    return init_wave * np.exp(1.j * phase)


def generate_iq_wave_from_multi_objects(
        times: NDArray, init_phases: list[float], delayed_phases: list[float],
        displacements: list[float], omegas: list[float], wave_number: float) \
            -> NDArray[np.complex128]:
    # main
    iq_wave = None
    for init_phase, delayed_phase, displacement, omega in zip(
            init_phases, delayed_phases, displacements, omegas):
        tmp_wave = generate_iq_wave(
            times, init_phase, delayed_phase, displacement, omega, wave_number)
        if iq_wave is None:
            iq_wave = tmp_wave
        else:
            iq_wave = iq_wave * tmp_wave
    # error
    if iq_wave is None:
        raise ValueError()
    return iq_wave


def apply_bandpass_filter(
        data: NDArray, fs: int, f_low: float, f_high: float, order: int = 4, zero_phase: bool = True) -> NDArray:
    nyquist = fs / 2.0
    cutoffs = (f_low / nyquist, f_high / nyquist)
    sos = signal.butter(order, cutoffs, btype='band', output='sos')
    if zero_phase:
        filtered = signal.sosfiltfilt(sos, data)
    else:
        filtered = signal.sosfilt(sos, data)
    return filtered  # type: ignore


def extract_side_band_frequency(
        data: NDArray, fs: int, f_low: float, f_high: float, order: int = 4, zero_phase: bool = True) -> NDArray:
    # 時刻軸の生成
    t = np.arange(len(data)) / fs

    # 帯域の中心周波数と半帯域幅の計算
    fc = (f_low + f_high) / 2.0

    # (1) 搬送波による周波数シフト：対象帯域をベースバンドに移動
    shifted = data * np.exp(-1j * 2 * np.pi * fc * t)

    # (2) 低域通過 IIR フィルタの設計：カットオフ周波数を半帯域幅に設定
    sos = _make_filter(fs, f_low, f_high, order)

    # フィルタ適用
    if zero_phase:
        filtered = signal.sosfiltfilt(sos, shifted)
    else:
        filtered = signal.sosfilt(sos, shifted)

    # (3) 逆搬送波で元の周波数位置に戻す
    result = filtered * np.exp(1j * 2 * np.pi * fc * t)
    return result  # type: ignore


def sideband_freqz(fs: int, f_low: float, f_high: float, order: int):
    sos = _make_filter(fs, f_low, f_high, order)
    w, h = signal.sosfreqz(sos, worN=2000, fs=fs)
    w2 = np.r_[-w[::-1][:-1], w]
    h2 = np.r_[h[::-1][:-1], h]  # type: ignore
    w2 = w2 + (f_low + f_high) / 2.0
    return w2, h2


def _make_filter(fs: int, f_low: float, f_high: float, order: int):
    nyquist = fs / 2.0
    bw_half = (f_high - f_low) / 2.0
    norm_cutoff = bw_half / nyquist  # 正規化カットオフ周波数
    sos = signal.butter(order, norm_cutoff, btype='lowpass', output='sos')
    return sos


def generate_time(start_: float, end_: float, fs: int) -> NDArray:
    time_step = 1 / fs
    times = np.arange(start_, end_, time_step)
    return times


def apply_highpass_filter(
        data: NDArray, cutoff_freq: float, fs: int, order: int = 4, zero_phase: bool = False) -> tuple[NDArray, NDArray]:
    # 正規化カットオフ周波数 (ナイキスト周波数で正規化)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    if not 0 < normal_cutoff < 1:
        raise ValueError("カットオフ周波数は 0 < cutoff_freq < Nyquist frequency である必要があります。")

    # SOS 形式のハイパスバターワースフィルタを設計
    sos = signal.butter(N=order, Wn=normal_cutoff, btype='high', analog=False, output='sos')

    # フィルタリングの種類に応じて適用
    if zero_phase:
        filtered_signal = signal.sosfiltfilt(sos, data)
    else:
        # 因果的フィルタリング（リアルタイム処理向き）
        filtered_signal = signal.sosfilt(sos, data)
    return filtered_signal, sos  # type: ignore


def make_positive_filter(
        fs: int, window_time: float, freq: float, amp: float, wave_number: float, remove_dc: bool) -> NDArray:
    # 奇数幅のフィルタ作成
    window_size = int(fs * window_time)
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    # np.exp(1j.k.a.sin(2pi.ft)), 2pi.ft ~ pi/2 付近とする
    times = np.arange(window_size) / fs - window_time / 2
    _filter = np.exp(1j * 2 * wave_number * amp * np.cos(2 * np.pi * freq * times))
    if remove_dc:
        _filter = _filter - np.mean(_filter)
    _filter = np.conjugate(_filter[::-1])
    return _filter


def main():
    # objects
    init_phases = [0., 0. * (2 * np.pi)]  # 物体までの距離依存 (同物体であれば同じ数値のはず)
    delayed_phases = [0., 0. * (2 * np.pi)]  # それぞれの位相ズレ
    # displacements = [0.001_0, 0.000_1]  # 振幅 [m]
    displacements = [0.000_0, 0.000_1]  # 振幅 [m]
    _frequencies = [0.2, 1.]  # [Hz]
    omegas = [2 * np.pi * f for f in _frequencies]
    # iq wave
    start_time = 0.
    end_time = 80.
    fs = 500
    times = generate_time(start_time, end_time, fs)
    iq_wave = generate_iq_wave_from_multi_objects(
        times, init_phases, delayed_phases, displacements, omegas, WAVE_NUMBER)

    # DC Cut
    dc_cutoff = 0.1
    iq_wave, _ = apply_highpass_filter(iq_wave, dc_cutoff, fs, order=4, zero_phase=False)
    iq_wave = iq_wave[int(60 * fs):]
    times = times[int(60 * fs):]

    # filter
    f_low = 3.
    f_high = 10.
    zero_filtered = apply_bandpass_filter(iq_wave, fs, f_low, f_high, order=4, zero_phase=True)
    side_filtered = apply_bandpass_filter(iq_wave, fs, f_low, f_high, order=4, zero_phase=False)
    one_zero_filtered = extract_side_band_frequency(iq_wave, fs, f_low, f_high, order=4, zero_phase=True)
    one_side_filtered = extract_side_band_frequency(iq_wave, fs, f_low, f_high, order=4, zero_phase=False)
    # one_side_filtered = extract_side_band_frequency(one_side_filtered, fs, f_low, f_high, order=4, zero_phase=False)

    angle_zero = np.unwrap(np.angle(zero_filtered))
    angle_side = np.unwrap(np.angle(side_filtered))
    angle_one_zero = np.unwrap(np.angle(one_zero_filtered))
    angle_one_side = np.unwrap(np.angle(one_side_filtered))

    # filter
    freq_filter = make_positive_filter(fs, 0.1, _frequencies[1], displacements[1], WAVE_NUMBER, True)
    freq_filtered = np.convolve(iq_wave, freq_filter, mode='same')

    # cepstrum
    # 0除算を防ぐために小さな値を加えます
    fft_abs, fft_freq = extract_frequency_info(iq_wave.real, fs)
    # valid_indices = fft_freq > 0.
    # fft_abs_p = fft_abs[valid_indices]
    # fft_freq_p = fft_freq[valid_indices]
    log_magX = np.log(fft_abs + np.finfo(float).eps)

    # 5. ケプストラムの計算（逆 FFT を実施）
    cepstrum = np.fft.ifft(log_magX).real
    que_frency = np.arange(len(cepstrum)) / fs
    cepstrum = cepstrum[:len(cepstrum)//2]
    que_frency = que_frency[:len(que_frency)//2]

    # plot
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(times, iq_wave.real, alpha=0.5)
    plt.plot(times, iq_wave.imag, alpha=0.5)

    ax = plt.subplot(3, 2, 3)
    ax2 = ax.twinx()
    ax.plot(times, side_filtered.real, c='C0', alpha=0.5, label='side I')
    ax.plot(times, side_filtered.imag, c='C1', alpha=0.5, label='side Q')
    ax2.plot(times, angle_side, c='black', alpha=0.5, label='angle')

    ax = plt.subplot(3, 2, 5)
    ax2 = ax.twinx()
    ax.plot(times, zero_filtered.real, c='blue', alpha=0.5, label='zero I')
    ax.plot(times, zero_filtered.imag, c='red', alpha=0.5, label='zero Q')
    ax2.plot(times, angle_zero, c='black', alpha=0.5, label='angle')

    ax = plt.subplot(3, 2, 2)
    ax2 = ax.twinx()
    ax.plot(times, one_side_filtered.real, c='C0', alpha=0.5, label='+ side I')
    ax.plot(times, one_side_filtered.imag, c='C1', alpha=0.5, label='+ side Q')
    ax.plot(times, np.abs(one_side_filtered), c='green', alpha=0.5, label='amp')
    ax2.plot(times, angle_one_side, c='black', alpha=0.5, label='angle')

    ax = plt.subplot(3, 2, 4)
    ax2 = ax.twinx()
    ax.plot(times, one_zero_filtered.real, c='blue', alpha=0.5, label='+ zero I')
    ax.plot(times, one_zero_filtered.imag, c='red', alpha=0.5, label='+ zero Q')
    ax.plot(times, np.abs(one_zero_filtered), c='green', alpha=0.5, label='amp')
    ax2.plot(times, angle_one_zero, c='black', alpha=0.5, label='angle')

    # ax = plt.subplot(3, 2, 6)
    # ax.plot(que_frency, cepstrum, c='blue', alpha=0.5, label='cepstrum')

    # ax = plt.subplot(3, 2, 6)
    # ax.plot(fft_freq, fft_abs, c='blue', alpha=0.5, label='fft')
    # ax.set_xlim(-10., 10.)
    # ax.set_yscale('log')

    ax = plt.subplot(3, 2, 6)
    ax.plot(times, np.abs(freq_filtered), c='blue', alpha=0.5, label='matched')
    ax2 = ax.twinx()
    ax2.plot(times, np.unwrap(np.angle(freq_filtered)), c='red', alpha=0.5, label='angle')
    ax.legend()
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()
