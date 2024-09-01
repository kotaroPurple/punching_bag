import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

#------------------------------------------------------------------------------
# テスト信号の生成
#------------------------------------------------------------------------------
dt = 1.0e-4
tms = 0.0
tme = 2.0
tm = np.arange(tms, tme, dt)

amp = abs(np.sin(2.0*np.pi*2.0*tm)) + np.tanh(tm)
freq = 10.0 + 10.0*tm*tm + np.cos(2.0*np.pi*3.0*tm)
sig = amp*np.sin(2.0*np.pi*freq*tm) + amp*5.0e-2*np.random.randn(len(tm))

#------------------------------------------------------------------------------
# 諸々の設定
freq_upper = 300.0 # in kHz
out_fig_path = "scalogram.png" 

#------------------------------------------------------------------------------
# 連続ウェーブレット変換
# --- n_cwt をlen(sig)よりも大きな2のべき乗の数になるように設定
n_cwt = int(2**(np.ceil(np.log2(len(sig)))))

# --- 後で使うパラメータを定義
dj = 0.125
omega0 = 6.0
s0 = 2.0*dt
J = int(np.log2(n_cwt*dt/s0)/dj)

# --- スケール
s = s0*2.0**(dj*np.arange(0, J+1, 1))

# --- n_cwt個のデータになるようにゼロパディングをして，DC成分を除く
x = np.zeros(n_cwt)
x[0:len(sig)] = sig[0:len(sig)] - np.mean(sig)

# --- omega array
omega = 2.0*np.pi*np.fft.fftfreq(n_cwt, dt)

# --- FFTを使って離散ウェーブレット変換する
X = np.fft.fft(x)
cwt = np.zeros((J+1, n_cwt), dtype=complex) # CWT array 

Hev = np.array(omega > 0.0)
for j in range(J+1):
    Psi = np.sqrt(2.0*np.pi*s[j]/dt)*np.pi**(-0.25)*np.exp(-(s[j]*omega-omega0)**2/2.0)*Hev
    cwt[j, :] = np.fft.ifft(X*np.conjugate(Psi))

s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi) 
freq_cwt = s_to_f / s
cwt = cwt[:, 0:len(sig)]

# --- cone of interference
COI = np.zeros_like(tm)
COI[0] = 0.5/dt
COI[1:len(tm)//2] = np.sqrt(2)*s_to_f/tm[1:len(tm)//2]
COI[len(tm)//2:-1] = np.sqrt(2)*s_to_f/(tm[-1]-tm[len(tm)//2:-1])
COI[-1] = 0.5/dt

#------------------------------------------------------------------------------
# 解析結果の可視化
figsize = (210/25.4, 294/25.4)
figsize = (12, 4)
dpi = 200
fig = plt.figure(figsize=figsize, dpi=dpi)

# --- 図の設定 (全体)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = 'Helvetica'

# プロット枠 (axes) の設定
ax1 = fig.add_axes([0.15, 0.55, 0.70, 0.3])
ax_sc = fig.add_axes([0.15, 0.2, 0.70, 0.30])
cx_sc = fig.add_axes([0.87, 0.2, 0.02, 0.30])

# 元データのプロット
ax1.set_xlim(tms, tme)
ax1.set_xlabel('')
ax1.tick_params(labelbottom=False)
ax1.set_ylabel('x')

ax1.plot(tm, sig, c='black')

# スペクトログラムのプロット
ax_sc.set_xlim(tms, tme)
ax_sc.set_xlabel('time (s)')
ax_sc.tick_params(labelbottom=True)
ax_sc.set_ylim(0, freq_upper)
ax_sc.set_ylabel('frequency\n(Hz)')

# カラーバーのレンジの設定
norm = mpl.colors.Normalize(vmin=np.log10(np.abs(cwt[freq_cwt < freq_upper, :])**2).min(), 
                            vmax=np.log10(np.abs(cwt[freq_cwt < freq_upper, :])**2).max())
# カラーマップ
cmap = mpl.cm.jet

# スカログラムのプロット
ax_sc.contourf(tm, freq_cwt, np.log10(np.abs(cwt)**2), 
                norm=norm, levels=256, cmap=cmap)

# Cone of Interference のプロット
ax_sc.fill_between(tm, COI, fc='w', hatch='x', alpha=0.5)
ax_sc.plot(tm, COI, '--', color='black')

# 右上の文字
ax_sc.text(0.99, 0.97, "scalogram", color='white', ha='right', va='top',
              path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                            path_effects.Normal()], 
              transform=ax_sc.transAxes)

mpl.colorbar.ColorbarBase(cx_sc, cmap=cmap,
                          norm=norm,
                          orientation="vertical",
                          label='log amplitude')

# 図を保存
# plt.savefig(out_fig_path, transparent=False)
plt.show()
