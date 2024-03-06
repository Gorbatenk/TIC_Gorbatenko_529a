import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

# Згенерований сигнал після обмеження частоти через ФНЧ.
n = 500
random = np.random.normal(0, 10, n)
Fs = 1000
t = np.arange(n) / Fs
F_max = 7
w = F_max / (Fs / 2)
b, a = signal.butter(3, w, 'low', output='ba')[:2]
filtered = signal.filtfilt(b, a, random)

# Змінні для збереження результатів
discrete_signals = []
discrete_spectrums = []
restored_signals = []
variances_diff = []
snr_ratios = []

# Цикл по різним крокам дискретизації
for Dt in [2, 4, 8, 16]:
    # Створення змінної для дискретного сигналу, сформованого з нулів
    discrete_signal = np.zeros(n)
    # Цикл для прорідження початкового сигналу
    for i in range(0, round(n / Dt)):
        # Формування дискретизованого сигналу з певним кроком
        discrete_signal[i * Dt] = filtered[i * Dt]
    # Збереження дискретизованого сигналу у список
    discrete_signals.append(list(discrete_signal))

    # Розрахунок спектру для дискретизованого сигналу
    spectrum = fft.fft(discrete_signal)
    spectrum = np.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(list(spectrum))

    # Розрахунок параметрів фільтру ФНЧ для відновлення
    F_filter = 14
    w_restore = F_filter / (Fs / 2)
    b_restore, a_restore = signal.butter(3, w_restore, 'low', output='ba')[:2]

    # Відновлення аналогового сигналу шляхом фільтрації дискретизованого
    restored_signal = signal.filtfilt(b_restore, a_restore, discrete_signal)
    restored_signals.append(list(restored_signal))

    # Розрахунок різниці між початковим та відновленим сигналами
    E1 = restored_signal - random

    # Розрахунок дисперсії різниці між початковим та відновленим сигналами
    variances_diff.append(np.var(E1))

    # Розрахунок співвідношення сигнал-шум як відношення дисперсій
    snr_ratio = np.var(random) / variances_diff[-1]
    snr_ratios.append(snr_ratio)

# Побудова графіків дискретизованих сигналів
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
fig.suptitle('Дискретизовані сигнали', fontsize=14)

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(t, discrete_signals[s], linewidth=1)
        ax[i][j].set_xlabel('Час, с', fontsize=14)
        ax[i][j].set_ylabel('Амплітуда', fontsize=14)
        s += 1

fig.savefig('./figures/discrete_signals.png', dpi=600)

# Побудова графіків спектрів
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
fig.suptitle('Спектри дискретизованих сигналів', fontsize=14)

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        freqs = fft.fftshift(fft.fftfreq(n, 1/Fs))
        ax[i][j].plot(freqs, discrete_spectrums[s], linewidth=1)
        ax[i][j].set_xlabel('Частота, Гц', fontsize=14)
        ax[i][j].set_ylabel('Амплітуда', fontsize=14)
        s += 1

fig.savefig('./figures/discrete_spectrums.png', dpi=600)

# Побудова графіків відновлених сигналів
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
fig.suptitle('Відновлені сигнали', fontsize=14)

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(t, restored_signals[s], linewidth=1)
        ax[i][j].set_xlabel('Час, с', fontsize=14)
        ax[i][j].set_ylabel('Амплітуда', fontsize=14)
        s += 1

fig.savefig('./figures/restored_signals.png', dpi=600)

# Графік залежності дисперсії різниці від кроку дискретизації
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot([2, 4, 8, 16], variances_diff, marker='o', linestyle='-', linewidth=1)  # Змінено на variances_diff
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('Дисперсія різниці', fontsize=14)
fig.savefig('./figures/variance_dependency.png', dpi=600)

# Графік залежності співвідношення сигнал-шум від кроку дискретизації
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot([2, 4, 8, 16], snr_ratios, marker='o', linestyle='-', linewidth=1)
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('Співвідношення сигнал-шум', fontsize=14)
fig.savefig('./figures/snr_ratio_dependency.png', dpi=600)

plt.show()
