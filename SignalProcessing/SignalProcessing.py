import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Згенерований сигнал після обмеження частоти через ФНЧ з попередньої практики
n = 500
Fs = 1000
t = np.arange(n) / Fs
F_max = 7
w = F_max / (Fs / 2)
b, a = signal.butter(3, w, 'low', output='ba')[:2]
random = np.random.normal(0, 10, n)
signal = signal.filtfilt(b, a, random)

# Змінні для збереження результатів
quantized_signals = []
variances = []
snr_ratios = []

# Створення figure для квантованих сигналів
fig_quantized, axes_quantized = plt.subplots(2, 2, figsize=(21 / 2.54, 21 / 2.54))
axes_quantized = axes_quantized.flatten()

# Цикл по різним рівням квантування
for idx, M in enumerate([4, 16, 64, 256]):
    # Змінні для збереження результатів в циклі
    bits = []

    # Розрахунок кроку квантування
    delta = (np.max(signal) - np.min(signal)) / (M - 1)

    # Квантування сигналу
    quantized_signal = delta * np.round(signal / delta)
    quantized_signals.append(quantized_signal)

    # Визначення квантованих рівнів
    quantized_levels = np.arange(np.min(quantized_signal), np.max(quantized_signal) + delta, delta)

    # Генерація бітових послідовностей для кодування рівнів
    quantized_bit = np.arange(0, M)
    quantized_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in quantized_bit]

    # Створення таблиці квантування
    quantized_table = np.c_[quantized_levels[:M], quantized_bit[:M]]

    # Відображення таблиці квантування
    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantized_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig(f'./figures/quantization_table_M={M}.png', dpi=600)

    # Перетворення значень сигналу на бітові послідовності
    for signal_value in quantized_signal:
        for index, value in enumerate(quantized_levels[:M]):
            if np.round(np.abs(signal_value - value), 0) == 0:
                bits.append(quantized_bit[index])
                break

    # Об'єднання бітових послідовностей в одну послідовність
    bits = [int(item) for item in list(''.join(bits))]

    # Побудова графіку бітової послідовності
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(np.arange(0, len(bits)), bits, linewidth=0.1)
    ax.set_xlabel('Відлік', fontsize=14)
    ax.set_ylabel('Біт', fontsize=14)
    ax.set_title(f'Бітова послідовність для M={M}', fontsize=14)
    fig.savefig(f'./figures/bit_sequence_M={M}.png', dpi=600)

    # Побудова графіку квантованого сигналу
    axes_quantized[idx].plot(t, quantized_signal, linewidth=1)
    axes_quantized[idx].set_xlabel('Час, с', fontsize=14)
    axes_quantized[idx].set_ylabel('Амплітуда', fontsize=14)
    axes_quantized[idx].set_title(f'Квантований сигнал для M={M}', fontsize=14)

    # Розрахунок дисперсії та співвідношення сигнал-шум
    variance = np.var(quantized_signal - signal)
    variances.append(variance)
    snr = np.var(signal) / variance
    snr_ratios.append(snr)

# Зберігання графіку квантованих сигналів
fig_quantized.tight_layout()
fig_quantized.savefig('./figures/quantized_signals.png', dpi=600)

# Побудова графіку залежності дисперсії від рівнів квантування
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], variances, marker='o', linestyle='-', linewidth=1)
ax.set_xlabel('Рівні квантування', fontsize=14)
ax.set_ylabel('Дисперсія', fontsize=14)
ax.set_title('Залежність дисперсії від рівнів квантування', fontsize=14)
fig.savefig('./figures/dispersion_dependency.png', dpi=600)

# Побудова графіку залежності співвідношення сигнал-шум від рівнів квантування
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], snr_ratios, marker='o', linestyle='-', linewidth=1)
ax.set_xlabel('Рівні квантування', fontsize=14)
ax.set_ylabel('Співвідношення сигнал-шум', fontsize=14)
ax.set_title('Залежність співвідношення сигнал-шум від рівнів квантування', fontsize=14)
fig.savefig('./figures/snr_dependency.png', dpi=600)

plt.show()
