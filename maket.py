#!/usr/bin/env python
# coding: utf-8
"""
ЛАБОРАТОРНАЯ РАБОТА: МЕТОДЫ РАЗДЕЛЕНИЯ ИСТОЧНИКОВ
Цель: Реализовать PCA, GED и ICA для разделения пространственно-временных сигналов.
Заполните все блоки с пометкой # <-- и запустите код.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import FastICA
import scipy.io as sio

# ======================== НАСТРОЙКИ ========================
np.random.seed(42)
N_CHANNELS = 10
N_TIMEPOINTS = 500
TIME_AXIS = np.arange(N_TIMEPOINTS)
AMP_1, FREQ_1 = 2.0, 0.05
AMP_2, FREQ_2 = 1.0, 0.10
PLOT_LIMIT = 100
VERT_OFFSET = 2.5

# Истинные паттерны (не менять)
spatial_1 = np.sin(np.linspace(0, 2*np.pi, N_CHANNELS))
spatial_2 = np.cos(np.linspace(0, 2*np.pi, N_CHANNELS))
temporal_1 = AMP_1 * np.sin(2*np.pi*FREQ_1*TIME_AXIS)
temporal_2 = AMP_2 * np.sin(2*np.pi*FREQ_2*TIME_AXIS)

def generate_data(noise_level):
    clean = np.outer(spatial_1, temporal_1) + np.outer(spatial_2, temporal_2)
    noise = np.random.randn(N_CHANNELS, N_TIMEPOINTS) * noise_level
    return clean + noise, clean, noise

def normalize_component(weights, temporal_pc, ref_signal=None, target_amp=None):
    scale = np.max(np.abs(weights))
    w_norm, t_norm = weights/scale, temporal_pc*scale
    if ref_signal is not None:
        if np.corrcoef(t_norm, ref_signal)[0,1] < 0:
            w_norm, t_norm = -w_norm, -t_norm
    if target_amp is not None and np.max(np.abs(t_norm)) > 0:
        t_norm = t_norm / np.max(np.abs(t_norm)) * target_amp
    return w_norm, t_norm

def plot_results(noisy_data, components, spatial_weights, titles, plot_limit=100):
    n_chan = noisy_data.shape[0]
    fig, ax = plt.subplots(figsize=(8,3))
    for ch in range(min(5, n_chan)):
        ax.plot(TIME_AXIS[:plot_limit], noisy_data[ch,:plot_limit] + ch*VERT_OFFSET, lw=0.8, alpha=0.7)
    ax.set_xlabel('Время'); ax.set_ylabel('Канал'); ax.set_title('Исходные данные'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    fig, ax = plt.subplots(figsize=(6,4))
    for w, lbl in zip(spatial_weights[:3], titles[:3]):
        ax.plot(np.arange(len(w)), w, marker='o', ms=3, label=lbl)
    ax.set_xlabel('Канал'); ax.set_ylabel('Вес'); ax.set_title('Пространственные паттерны'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    fig, ax = plt.subplots(figsize=(8,4))
    for t, lbl in zip(components[:3], titles[:3]):
        ax.plot(TIME_AXIS[:plot_limit], t[:plot_limit], label=lbl, lw=1.5)
    ax.plot(TIME_AXIS[:plot_limit], temporal_1[:plot_limit], 'k--', label='Источник 1', alpha=0.7)
    ax.plot(TIME_AXIS[:plot_limit], temporal_2[:plot_limit], 'r--', label='Источник 2', alpha=0.7)
    ax.set_xlabel('Время'); ax.set_ylabel('Амплитуда'); ax.set_title('Временные компоненты'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# ======================== ЗАДАНИЕ 1: PCA ========================
print("\n=== ЗАДАНИЕ 1: PCA ===")
NOISE_1 = 0.3
noisy_1, clean_1, noise_1 = generate_data(NOISE_1)

# Центрирование
data_centered_1 = noisy_1 - np.mean(noisy_1, axis=1, keepdims=True)

# Ковариационная матрица
cov_matrix_1 = np.cov(data_centered_1)

# Собственные значения и векторы
eigenvalues_1, eigenvectors_1 = np.linalg.eigh(cov_matrix_1)

# Сортировка по убыванию
sorted_idx_1 = np.argsort(eigenvalues_1)[::-1]
eigenvalues_1 = eigenvalues_1[sorted_idx_1]
eigenvectors_1 = eigenvectors_1[:, sorted_idx_1]

# Проекция на компоненты
principal_components_1 = eigenvectors_1.T @ data_centered_1

# Визуализация
w1, t1 = normalize_component(eigenvectors_1[:,0], principal_components_1[0], temporal_1, AMP_1)
w2, t2 = normalize_component(eigenvectors_1[:,1], principal_components_1[1], temporal_2, AMP_2)
w3, t3 = normalize_component(eigenvectors_1[:,2], principal_components_1[2])
plot_results(noisy_1, [t1,t2,t3], [w1,w2,w3], ['PCA1','PCA2','PCA3'])

corr1_pca = np.abs(np.corrcoef(t1, temporal_1)[0,1])
corr2_pca = np.abs(np.corrcoef(t2, temporal_2)[0,1])
print(f"PCA: corr1 = {corr1_pca:.4f}, corr2 = {corr2_pca:.4f}")

# ======================== ЗАДАНИЕ 2: GED ========================
print("\n=== ЗАДАНИЕ 2: GED ===")
NOISE_2 = 0.8  # Здесь шум выше, PCA будет работать хуже
noisy_2, clean_2, noise_2 = generate_data(NOISE_2)

data_c_2 = noisy_2 - noisy_2.mean(axis=1, keepdims=True)
noise_c_2 = noise_2 - noise_2.mean(axis=1, keepdims=True)

# Ковариации
cov_sig_2 = np.cov(data_c_2)
cov_noise_2 = np.cov(noise_c_2)

# Регуляризация
reg = 1e-8 * np.trace(cov_noise_2) / N_CHANNELS
cov_noise_reg_2 = cov_noise_2 + reg * np.eye(N_CHANNELS)

# GED
evals_2, evecs_2 = scipy.linalg.eigh(cov_sig_2, cov_noise_reg_2)

# Сортировка
idx_ged = np.argsort(evals_2)[::-1]
evals_2 = evals_2[idx_ged]
evecs_2 = evecs_2[:, idx_ged]
comp_ts_2 = evecs_2.T @ data_c_2

# Форвард-модель (пространственный паттерн)
def compute_forward(w, C_sig, C_noise):
    denom = w.T @ C_noise @ w
    return (C_sig @ w) / denom if np.abs(denom) > 1e-12 else np.zeros_like(w)

C_sig_diff_2 = cov_sig_2 - cov_noise_2
f1 = compute_forward(evecs_2[:,0], C_sig_diff_2, cov_noise_reg_2)
f2 = compute_forward(evecs_2[:,1], C_sig_diff_2, cov_noise_reg_2)

w1g, t1g = normalize_component(f1, comp_ts_2[0], temporal_1, AMP_1)
w2g, t2g = normalize_component(f2, comp_ts_2[1], temporal_2, AMP_2)
plot_results(noisy_2, [t1g,t2g,comp_ts_2[2]], [w1g,w2g,evecs_2[:,2]], ['GED1','GED2','GED3'])

corr1_ged = np.abs(np.corrcoef(t1g, temporal_1)[0,1])
corr2_ged = np.abs(np.corrcoef(t2g, temporal_2)[0,1])
print(f"GED: corr1 = {corr1_ged:.4f}, corr2 = {corr2_ged:.4f}")

# ======================== ЗАДАНИЕ 3: ICA ========================
print("\n=== ЗАДАНИЕ 3: ICA ===")
X_ica_input = noisy_2 - noisy_2.mean(axis=1, keepdims=True)
X_ica = X_ica_input.T 

# FastICA
ica = FastICA(n_components=N_CHANNELS, random_state=42, whiten='arbitrary-variance')
sources = ica.fit_transform(X_ica).T  # Транспонируем в (компоненты, время)
spatial_maps = ica.mixing_.T

# Сортировка по энергии
energy = np.sum(spatial_maps**2, axis=1)
sorted_idx = np.argsort(energy)[::-1]
spatial_maps = spatial_maps[sorted_idx]
sources = sources[sorted_idx]

# Поиск соответствия компонентам (максимальная корреляция)
corrs1 = [np.abs(np.corrcoef(s, temporal_1)[0,1]) for s in sources]
corrs2 = [np.abs(np.corrcoef(s, temporal_2)[0,1]) for s in sources]
idx1, idx2 = np.argmax(corrs1), np.argmax(corrs2)

t1i, w1i = normalize_component(spatial_maps[idx1], sources[idx1], temporal_1, AMP_1)
t2i, w2i = normalize_component(spatial_maps[idx2], sources[idx2], temporal_2, AMP_2)
plot_results(noisy_2, [sources[idx1], sources[idx2], sources[0]], 
             [w1i, w2i, spatial_maps[0]], [f'ICA{idx1+1}', f'ICA{idx2+1}', 'ICA-шум'])

corr1_ica = np.abs(np.corrcoef(sources[idx1], temporal_1)[0,1])
corr2_ica = np.abs(np.corrcoef(sources[idx2], temporal_2)[0,1])
print(f"ICA: corr1 = {corr1_ica:.4f}, corr2 = {corr2_ica:.4f}")

# ======================== ЗАДАНИЕ 4*: ЭЭГ ========================
# Для выполнения этого блока убедитесь, что файлы emptyEEG.mat и pytopo.py в папке
# if 'matfile' in locals():
#     print("\n=== ЗАДАНИЕ 4*: ЭЭГ (запуск моделирования) ===")
#     from pytopo import topoplotIndie
    
#     # Сигналы диполей для всех эпох
#     for trial in range(EEG['trials']):
#         phi1, phi2 = np.random.rand(2) * 2 * np.pi
#         dip_act = np.zeros((n_dip, EEG['pnts']))
#         dip_act[DIPOLE_LOC1, tidx:] = AMP1 * np.sin(omega1 + phi1)
#         dip_act[DIPOLE_LOC2, tidx:] = AMP2 * np.sin(omega2 + phi2)
        
#         # Проекция + шум (сильный)
#         noise = np.random.randn(n_chan, EEG['pnts']) * 5
#         EEG['data'][:, :, trial] = (Gain @ dip_act) + noise
#         true_dipoles[trial, :, :] = dip_act.T

#     # GED на ЭЭГ данных
#     # Матрица R (пре-стимул), Матрица S (пост-стимул)
#     R = np.zeros((n_chan, n_chan))
#     S = np.zeros((n_chan, n_chan))
#     for tr in range(EEG['trials']):
#         R += np.cov(EEG['data'][:, :tidx, tr])
#         S += np.cov(EEG['data'][:, tidx:, tr])
    
#     evals_eeg, evecs_eeg = scipy.linalg.eigh(S, R + REG_PARAM * np.eye(n_chan))
#     # Визуализация топографии первой компоненты
#     fig, ax = plt.subplots()
#     topoplotIndie(evecs_eeg[:, -1], EEG['chanlocs'], ax=ax, title='GED Topo (Source 1)')
#     plt.show()