#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节假日/战役冲击可视化分析
==========================
Fig 1 — 战役时间线甘特图 (横轴=日期, 纵轴=战役序号, 蓝→红烈度, 全标注)
Fig 2 — ARIMA+GPR 预测 + 战役叠加 (左轴=损失, 右轴=战役序号, 矩形=战役区间)
       同时表达: 某区间发生了什么战役 & 伤亡人数是多少
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from statsmodels.tsa.arima.model import ARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 配色 — shulitongji 体系 + 蓝→红烈度
# ══════════════════════════════════════════════════════════════
ROYAL_BLUE  = '#4169E1'
DARK_ORANGE = '#FF8C00'
CRIMSON     = '#DC143C'
GRAY_LINE   = '#808080'
DODGER_BLUE = '#1E90FF'

BLUE_TO_RED = LinearSegmentedColormap.from_list('BlueRed', [
    '#D6E4F0', '#6BAED6', '#3182BD', '#C9898F', '#E05554', '#B71C1C'
])

plt.rcParams.update({
    'font.size': 9.5, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 200, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
})

# ══════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("Loading data...")

tank = pd.read_csv('tank.csv')
tank.columns = ['date', 'day', 'direction', 'tank']
tank['date'] = pd.to_datetime(tank['date'])
tank = tank.set_index('date').sort_index()
tank['tank'] = tank['tank'].apply(
    lambda x: float(str(x).replace(',', '').replace('"', '').strip()) if isinstance(x, str) else float(x)
)
tank.loc[tank['tank'] < 0, 'tank'] = 0
y = tank['tank'].fillna(0)
n_samples = len(y)

battles = pd.read_csv('russia_ukraine_battles.csv')
battles['start_date'] = pd.to_datetime(battles['start_date'])
battles['end_date'] = battles.apply(
    lambda row: row['start_date'] + pd.Timedelta(days=int(row['duration_days'])), axis=1
)
b_sorted = battles.sort_values('start_date').reset_index(drop=True)
b_sorted['battle_idx'] = range(1, len(b_sorted) + 1)
n_battles = len(b_sorted)

tank_ma30 = tank['tank'].rolling(30).mean()

print(f"Tank: {n_samples} days | Battles: {n_battles}")
print(f"Date: {tank.index[0].date()} → {tank.index[-1].date()}")

# ══════════════════════════════════════════════════════════════
# Fig 1 — 战役时间线甘特图
#   横轴 = 日期, 纵轴 = 战役序号, 每场战役 = 彩色矩形, 全标注
# ══════════════════════════════════════════════════════════════
print("\n[Fig 1] Battle Gantt — time axis, every battle labeled...")

fig1, ax1 = plt.subplots(figsize=(22, 12))

for i, (_, row) in enumerate(b_sorted.iterrows()):
    s, e = row['start_date'], row['end_date']
    c = BLUE_TO_RED(row['daily_intensity'])
    bar = ax1.barh(i, (e - s).days, left=s, height=0.72,
                   color=c, alpha=0.82, linewidth=0)
    lbl = f"  {row['war_name']}"
    ax1.text(s + pd.Timedelta(days=2), i, lbl, va='center',
             fontsize=7.2, fontweight='bold',
             color='white' if row['daily_intensity'] > 0.45 else '#333333',
             alpha=0.9)

# 色标
sm1 = plt.cm.ScalarMappable(cmap=BLUE_TO_RED, norm=plt.Normalize(0, 1))
sm1.set_array([])
cbar1 = fig1.colorbar(sm1, ax=ax1, shrink=0.5, aspect=35, pad=0.02)
cbar1.set_label('Intensity', fontsize=10)
cbar1.outline.set_visible(False)

ax1.set_yticks(range(n_battles))
ax1.set_yticklabels([f'#{i}' for i in b_sorted['battle_idx']], fontsize=6.8)
ax1.invert_yaxis()
ax1.set_xlabel('Date', fontsize=11)
ax1.set_title('Russo-Ukrainian War — Major Battles Timeline\n'
              '(Color = Intensity: Blue = Low, Red = High)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.2, color='#cccccc', linewidth=0.35, axis='x')
for s in ['top', 'right', 'left']: ax1.spines[s].set_visible(False)
ax1.spines['bottom'].set_color('#cccccc')

fig1.tight_layout(pad=1.5)
fig1.savefig('fig1_battle_gantt.png', dpi=250, facecolor='white', edgecolor='none')
print("  → fig1_battle_gantt.png")

# ══════════════════════════════════════════════════════════════
# Fig 2 — ARIMA+GPR 预测 + 战役矩形叠加
#   左轴 = 坦克损失 (Actual + ARIMA fit + Hybrid + Forecast)
#   右轴 = 战役序号 (每场战役 = 彩色矩形 = 该区间发生了什么)
# ══════════════════════════════════════════════════════════════
print("\n[Fig 2] ARIMA+GPR forecast + battle overlay...")

# --- ARIMA 基线 (同 drone_arima_gpr.py) ---
print("  Fitting ARIMA(0,2,3) baseline...")
arima_order = (0, 2, 3)
arima_model = ARIMA(y, order=arima_order)
arima_fitted = arima_model.fit()
arima_hist = arima_fitted.predict(start=y.index[0], end=y.index[-1], dynamic=False)
arima_hist.iloc[:arima_order[1]] = y.iloc[:arima_order[1]]

future_steps = 30
arima_forecast = arima_fitted.forecast(steps=future_steps)

# --- 残差 + GPR ---
print("  Fitting GPR on residuals...")
residuals = y - arima_hist
X_train = np.arange(n_samples).reshape(-1, 1) / n_samples
X_future = np.arange(n_samples, n_samples + future_steps).reshape(-1, 1) / n_samples

kernel = (ConstantKernel(1.0, (1e-3, 1e3))
          * Matern(length_scale=0.1, nu=1.5)
          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2)))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
gpr.fit(X_train, residuals.values)

gpr_hist, _ = gpr.predict(X_train, return_std=True)
gpr_future, gpr_future_std = gpr.predict(X_future, return_std=True)

# --- 混合 ---
hybrid_hist = arima_hist + gpr_hist
hybrid_future = arima_forecast + gpr_future
lower = np.maximum(0, hybrid_future - 2.326 * gpr_future_std)  # 99% CI
upper = hybrid_future + 2.326 * gpr_future_std

future_idx = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=future_steps)

# --- 绘图 ---
print("  Drawing...")
fig2 = plt.figure(figsize=(22, 12))
ax2 = fig2.add_subplot(111)

# ---- 左轴: 坦克损失 ----
# Actual data
ax2.plot(y.index, y, color=ROYAL_BLUE, alpha=0.4, linewidth=0.7, label='Actual Data', zorder=2)
# ARIMA 基线
ax2.plot(y.index, arima_hist, color=DODGER_BLUE, alpha=0.55, linestyle='--',
         linewidth=1.0, label='ARIMA Linear Baseline', zorder=2)
# 混合拟合
ax2.plot(y.index, hybrid_hist, color=DARK_ORANGE, linewidth=1.3,
         label='Hybrid Fit (ARIMA+GPR)', zorder=3)
# 未来预测
ax2.plot(future_idx, hybrid_future, color=CRIMSON, linewidth=2.2, linestyle='--',
         label='Hybrid Forecast', zorder=4)
ax2.fill_between(future_idx, lower, upper, color=CRIMSON, alpha=0.15,
                 label='99% Confidence Interval', zorder=3)
# 预测起点
ax2.axvline(x=y.index[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)

ax2.set_ylabel('Daily Tank Loss', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)

# ---- 右轴: 战役矩形 ----
ax2_r = ax2.twinx()
ax2_r.set_ylim(-1.0, n_battles + 0.3)
ax2_r.invert_yaxis()
ax2_r.set_ylabel('Battle Index', fontsize=11)
ax2_r.set_yticks(range(1, n_battles + 1))
ax2_r.set_yticklabels([f'#{i}' for i in b_sorted['battle_idx']], fontsize=6.5)

# 画战役矩形 (高=0.55, 紧凑不占满屏) + 标注名称
for i, (_, row) in enumerate(b_sorted.iterrows()):
    s, e = row['start_date'], row['end_date']
    c = BLUE_TO_RED(row['daily_intensity'])
    y_center = i + 1  # battle index
    ax2_r.barh(y_center, (e - s).days, left=s, height=0.55,
               color=c, alpha=0.78, linewidth=0, zorder=1)
    # 战役名称 (标注在矩形右侧)
    ax2_r.text(e + pd.Timedelta(days=3), y_center,
               row['war_name'], va='center', fontsize=6.5,
               fontweight='bold' if row['daily_intensity'] > 0.3 else 'normal',
               color='#333333', alpha=0.82, zorder=5)

# 色标
sm2 = plt.cm.ScalarMappable(cmap=BLUE_TO_RED, norm=plt.Normalize(0, 1))
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=ax2_r, shrink=0.45, aspect=35, pad=0.04)
cbar2.set_label('Battle Intensity', fontsize=10)
cbar2.outline.set_visible(False)

# --- 修饰 ---
ax2.set_title('ARIMA+GPR Hybrid Forecast with Battle Context\n'
              '(Left axis = Tank Loss  |  Right axis = Battle Index — '
              'each colored bar = one battle period)',
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9.5, framealpha=0.9, ncol=2)
ax2.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
ax2.set_ylim(bottom=-4)
# 右侧留白给战役名称
x_extra = pd.Timedelta(days=80)
ax2.set_xlim(tank.index[0] - pd.Timedelta(days=10),
             future_idx[-1] + x_extra)

# 隐藏右边框
for s in ['top', 'right']: ax2.spines[s].set_visible(False)
for s in ['top', 'left']: ax2_r.spines[s].set_visible(False)

fig2.tight_layout(pad=1.5)
fig2.savefig('fig2_arima_gpr_battles.png', dpi=250, facecolor='white', edgecolor='none')
print("  → fig2_arima_gpr_battles.png")

# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*50}")
print(f"Battles: {n_battles} | ARIMA{arima_order} | GPR: Matern(nu=1.5)")
print(f"Output: fig1_battle_gantt.png")
print(f"        fig2_arima_gpr_battles.png")
print(f"{'─'*50}")
print("Done.")
plt.show()
