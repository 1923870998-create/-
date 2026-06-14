#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone ARIMAX — 无人机日损失预测 (drone dataset)
================================================
基于 ARIMA BIC 框架, 升级为 ARIMAX:
  - 外生变量: 逐日战役强度 (重叠战役取 max, 确保信息完整 + 不溢出)
  - BIC 网格搜索 (p,q) + 战役强度 exog
  - 单情景预测: 冲突持续 (I=last_I)

Figure 1: ARIMAX 回归曲线 (纯时序视角, 无战役叠加)
Figure 2: 战役强度-ARIMAX 重叠图 (双轴叠加, 展示外生变量与预测的时空对齐)

视觉效果: 统一风格
  - 图1: ARIMAX 纯回归视角 → 单轴, 无置信区间(不含GPR)
  - 图2: 外生战役变量 → 双轴叠加
  - 不含 I=0 情景; 不含置信区间 (无GPR)

数据:
  - 训练集: drone.csv (2022-02-25 ~ 2026-05-31, 1557天)
  - 战役数据: russia_ukraine_battles.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 配色体系 — 柔和红方案 (统一风格)
# ==============================================================
ROYAL_BLUE  = '#4169E1'
DARK_ORANGE = '#FF8C00'
CRIMSON     = '#DC143C'
DODGER_BLUE = '#1E90FF'
GRAY_LINE   = '#808080'
SOFT_GREEN  = '#66BB6A'

BLUE_TO_RED = LinearSegmentedColormap.from_list('BlueRed', [
    '#D6E4F0', '#6BAED6', '#3182BD', '#C9898F', '#E05554', '#B71C1C'
])

plt.rcParams.update({
    'font.size': 9.5, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 200, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
})

# ==============================================================
# 1. 数据加载与预处理
# ==============================================================
print("=" * 70)
print("ARIMAX — Drone Daily Loss Prediction with Battle Intensity Exog")
print("=" * 70)

print("\n[1/5] Loading data...")

# --- drone 数据 ---
drone = pd.read_csv('drone.csv')
drone.columns = ['date', 'day', 'direction', 'drone']
drone['date'] = pd.to_datetime(drone['date'])
drone = drone.set_index('date').sort_index()
drone['drone'] = drone['drone'].apply(
    lambda x: float(str(x).replace(',', '').replace('"', '').strip())
    if isinstance(x, str) else float(x)
)
drone.loc[drone['drone'] < 0, 'drone'] = 0
y = drone['drone'].fillna(0)
n = len(y)
dates = y.index
print(f"  Drone data: {n} days [{dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')}]")
print(f"  Mean={y.mean():.1f}  Std={y.std():.1f}  Max={y.max():.0f}  Min={y.min():.0f}")

# --- 战役数据 ---
battles = pd.read_csv('russia_ukraine_battles.csv')
battles['start_date'] = pd.to_datetime(battles['start_date'])
battles['end_date'] = battles.apply(
    lambda r: r['start_date'] + pd.Timedelta(days=int(r['duration_days'])), axis=1
)
b_by_start = battles.sort_values('start_date').reset_index(drop=True)
nB = len(b_by_start)
print(f"  Battles: {nB} campaigns loaded")

# ==============================================================
# 2. 构建逐日战役强度序列 — 重叠求和
# ==============================================================
print("\n[2/5] Building daily battle intensity (sum of overlapping)...")

battle_intensity = pd.Series(0.0, index=dates)
for _, row in battles.iterrows():
    mask = (dates >= row['start_date']) & (dates <= row['end_date'])
    if mask.sum() > 0:
        # 重叠战役取和 — 多战役并行时总压力更大
        battle_intensity.loc[mask] += row['daily_intensity']

# Min-Max 归一化 → [0, 1]
bi_min, bi_max = battle_intensity.min(), battle_intensity.max()
battle_intensity = (battle_intensity - bi_min) / (bi_max - bi_min + 1e-8)

exog_train = battle_intensity.values.reshape(-1, 1)
active_days = (battle_intensity > 0).sum()
avg_I = battle_intensity[battle_intensity > 0].mean()
last_I = battle_intensity.iloc[-1]
print(f"  Active battle days: {active_days}/{n}")
print(f"  Mean intensity (active days): {avg_I:.4f}")
print(f"  Current intensity (last day): {last_I:.4f}")

# --- 双次滚动均值平滑战役烈度 ---
battle_I_smooth = battle_intensity.rolling(window=21, center=True, min_periods=1).mean()
battle_I_fitted = battle_I_smooth.rolling(window=11, center=True, min_periods=1).mean()
print(f"  Battle intensity dual-rolling-mean smoothed")

# ==============================================================
# 3. ADF 平稳性检验 -> d
# ==============================================================
print("\n[3/5] ADF stationarity test...")

y_diff = y.diff().dropna()
adf_result = adfuller(y_diff)
print(f"  ADF statistic: {adf_result[0]:.4f}")
print(f"  p-value:       {adf_result[1]:.4e}")
d = 2  # 固定二阶差分
print(f"  ADF p={adf_result[1]:.2e} (ref), fixed d={d}")

# ==============================================================
# 4. ARIMAX — BIC 网格搜索 + 战役强度外生变量
# ==============================================================
print("\n[4/5] ARIMAX BIC grid search (p,q) in [0,5]x[0,5] with exog...")

best_bic = float("inf")
best_order = None
best_model = None
beta_battle = 0.0

for p, q in itertools.product(range(0, 6), range(0, 6)):
    try:
        model = SARIMAX(
            y, exog=exog_train,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        if fitted.bic < best_bic:
            best_bic = fitted.bic
            best_order = (p, d, q)
            best_model = fitted
    except Exception:
        continue

if best_model is None:
    best_order = (2, 1, 2)
    best_model = SARIMAX(
        y, exog=exog_train, order=best_order,
        enforce_stationarity=False
    ).fit(disp=False)

# 提取战役效应系数
for i, nm in enumerate(best_model.param_names):
    if 'x1' in nm.lower():
        beta_battle = best_model.params.iloc[i]
        break

print(f"  Best ARIMAX{best_order}  BIC={best_bic:.0f}  beta_battle={beta_battle:.3f}")
print(f"  -> Each 0.1 intensity adds ~{beta_battle*0.1:.1f} drones/day to baseline")

# 历史拟合
arimax_fit = pd.Series(
    best_model.predict(start=0, end=n - 1, exog=exog_train),
    index=y.index
)
arimax_residuals = y - arimax_fit
rmse_arimax = np.sqrt(np.mean(arimax_residuals ** 2))
sigma_resid = np.std(arimax_residuals)
print(f"  RMSE={rmse_arimax:.2f}  Residual std={sigma_resid:.2f}")

# 未来预测 (30天, 单情景: I=last_I, 冲突持续)
future_steps = 30
future_idx = pd.date_range(
    start=y.index[-1] + pd.Timedelta(days=1), periods=future_steps
)

# 单情景: I=last_I (冲突持续)
exog_fut = np.full((future_steps, 1), last_I)
fut = best_model.get_forecast(steps=future_steps, exog=exog_fut).predicted_mean

print(f"  Forecast (I={last_I:.2f}): mean = {fut.mean():.1f}")

# ==============================================================
# 5. 可视化 — 统一风格
# ==============================================================
print("\n[5/5] Drawing figures...")

# ─────────────────────────────────────────────────────────────
# Figure 1: ARIMAX 纯回归曲线 (无战役叠加, 无置信区间)
# ─────────────────────────────────────────────────────────────
print("  -> Figure 1: ARIMAX Regression (no battle overlay)")
fig1, ax1 = plt.subplots(figsize=(22, 12))

# 实际数据 (背景)
ax1.plot(dates, y,
         color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Drone Loss', zorder=2)

# ARIMAX 历史拟合
ax1.plot(dates, arimax_fit,
         color=DARK_ORANGE, linewidth=1.3,
         label=f'ARIMAX{best_order} Fit', zorder=3)

# 未来预测 (单情景: I=last_I)
ax1.plot(future_idx, fut,
         color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)

# 分割线
ax1.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':',
            linewidth=1.0, alpha=0.6)

# 装饰
ax1.set_ylabel('Daily Drone Loss', fontsize=12)
ax1.set_title(
    f'ARIMAX{best_order} + Battle Intensity  |  Drone Daily Loss  |  '
    f'beta_battle={beta_battle:.3f}  BIC={best_bic:.0f}  RMSE={rmse_arimax:.2f}  '
    f'Forecast={future_steps}d',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax1.set_ylim(bottom=-20)
for s in ['top', 'right']:
    ax1.spines[s].set_visible(False)

fig1.tight_layout(pad=2)
fig1.savefig('drone_ARIMAX_Regression.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_ARIMAX_Regression.png saved")

# ─────────────────────────────────────────────────────────────
# Figure 2: 战役强度-ARIMAX 重叠图 (双轴叠加)
# ─────────────────────────────────────────────────────────────
print("  -> Figure 2: Battle Intensity + ARIMAX Overlay")
fig2 = plt.figure(figsize=(24, 14))
gs2 = fig2.add_gridspec(2, 1, height_ratios=[2.8, 1], hspace=0.3)

# --- 上图: ARIMAX 预测 + 战役矩形 ---
ax2 = fig2.add_subplot(gs2[0])

# 左轴: 预测内容
ax2.plot(dates, y,
         color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Drone Loss', zorder=2)
ax2.plot(dates, arimax_fit,
         color=DARK_ORANGE, linewidth=1.3,
         label=f'ARIMAX{best_order} Fit', zorder=3)

# 单情景: I=last_I
ax2.plot(future_idx, fut,
         color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)

# 分割线
ax2.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':',
            linewidth=1.0, alpha=0.6)

# 左轴装饰
ax2.set_ylabel('Daily Drone Loss', fontsize=12)
ax2.set_title(
    f'ARIMAX{best_order} + Battle Intensity  |  Left=Loss  Right=Intensity  |  '
    f'beta_battle={beta_battle:.3f}  RMSE={rmse_arimax:.2f}  BIC={best_bic:.0f}',
    fontsize=13, fontweight='bold'
)
ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax2.set_ylim(bottom=-20)
for s in ['top', 'right']:
    ax2.spines[s].set_visible(False)

# 右轴: Smoothed 拟合战役烈度线 (柔和绿色)
ax2_r = ax2.twinx()
ax2_r.set_ylabel('Battle Intensity (Smoothed)', fontsize=11, color=SOFT_GREEN)
ax2_r.plot(dates, battle_I_fitted.values,
           color=SOFT_GREEN, linewidth=1.2, alpha=0.9, zorder=5)
ax2_r.set_ylim(0, 1.05)
ax2_r.tick_params(axis='y', colors=SOFT_GREEN)
for s in ['top', 'left']:
    ax2_r.spines[s].set_visible(False)

# --- 下图: Smoothed 拟合战役强度 ---
ax2b = fig2.add_subplot(gs2[1])
ax2b.fill_between(dates, 0, battle_intensity.values,
                  color=SOFT_GREEN, alpha=0.18, linewidth=0,
                  label='Raw Intensity')
ax2b.plot(dates, battle_I_fitted.values,
          color=SOFT_GREEN, alpha=0.95, linewidth=1.0,
          label='Smoothed Fitted')
ax2b.set_ylabel('Battle Intensity', fontsize=11)
ax2b.set_xlabel('Date', fontsize=12)
ax2b.set_title(
    f'Smoothed-Fitted Battle Intensity (Exogenous)  |  '
    f'Active days: {active_days}/{n}  Mean I={avg_I:.3f}',
    fontsize=12, fontweight='bold'
)
ax2b.legend(loc='upper left', fontsize=8, framealpha=0.8)
ax2b.set_ylim(bottom=0, top=1.05)
ax2b.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
for s in ['top', 'right']:
    ax2b.spines[s].set_visible(False)

fig2.tight_layout(pad=2)
fig2.savefig('drone_ARIMAX_BattleOverlay.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_ARIMAX_BattleOverlay.png saved")

# ==============================================================
# 6. 汇总
# ==============================================================
print("\n" + "=" * 70)
print("ARIMAX DRONE PREDICTION SUMMARY")
print("=" * 70)
print(f"  Data: {n} daily drone loss observations")
print(f"  Best model: ARIMAX{best_order}  BIC={best_bic:.0f}")
print(f"  Battle effect: beta = {beta_battle:.3f}")
print(f"    -> I=1.0 adds {beta_battle:.1f} drones/day")
print(f"    -> I=0.5 adds {beta_battle/2:.1f} drones/day")
print(f"  RMSE: {rmse_arimax:.2f}")
print(f"  Forecast horizon: {future_steps} days")
print(f"  Forecast (I={last_I:.2f}): {fut.iloc[0]:.1f} -> {fut.iloc[-1]:.1f}")
print()
print("Output files:")
print("  - drone_ARIMAX_Regression.png   (ARIMAX regression curve only, no CI)")
print("  - drone_ARIMAX_BattleOverlay.png (battle intensity overlay, no CI)")
print()
print("Done.")
# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(dates) + list(future_idx),
    'actual': list(y.values) + [np.nan]*len(future_idx),
    'predicted': list(arimax_fit.values) + list(fut.values)
})
csv_df.to_csv('pred_drone_ARIMAX.csv', index=False)
print(f"  -> pred_drone_ARIMAX.csv saved")

plt.show()
