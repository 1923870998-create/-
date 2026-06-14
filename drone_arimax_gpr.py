#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMAX+GPR 混合模型 — 引入战役效应
========================================
思路: 将战役视为"长时间序列效应"而非点冲击, 展开为逐日强度序列,
     作为 ARIMAX 的外生变量 + 2D GPR 的第二维输入.

两步法:
  Step 1 — ARIMAX(exog=战役强度) : 捕捉线性基线 + 战役的线性边际效应
  Step 2 — 2D GPR(time, intensity): 拟合 ARIMAX 遗漏的非线性残差
  最终预测 = ARIMAX + GPR_residual

对比纯 ARIMA:
  - ARIMA 只能通过时间间接"感知"战役 (如 Bakhmut 292天的高损耗期)
  - ARIMAX 明确告知模型: 当战役强度=0.8时, 基线自动上调
  - GPR 进一步在2D空间 [时间, 强度] 上修正非线性交互

输出:
  Figure 1: 非重叠图 (纯预测曲线, 无战役矩形, 含GPR置信区间)
  Figure 2: 重叠图 (左轴=损失, 右轴=战役序号, 含GPR置信区间)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import itertools
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# 统一配色 + 蓝→红烈度
# ══════════════════════════════════════════════════════
ROYAL_BLUE  = '#4169E1'
DARK_ORANGE = '#FF8C00'
CRIMSON     = '#DC143C'
DODGER_BLUE = '#1E90FF'
GRAY_LINE   = '#808080'
SOFT_GREEN  = '#66BB6A'

BLUE_TO_RED = LinearSegmentedColormap.from_list('BlueRed', [
    '#D6E4F0', '#6BAED6', '#3182BD', '#C9898F', '#E05554', '#B71C1C'
])

plt.rcParams.update({'font.size':9.5, 'axes.titlesize':12, 'axes.labelsize':10,
                     'figure.dpi':200, 'savefig.dpi':250, 'savefig.bbox':'tight',
                     'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False, 'figure.facecolor':'white'})

# ══════════════════════════════════════════════════════
# 1. 数据
# ══════════════════════════════════════════════════════
print("ARIMAX+GPR — Loading...")
drone = pd.read_csv('drone.csv')
drone.columns = ['date','day','direction','drone']
drone['date'] = pd.to_datetime(drone['date']); drone = drone.set_index('date').sort_index()
drone['drone'] = drone['drone'].apply(lambda x: float(str(x).replace(',','').replace('"','').strip()) if isinstance(x,str) else float(x))
drone.loc[drone['drone']<0, 'drone'] = 0
y = drone['drone'].fillna(0); n = len(y)

battles = pd.read_csv('russia_ukraine_battles.csv')
battles['start_date'] = pd.to_datetime(battles['start_date'])
battles['end_date'] = battles.apply(lambda r: r['start_date']+pd.Timedelta(days=int(r['duration_days'])), axis=1)
b_by_start = battles.sort_values('start_date').reset_index(drop=True)

# ══════════════════════════════════════════════════════
# 2. 逐日战役强度 (长时间序列效应)
# ══════════════════════════════════════════════════════
dates = y.index
battle_intensity = pd.Series(0.0, index=dates)
for _, row in battles.iterrows():
    mask = (dates >= row['start_date']) & (dates <= row['end_date'])
    if mask.sum()>0: battle_intensity.loc[mask] += row['daily_intensity']  # 重叠求和
bi_min, bi_max = battle_intensity.min(), battle_intensity.max()
battle_intensity = (battle_intensity - bi_min) / (bi_max - bi_min + 1e-8)  # Min-Max归一化
exog_train = battle_intensity.values.reshape(-1,1)

# --- 双次滚动均值平滑战役烈度 ---
battle_I_smooth = battle_intensity.rolling(window=21, center=True, min_periods=1).mean()
battle_I_fitted = battle_I_smooth.rolling(window=11, center=True, min_periods=1).mean()

# ══════════════════════════════════════════════════════
# 3. ARIMAX — BIC 搜索 (p,q) + 战役强度 exog
# ══════════════════════════════════════════════════════
print("  BIC search ARIMAX(p,d,q)+exog...")
y_diff = y.diff().dropna(); d = 2  # 固定二阶差分适应强趋势
best_bic, best_order, best_model = float("inf"), None, None

for p,q in itertools.product(range(0,6), range(0,6)):
    try:
        m = SARIMAX(y, exog=exog_train, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
        f = m.fit(disp=False, maxiter=200)
        if f.bic < best_bic: best_bic, best_order, best_model = f.bic, (p,d,q), f
    except: continue

if best_model is None: best_order=(2,1,2); best_model=SARIMAX(y,exog=exog_train,order=best_order,enforce_stationarity=False).fit(disp=False)

# 战役系数
beta = best_model.params.iloc[[i for i,nm in enumerate(best_model.param_names) if 'x1' in nm.lower()][0]] if any('x1' in nm.lower() for nm in best_model.param_names) else 0
print(f"  ARIMAX{best_order} BIC={best_bic:.0f}  β_battle={beta:.3f}")

# 拟合 & 残差
arimax_fit = pd.Series(best_model.predict(start=0, end=n-1, exog=exog_train), index=y.index)
residuals = y - arimax_fit

# ══════════════════════════════════════════════════════
# 4. 2D GPR 拟合残差: X = [time, intensity]
# ══════════════════════════════════════════════════════
print("  Fitting 2D GPR on residuals...")
future_steps = 30
X_train = np.hstack([np.arange(n).reshape(-1,1)/n, battle_intensity.values.reshape(-1,1)])

# 未来: 单情景 (I=last_I, 冲突持续)
last_I = battle_intensity.iloc[-1]
t_fut = np.arange(n, n+future_steps).reshape(-1,1)/n
X_fut = np.hstack([t_fut, np.full((future_steps,1), last_I)])

kernel = (ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=[1.0,1.0],length_scale_bounds=(1e-3,1e3),nu=1.5)
          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3,1e3)))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, alpha=0.01)
gpr.fit(X_train, residuals.values)

gpr_fit, gpr_std_hist = gpr.predict(X_train, return_std=True)
gpr_fut, gpr_std_fut = gpr.predict(X_fut, return_std=True)

# ARIMAX 未来 (单情景)
arimax_fut = best_model.get_forecast(steps=future_steps, exog=np.full((future_steps,1),last_I)).predicted_mean

# 混合
hybrid_hist = arimax_fit + gpr_fit
hybrid_fut = arimax_fut + gpr_fut
lo_fut = np.maximum(0, hybrid_fut - 2.326*gpr_std_fut)
hi_fut = hybrid_fut + 2.326*gpr_std_fut

future_idx = pd.date_range(start=y.index[-1]+pd.Timedelta(days=1), periods=future_steps)
rmse_arimax = np.sqrt(np.mean((y-arimax_fit)**2))
rmse_hybrid = np.sqrt(np.mean((y-hybrid_hist)**2))
print(f"  RMSE: ARIMAX={rmse_arimax:.2f} → Hybrid={rmse_hybrid:.2f} ({(rmse_arimax-rmse_hybrid)/rmse_arimax*100:.1f}% improvement)")

# ══════════════════════════════════════════════════════
# 5. 图 — 统一风格
# ══════════════════════════════════════════════════════
print("  Drawing...")
nB = len(b_by_start)
z = 2.326  # 99% CI

# ────────────────────────────────────────────────────
# Figure 1: 非重叠图 (纯预测曲线, 无战役矩形, 含GPR置信区间)
# ────────────────────────────────────────────────────
print("  -> Figure 1: ARIMA+GPR Regression (no battle overlay)")
fig1, ax1 = plt.subplots(figsize=(22, 12))

# 实际数据
ax1.plot(y.index, y, color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Data', zorder=2)
# ARIMAX 基线 (虚线)
ax1.plot(y.index, arimax_fit, color=DODGER_BLUE, alpha=0.55, linestyle='--',
         linewidth=1.0, label='ARIMAX Baseline', zorder=2)
# 混合拟合
ax1.plot(y.index, hybrid_hist, color=DARK_ORANGE, linewidth=1.3,
         label='Hybrid Fit (ARIMAX+GPR)', zorder=3)
# 历史 99% CI
lo_hist = np.maximum(0, hybrid_hist - z*gpr_std_hist)
hi_hist = hybrid_hist + z*gpr_std_hist
ax1.fill_between(y.index, lo_hist, hi_hist, color=DARK_ORANGE, alpha=0.10,
                 label='99% CI (Historical)', zorder=2)

# 未来预测 (单情景, 含CI)
ax1.plot(future_idx, hybrid_fut, color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)
ax1.fill_between(future_idx, lo_fut, hi_fut, color=CRIMSON, alpha=0.12,
                 label='99% CI (Future)', zorder=3)

# 分割线
ax1.axvline(x=y.index[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)

ax1.set_ylabel('Daily Drone Loss', fontsize=12)
ax1.set_title(
    f'ARIMAX{best_order}+2D GPR  |  Drone Daily Loss  |  '
    f'β_battle={beta:.3f}  RMSE: {rmse_arimax:.2f}→{rmse_hybrid:.2f}  '
    f'Forecast={future_steps}d',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)
ax1.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax1.set_ylim(bottom=-5)
for s in ['top','right']: ax1.spines[s].set_visible(False)

fig1.tight_layout(pad=2)
fig1.savefig('drone_ARIMAX_GPR_Regression.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_ARIMAX_GPR_Regression.png saved")

# ────────────────────────────────────────────────────
# Figure 2: 重叠图 (左轴=损失, 右轴=战役序号, 含GPR置信区间)
# ────────────────────────────────────────────────────
print("  -> Figure 2: ARIMA+GPR + Battle Overlay")
fig2 = plt.figure(figsize=(24, 14))
gs = fig2.add_gridspec(2, 1, height_ratios=[2.8, 1], hspace=0.3)

# --- 上图: 预测 + 右侧战役矩形 ---
ax = fig2.add_subplot(gs[0])

# 左轴: 预测内容 (统一配色)
ax.plot(y.index, y, color=ROYAL_BLUE, alpha=0.4, linewidth=0.7, label='Actual Data', zorder=2)
ax.plot(y.index, arimax_fit, color=DODGER_BLUE, alpha=0.55, linestyle='--', linewidth=1.0, label='ARIMAX Baseline', zorder=2)
ax.plot(y.index, hybrid_hist, color=DARK_ORANGE, linewidth=1.3, label='Hybrid Fit (ARIMAX+GPR)', zorder=3)
ax.fill_between(y.index, lo_hist, hi_hist, color=DARK_ORANGE, alpha=0.10, zorder=2)
ax.plot(future_idx, hybrid_fut, color=CRIMSON, linewidth=2.2, linestyle='--',
        label=f'Forecast: I={last_I:.2f}', zorder=4)
ax.fill_between(future_idx, lo_fut, hi_fut, color=CRIMSON, alpha=0.12, zorder=3)
ax.axvline(x=y.index[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)
ax.set_ylabel('Daily Drone Loss', fontsize=12)
ax.set_title(f'ARIMAX{best_order}+2D GPR  |  Left=Loss  Right=Intensity  |  '
             f'β_battle={beta:.3f}  RMSE: {rmse_arimax:.2f}→{rmse_hybrid:.2f}',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=3)
ax.grid(True, linestyle='--', alpha=0.35, color='#cccccc'); ax.set_ylim(bottom=-5)
for s in ['top','right']: ax.spines[s].set_visible(False)

# 右轴: ARIMA 拟合战役烈度线 (柔和绿色)
ax_r = ax.twinx()
ax_r.set_ylabel('Battle Intensity (Smoothed)', fontsize=11, color=SOFT_GREEN)
ax_r.plot(dates, battle_I_fitted.values,
          color=SOFT_GREEN, linewidth=1.2, alpha=0.9, zorder=5)
ax_r.set_ylim(0, 1.05)
ax_r.tick_params(axis='y', colors=SOFT_GREEN)
for s in ['top','left']: ax_r.spines[s].set_visible(False)

# --- 下图: 残差分解 ---
ax2 = fig2.add_subplot(gs[1])
ax2.plot(y.index, residuals, color='gray', alpha=0.45, linewidth=0.5, label=f'ARIMAX Residuals (RMSE={rmse_arimax:.2f})')
ax2.plot(y.index, gpr_fit, color=DARK_ORANGE, linewidth=1.3, label='2D GPR Correction')
ax2.fill_between(y.index, gpr_fit-2.326*gpr_std_hist, gpr_fit+2.326*gpr_std_hist, color=DARK_ORANGE, alpha=0.12)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylabel('Residual', fontsize=11)
ax2.set_title('2D GPR Nonlinear Correction on ARIMAX Residuals', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
for s in ['top','right']: ax2.spines[s].set_visible(False)

fig2.tight_layout(pad=2)
fig2.savefig('drone_ARIMAX_GPR_Overlay.png', dpi=250, facecolor='white', edgecolor='none')
print(f"     -> drone_ARIMAX_GPR_Overlay.png")

print(f"  ARIMAX{best_order}+exog  |  2D GPR Matern(1.5) ARD  |  β={beta:.3f}")
print("Done.")
# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(y.index) + list(future_idx),
    'actual': list(y.values) + [np.nan]*len(future_idx),
    'predicted': list(hybrid_hist.values) + list(hybrid_fut.values)
})
csv_df.to_csv('pred_drone_ARIMAX_GPR.csv', index=False)
print(f"  -> pred_drone_ARIMAX_GPR.csv saved")

plt.show()
