#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure GPR 2D — 引入战役效应
=============================
思路: 不使用 ARIMA, GPR 直接在 2D 空间 [时间, 战役强度] 上建模无人机损失.

与 ARIMAX+GPR 的区别:
  ARIMAX+GPR : ARIMAX先提取线性结构(含战役线性效应) → GPR修正非线性残差
             优势: 外推稳定; 劣势: 需正确指定ARIMAX阶数
  Pure GPR  : GPR直接从数据学习所有结构(线性+非线性+战役效应)
             优势: 更灵活, 无参数假设, ARD自动学习维度重要性
             劣势: O(N³)计算, 外推依赖核函数

核心机制 — ARD (Automatic Relevance Determination):
  复合核中的每个子核都有独立的长度尺度 [ℓ_time, ℓ_intensity]
  优化后:
    - ℓ_time 小 → 时间维度变化敏感 (趋势/季节性)
    - ℓ_intensity 小 → 战役强度变化敏感 (战役冲击效应强)
  模型自动判断哪个维度对预测更重要.

输出:
  Figure 1: 非重叠图 (纯预测曲线, 无战役矩形, 含GPR置信区间)
  Figure 2: 重叠图 (左轴=损失, 右轴=战役序号, 含GPR置信区间)
  Figure 3: 边际效应图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
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
print("Pure GPR 2D — Loading...")
drone = pd.read_csv('drone.csv')
drone.columns = ['date','day','direction','drone']
drone['date'] = pd.to_datetime(drone['date']); drone = drone.set_index('date').sort_index()
drone['drone'] = drone['drone'].apply(lambda x: float(str(x).replace(',','').replace('"','').strip()) if isinstance(x,str) else float(x))
drone.loc[drone['drone']<0, 'drone'] = 0
y_raw = drone['drone'].fillna(0); n = len(y_raw); dates = y_raw.index

battles = pd.read_csv('russia_ukraine_battles.csv')
battles['start_date'] = pd.to_datetime(battles['start_date'])
battles['end_date'] = battles.apply(lambda r: r['start_date']+pd.Timedelta(days=int(r['duration_days'])), axis=1)
b_by_start = battles.sort_values('start_date').reset_index(drop=True)

# ══════════════════════════════════════════════════════
# 2. 2D 输入: X = [time, battle_intensity]
# ══════════════════════════════════════════════════════
battle_I = pd.Series(0.0, index=dates)
for _, row in battles.iterrows():
    mask = (dates >= row['start_date']) & (dates <= row['end_date'])
    if mask.sum()>0: battle_I.loc[mask] += row['daily_intensity']  # 重叠求和

# Min-Max 归一化 → [0, 1]
bi_min, bi_max = battle_I.min(), battle_I.max()
battle_I = (battle_I - bi_min) / (bi_max - bi_min + 1e-8)

# --- 双次滚动均值平滑战役烈度 ---
battle_I_smooth = battle_I.rolling(window=21, center=True, min_periods=1).mean()
battle_I_fitted = battle_I_smooth.rolling(window=11, center=True, min_periods=1).mean()

# 标准化 y
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y_raw.values.reshape(-1,1)).ravel()

# 2D 输入
X_train = np.hstack([np.arange(n).reshape(-1,1)/n, battle_I.values.reshape(-1,1)])

# 未来 (单情景: I=last_I, 冲突持续)
future_steps = 30; last_I = battle_I.iloc[-1]
t_fut = np.arange(n, n+future_steps).reshape(-1,1)/n
X_fut = np.hstack([t_fut, np.full((future_steps,1), last_I)])

# ══════════════════════════════════════════════════════
# 3. 2D 复合核 + 训练
# ══════════════════════════════════════════════════════
print("  Training 2D GPR (RBF+Matern ARD, n=1556, ~1 min)...")
kernel = (
    ConstantKernel(1.0,(1e-3,1e3))*RBF(length_scale=[1.0,1.0],length_scale_bounds=(1e-2,1e2)) +
    ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=[1.0,1.0],length_scale_bounds=(1e-2,1e2),nu=1.5) +
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3,1e2))
)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01, normalize_y=False, random_state=42)
gpr.fit(X_train, y_scaled)
print(f"  LML={gpr.log_marginal_likelihood_value_:.1f}")

# 解读 ARD 长度尺度
for nm, val in gpr.kernel_.get_params().items():
    if 'length_scale' in nm and hasattr(val,'__len__') and len(val)==2:
        print(f"  ARD [{nm}]: L_time={val[0]:.4f}  L_intensity={val[1]:.4f}  "
              f"({'intensity dominates' if val[1]<val[0] else 'time dominates'})")

# ══════════════════════════════════════════════════════
# 4. 预测
# ══════════════════════════════════════════════════════
y_hist_s, s_hist = gpr.predict(X_train, return_std=True)
y_fut_s, s_fut = gpr.predict(X_fut, return_std=True)

y_hist  = scaler.inverse_transform(y_hist_s.reshape(-1,1)).ravel()
s_hist  = s_hist * scaler.scale_[0]
y_fut   = scaler.inverse_transform(y_fut_s.reshape(-1,1)).ravel()
s_fut   = s_fut * scaler.scale_[0]

future_idx = pd.date_range(start=dates[-1]+pd.Timedelta(days=1), periods=future_steps)
rmse = np.sqrt(np.mean((y_raw.values-y_hist)**2))
mae  = np.mean(np.abs(y_raw.values-y_hist))
print(f"  Fit: RMSE={rmse:.2f}  MAE={mae:.2f}")

# ══════════════════════════════════════════════════════
# 5. 图 — 统一风格
# ══════════════════════════════════════════════════════
print("  Drawing...")
z = 2.326; nB = len(b_by_start)

# ────────────────────────────────────────────────────
# Figure 1: 非重叠图 (纯预测曲线, 无战役矩形, 含GPR置信区间)
# ────────────────────────────────────────────────────
print("  -> Figure 1: Pure 2D GPR Regression (no battle overlay)")
fig1, ax1 = plt.subplots(figsize=(22, 12))

# 实际数据
ax1.plot(dates, y_raw, color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Data', zorder=2)
# GPR 历史拟合 + CI
ax1.plot(dates, y_hist, color=DARK_ORANGE, linewidth=1.3,
         label='2D GPR Fit', zorder=3)
ax1.fill_between(dates, np.maximum(0, y_hist-z*s_hist), y_hist+z*s_hist,
                 color=DARK_ORANGE, alpha=0.12, label='99% CI (Historical)', zorder=2)
# 未来预测 + CI
ax1.plot(future_idx, y_fut, color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)
ax1.fill_between(future_idx, np.maximum(0, y_fut-z*s_fut), y_fut+z*s_fut,
                 color=CRIMSON, alpha=0.12, label='99% CI (Future)', zorder=3)
# 分割线
ax1.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)

ax1.set_ylabel('Daily Drone Loss', fontsize=12)
ax1.set_title(
    f'Pure 2D GPR  |  Drone Daily Loss  |  '
    f'RBF+Matern(ARD)  RMSE={rmse:.2f}  LML={gpr.log_marginal_likelihood_value_:.0f}  '
    f'Forecast={future_steps}d',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)
ax1.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax1.set_ylim(bottom=-5)
for s in ['top','right']: ax1.spines[s].set_visible(False)

fig1.tight_layout(pad=2)
fig1.savefig('drone_PureGPR_2D_Regression.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_PureGPR_2D_Regression.png saved")

# ────────────────────────────────────────────────────
# Figure 2: 重叠图 (左轴=损失, 右轴=战役序号, 含GPR置信区间)
# ────────────────────────────────────────────────────
print("  -> Figure 2: Pure 2D GPR + Battle Overlay")
fig2 = plt.figure(figsize=(24, 14))
gs = fig2.add_gridspec(2, 1, height_ratios=[2.8, 1], hspace=0.3)

# --- 上图: 预测 + 右侧战役矩形 ---
ax = fig2.add_subplot(gs[0])

# 左轴: 预测内容
ax.plot(dates, y_raw, color=ROYAL_BLUE, alpha=0.4, linewidth=0.7, label='Actual Data', zorder=2)
ax.plot(dates, y_hist, color=DARK_ORANGE, linewidth=1.3, label='2D GPR Fit', zorder=3)
ax.fill_between(dates, np.maximum(0, y_hist-z*s_hist), y_hist+z*s_hist,
                color=DARK_ORANGE, alpha=0.12, label='99% CI', zorder=2)
ax.plot(future_idx, y_fut, color=CRIMSON, linewidth=2.2, linestyle='--',
        label=f'Forecast: I={last_I:.2f}', zorder=4)
ax.fill_between(future_idx, np.maximum(0, y_fut-z*s_fut), y_fut+z*s_fut,
                color=CRIMSON, alpha=0.12, zorder=3)
ax.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)
ax.set_ylabel('Daily Drone Loss', fontsize=12)
ax.set_title('Pure 2D GPR  |  Left=Loss  Right=Intensity  |  '
             f'RBF+Matern(ARD)  RMSE={rmse:.2f}  LML={gpr.log_marginal_likelihood_value_:.0f}',
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

# --- 下图: 边际效应 ---
ax2 = fig2.add_subplot(gs[1])
intensities = np.linspace(0, 1, 100)
time_slices = [0.25, 0.50, 0.75, 0.95]
colors_t = ['#7BA7BC','#4A6A7D','#D4786E','#B71C1C']
labels_t  = [dates[int(t*n)].strftime('%Y-%m') if int(t*n)<n else 'Future' for t in time_slices]

for t, c, lbl in zip(time_slices, colors_t, labels_t):
    X_curve = np.column_stack([np.full(100, min(t,0.999)), intensities])
    y_c_s, s_c = gpr.predict(X_curve, return_std=True)
    y_c = scaler.inverse_transform(y_c_s.reshape(-1,1)).ravel(); sc = s_c*scaler.scale_[0]
    ax2.plot(intensities, y_c, color=c, linewidth=1.8, label=lbl)
    ax2.fill_between(intensities, np.maximum(0,y_c-2.326*sc), y_c+2.326*sc, color=c, alpha=0.08)

ax2.set_xlabel('Battle Intensity (normalized)', fontsize=11)
ax2.set_ylabel('Predicted Drone Loss', fontsize=11)
ax2.set_title('Marginal Effect: How intensity changes predicted loss across time',
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=4)
ax2.grid(True, linestyle='--', alpha=0.35, color='#cccccc'); ax2.set_ylim(bottom=0)
for s in ['top','right']: ax2.spines[s].set_visible(False)

fig2.tight_layout(pad=2)
fig2.savefig('drone_PureGPR_2D_Overlay.png', dpi=250, facecolor='white', edgecolor='none')
print(f"     -> drone_PureGPR_2D_Overlay.png")

print(f"  2D GPR RBF+Matern(ARD)  |  LML={gpr.log_marginal_likelihood_value_:.0f}  |  RMSE={rmse:.2f}")
print("Done.")
# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(dates) + list(future_idx),
    'actual': list(y_raw.values) + [np.nan]*len(future_idx),
    'predicted': list(y_hist) + list(y_fut)
})
csv_df.to_csv('pred_drone_PureGPR_2D.csv', index=False)
print(f"  -> pred_drone_PureGPR_2D.csv saved")

plt.show()
