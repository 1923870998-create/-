#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone Prophet-X — 无人机日损失预测 (含战役强度外生变量)
=========================================================
参考 ARIMAX 框架, 使用 Prophet 并加入
战役强度作为外生变量 (extra regressor):
  - 外生变量: 逐日战役强度 (重叠战役取 max)
  - Prophet 自动分解: 趋势 + 季节性 + 战役效应
  - 单情景预测: I=last_I (冲突持续)

Prophet-X vs Prophet:
  - Prophet:  仅使用时间和目标值
  - Prophet-X: 加入战役强度 regressor, 模型可学习战役对损失的影响

Prophet-X vs ARIMAX:
  - ARIMAX:  线性模型 + 外生变量, 需平稳性
  - Prophet-X: 非线性趋势 + 季节性 + 外生变量, 无需平稳性

数据:
  - 训练集: drone.csv (2022-02-25 ~ 2026-05-31)
  - 战役数据: russia_ukraine_battles.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 配色体系 — 统一风格
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
print("Prophet-X — Drone Daily Loss Prediction with Battle Intensity")
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
# 2. 构建逐日战役强度 — 重叠求和
# ==============================================================
print("\n[2/5] Building daily battle intensity (sum of overlapping)...")

battle_intensity = pd.Series(0.0, index=dates)
for _, row in battles.iterrows():
    mask = (dates >= row['start_date']) & (dates <= row['end_date'])
    if mask.sum() > 0:
        battle_intensity.loc[mask] += row['daily_intensity']  # 重叠求和

# Min-Max 归一化 → [0, 1]
bi_min, bi_max = battle_intensity.min(), battle_intensity.max()
battle_intensity = (battle_intensity - bi_min) / (bi_max - bi_min + 1e-8)

active_days = (battle_intensity > 0).sum()
avg_I = battle_intensity[battle_intensity > 0].mean()
last_I = battle_intensity.iloc[-1]
print(f"  Active battle days: {active_days}/{n}")
print(f"  Mean intensity (active days): {avg_I:.4f}")
print(f"  Current intensity (last day): {last_I:.4f}")

# --- 双次滚动均值平滑战役烈度 ---
battle_I_smooth = battle_intensity.rolling(window=21, center=True, min_periods=1).mean()
battle_I_fitted = battle_I_smooth.rolling(window=11, center=True, min_periods=1).mean()

# ==============================================================
# 3. 构建 Prophet 数据框 + 外生变量
# ==============================================================
print("\n[3/5] Preparing Prophet-X data with battle intensity regressor...")

df_px = pd.DataFrame({
    'ds': dates,
    'y': y.values,
    'battle_intensity': battle_intensity.values
})
print(f"  Prophet-X dataframe: {len(df_px)} rows")

# ==============================================================
# 4. 构建与拟合 Prophet-X 模型
# ==============================================================
print("\n[4/5] Fitting Prophet-X model with battle intensity...")

model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.8,          # 适度平滑趋势
    seasonality_prior_scale=8.0,           # 适度季节性
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_range=0.9,                 # 最后10%不设变点
    seasonality_mode='additive',
)

# 添加战役强度作为外生 regressor
model.add_regressor('battle_intensity')

model.fit(df_px)
print(f"  Prophet-X fitted successfully")
print(f"  Detected changepoints: {len(model.changepoints)}")

# 查看战役强度的拟合系数
# Prophet 将 regressor 系数存在 model.params 中
for col in ['battle_intensity']:
    # 从训练后的模型中提取 regressor 效应
    pass

# ==============================================================
# 5. 未来预测 (单情景: I=last_I)
# ==============================================================
future_steps = 30
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_steps)

# 构建未来 dataframe (包含 regressor)
future = model.make_future_dataframe(periods=future_steps)
future['battle_intensity'] = np.concatenate([
    battle_intensity.values,                          # 历史: 实际战役强度
    np.full(future_steps, last_I)                      # 未来: 假设冲突持续
])

forecast = model.predict(future)

# 提取预测值
yhat = forecast['yhat'].values

# 分离历史拟合与未来预测
hist_fit = yhat[:n]
future_forecast = yhat[n:]

# 计算 RMSE
residuals = y.values - hist_fit
rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))
print(f"  In-sample RMSE={rmse:.2f}  MAE={mae:.2f}")
print(f"  Forecast (I={last_I:.2f}, {future_steps}d) mean: {future_forecast.mean():.1f}")

# 尝试提取 battle_intensity 的边际效应
# Prophet 的 regressor 系数可以通过 model.params 查看
beta_prophet = None
try:
    from prophet.serialize import model_to_json
    import json
    model_json = model_to_json(model)
    model_dict = json.loads(model_json)
    # 从模型参数中提取 beta
    if 'extra_regressors' in model_dict:
        er = model_dict['extra_regressors']
        if 'battle_intensity' in er:
            beta_info = er['battle_intensity']
            print(f"  Regressor 'battle_intensity' info: {beta_info}")
except Exception:
    pass

# 简单地通过数值近似估计 beta (在线性区域的斜率)
# 在均值附近计算偏效应
mean_I = battle_intensity.mean()
df_test_low = pd.DataFrame({
    'ds': [dates[len(dates)//2]] * 2,
    'battle_intensity': [mean_I - 0.1, mean_I + 0.1]
})
fc_test = model.predict(df_test_low)
beta_approx = (fc_test['yhat'].values[1] - fc_test['yhat'].values[0]) / 0.2
print(f"  Approx beta_battle (marginal effect): {beta_approx:.3f}")
print(f"  -> Each 0.1 intensity adds ~{beta_approx*0.1:.1f} drones/day")

# ==============================================================
# 6. 可视化 — 统一风格
# ==============================================================
print("\n[5/5] Drawing figures...")

# ────────────────────────────────────────────────────
# Figure 1: Prophet-X 纯回归曲线 (非重叠, 无CI)
# ────────────────────────────────────────────────────
print("  -> Figure 1: Prophet-X Regression (no battle overlay)")
fig1, ax1 = plt.subplots(figsize=(22, 12))

# 实际数据 (背景)
ax1.plot(dates, y.values,
         color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Drone Loss', zorder=2)

# Prophet-X 历史拟合
ax1.plot(dates, hist_fit,
         color=DARK_ORANGE, linewidth=1.3,
         label='Prophet-X Fit', zorder=3)

# 未来预测 (单情景: I=last_I)
ax1.plot(future_dates, future_forecast,
         color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)

# 分割线
ax1.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':',
            linewidth=1.0, alpha=0.6)

# 装饰
ax1.set_ylabel('Daily Drone Loss', fontsize=12)
ax1.set_title(
    f'Prophet-X + Battle Intensity  |  Drone Daily Loss  |  '
    f'beta_battle≈{beta_approx:.2f}  RMSE={rmse:.2f}  MAE={mae:.2f}  '
    f'Forecast={future_steps}d',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax1.set_ylim(bottom=-20)
for s in ['top', 'right']:
    ax1.spines[s].set_visible(False)

fig1.tight_layout(pad=2)
fig1.savefig('drone_ProphetX_Regression.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_ProphetX_Regression.png saved")

# ────────────────────────────────────────────────────
# Figure 2: 战役强度-Prophet-X 重叠图 (双轴叠加)
# ────────────────────────────────────────────────────
print("  -> Figure 2: Battle Intensity + Prophet-X Overlay")
fig2 = plt.figure(figsize=(24, 14))
gs2 = fig2.add_gridspec(2, 1, height_ratios=[2.8, 1], hspace=0.3)

# --- 上图: Prophet-X 预测 + 战役矩形 ---
ax2 = fig2.add_subplot(gs2[0])

# 左轴: 预测内容
ax2.plot(dates, y.values,
         color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
         label='Actual Drone Loss', zorder=2)
ax2.plot(dates, hist_fit,
         color=DARK_ORANGE, linewidth=1.3,
         label='Prophet-X Fit', zorder=3)

# 单情景: I=last_I
ax2.plot(future_dates, future_forecast,
         color=CRIMSON, linewidth=2.2, linestyle='--',
         label=f'Forecast: I={last_I:.2f} (Conflict)', zorder=4)

# 分割线
ax2.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':',
            linewidth=1.0, alpha=0.6)

# 左轴装饰
ax2.set_ylabel('Daily Drone Loss', fontsize=12)
ax2.set_title(
    f'Prophet-X + Battle Intensity  |  Left=Loss  Right=Intensity  |  '
    f'beta_battle≈{beta_approx:.2f}  RMSE={rmse:.2f}',
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
                  color=SOFT_GREEN, alpha=0.15, linewidth=0)
ax2b.plot(dates, battle_I_fitted.values,
          color=SOFT_GREEN, alpha=0.95, linewidth=1.0)
ax2b.set_ylabel('Battle Intensity', fontsize=11)
ax2b.set_xlabel('Date', fontsize=12)
ax2b.set_title(
    f'Smoothed-Fitted Battle Intensity (Regressor)  |  '
    f'Active days: {active_days}/{n}  Mean I={avg_I:.3f}  '
    f'Forecast assumes I={last_I:.2f}',
    fontsize=12, fontweight='bold'
)
ax2b.set_ylim(bottom=0, top=1.05)
ax2b.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
for s in ['top', 'right']:
    ax2b.spines[s].set_visible(False)

fig2.tight_layout(pad=2)
fig2.savefig('drone_ProphetX_BattleOverlay.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"     -> drone_ProphetX_BattleOverlay.png saved")

# ==============================================================
# 7. 汇总
# ==============================================================
print("\n" + "=" * 70)
print("PROPHET-X DRONE PREDICTION SUMMARY")
print("=" * 70)
print(f"  Data: {n} daily drone loss observations")
print(f"  Battles: {nB} campaigns")
print(f"  Changepoints detected: {len(model.changepoints)}")
print(f"  Battle effect (approx beta): {beta_approx:.3f}")
print(f"    -> I=1.0 adds ~{beta_approx:.1f} drones/day")
print(f"    -> I=0.5 adds ~{beta_approx/2:.1f} drones/day")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  Forecast horizon: {future_steps} days")
print(f"  Forecast (I={last_I:.2f}): {future_forecast[0]:.1f} -> {future_forecast[-1]:.1f}")
print()
print("Output files:")
print("  - drone_ProphetX_Regression.png     (Prophet-X regression curve)")
print("  - drone_ProphetX_BattleOverlay.png  (battle intensity overlay)")
print()
print("Done.")
# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(dates) + list(future_dates),
    'actual': list(y.values) + [np.nan]*len(future_dates),
    'predicted': list(hist_fit) + list(future_forecast)
})
csv_df.to_csv('pred_drone_ProphetX.csv', index=False)
print(f"  -> pred_drone_ProphetX.csv saved")

plt.show()
