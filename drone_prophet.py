#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone Prophet — 无人机日损失预测 (drone dataset)
=================================================
参考 ARIMA BIC 框架, 使用 Prophet 进行时间序列建模:
  - Prophet 自动分解: 趋势(growth) + 季节性(seasonality) + 节假日效应
  - 自动检测变点 (changepoints)
  - 提供不确定性区间 (uncertainty intervals)

Prophet vs ARIMA:
  - Prophet: 加法模型, 对趋势变化、季节性和节假日有良好支持
  - ARIMA:  自回归移动平均, 需要平稳性假设

数据:
  - 训练集: drone.csv (2022-02-25 ~ 2026-05-31)
  - 预测: 未来 30 天
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
print("Prophet — Drone Daily Loss Prediction")
print("=" * 70)

print("\n[1/4] Loading data...")

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

# ==============================================================
# 2. 转换为 Prophet 格式 (必须包含 ds 和 y 列)
# ==============================================================
print("\n[2/4] Preparing data for Prophet...")

df_prophet = pd.DataFrame({
    'ds': dates,
    'y': y.values
})
print(f"  Prophet dataframe: {len(df_prophet)} rows")

# ==============================================================
# 3. 构建与拟合 Prophet 模型
# ==============================================================
print("\n[3/4] Fitting Prophet model...")

# Prophet 参数说明:
#   - growth='linear': 线性趋势 (适合长期增长趋势)
#   - changepoint_prior_scale=2.0: 控制趋势变化灵敏度 (默认0.05, 极大值以捕捉战争脉冲)
#   - seasonality_prior_scale=20.0: 控制季节性强度 (增大以允许更灵活的季节性)
#   - yearly_seasonality=True: 启用年周期
#   - n_changepoints=50: 潜在变点数量 (默认25, 加倍以捕捉更多突变)
#   - changepoint_range=1.0: 全范围可变点检测 (包括最近数据)
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

model.fit(df_prophet)
print(f"  Prophet fitted successfully")
print(f"  Detected changepoints: {len(model.changepoints)}")

# ==============================================================
# 4. 未来预测
# ==============================================================
future_steps = 30
future = model.make_future_dataframe(periods=future_steps)
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
print(f"  Forecast ({future_steps}d) mean: {future_forecast.mean():.1f}")

# ==============================================================
# 5. 可视化 — 统一风格
# ==============================================================
print("\n[4/4] Drawing figure...")

fig, ax = plt.subplots(figsize=(22, 12))

# 实际数据 (背景)
ax.plot(dates, y.values,
        color=ROYAL_BLUE, alpha=0.4, linewidth=0.7,
        label='Actual Drone Loss', zorder=2)

# Prophet 历史拟合
ax.plot(dates, hist_fit,
        color=DARK_ORANGE, linewidth=1.3,
        label='Prophet Fit', zorder=3)

# 未来预测
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_steps)
ax.plot(future_dates, future_forecast,
        color=CRIMSON, linewidth=2.2, linestyle='--',
        label=f'Prophet Forecast ({future_steps}d)', zorder=4)

# 分割线
ax.axvline(x=dates[-1], color=GRAY_LINE, linestyle=':', linewidth=1.0, alpha=0.6)

# 装饰
ax.set_ylabel('Daily Drone Loss', fontsize=12)
ax.set_title(
    f'Prophet Forecast  |  Drone Daily Loss  |  '
    f'RMSE={rmse:.2f}  MAE={mae:.2f}  '
    f'Changepoints={len(model.changepoints)}  Forecast={future_steps}d',
    fontsize=13, fontweight='bold'
)
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax.set_ylim(bottom=-20)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=2)
fig.savefig('drone_Prophet_Forecast_drone.png', dpi=250,
            facecolor='white', edgecolor='none')
print(f"  -> drone_Prophet_Forecast_drone.png saved")

# ==============================================================
# 6. 分量分解图 (Prophet 特色)
# ==============================================================
print("  -> Drawing components figure...")
fig2 = model.plot_components(forecast, figsize=(14, 10))
fig2.savefig('drone_Prophet_Components_drone.png', dpi=250,
             facecolor='white', edgecolor='none')
print(f"  -> drone_Prophet_Components_drone.png saved")

# ==============================================================
# 7. 汇总
# ==============================================================
print("\n" + "=" * 70)
print("PROPHET DRONE PREDICTION SUMMARY")
print("=" * 70)
print(f"  Data: {n} daily drone loss observations")
print(f"  Changepoints detected: {len(model.changepoints)}")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  Forecast horizon: {future_steps} days")
print(f"  Forecast: {future_forecast[0]:.1f} -> {future_forecast[-1]:.1f}")
print()
print("Output files:")
print("  - drone_Prophet_Forecast_drone.png    (forecast plot)")
print("  - drone_Prophet_Components_drone.png  (trend + seasonality decomposition)")
print()
print("Done.")
# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(dates) + list(future_dates),
    'actual': list(y.values) + [np.nan]*len(future_dates),
    'predicted': list(hist_fit) + list(future_forecast)
})
csv_df.to_csv('pred_drone_Prophet.csv', index=False)
print(f"  -> pred_drone_Prophet.csv saved")

plt.show()
