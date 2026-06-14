#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_actual — 无人机日损失完整数据集展示
============================================
从 drone.csv 读取训练集，追加 5.31-6.8 的 7 天测试数据，
生成 dronetotal.csv 并绘制全时段图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROYAL_BLUE  = '#4169E1'
DARK_ORANGE = '#FF8C00'
CRIMSON     = '#DC143C'
GRAY_LINE   = '#808080'

plt.rcParams.update({
    'font.size': 9.5, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 200, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
})

# ==============================================================
# 1. 加载 drone.csv 训练集
# ==============================================================
print("=" * 60)
print("Drone Actual Data")
print("=" * 60)

drone = pd.read_csv('drone.csv')
drone.columns = ['date', 'day', 'direction', 'drone']
drone['date'] = pd.to_datetime(drone['date'])
drone = drone.set_index('date').sort_index()
drone['drone'] = drone['drone'].apply(
    lambda x: float(str(x).replace(',', '').replace('"', '').strip())
    if isinstance(x, str) else float(x)
)
drone.loc[drone['drone'] < 0, 'drone'] = 0
train = drone['drone'].fillna(0)
print(f"  Train: {len(train)} days [{train.index[0].date()} → {train.index[-1].date()}]")

# ==============================================================
# 2. 追加 5.31-6.7 测试数据 (7天)
# ==============================================================
extra = [
    ('2026-05-31', 1894),
    ('2026-06-01', 1852),
    ('2026-06-02', 1583),
    ('2026-06-03', 1853),
    ('2026-06-04', 2111),
    ('2026-06-05', 2046),
    ('2026-06-06', 2046),
    ('2026-06-07', 2245),
]
test_dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in extra])
test_vals  = np.array([v for _, v in extra], dtype=float)
test = pd.Series(test_vals, index=test_dates, name='drone')
print(f"  Test:  {len(test)} days [{test.index[0].date()} → {test.index[-1].date()}]")

# 合并
full = pd.concat([train, test])
print(f"  Total: {len(full)} days [{full.index[0].date()} → {full.index[-1].date()}]")

# 导出
full.to_csv('dronetotal.csv', header=['drone'])
print("  -> dronetotal.csv saved")

# ==============================================================
# 3. 统计
# ==============================================================
split_date = pd.Timestamp('2026-05-31')
train_mask = full.index < split_date
test_mask  = full.index >= split_date

print(f"\n  Train — Mean={full[train_mask].mean():.1f}  Std={full[train_mask].std():.1f}  Max={full[train_mask].max():.0f}")
print(f"  Test  — Mean={full[test_mask].mean():.1f}  Std={full[test_mask].std():.1f}  Max={full[test_mask].max():.0f}")

# ==============================================================
# 4. 绘图
# ==============================================================
print("\n  Drawing...")
fig, ax = plt.subplots(figsize=(24, 12))

# 训练集
ax.plot(full.index[train_mask], full.values[train_mask],
        color=ROYAL_BLUE, alpha=0.55, linewidth=0.8,
        label=f'Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)}d)',
        zorder=2)

# 测试集
ax.plot(full.index[test_mask], full.values[test_mask],
        color=CRIMSON, alpha=0.95, linewidth=2.0, marker='o', markersize=6,
        label=f'Test: {test.index[0].date()} → {test.index[-1].date()}  ({len(test)}d)',
        zorder=3)

for dt, val in zip(test_dates, test_vals):
    ax.annotate(f'{int(val)}',
                xy=(dt, val), xytext=(0, 14), textcoords='offset points',
                fontsize=7.5, color=CRIMSON, ha='center', fontweight='bold')

# 30日均线
ma30 = full.rolling(30, min_periods=1).mean()
ax.plot(full.index, ma30.values,
        color=DARK_ORANGE, linewidth=1.3, alpha=0.85,
        label='30-Day MA', zorder=4)

# 分割线
ax.axvline(x=split_date, color=GRAY_LINE, linestyle='--', linewidth=1.8, alpha=0.9,
           label=f'Split: {split_date.date()}')
ax.axvspan(split_date, full.index[-1] + pd.Timedelta(hours=12),
           color=CRIMSON, alpha=0.05, zorder=0)

# 装饰
ax.set_ylabel('Daily Drone Loss', fontsize=12)
ax.set_title(
    f'Drone Daily Loss — Full Dataset  |  '
    f'{full.index[0].date()} → {full.index[-1].date()}  |  '
    f'Train={len(train)}d  Test={len(test)}d  Mean={full.mean():.1f}  Max={full.max():.0f}',
    fontsize=13, fontweight='bold'
)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
ax.grid(True, linestyle='--', alpha=0.35, color='#cccccc')
ax.set_ylim(bottom=-20)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.tight_layout(pad=2)
fig.savefig('drone_Actual_Data.png', dpi=250, facecolor='white', edgecolor='none')
print("  -> drone_Actual_Data.png saved")
plt.show()
print("Done.")
