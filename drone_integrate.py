#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_integrate — 整合所有模型预测 CSV
========================================
读取 pred_drone_*.csv → 按 date 合并 → 输出 pred_drone_ALL.csv + 对比图
"""

import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROYAL_BLUE = '#4169E1'; CRIMSON = '#DC143C'; GRAY_LINE = '#808080'

plt.rcParams.update({'font.size':9.5,'axes.titlesize':12,'axes.labelsize':10,
    'figure.dpi':200,'savefig.dpi':250,'savefig.bbox':'tight',
    'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
    'axes.unicode_minus':False,'figure.facecolor':'white'})

CSV_DIR = '.'
PATTERN = 'pred_drone_*.csv'
OUT_CSV  = 'pred_drone_ALL.csv'
OUT_PNG  = 'drone_Model_Comparison.png'

# ---- 读取所有 CSV ----
files = sorted(glob.glob(os.path.join(CSV_DIR, PATTERN)))
print(f"Found {len(files)} prediction CSVs:")
for f in files:
    print(f"  {os.path.basename(f)}")

if not files:
    print("No pred_drone_*.csv found. Run prediction scripts first.")
    exit()

merged = None
for f in files:
    df = pd.read_csv(f, parse_dates=['date'])
    # 提取模型名
    model_name = os.path.basename(f).replace('pred_drone_', '').replace('.csv', '')
    df = df.rename(columns={'predicted': model_name})
    cols = ['date', model_name]
    if 'actual' in df.columns:
        cols.insert(1, 'actual')
    df = df[cols]
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on='date', how='outer')
        # 合并 actual 列
        if 'actual_x' in merged.columns:
            merged['actual'] = merged['actual_x'].combine_first(merged['actual_y'])
            merged = merged.drop(columns=['actual_x', 'actual_y'])

merged = merged.sort_values('date').reset_index(drop=True)
merged.to_csv(OUT_CSV, index=False)
print(f"\n-> {OUT_CSV}  ({len(merged)} rows, {len(merged.columns)-1} models)")

# ---- 对比图 ----
model_cols = [c for c in merged.columns if c not in ('date', 'actual')]
colors = plt.cm.tab20(np.linspace(0, 1, len(model_cols)))

fig, ax = plt.subplots(figsize=(26, 14))

# 实际值
if 'actual' in merged.columns:
    ax.plot(merged['date'], merged['actual'], color=ROYAL_BLUE, alpha=0.35, lw=0.8,
            label='Actual Drone Loss', zorder=2)

# 各模型预测
for col, c in zip(model_cols, colors):
    ax.plot(merged['date'], merged[col], color=c, lw=1.0, alpha=0.8,
            label=col, zorder=3)

split = pd.Timestamp('2026-05-31')
ax.axvline(x=split, color=GRAY_LINE, ls='--', lw=1.5, alpha=0.8)

ax.set_ylabel('Daily Drone Loss', fontsize=12)
ax.set_title(f'All Model Predictions Comparison  |  {len(model_cols)} models',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=7, framealpha=0.85, ncol=3)
ax.grid(True, ls='--', alpha=0.35, color='#cccccc')
ax.set_ylim(bottom=-20)
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout(pad=2)
fig.savefig(OUT_PNG, dpi=250, facecolor='white', edgecolor='none')
print(f"-> {OUT_PNG}")
plt.show()
print("Done.")
