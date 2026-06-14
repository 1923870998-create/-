#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_compare — 多模型系统比较
================================
维度:
  1. 模型特性    — 类型 / 参数 / 假设 / 是否外生变量 / 是否含GPR
  2. 训练拟合    — RMSE / MAE / sMAPE (训练窗口: 2022-02-25 ~ 2026-05-30)
  3. 测试预测    — RMSE / MAE / sMAPE (测试窗口: 2026-05-31 ~ 2026-06-07, 8天)
  4. 残差诊断    — 残差均值 / 标准差 / Ljung-Box Q / 正态性
  5. 未来预测    — 30日预测均值 / 趋势方向 / 离散度

数据需求: 所有 pred_drone_*.csv 文件已经生成
"""

import glob, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==============================================================
# Config
# ==============================================================
SPLIT_DATE = pd.Timestamp('2026-05-31')
TEST_DAYS  = 8   # 5/31 ~ 6/7 (real held-out test data)

FILES = {
    'ARIMA_BIC':          'pred_drone_ARIMA_BIC.csv',
    'ARIMA_NoCI':         'pred_drone_ARIMA_NoCI.csv',
    'ARIMA_GPR':          'pred_drone_ARIMA_GPR.csv',
    'PureGPR':            'pred_drone_PureGPR.csv',
    'ARIMAX':             'pred_drone_ARIMAX.csv',
    'ARIMAX_GPR':         'pred_drone_ARIMAX_GPR.csv',
    'PureGPR_2D':         'pred_drone_PureGPR_2D.csv',
    'Prophet':            'pred_drone_Prophet.csv',
    'ProphetX':           'pred_drone_ProphetX.csv',
    'LSTM_TCN_XGB':       'pred_drone_LSTM_TCN_XGB.csv',
}

MODEL_META = {
    'ARIMA_BIC':          {'type':'Statistical','exog':False,'gpr':False,'params':'ARIMA(0,2,3)'},
    'ARIMA_NoCI':         {'type':'Statistical','exog':False,'gpr':False,'params':'ARIMA(0,2,3)'},
    'ARIMA_GPR':          {'type':'Statistical','exog':False,'gpr':True, 'params':'ARIMA(0,2,3)+GPR'},
    'PureGPR':            {'type':'Statistical','exog':False,'gpr':True, 'params':'RBF+Matern+White'},
    'ARIMAX':             {'type':'Statistical','exog':True, 'gpr':False,'params':'ARIMAX+BIC'},
    'ARIMAX_GPR':         {'type':'Statistical','exog':True, 'gpr':True, 'params':'ARIMAX+2D GPR'},
    'PureGPR_2D':         {'type':'Statistical','exog':True, 'gpr':True, 'params':'2D GPR ARD'},
    'Prophet':            {'type':'ML','exog':False,'gpr':False,'params':'changepoint=0.8'},
    'ProphetX':           {'type':'ML','exog':True, 'gpr':False,'params':'changepoint=0.8+regressor'},
    'LSTM_TCN_XGB':       {'type':'DL','exog':True, 'gpr':False,'params':'LSTM(20)+TCN(8ch)+XGBoost'},
}

plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,
    'figure.dpi':180,'savefig.dpi':220,'savefig.bbox':'tight',
    'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
    'axes.unicode_minus':False,'figure.facecolor':'white'})

# ==============================================================
# 1. 读取完整实际数据 (含测试期 5/31-6/7 的真实值)
# ==============================================================
print("="*70)
print("Multi-Model Comparison Framework")
print("="*70)

# 优先从 dronetotal.csv 读取 (含测试真值), 否则回退到 drone.csv
if os.path.exists('dronetotal.csv'):
    df_total = pd.read_csv('dronetotal.csv', index_col=0, parse_dates=True)
    df_total = df_total.reset_index()  # date becomes a column
    df_total.columns = ['date', 'actual']
    print(f"  Using dronetotal.csv: {len(df_total)} rows")
else:
    # 回退: 用 drone_actual.py 内嵌的测试数据
    print("  dronetotal.csv not found, using built-in test data")
    drone = pd.read_csv('drone.csv')
    drone.columns = ['date','day','direction','drone']
    drone['date'] = pd.to_datetime(drone['date'])
    drone = drone.set_index('date').sort_index()
    drone['drone'] = drone['drone'].apply(
        lambda x: float(str(x).replace(',','').replace('"','').strip()) if isinstance(x,str) else float(x))
    drone.loc[drone['drone']<0, 'drone'] = 0
    train_s = drone['drone'].fillna(0)
    extra = [('2026-05-31',1894),('2026-06-01',1852),('2026-06-02',1583),
             ('2026-06-03',1853),('2026-06-04',2111),('2026-06-05',2046),
             ('2026-06-06',2046),('2026-06-07',2245)]
    test_dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d,_ in extra])
    test_s = pd.Series([v for _,v in extra], index=test_dates_idx)
    full_s = pd.concat([train_s, test_s])
    df_total = pd.DataFrame({'date': full_s.index, 'actual': full_s.values})

df_total['date'] = pd.to_datetime(df_total['date'])
full_actual = df_total.set_index('date')['actual'] if 'actual' in df_total.columns else df_total.set_index('date').iloc[:,0]

train_actual = full_actual[full_actual.index < SPLIT_DATE]
test_actual  = full_actual[(full_actual.index >= SPLIT_DATE) &
                            (full_actual.index < SPLIT_DATE + pd.Timedelta(days=TEST_DAYS))]
print(f"  Train actual: {len(train_actual)} days  |  Test actual: {len(test_actual)} days")
print(f"  Test period: {test_actual.index[0].date()} ~ {test_actual.index[-1].date()}")
print(f"  Test mean={test_actual.mean():.1f}  std={test_actual.std():.1f}")

# ==============================================================
# 2. 遍历各模型, 对齐日期计算指标
# ==============================================================
all_metrics = []
all_residuals = {}
all_forecasts = {}

# Helper functions
def safe_rmse(y, p): return np.sqrt(np.mean((y-p)**2)) if len(y)>0 else np.nan
def safe_mae(y, p):  return np.mean(np.abs(y-p)) if len(y)>0 else np.nan
def safe_smape(y, p):
    if len(y) == 0: return np.nan
    denom = np.abs(y) + np.abs(p) + 1e-8
    return np.mean(2*np.abs(y-p)/denom)

for name, fname in FILES.items():
    if not os.path.exists(fname):
        print(f"  SKIP {name}: {fname} not found")
        continue
    df = pd.read_csv(fname, parse_dates=['date'])
    df = df.set_index('date')
    pred_s = df['predicted']

    # Align with full_actual by date
    common_dates = full_actual.index.intersection(pred_s.index)
    actual_aligned = full_actual.loc[common_dates]
    pred_aligned   = pred_s.loc[common_dates]

    # Split train / test
    train_mask = common_dates < SPLIT_DATE
    test_mask  = (common_dates >= SPLIT_DATE) & (common_dates < SPLIT_DATE + pd.Timedelta(days=TEST_DAYS))
    future_mask = common_dates >= SPLIT_DATE + pd.Timedelta(days=TEST_DAYS)

    y_train = actual_aligned[train_mask].values
    p_train = pred_aligned[train_mask].values
    y_test  = actual_aligned[test_mask].values
    p_test  = pred_aligned[test_mask].values
    test_dates_arr = common_dates[test_mask]

    # Store test predictions for visualization
    all_forecasts[name] = {'dates': test_dates_arr, 'pred': p_test}

    resid = y_train - p_train

    # Ljung-Box on training residuals (lag=10)
    lb_q, lb_p = np.nan, np.nan
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        if len(resid) > 20:
            lb_res = acorr_ljungbox(resid[~np.isnan(resid)], lags=[10], return_df=True)
            lb_q = lb_res['lb_stat'].values[0]
            lb_p = lb_res['lb_pvalue'].values[0]
    except: pass

    # Shapiro-Wilk normality
    sw_stat, sw_p = np.nan, np.nan
    try:
        if len(resid) > 3 and len(resid) < 5000:
            sw_stat, sw_p = stats.shapiro(resid[~np.isnan(resid)][:500])
    except: pass

    # Train / Future trend direction
    if len(p_test) >= 2:
        fut_trend = np.polyfit(range(min(30, len(p_test))), p_test[:min(30, len(p_test))], 1)[0]
    else:
        fut_trend = np.nan
    if len(p_train) >= 30:
        train_trend = np.polyfit(range(30), p_train[-30:], 1)[0]
    else:
        train_trend = np.nan

    meta = MODEL_META.get(name, {})
    # Future 30-day mean (from future_mask)
    p_future = pred_aligned[future_mask].values if future_mask.sum() > 0 else p_test
    fut_mean30 = np.mean(p_future[:30]) if len(p_future) >= 1 else np.nan

    all_metrics.append({
        'Model': name,
        'Type': meta.get('type','?'),
        'Exog': 'Y' if meta.get('exog') else 'N',
        'GPR':  'Y' if meta.get('gpr') else 'N',
        'Params': meta.get('params','?'),
        'Train_RMSE': safe_rmse(y_train, p_train),
        'Train_MAE':  safe_mae(y_train, p_train),
        'Train_sMAPE': safe_smape(y_train, p_train),
        'Test_RMSE':  safe_rmse(y_test, p_test),
        'Test_MAE':   safe_mae(y_test, p_test),
        'Test_sMAPE': safe_smape(y_test, p_test),
        'Resid_Mean': np.mean(resid) if len(resid) > 0 else np.nan,
        'Resid_Std':  np.std(resid) if len(resid) > 0 else np.nan,
        'LjungBox_Q': lb_q,
        'LjungBox_p': lb_p,
        'Shapiro_W':  sw_stat,
        'Shapiro_p':  sw_p,
        'TrainTrend': train_trend,
        'FutTrend':   fut_trend,
        'FutMean30':  fut_mean30,
    })
    all_residuals[name] = {'train': resid}

print(f"\n  Models loaded: {len(all_metrics)}")

# ==============================================================
# 3. 比较表
# ==============================================================
tbl = pd.DataFrame(all_metrics).sort_values('Test_RMSE')
tbl = tbl.round({'Train_RMSE':1,'Train_MAE':1,'Train_sMAPE':3,
                 'Test_RMSE':1,'Test_MAE':1,'Test_sMAPE':3,
                 'Resid_Mean':1,'Resid_Std':1,'LjungBox_Q':1,'LjungBox_p':3,
                 'Shapiro_W':3,'Shapiro_p':3,'FutMean30':1})

print("\n" + "="*70)
print("MODEL COMPARISON (sorted by Test RMSE)")
print("="*70)
cols_show = ['Model','Type','Exog','GPR','Train_RMSE','Test_RMSE','Test_MAE',
             'Resid_Std','FutMean30']
print(tbl[cols_show].to_string(index=False))

tbl.to_csv('drone_Model_Comparison.csv', index=False, encoding='utf-8-sig')
print("\n-> drone_Model_Comparison.csv")

# ==============================================================
# 4. 可视化 — 六合一比较图
# ==============================================================
fig, axes = plt.subplots(2, 3, figsize=(26, 16))
colors = plt.cm.tab10(np.linspace(0, 1, len(tbl)))
model_names = tbl['Model'].values

# (a) Train vs Test RMSE 散点
ax = axes[0,0]
for i, row in tbl.iterrows():
    ax.scatter(row['Train_RMSE'], row['Test_RMSE'], c=[colors[i]], s=120, zorder=3)
    ax.annotate(row['Model'], (row['Train_RMSE'], row['Test_RMSE']),
                fontsize=6.5, alpha=0.85, xytext=(3,3), textcoords='offset points')
ax.set_xlabel('Train RMSE'); ax.set_ylabel('Test RMSE')
ax.set_title('(a) Train vs Test RMSE'); ax.grid(alpha=0.3)
ax.axline((0,0), slope=1, color='gray', ls='--', alpha=0.5)

# (b) Test RMSE bar
ax = axes[0,1]
idx = np.argsort(tbl['Test_RMSE'].values)
sorted_names = tbl['Model'].values[idx]
sorted_rmse  = tbl['Test_RMSE'].values[idx]
bars = ax.barh(range(len(sorted_names)), sorted_rmse,
               color=[colors[i] for i in idx])
ax.set_yticks(range(len(sorted_names))); ax.set_yticklabels(sorted_names, fontsize=7.5)
ax.set_xlabel('Test RMSE'); ax.set_title('(b) Test RMSE Ranking')
ax.grid(axis='x', alpha=0.3)

# (c) Test predictions vs actual
ax = axes[0,2]
if len(test_actual) >= TEST_DAYS:
    test_vals = test_actual.values[:TEST_DAYS]
    test_dates_plot = test_actual.index[:TEST_DAYS]
    ax.plot(range(TEST_DAYS), test_vals,
            'ko-', lw=2.5, ms=8, label='Actual', zorder=5)
    for i, row in tbl.iterrows():
        name = row['Model']
        if name in all_forecasts:
            p = all_forecasts[name]['pred']
            if len(p) >= TEST_DAYS:
                ax.plot(range(TEST_DAYS), p[:TEST_DAYS], 'o-', color=colors[i],
                        lw=1.2, ms=5, alpha=0.8, label=name)
ax.set_xticks(range(TEST_DAYS))
ax.set_xticklabels([d.strftime('%m-%d') for d in test_dates_plot], fontsize=7)
ax.set_title('(c) Test Period Predictions (5/31-6/7)'); ax.legend(fontsize=5.5, ncol=2, loc='best')
ax.grid(alpha=0.3)

# (d) Residual distribution
ax = axes[1,0]
for i, (name, resid_dict) in enumerate(all_residuals.items()):
    r = resid_dict['train'][~np.isnan(resid_dict['train'])]
    if len(r) > 10:
        ax.hist(r, bins=40, alpha=0.3, color=colors[i], density=True, label=name)
ax.set_xlabel('Residual'); ax.set_title('(d) Residual Distribution (Train)')
ax.legend(fontsize=5, ncol=2, loc='best'); ax.grid(alpha=0.3)

# (e) Exog vs No-Exog Test RMSE boxplot
ax = axes[1,1]
exog_rmse = [tbl[tbl['Exog']=='Y']['Test_RMSE'].values,
             tbl[tbl['Exog']=='N']['Test_RMSE'].values]
bp = ax.boxplot(exog_rmse, tick_labels=['With Exog','No Exog'], patch_artist=True)
bp['boxes'][0].set_facecolor('#A5D6A7'); bp['boxes'][1].set_facecolor('#90CAF9')
ax.set_ylabel('Test RMSE'); ax.set_title('(e) Exog vs No-Exog')
ax.grid(axis='y', alpha=0.3)

# (f) Forecast 30-day overlay
ax = axes[1,2]
for i, (name, fc) in enumerate(all_forecasts.items()):
    if len(fc['pred']) >= 30:
        ax.plot(range(30), fc['pred'][:30], color=colors[i], lw=1.0, alpha=0.85, label=name)
ax.set_xlabel('Days ahead'); ax.set_ylabel('Forecast'); ax.set_title('(f) 30-Day Forecast')
ax.legend(fontsize=5, ncol=2, loc='best'); ax.grid(alpha=0.3)

fig.suptitle('Multi-Model Comparison: Statistical × ML × DL for Drone Daily Loss',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('drone_Model_Comparison.png', dpi=250, facecolor='white', edgecolor='none')
print("-> drone_Model_Comparison.png")

# ==============================================================
# 4. 总结报告
# ==============================================================
report = {
    'best_test_rmse': {
        'model': tbl.iloc[0]['Model'],
        'value': float(tbl.iloc[0]['Test_RMSE']) if pd.notna(tbl.iloc[0]['Test_RMSE']) else None
    },
    'best_test_mae': {
        'model': tbl.loc[tbl['Test_MAE'].idxmin(skipna=True), 'Model'] if tbl['Test_MAE'].notna().any() else 'N/A',
        'value': float(tbl['Test_MAE'].min()) if tbl['Test_MAE'].notna().any() else None
    },
    'exog_improvement': {
        'with_exog_mean_rmse': float(tbl[tbl['Exog']=='Y']['Test_RMSE'].mean()),
        'no_exog_mean_rmse':   float(tbl[tbl['Exog']=='N']['Test_RMSE'].mean()),
    },
    'gpr_improvement': {
        'with_gpr_mean_rmse': float(tbl[tbl['GPR']=='Y']['Test_RMSE'].mean()),
        'no_gpr_mean_rmse':   float(tbl[tbl['GPR']=='N']['Test_RMSE'].mean()),
    },
    'model_count': len(tbl),
    'test_days': TEST_DAYS,
    'split_date': str(SPLIT_DATE.date()),
}
print("\n" + json.dumps(report, indent=2, ensure_ascii=False))

plt.show()
print("\nDone.")
