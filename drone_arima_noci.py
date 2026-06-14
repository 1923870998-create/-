import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略网格搜索过程中产生的大量收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础配置与数据加载
# ==========================================
file_path = r"E:\drone.csv"
print(f"[*] 正在加载数据: {file_path}")

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

target_col = df.columns[-1]
df[target_col] = df[target_col].apply(
    lambda x: float(str(x).replace(',', '').replace('"', '').strip()) if isinstance(x, str) else float(x)
)
y = df[target_col].fillna(0)
print(f"[*] 自动识别到预测目标列: [{target_col}]\n")

# ==========================================
# 2. 对差分结果进行 ADF 单位根检验
# ==========================================
print("--- 【ADF 检验: 一阶差分序列】 ---")
y_diff = y.diff().dropna()
adf_result = adfuller(y_diff)

print(f"ADF 统计量: {adf_result[0]:.4f}")
print(f"P-value (P值): {adf_result[1]:.4e}")
d = 2  # 固定二阶差分, 更好捕捉长期增长趋势
print(f"结论: 固定 d={d} (二阶差分, 适应强趋势战争数据)。\n")

# ==========================================
# 3. 固定 ARIMA(0, 2, 3) 参数拟合
# ==========================================
best_order = (0, d, 3)
print(f"[*] 使用固定参数 ARIMA{best_order} 进行拟合...")
best_model = ARIMA(y, order=best_order).fit()
best_bic = best_model.bic
print(f"[*] 拟合完成！BIC 值为: {best_bic:.2f}\n")

# ==========================================
# 4. 获取历史拟合与未来预测 (无置信区间，因为不含GPR)
# ==========================================
future_steps = 30

# 4.1 未来预测
forecast_result = best_model.get_forecast(steps=future_steps)
future_forecast = forecast_result.predicted_mean

# 4.2 历史拟合
historical_result = best_model.get_prediction(start=y.index[0], end=y.index[-1], dynamic=False)
historical_fit = historical_result.predicted_mean

# ==========================================
# 5. 绘制主图 (无置信区间)
# ==========================================
plt.figure(figsize=(15, 8))

# 画出实际数据
plt.plot(y.index, y, label='Actual Data', color='royalblue', alpha=0.5, linewidth=1.5)

# 画出历史拟合
plt.plot(historical_fit.index, historical_fit, label='ARIMA Historical Fit', color='darkorange', linewidth=1.5, alpha=0.9)

# 画出未来预测
plt.plot(future_forecast.index, future_forecast, label='Future Forecast', color='crimson', linewidth=2, linestyle='--')

# 分界线
plt.axvline(x=y.index[-1], color='gray', linestyle=':', label='Forecast Start Point')

# 图表装饰
plt.title(f'ARIMA {best_order} Full Cycle Analysis & Forecast', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Loss', fontsize=12)
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)

# 限制Y轴下限不为负，因为装备损耗不能是负数
plt.ylim(bottom=0)
plt.tight_layout()

# 保存图表
output_filename = f"drone_ARIMA_Forecast_{target_col}.png"
plt.savefig(output_filename, dpi=300)
print(f"[*] 图表绘制完毕，已保存至当前目录: {output_filename}")

# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(y.index) + list(future_forecast.index),
    'actual': list(y.values) + [np.nan]*len(future_forecast),
    'predicted': list(historical_fit.values) + list(future_forecast.values)
})
csv_df.to_csv('pred_drone_ARIMA_NoCI.csv', index=False)
print(f"[*] -> pred_drone_ARIMA_NoCI.csv saved")

plt.show()
