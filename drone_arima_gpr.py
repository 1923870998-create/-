import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings

# 忽略收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 数据加载与预处理
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
print(f"[*] 预测目标: [{target_col}]\n")

# ==========================================
# 2. 阶段一：ARIMA 线性基线拟合
# ==========================================
print("[*] 阶段一：拟合 ARIMA 线性基线...")
# ARIMA(0,2,3) 固定参数
arima_order = (0, 2, 3)
arima_model = ARIMA(y, order=arima_order, trend='n')
arima_fitted = arima_model.fit()

# 获取 ARIMA 历史拟合值
# 注意：差分模型的前 d 个拟合值可能存在巨大偏差，将其强制对齐原始数据以避免污染后续残差
arima_hist_fit = arima_fitted.predict(start=y.index[0], end=y.index[-1], dynamic=False)
arima_hist_fit.iloc[:arima_order[1]] = y.iloc[:arima_order[1]]

# 获取 ARIMA 未来 30 天的基线预测
future_steps = 30
arima_forecast = arima_fitted.forecast(steps=future_steps)

# ==========================================
# 3. 提取非线性残差
# ==========================================
# 核心逻辑：真实的损耗 - ARIMA认为的损耗 = 突发事件引发的非线性残差
residuals = y - arima_hist_fit

# ==========================================
# 4. 阶段二：GPR 拟合残差
# ==========================================
print("[*] 阶段二：使用 GPR (Matern 核) 拟合非线性残差...")

# 构建 GPR 所需的时间特征矩阵 X，并进行 [0, 1] 归一化（极其重要！）
n_samples = len(y)
X_train = np.arange(n_samples).reshape(-1, 1) / n_samples

# 构建未来的时间特征矩阵
X_future = np.arange(n_samples, n_samples + future_steps).reshape(-1, 1) / n_samples

# 定义核函数 (Kernel)
# ConstantKernel: 调节整体振幅
# Matern (nu=1.5): 允许突变的粗糙核函数，完美契合战争突发脉冲
# WhiteKernel: 吸收纯粹的无意义白噪声，防止 GPR 严重过拟合
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.1, nu=1.5) + \
         WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))

# 初始化并训练 GPR
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
gpr.fit(X_train, residuals.values)

print(f"[*] GPR 优化后的核函数: {gpr.kernel_}")

# 预测历史残差的拟合，并向外预测未来 30 天残差的走势及不确定性 (std)
gpr_hist_fit, gpr_hist_std = gpr.predict(X_train, return_std=True)
gpr_future_forecast, gpr_future_std = gpr.predict(X_future, return_std=True)

# ==========================================
# 5. 阶段三：混合模型的重组 (ARIMA + GPR)
# ==========================================
print("[*] 阶段三：合并双模型输出，生成最终混合预测与不确定性带...")

# 最终拟合 = 线性基线 + 非线性残差拟合
final_hist_fit = arima_hist_fit + gpr_hist_fit
final_future_forecast = arima_forecast + gpr_future_forecast

# 计算未来预测的 99% 置信区间 (2.326 个标准差)
# 战争数据损耗不能为负数，因此下边界使用 np.maximum 兜底
lower_bound = np.maximum(0, final_future_forecast - 2.326 * gpr_future_std)
upper_bound = final_future_forecast + 2.326 * gpr_future_std

# 计算历史拟合的 99% 置信区间
hist_lower = np.maximum(0, final_hist_fit - 2.326 * gpr_hist_std)
hist_upper = final_hist_fit + 2.326 * gpr_hist_std

# ==========================================
# 6. 数据可视化
# ==========================================
plt.figure(figsize=(15, 8))

# 6.1 绘制历史区间
plt.plot(y.index, y, label='Actual Data', color='gray', alpha=0.4, linewidth=1.5)
plt.plot(y.index, arima_hist_fit, label='ARIMA Linear Baseline', color='dodgerblue', alpha=0.6, linestyle='--')
plt.plot(y.index, final_hist_fit, label='Hybrid Model (ARIMA + GPR) Fit', color='darkorange', linewidth=1.5)

# 历史 99% 置信区间 (GPR)
plt.fill_between(y.index, hist_lower, hist_upper, color='darkorange', alpha=0.18, zorder=1, label='99% Historical CI (GPR)')

# 6.2 绘制未来区间
plt.plot(arima_forecast.index, final_future_forecast, label='Hybrid Future Forecast', color='crimson', linewidth=2)

# 绘制 GPR 提供的未来不确定性带 (99% CI)
plt.fill_between(arima_forecast.index, lower_bound, upper_bound, color='crimson', alpha=0.22, zorder=1, label='99% Future CI (GPR)')

# 纵向分割线
plt.axvline(x=y.index[-1], color='black', linestyle=':', label='Forecast Start', alpha=0.5)

plt.title(f'ARIMA-GPR Hybrid Time Series Forecast for [{target_col.upper()}]', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Loss', fontsize=12)
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_filename = f"drone_Hybrid_ARIMA_GPR_{target_col}.png"
plt.savefig(output_filename, dpi=300)
print(f"[*] 混合模型预测图表已保存: {output_filename}")

# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(y.index) + list(arima_forecast.index),
    'actual': list(y.values) + [np.nan]*len(arima_forecast),
    'predicted': list(final_hist_fit.values) + list(final_future_forecast.values)
})
csv_df.to_csv('pred_drone_ARIMA_GPR.csv', index=False)
print(f"[*] -> pred_drone_ARIMA_GPR.csv saved")

plt.show()