import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 数据加载与时间轴映射
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
y_raw = df[target_col].fillna(0).values.reshape(-1, 1)
print(f"[*] 自动识别到预测目标列: [{target_col}]\n")

# 时间序列本质是 1D 数据，我们将时间转换为 [0, 1] 区间的连续坐标 X
n_samples = len(y_raw)
X_train = np.arange(n_samples).reshape(-1, 1) / n_samples

# ==========================================
# 2. 目标变量的标准化 (纯 GPR 成功的绝对关键)
# ==========================================
# 将损耗数据缩放到均值为 0，方差为 1 的标准正态分布
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y_raw)

# ==========================================
# 3. 构建复合核函数 (Composite Kernel)
# ==========================================
print("[*] 正在构建复合核函数并训练 GPR (这可能需要 1~2 分钟，请耐心等待)...")

# 1. RBF 核: 负责捕捉长期的、平滑的宏观战争趋势
kernel_trend = ConstantKernel(1.0) * RBF(length_scale=0.5)

# 2. Matern 核 (nu=1.5): 负责捕捉粗糙的、突发的局部战役脉冲
kernel_irregular = ConstantKernel(1.0) * Matern(length_scale=0.05, nu=1.5)

# 3. WhiteKernel: 吸收纯随机的白噪声，防止模型死记硬背（过拟合）
kernel_noise = WhiteKernel(noise_level=0.1)

# 将三者相加，使得 GPR 同时具备看长线、抓突变和抗干扰的能力
composite_kernel = kernel_trend + kernel_irregular + kernel_noise

# 实例化 GPR 模型
gpr = GaussianProcessRegressor(kernel=composite_kernel, n_restarts_optimizer=5, alpha=0.0)

# 训练模型 (这一步由于时间复杂度 O(N^3)，数据量大时会有稍许计算延迟)
gpr.fit(X_train, y_scaled)
print(f"[*] GPR 优化完毕！最终核函数参数: \n{gpr.kernel_}\n")

# ==========================================
# 4. 预测历史拟合与未来走势
# ==========================================
future_steps = 30
# 构造未来的时间坐标 X
X_future = np.arange(n_samples, n_samples + future_steps).reshape(-1, 1) / n_samples
X_all = np.vstack([X_train, X_future])

# 一次性预测全周期（包含历史和未来），返回均值和标准差
y_pred_scaled, sigma_scaled = gpr.predict(X_all, return_std=True)

# 将缩放后的预测均值和标准差【反向还原】到真实的战损数量级
y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
# 标准差的还原只需要乘以缩放器的标准差比例 (scaler.scale_)
sigma_real = sigma_scaled * scaler.scale_[0]

# 分离历史和未来部分
hist_fit = y_pred_real[:n_samples]
future_forecast = y_pred_real[n_samples:]

hist_sigma = sigma_real[:n_samples]
future_sigma = sigma_real[n_samples:]

# ==========================================
# 5. 可视化
# ==========================================
plt.figure(figsize=(15, 8))

# 生成完整的日期索引
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps)
all_dates = df.index.append(future_dates)

# 绘制原始数据
plt.plot(df.index, y_raw, label='Actual Data', color='royalblue', alpha=0.5, linewidth=1.5)

# 绘制 GPR 历史拟合线及 99% 置信区间
plt.plot(df.index, hist_fit, label='Pure GPR Historical Fit', color='darkorange', linewidth=1.5)
plt.fill_between(df.index,
                 np.maximum(0, hist_fit - 2.326 * hist_sigma),
                 hist_fit + 2.326 * hist_sigma,
                 color='darkorange', alpha=0.2, label='99% Historical CI')

# 绘制 GPR 未来预测线及 99% 置信区间
plt.plot(future_dates, future_forecast, label='Pure GPR Future Forecast', color='crimson', linewidth=2, linestyle='--')
plt.fill_between(future_dates,
                 np.maximum(0, future_forecast - 2.326 * future_sigma),
                 future_forecast + 2.326 * future_sigma,
                 color='crimson', alpha=0.2, label='99% Future CI')

# 纵向分割线
plt.axvline(x=df.index[-1], color='black', linestyle=':', label='Forecast Start Point', alpha=0.7)

# 图表装饰
plt.title(f'Pure GPR Time Series Forecast for [{target_col.upper()}]', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Loss', fontsize=12)
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)

# 限制 Y 轴底部为 0
plt.ylim(bottom=0)
plt.tight_layout()

# 输出保存
output_filename = f"drone_Pure_GPR_Forecast_{target_col}.png"
plt.savefig(output_filename, dpi=300)
print(f"[*] 纯 GPR 预测图表已保存至当前目录: {output_filename}")

# ---- CSV output ----
csv_df = pd.DataFrame({
    'date': list(df.index) + list(future_dates),
    'actual': list(y_raw.flatten()) + [np.nan]*len(future_dates),
    'predicted': list(hist_fit) + list(future_forecast)
})
csv_df.to_csv('pred_drone_PureGPR.csv', index=False)
print(f"[*] -> pred_drone_PureGPR.csv saved")

plt.show()