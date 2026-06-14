#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone LSTM-TCN + XGBoost 融合预测 (Pure NumPy)
================================================
参考 ml_hybrid 架构重写:
  1. 预处理: MA30 局部平台 + 对数尺度相对偏离
     MA30_t = mean(y_{t-30:t}),  目标 r_t = ln(1+y_t) - ln(1+MA30_t)

  2. LSTM-TCN 联合训练 (纯NumPy, 零框架依赖):
     - LSTM: 64隐层, 门控记忆捕捉长期依赖
     - TCN:  多尺度因果卷积 d=1,2,4,8,16
     - 融合: 拼接 → Dense → r_pred

  3. 逆变换还原: y_pred = exp(ln(1+MA30_t) + r_pred) - 1

  4. XGBoost 残差修正 (专门修正LSTM-TCN遗漏的系统偏差):
     - 特征: 基础预测, 滞后损失/残差, 滚动均值/波动, 日历, 战役烈度
     - e_xgb = XGBoost(Z_t),  y_final = max(0, y_base + e_xgb)

  5. 递推30日预测 + 统一风格可视化

数据: drone.csv + russia_ukraine_battles.csv
"""

import math, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ================================================================
# Config
# ================================================================
SEQ_LEN = 30; HIDDEN = 20; TCN_CH = 8; FUTURE = 30; SEED = 42
DILATIONS = (1, 2, 4, 8, 16); Z = 2.326  # 99% CI for future GPR (kept for compat)

ROYAL_BLUE = "#4169E1"; DARK_ORANGE = "#FF8C00"; CRIMSON = "#DC143C"
DODGER_BLUE = "#1E90FF"; GRAY_LINE = "#808080"; SOFT_GREEN = "#66BB6A"
random.seed(SEED); np.random.seed(SEED)

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 12, "axes.labelsize": 10,
                     "figure.dpi": 200, "savefig.dpi": 250, "savefig.bbox": "tight",
                     "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.facecolor": "white"})

# ================================================================
# Pure-NumPy LSTM-TCN Fusion
# ================================================================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

class LstmTcnFusion:
    """Jointly trained LSTM(dim=HIDDEN) + TCN(d=1,2,4,8,16, ch=TCN_CH) → linear head."""

    def __init__(self, in_dim, hidden=HIDDEN, tcn_ch=TCN_CH, dilations=DILATIONS,
                 lr=0.005, epochs=120, batch=32, patience=15):
        self.in_dim = in_dim; self.hidden = hidden; self.tcn_ch = tcn_ch
        self.dilations = dilations; self.lr = lr; self.epochs = epochs
        self.batch = batch; self.patience = patience
        self.rng = np.random.default_rng(SEED + 1)
        self._init()

    def _init(self):
        s_lstm = 1.0 / math.sqrt(self.in_dim + self.hidden)
        def lw(): return self.rng.normal(0, s_lstm, (self.in_dim + self.hidden, self.hidden))
        s_tcn = 1.0 / math.sqrt(self.in_dim * 3)
        self.Wf, self.Wi, self.Wo, self.Wg = lw(), lw(), lw(), lw()
        self.bf = np.zeros(self.hidden) + 0.5; self.bi = np.zeros(self.hidden)
        self.bo = np.zeros(self.hidden); self.bg = np.zeros(self.hidden)
        nd = len(self.dilations)
        self.Wtcn = self.rng.normal(0, s_tcn, (nd, 3, self.in_dim, self.tcn_ch))
        self.btcn = np.zeros((nd, self.tcn_ch))
        fdim = self.hidden + nd * self.tcn_ch
        self.Wy = self.rng.normal(0, 1.0/math.sqrt(fdim), fdim)
        self.by = 0.0

    def _forward(self, x):
        B, T, _ = x.shape
        h, c = np.zeros((B, self.hidden)), np.zeros((B, self.hidden))
        cache = {"h": [h.copy()], "f": [], "i": [], "o": [], "g": [], "c_prev": [], "c": [c.copy()]}
        for t in range(T):
            z = np.concatenate([x[:, t, :], h], axis=1)
            f = sigmoid(z @ self.Wf + self.bf); i = sigmoid(z @ self.Wi + self.bi)
            o = sigmoid(z @ self.Wo + self.bo); g = np.tanh(z @ self.Wg + self.bg)
            cp = c; c = f * cp + i * g; h = o * np.tanh(c)
            for k, v in zip(["f","i","o","g","c_prev","c","h"], [f,i,o,g,cp,c.copy(),h]):
                cache[k].append(v)
        z_tcn = np.zeros((B, len(self.dilations), self.tcn_ch)); selected = []
        for di, d in enumerate(self.dilations):
            sd = []; fmap = np.zeros((3, B, self.in_dim, self.tcn_ch))
            for k in range(3):
                idx = np.clip(T - 1 - k * d, 0, T - 1)
                xk = x[:, idx, :]; sd.append(xk)
                z_tcn[:, di, :] += xk @ self.Wtcn[di, k]
            z_tcn[:, di, :] += self.btcn[di]; selected.append(sd)
        ht = np.tanh(z_tcn).reshape(B, -1)
        fusion = np.concatenate([h, ht], axis=1)
        return fusion @ self.Wy + self.by, {"cache": cache, "selected": selected, "h": h, "ht3": np.tanh(z_tcn), "fusion": fusion}

    def _grads(self, x, y):
        pred, meta = self._forward(x); B = len(x); dy = (2.0/B) * (pred - y)
        grads = {k: np.zeros_like(getattr(self, k)) for k in
                 ["Wf","Wi","Wo","Wg","bf","bi","bo","bg","Wtcn","btcn","Wy"]}
        grads["by"] = 0.0; grads["Wy"] = meta["fusion"].T @ dy; grads["by"] += float(np.sum(dy))
        d_fusion = np.outer(dy, self.Wy)
        d_lstm = d_fusion[:, :self.hidden]
        d_tcn = d_fusion[:, self.hidden:].reshape(B, len(self.dilations), self.tcn_ch)
        cache = meta["cache"]; dh = d_lstm; dc = np.zeros_like(dh)
        for t in reversed(range(x.shape[1])):
            f = cache["f"][t]; i = cache["i"][t]; o = cache["o"][t]; g = cache["g"][t]
            _c = cache["c"][t]; cp = cache["c_prev"][t]
            z = np.concatenate([x[:, t, :], cache["h"][t]], axis=1)
            tanh_c = np.tanh(_c)
            do = dh * tanh_c; dao = do * o * (1 - o)
            dc = dh * o * (1 - tanh_c**2) + dc
            daf = (dc * cp) * f * (1 - f); dai = (dc * g) * i * (1 - i)
            dag = (dc * i) * (1 - g**2)
            grads["Wf"] += z.T @ daf; grads["Wi"] += z.T @ dai
            grads["Wo"] += z.T @ dao; grads["Wg"] += z.T @ dag
            grads["bf"] += daf.sum(0); grads["bi"] += dai.sum(0)
            grads["bo"] += dao.sum(0); grads["bg"] += dag.sum(0)
            dh = (daf @ self.Wf.T + dai @ self.Wi.T + dao @ self.Wo.T + dag @ self.Wg.T)[:, self.in_dim:]
            dc = dc * f
        ht3 = meta["ht3"]; dz = d_tcn * (1 - ht3**2)
        for di in range(len(self.dilations)):
            grads["btcn"][di] += dz[:, di, :].sum(0)
            for k in range(3):
                grads["Wtcn"][di, k] += meta["selected"][di][k].T @ dz[:, di, :]
        gnorm = math.sqrt(sum(float((g**2).sum()) for g in grads.values() if not np.isscalar(g)) + float(grads["by"])**2)
        if gnorm > 5.0: s = 5.0 / (gnorm + 1e-8)
        else: s = 1.0
        scaled = {k: v * s for k, v in grads.items()}
        return scaled, float(np.mean((pred - y)**2))

    def fit(self, x, y, xv, yv):
        m = {k: np.zeros_like(getattr(self, k)) for k in
             ["Wf","Wi","Wo","Wg","bf","bi","bo","bg","Wtcn","btcn","Wy"]}
        m["by"] = 0.0; v = {k: 0.0 for k in m}; b1, b2 = 0.9, 0.999
        best_loss = float("inf"); best_state = {}; stag = 0; step = 0
        for epoch in range(self.epochs):
            perm = self.rng.permutation(len(x))
            for start in range(0, len(perm), self.batch):
                idx = perm[start:start+self.batch]; g, _ = self._grads(x[idx], y[idx]); step += 1
                for k in m:
                    m[k] = b1 * m[k] + (1-b1) * g[k]; v[k] = b2 * v[k] + (1-b2) * (g[k]**2)
                    mh = m[k] / (1 - b1**step); vh = v[k] / (1 - b2**step)
                    setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + 1e-8))
            vl = float(np.mean((self.predict(xv) - yv)**2))
            if vl < best_loss - 1e-6: best_loss = vl; best_state = {k: getattr(self, k).copy() if hasattr(getattr(self, k), 'copy') else getattr(self, k) for k in g}; stag = 0
            else: stag += 1
            if stag >= self.patience: break
        if best_state:
            for k, val in best_state.items(): setattr(self, k, val)
        return self

    def predict(self, x):
        preds = []
        for s in range(0, len(x), 256): p, _ = self._forward(x[s:s+256]); preds.append(p)
        return np.concatenate(preds)

# ================================================================
# Data pipeline
# ================================================================
def load_and_build_features():
    """Load drone.csv + battles, build feature frame with MA30 log-transform."""
    print("=" * 70)
    print("LSTM-TCN + XGBoost (Pure NumPy) — Drone Daily Loss")
    print("=" * 70)
    print("\n[1/6] Loading data & building features...")

    drone = pd.read_csv("drone.csv")
    drone.columns = ["date", "day", "direction", "drone"]
    drone["date"] = pd.to_datetime(drone["date"]); drone = drone.set_index("date").sort_index()
    drone["drone"] = drone["drone"].apply(lambda x: float(str(x).replace(",","").replace('"',"").strip()) if isinstance(x, str) else float(x))
    drone.loc[drone["drone"] < 0, "drone"] = 0
    y_raw = drone["drone"].fillna(0); dates = y_raw.index; n = len(y_raw)

    # Battle intensity
    battles = pd.read_csv("russia_ukraine_battles.csv")
    battles["start_date"] = pd.to_datetime(battles["start_date"])
    battles["end_date"] = battles.apply(lambda r: r["start_date"] + pd.Timedelta(days=int(r["duration_days"])), axis=1)
    battle_I = pd.Series(0.0, index=dates)
    for _, row in battles.iterrows():
        mask = (dates >= row["start_date"]) & (dates <= row["end_date"])
        if mask.sum() > 0: battle_I.loc[mask] += row["daily_intensity"]  # 重叠求和

    # Min-Max 归一化 → [0, 1]
    bi_min, bi_max = battle_I.min(), battle_I.max()
    battle_I = (battle_I - bi_min) / (bi_max - bi_min + 1e-8)

    # Dual rolling-mean smooth battle intensity
    battle_I_smooth = battle_I.rolling(window=21, center=True, min_periods=1).mean()
    battle_I_fitted = battle_I_smooth.rolling(window=11, center=True, min_periods=1).mean()

    y = y_raw.values.astype(float)
    # MA30
    ma30 = np.full(n, np.nan)
    for t in range(30, n): ma30[t] = y[t-30:t].mean()
    ma30[:30] = y[:30].mean()

    # Log-transform: r_t = ln(1+y_t) - ln(1+MA30_t)
    log_y = np.log1p(y); log_ma = np.log1p(ma30)
    r_target = log_y - log_ma

    # Build feature frame
    frame = pd.DataFrame(index=dates)
    frame["target"] = y; frame["y_log"] = log_y
    frame["ma30_log"] = log_ma; frame["resid_log"] = r_target
    frame["drone_diff1_log"] = np.diff(log_y, prepend=log_y[0])
    frame["drone_zero_lag1"] = (y_raw.shift(1).fillna(0) <= 0).astype(float).values
    for w in (7, 14, 30):
        roll = y_raw.shift(1).rolling(w, min_periods=2)
        frame[f"roll{w}_mean_log"] = np.log1p(roll.mean().fillna(y_raw.shift(1).expanding().mean()).fillna(y_raw.mean())).values
    frame["roll7_std_log"] = np.log1p(y_raw.shift(1).rolling(7, min_periods=2).std().fillna(0)).values
    dow = dates.dayofweek; month = dates.month; ord_d = (dates - dates[0]).days.astype(float)
    frame["dow_sin"] = np.sin(2*np.pi*dow/7); frame["dow_cos"] = np.cos(2*np.pi*dow/7)
    frame["month_sin"] = np.sin(2*np.pi*month/12); frame["month_cos"] = np.cos(2*np.pi*month/12)
    frame["time_idx"] = ord_d / max(1, ord_d.max())
    frame["battle_intensity"] = battle_I.values
    frame = frame.fillna(0)

    seq_cols = ["y_log", "ma30_log", "resid_log", "drone_diff1_log", "drone_zero_lag1",
                "roll7_mean_log", "roll14_mean_log", "roll30_mean_log", "roll7_std_log",
                "dow_sin", "dow_cos", "month_sin", "month_cos", "time_idx", "battle_intensity"]

    print(f"  Data: {n} days [{str(dates[0].date())} → {str(dates[-1].date())}]")
    print(f"  Target mean={y.mean():.1f}  max={y.max():.0f}   features={len(seq_cols)}")
    return frame, seq_cols, y, dates, battle_I, battle_I_fitted, n

def make_sequences(frame, seq_cols, x_scaler, y_scaler):
    xv = x_scaler.transform(frame[seq_cols].values.astype(float))
    ys = y_scaler.transform(frame[["resid_log"]].values.astype(float)).ravel()
    xs, ys2, dts, ry, bl = [], [], [], [], []
    for end in range(SEQ_LEN, len(frame)):
        xs.append(xv[end-SEQ_LEN:end]); ys2.append(ys[end])
        dts.append(frame.index[end]); ry.append(frame["target"].iloc[end])
        bl.append(frame["ma30_log"].iloc[end])
    return {"X": np.array(xs, dtype=float), "y": np.array(ys2, dtype=float),
            "date": np.array(dts), "y_raw": np.array(ry, dtype=float),
            "baseline_log": np.array(bl, dtype=float)}

def inverse_pred(pred_scaled, y_scaler, baseline_log):
    pr = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(np.expm1(baseline_log + pr), 0, None)

# ================================================================
# XGBoost residual features
# ================================================================
def build_xgb_features(frame, y_base_all, e_all, battle_I, start_idx, end_idx):
    """y_base_all[i] aligns to frame index i+start_idx. Build features for indices [start_idx, end_idx)."""
    y = frame["target"].values; offset = start_idx; rows = []
    for t in range(start_idx, end_idx):
        row = {"base_pred": y_base_all[t - offset], "y_lag1": y[t-1] if t>0 else 0,
               "y_lag7": y[t-7] if t>=7 else 0, "e_lag1": e_all[t-offset-1] if t>offset else 0,
               "e_lag7": e_all[t-offset-7] if t>=offset+7 else 0}
        for w in (7, 30):
            st = max(0, t-w); row[f"roll_mean_{w}"] = np.mean(y[st:t])
            row[f"roll_std_{w}"] = np.std(y[st:t]) if t-st>1 else 0
        row["battle_intensity"] = battle_I.iloc[t]
        dt = frame.index[t]; row["doy"] = dt.dayofyear; row["month"] = dt.month
        rows.append(row)
    return pd.DataFrame(rows)

# ================================================================
# Recursive forecast — builds full 15-dim feature rows each step
# ================================================================
def recursive_forecast(model, xgb_model, frame, seq_cols, x_scaler, y_scaler,
                       xgb_cols, battle_I, n, horizon=30):
    y_vals = frame["target"].values; dates = frame.index
    last_I = battle_I.iloc[-1]
    # Rolling feature buffer
    y_buf = y_vals[-30:].tolist()
    ma30_val = np.mean(y_buf)
    # Last feature window (SEQ_LEN x 15)
    feat_win = frame[seq_cols].values[-SEQ_LEN:].copy()
    y_scaler_mean = y_scaler.mean_[0]; y_scaler_std = y_scaler.scale_[0]

    fut_base, fut_final = [], []

    for step in range(horizon):
        # --- Build next feature row ---
        # Calendar for future date
        fut_dt = dates[-1] + pd.Timedelta(days=step + 1)
        dow = fut_dt.dayofweek; mth = fut_dt.month
        od = (fut_dt - dates[0]).days / max(1, (dates[-1] - dates[0]).days)

        # Log terms (use last known MA30 + placeholder y)
        y_est = ma30_val  # initial estimate
        log_y_est = np.log1p(y_est); log_ma = np.log1p(ma30_val)
        resid_est = log_y_est - log_ma

        # Rolling features from buffer
        def roll_mean(buf, w): return np.mean(buf[-w:]) if len(buf) >= w else np.mean(buf)
        def roll_std(buf, w): return np.std(buf[-w:]) if len(buf) >= max(w, 2) else 0.0
        diff_log = log_y_est - np.log1p(y_buf[-1]) if y_buf[-1] > 0 else 0.0

        next_feat = np.array([
            log_y_est,                   # y_log
            log_ma,                       # ma30_log
            resid_est,                    # resid_log
            diff_log,                     # drone_diff1_log
            0.0 if y_buf[-1] > 0 else 1.0,  # drone_zero_lag1
            np.log1p(roll_mean(y_buf, 7)),   # roll7_mean_log
            np.log1p(roll_mean(y_buf, 14)),  # roll14_mean_log
            np.log1p(roll_mean(y_buf, 30)),  # roll30_mean_log
            np.log1p(roll_std(y_buf, 7)),     # roll7_std_log
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),  # dow
            np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),  # month
            od,                            # time_idx
            last_I,                        # battle_intensity
        ])

        # --- LSTM-TCN predict ---
        # Roll window: drop oldest, append next_feat
        feat_win = np.vstack([feat_win[1:], next_feat.reshape(1, -1)])
        feat_win_scaled = x_scaler.transform(feat_win)
        inp = feat_win_scaled.reshape(1, SEQ_LEN, -1)
        r_scaled = float(model.predict(inp)[0])
        r_pred = r_scaled * y_scaler_std + y_scaler_mean  # inverse scale resid_log

        # Inverse transform to y
        y_base = np.expm1(log_ma + r_pred)

        # --- XGBoost correct ---
        xgb_row = pd.DataFrame([{
            "base_pred": y_base,
            "y_lag1": y_buf[-1],
            "y_lag7": y_buf[-7] if len(y_buf) >= 7 else y_buf[-1],
            "e_lag1": 0, "e_lag7": 0,
            "roll_mean_7": roll_mean(y_buf, 7),
            "roll_mean_30": ma30_val,
            "roll_std_7": roll_std(y_buf, 7),
            "roll_std_30": roll_std(y_buf, 30),
            "battle_intensity": last_I,
            "doy": fut_dt.dayofyear, "month": mth,
        }])
        e_next = float(xgb_model.predict(xgb_row[xgb_cols])[0])
        y_final = max(0.0, y_base + e_next)

        fut_base.append(y_base); fut_final.append(y_final)

        # Update buffers
        y_buf.append(y_final); y_buf = y_buf[-30:]
        ma30_val = np.mean(y_buf)
        # Update feature window's last row with actual y_final
        feat_win[-1, 0] = np.log1p(y_final)        # y_log
        feat_win[-1, 1] = np.log1p(ma30_val)        # ma30_log
        feat_win[-1, 2] = feat_win[-1, 0] - feat_win[-1, 1]  # resid_log

    return np.array(fut_base), np.array(fut_final)

# ================================================================
# Plotting
# ================================================================
def plot_regression(dates, y, hist_dates, y_base, y_final, fut_dates, fut_final,
                    rmse_b, rmse_f, mae_f):
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.plot(dates, y, color=ROYAL_BLUE, alpha=0.4, lw=0.7, label="Actual Drone Loss", zorder=2)
    ax.plot(hist_dates, y_base, color=DODGER_BLUE, alpha=0.55, ls="--", lw=1.0, label="LSTM-TCN Base", zorder=2)
    ax.plot(hist_dates, y_final, color=DARK_ORANGE, lw=1.3, label="LSTM-TCN+XGBoost Final", zorder=3)
    ax.plot(fut_dates, fut_final, color=CRIMSON, lw=2.2, ls="--", label=f"Forecast ({FUTURE}d)", zorder=4)
    ax.axvline(x=dates[-1], color=GRAY_LINE, ls=":", lw=1.0, alpha=0.6)
    ax.set_ylabel("Daily Drone Loss", fontsize=12)
    ax.set_title(f"LSTM-TCN + XGBoost  |  Drone Daily Loss  |  "
                 f"RMSE: {rmse_b:.1f}→{rmse_f:.1f}  MAE={mae_f:.1f}  Forecast={FUTURE}d",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(True, ls="--", alpha=0.35, color="#cccccc"); ax.set_ylim(bottom=-20)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(pad=2)
    fig.savefig("drone_LSTM_TCN_XGB_Regression.png", dpi=250, facecolor="white", edgecolor="none")
    print("  -> drone_LSTM_TCN_XGB_Regression.png"); plt.close(fig)

def plot_overlay(dates, y, hist_dates, y_base, y_final, fut_dates, fut_final,
                 battle_I_fitted, rmse_b, rmse_f, mae_f):
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax.plot(dates, y, color=ROYAL_BLUE, alpha=0.4, lw=0.7, label="Actual Drone Loss", zorder=2)
    ax.plot(hist_dates, y_base, color=DODGER_BLUE, alpha=0.55, ls="--", lw=1.0, label="LSTM-TCN Base", zorder=2)
    ax.plot(hist_dates, y_final, color=DARK_ORANGE, lw=1.3, label="LSTM-TCN+XGBoost Final", zorder=3)
    ax.plot(fut_dates, fut_final, color=CRIMSON, lw=2.2, ls="--", label=f"Forecast ({FUTURE}d)", zorder=4)
    ax.axvline(x=dates[-1], color=GRAY_LINE, ls=":", lw=1.0, alpha=0.6)
    ax.set_ylabel("Daily Drone Loss", fontsize=12)
    ax.set_title(f"LSTM-TCN+XGBoost  |  Left=Loss  Right=Intensity  |  "
                 f"RMSE: {rmse_b:.1f}→{rmse_f:.1f}  MAE={mae_f:.1f}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(True, ls="--", alpha=0.35, color="#cccccc"); ax.set_ylim(bottom=-20)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    # Right axis: ARIMA-fitted battle intensity (soft green)
    ax_r = ax.twinx(); ax_r.set_ylabel("Battle Intensity (Smoothed)", fontsize=11, color=SOFT_GREEN)
    ax_r.plot(battle_I_fitted.index, battle_I_fitted.values, color=SOFT_GREEN, lw=1.2, alpha=0.9, zorder=5)
    ax_r.set_ylim(0, 1.05); ax_r.tick_params(axis="y", colors=SOFT_GREEN)
    for s in ["top", "left"]: ax_r.spines[s].set_visible(False)
    # Lower: residuals
    ax2 = fig.add_subplot(gs[1])
    resid = y[SEQ_LEN:] - y_final
    ax2.plot(hist_dates, resid, color="gray", alpha=0.45, lw=0.5, label=f"Final Residuals (RMSE={rmse_f:.2f})")
    ax2.axhline(y=0, color="black", ls="-", alpha=0.3)
    ax2.set_ylabel("Residual", fontsize=11); ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("LSTM-TCN+XGBoost Final Residuals", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(True, ls="--", alpha=0.35, color="#cccccc")
    for s in ["top", "right"]: ax2.spines[s].set_visible(False)
    fig.tight_layout(pad=2)
    fig.savefig("drone_LSTM_TCN_XGB_Overlay.png", dpi=250, facecolor="white", edgecolor="none")
    print("  -> drone_LSTM_TCN_XGB_Overlay.png"); plt.close(fig)

# ================================================================
# Main
# ================================================================
def main():
    frame, seq_cols, y, dates, battle_I, battle_I_fitted, n = load_and_build_features()
    train_end = n - 30

    # ---- Scale & build sequences ----
    print("\n[2/6] Building sequences...")
    x_scaler = StandardScaler(); y_scaler = StandardScaler()
    x_scaler.fit(frame[seq_cols].iloc[:train_end].values.astype(float))
    y_scaler.fit(frame[["resid_log"]].iloc[:train_end].values.astype(float))
    seq = make_sequences(frame, seq_cols, x_scaler, y_scaler)

    X_all, y_all = seq["X"], seq["y"]
    y_raw_all = seq["y_raw"]; bl_all = seq["baseline_log"]
    seq_dates = pd.DatetimeIndex(seq["date"])

    train_mask = seq_dates < dates[train_end]
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    # Val = last 15% of training
    n_val = max(24, int(len(X_tr) * 0.85))
    X_t, y_t = X_tr[:n_val], y_tr[:n_val]
    X_v, y_v = X_tr[n_val:], y_tr[n_val:]

    # ---- LSTM-TCN fusion ----
    print(f"\n[3/6] Training LSTM-TCN fusion ({len(X_t)} train / {len(X_v)} val)...")
    model = LstmTcnFusion(in_dim=len(seq_cols))
    model.fit(X_t, y_t, X_v, y_v)

    # Base predictions (full history)
    y_pred_scaled = model.predict(X_all)
    y_base_all = inverse_pred(y_pred_scaled, y_scaler, bl_all)

    # RMSE base (train only)
    rmse_b = float(np.sqrt(np.mean((y_raw_all[train_mask] - y_base_all[train_mask])**2)))
    print(f"  LSTM-TCN base RMSE (train): {rmse_b:.2f}")

    # ---- XGBoost residual ----
    print("\n[4/6] Training XGBoost residual corrector...")
    e_all = y_raw_all - y_base_all
    # y_base_all/e_all aligned to frame indices [SEQ_LEN : n]
    X_xgb = build_xgb_features(frame, y_base_all, e_all, battle_I, SEQ_LEN, n)
    xgb_cols = list(X_xgb.columns)
    # Training mask for XGBoost: align with train_end
    train_end_seq = train_end - SEQ_LEN
    X_xgb_tr = X_xgb.iloc[:train_end_seq]
    e_tr = e_all[:train_end_seq]

    xgb_model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.03,
                             subsample=0.85, colsample_bytree=0.8, reg_lambda=5.0,
                             random_state=SEED, verbosity=0)
    xgb_model.fit(X_xgb_tr, e_tr)

    e_xgb_pred = xgb_model.predict(X_xgb)
    y_final_all = np.clip(y_base_all + e_xgb_pred, 0, None)

    rmse_f = float(np.sqrt(np.mean((y_raw_all[train_mask] - y_final_all[train_mask])**2)))
    mae_f = float(np.mean(np.abs(y_raw_all[train_mask] - y_final_all[train_mask])))
    print(f"  +XGBoost RMSE: {rmse_f:.2f}  MAE: {mae_f:.2f}  ({'↓' if rmse_f<rmse_b else '↑'}{abs(rmse_b-rmse_f)/rmse_b*100:.1f}%)")

    # ---- Recursive forecast ----
    print(f"\n[5/6] Recursive {FUTURE}-day forecast...")
    # Get last r_t sequence for recursive prediction
    fut_base, fut_final = recursive_forecast(model, xgb_model, frame, seq_cols,
                                             x_scaler, y_scaler, xgb_cols,
                                             battle_I, n, FUTURE)
    fut_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=FUTURE)
    print(f"  Forecast: {fut_final[0]:.0f} → {fut_final[-1]:.0f}")

    # ---- Plot ----
    print("\n[6/6] Drawing figures...")
    hist_dates = dates[SEQ_LEN:]
    plot_regression(dates, y, hist_dates, y_base_all, y_final_all,
                    fut_dates, fut_final, rmse_b, rmse_f, mae_f)
    plot_overlay(dates, y, hist_dates, y_base_all, y_final_all,
                 fut_dates, fut_final, battle_I_fitted, rmse_b, rmse_f, mae_f)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("LSTM-TCN + XGBoost SUMMARY")
    print("=" * 70)
    print(f"  Data: {n} obs  |  LOOKBACK={SEQ_LEN}d  |  Forecast={FUTURE}d")
    print(f"  Model: LSTM({HIDDEN}) + TCN(d=1,2,4,8,16, ch={TCN_CH}) + XGBoost(150 trees)")
    print(f"  RMSE: base={rmse_b:.2f} → +XGBoost={rmse_f:.2f}")
    print(f"  MAE:  {mae_f:.2f}")
    print(f"  Forecast: {fut_final[0]:.0f} → {fut_final[-1]:.0f}")
    print("  Output: drone_LSTM_TCN_XGB_Regression.png, drone_LSTM_TCN_XGB_Overlay.png")
    # ---- CSV output ----
    csv_df = pd.DataFrame({
        'date': list(hist_dates) + list(fut_dates),
        'actual': list(y[SEQ_LEN:]) + [np.nan]*len(fut_dates),
        'predicted': list(y_final_all) + list(fut_final)
    })
    csv_df.to_csv('pred_drone_LSTM_TCN_XGB.csv', index=False)
    print("  -> pred_drone_LSTM_TCN_XGB.csv saved")
    print("Done.")

if __name__ == "__main__":
    main()
