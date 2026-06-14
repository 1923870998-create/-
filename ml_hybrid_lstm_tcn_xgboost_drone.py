#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM-TCN fusion with XGBoost residual correction — drone target.

Target: ``drone_loss`` from ``campaign_feature_wide_by_date.csv``.
External regressor: battle intensity built from ``russia_ukraine_battles.csv``
(replaces per-campaign dummy columns).

Output (original preserved under ml_residual_outputs/):
  figs + tables + manifest

Additional PNG (shulitongji style):
  drone_ML_Regression.png       — non-overlay regression curve
  drone_ML_Overlay.png          — overlay with battle-intensity right axis
"""

from __future__ import annotations

import json, math, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
CSV_PATH = PROJECT_DIR / "campaign_feature_wide_by_date.csv"
BATTLES_CSV = PROJECT_DIR / "russia_ukraine_battles.csv"
OUT_DIR = BASE_DIR / "ml_residual_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MPL_DIR = OUT_DIR / ".mplconfig"
for path in (OUT_DIR, FIG_DIR, TABLE_DIR, MPL_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR / "xdg-cache"))

import matplotlib; matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEQ_LEN = 30; EVAL_DAYS = 8; FORECAST_HORIZON = 30
SEED = 20260611
random.seed(SEED); np.random.seed(SEED)

ROYAL_BLUE = "#4169E1"; DARK_ORANGE = "#FF8C00"; CRIMSON = "#DC143C"
DODGER_BLUE = "#1E90FF"; GRAY_LINE = "#808080"; GRID_GRAY = "#CCCCCC"
TEXT_DARK = "#333333"
BLUE_TO_RED = LinearSegmentedColormap.from_list("BlueRed",
    ["#D6E4F0","#6BAED6","#3182BD","#C9898F","#E05554","#B71C1C"])

CH_FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")
if CH_FONT_PATH.exists():
    CH_FONT = FontProperties(fname=str(CH_FONT_PATH))
    plt.rcParams["font.family"] = CH_FONT.get_name()
plt.rcParams.update({"font.size":9.5,"axes.titlesize":12,"axes.labelsize":10,
                     "figure.dpi":200,"savefig.dpi":250,"savefig.bbox":"tight",
                     "axes.unicode_minus":False,"figure.facecolor":"white"})

# ================================================================
# Battle intensity builder
# ================================================================
_battle_intensity_cache = None

def get_battle_intensity(date_index: pd.DatetimeIndex) -> pd.Series:
    global _battle_intensity_cache
    if _battle_intensity_cache is not None and len(_battle_intensity_cache) == len(date_index):
        return _battle_intensity_cache
    battles = pd.read_csv(BATTLES_CSV)
    battles["start_date"] = pd.to_datetime(battles["start_date"])
    battles["end_date"] = battles.apply(
        lambda r: r["start_date"] + pd.Timedelta(days=int(r["duration_days"])), axis=1)
    bi = pd.Series(0.0, index=date_index)
    for _, row in battles.iterrows():
        mask = (date_index >= row["start_date"]) & (date_index <= row["end_date"])
        if mask.sum() > 0:
            bi.loc[mask] = np.maximum(bi.loc[mask], row["daily_intensity"])
    _battle_intensity_cache = bi
    return bi

@dataclass
class Scaler:
    mean: np.ndarray; std: np.ndarray
    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        mean = np.nanmean(values, axis=0); std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std); return cls(mean=mean, std=std)
    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std
    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40); return 1.0 / (1.0 + np.exp(-x))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true-y_pred)**2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true-y_pred)))
def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0*np.abs(y_true-y_pred)/denom))
def coverage(y_true, lower, upper):
    return float(np.mean((y_true>=lower)&(y_true<=upper)))

# ================================================================
# NumpyLSTMRegressor — unchanged from original
# ================================================================
class NumpyLSTMRegressor:
    def __init__(self, input_dim: int, hidden_dim: int = 10, lr: float = 0.007,
                 epochs: int = 135, batch_size: int = 32, patience: int = 18,
                 seed: int = SEED) -> None:
        self.input_dim=input_dim; self.hidden_dim=hidden_dim; self.lr=lr
        self.epochs=epochs; self.batch_size=batch_size; self.patience=patience
        self.rng=np.random.default_rng(seed); self.params=self._init_params()
    def _init_params(self):
        in_dim=self.input_dim+self.hidden_dim; scale=1.0/math.sqrt(in_dim)
        def w(): return self.rng.normal(0.0, scale, size=(in_dim, self.hidden_dim))
        params={"Wf":w(),"Wi":w(),"Wo":w(),"Wg":w(),
                "bf":np.zeros(self.hidden_dim),"bi":np.zeros(self.hidden_dim),
                "bo":np.zeros(self.hidden_dim),"bg":np.zeros(self.hidden_dim),
                "Wy":self.rng.normal(0.0,1.0/math.sqrt(self.hidden_dim),size=(self.hidden_dim,)),
                "by":np.zeros(1)}
        params["bf"]+=0.5; return params
    def _forward(self, x):
        batch, seq_len, _ = x.shape
        h=np.zeros((batch,self.hidden_dim)); c=np.zeros((batch,self.hidden_dim))
        cache={"concat":[],"f":[],"i":[],"o":[],"g":[],"c":[],"c_prev":[],"h":[]}
        for t in range(seq_len):
            xt=x[:,t,:]; concat=np.concatenate([xt,h],axis=1)
            f=sigmoid(concat@self.params["Wf"]+self.params["bf"])
            i=sigmoid(concat@self.params["Wi"]+self.params["bi"])
            o=sigmoid(concat@self.params["Wo"]+self.params["bo"])
            g=np.tanh(concat@self.params["Wg"]+self.params["bg"])
            c_prev=c; c=f*c_prev+i*g; h=o*np.tanh(c)
            for k,v in zip(["concat","f","i","o","g","c_prev","c","h"],
                           [concat,f,i,o,g,c_prev,c,h]): cache[k].append(v)
        y_hat=h@self.params["Wy"]+self.params["by"][0]
        return y_hat, cache
    def _loss_and_grads(self, x, y):
        y_hat, cache = self._forward(x); batch=x.shape[0]; diff=y_hat-y
        loss=float(np.mean(diff**2)); dy=(2.0/batch)*diff
        grads={n:np.zeros_like(v) for n,v in self.params.items()}
        grads["Wy"]+=cache["h"][-1].T@dy; grads["by"]+=np.array([np.sum(dy)])
        dh_next=np.outer(dy,self.params["Wy"]); dc_next=np.zeros_like(dh_next)
        for t in reversed(range(x.shape[1])):
            f=cache["f"][t]; i=cache["i"][t]; o=cache["o"][t]; g=cache["g"][t]
            c=cache["c"][t]; c_prev=cache["c_prev"][t]; concat=cache["concat"][t]
            tanh_c=np.tanh(c)
            do=dh_next*tanh_c; dao=do*o*(1.0-o)
            dc=dh_next*o*(1.0-tanh_c**2)+dc_next
            df=dc*c_prev; daf=df*f*(1.0-f)
            di=dc*g; dai=di*i*(1.0-i)
            dg=dc*i; dag=dg*(1.0-g**2)
            grads["Wf"]+=concat.T@daf; grads["Wi"]+=concat.T@dai
            grads["Wo"]+=concat.T@dao; grads["Wg"]+=concat.T@dag
            grads["bf"]+=np.sum(daf,axis=0); grads["bi"]+=np.sum(dai,axis=0)
            grads["bo"]+=np.sum(dao,axis=0); grads["bg"]+=np.sum(dag,axis=0)
            dconcat=(daf@self.params["Wf"].T+dai@self.params["Wi"].T+
                     dao@self.params["Wo"].T+dag@self.params["Wg"].T)
            dh_next=dconcat[:,self.input_dim:]; dc_next=dc*f
        gn=math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
        if gn>5.0: s=5.0/(gn+1e-8); grads={k:v*s for k,v in grads.items()}
        return loss, grads
    def fit(self, x, y):
        split=max(24,int(len(x)*0.85)); split=min(split,len(x)-12)
        xt,yt=x[:split],y[:split]; xv,yv=x[split:],y[split:]
        m={n:np.zeros_like(v) for n,v in self.params.items()}
        v={n:np.zeros_like(v) for n,v in self.params.items()}
        b1,b2=0.9,0.999; best_loss=float("inf")
        best={k:v.copy() for k,v in self.params.items()}; stag=0; step=0
        for epoch in range(self.epochs):
            order=self.rng.permutation(len(xt))
            for start in range(0,len(order),self.batch_size):
                idx=order[start:start+self.batch_size]
                _,g=self._loss_and_grads(xt[idx],yt[idx]); step+=1
                for n in self.params:
                    m[n]=b1*m[n]+(1.0-b1)*g[n]; v[n]=b2*v[n]+(1.0-b2)*(g[n]**2)
                    mh=m[n]/(1.0-b1**step); vh=v[n]/(1.0-b2**step)
                    self.params[n]-=self.lr*mh/(np.sqrt(vh)+1e-8)
            vl=float(np.mean((self.predict_scaled(xv)-yv)**2))
            if vl<best_loss-1e-5: best_loss=vl; best={k:v.copy() for k,v in self.params.items()}; stag=0
            else: stag+=1
            if stag>=self.patience: break
        self.params=best; self.best_val_loss_=best_loss; self.epochs_run_=epoch+1; return self
    def predict_scaled(self, x):
        preds=[]
        for start in range(0,len(x),256): yh,_=self._forward(x[start:start+256]); preds.append(yh)
        return np.concatenate(preds,axis=0)

# ================================================================
# NumpyTCNRegressor — unchanged from original
# ================================================================
class NumpyTCNRegressor:
    def __init__(self, input_dim, channels=8, kernel_size=3, dilations=(1,2,4,8,16),
                 lr=0.018, epochs=260, batch_size=64, patience=28, seed=SEED+101):
        self.input_dim=input_dim; self.channels=channels; self.kernel_size=kernel_size
        self.dilations=tuple(dilations); self.lr=lr; self.epochs=epochs
        self.batch_size=batch_size; self.patience=patience
        self.rng=np.random.default_rng(seed)
        scale=1.0/math.sqrt(input_dim*kernel_size)
        self.W=self.rng.normal(0.0,scale,size=(len(self.dilations),kernel_size,input_dim,channels))
        self.b=np.zeros((len(self.dilations),channels))
        self.Wy=self.rng.normal(0.0,1.0/math.sqrt(len(self.dilations)*channels),size=(len(self.dilations)*channels,))
        self.by=0.0
    def _features(self, x):
        batch,seq_len,_=x.shape; z=np.zeros((batch,len(self.dilations),self.channels))
        selected=[]
        for d_i,dilation in enumerate(self.dilations):
            sd=[]
            for k in range(self.kernel_size):
                idx=max(0,seq_len-1-k*dilation); xk=x[:,idx,:]; sd.append(xk)
                z[:,d_i,:]+=xk@self.W[d_i,k]
            z[:,d_i,:]+=self.b[d_i]; selected.append(sd)
        h=np.tanh(z).reshape(batch,-1); return h,selected
    def _loss_and_grads(self, x, y):
        h,selected=self._features(x); pred=h@self.Wy+self.by; batch=len(x)
        diff=pred-y; loss=float(np.mean(diff**2)); dy=(2.0/batch)*diff
        gWy=h.T@dy; gby=float(np.sum(dy))
        dh=np.outer(dy,self.Wy).reshape(batch,len(self.dilations),self.channels)
        dz=dh*(1.0-h.reshape(batch,len(self.dilations),self.channels)**2)
        gW=np.zeros_like(self.W); gb=np.zeros_like(self.b)
        for d_i in range(len(self.dilations)):
            gb[d_i]+=np.sum(dz[:,d_i,:],axis=0)
            for k in range(self.kernel_size): gW[d_i,k]+=selected[d_i][k].T@dz[:,d_i,:]
        norm=math.sqrt(float(np.sum(gW**2)+np.sum(gb**2)+np.sum(gWy**2)+gby**2))
        if norm>5.0: s=5.0/(norm+1e-8); gW,gGb,gWy,gby=gW*s,gb*s,gWy*s,gby*s
        return loss,{"W":gW,"b":gb,"Wy":gWy,"by":gby}
    def fit(self, x, y):
        split=max(24,int(len(x)*0.85)); split=min(split,len(x)-12)
        xt,yt=x[:split],y[:split]; xv,yv=x[split:],y[split:]
        best_loss=float("inf"); best=(self.W.copy(),self.b.copy(),self.Wy.copy(),float(self.by)); stag=0
        for epoch in range(self.epochs):
            order=self.rng.permutation(len(xt))
            for start in range(0,len(order),self.batch_size):
                idx=order[start:start+self.batch_size]
                _,g=self._loss_and_grads(xt[idx],yt[idx])
                self.W-=self.lr*g["W"]; self.b-=self.lr*g["b"]
                self.Wy-=self.lr*g["Wy"]; self.by-=self.lr*g["by"]
            vl=float(np.mean((self.predict_scaled(xv)-yv)**2))
            if vl<best_loss-1e-5: best_loss=vl; best=(self.W.copy(),self.b.copy(),self.Wy.copy(),float(self.by)); stag=0
            else: stag+=1
            if stag>=self.patience: break
        self.W,self.b,self.Wy,self.by=best; self.best_val_loss_=best_loss; self.epochs_run_=epoch+1; return self
    def predict_scaled(self, x): h,_=self._features(x); return h@self.Wy+self.by

# ================================================================
# LstmTcnFusionRegressor — unchanged from original
# ================================================================
class LstmTcnFusionRegressor:
    def __init__(self, input_dim, hidden_dim=10, tcn_channels=8, kernel_size=3,
                 dilations=(1,2,4,8,16), lr=0.008, epochs=125, batch_size=32,
                 patience=20, seed=SEED+500):
        self.input_dim=input_dim; self.hidden_dim=hidden_dim; self.tcn_channels=tcn_channels
        self.kernel_size=kernel_size; self.dilations=tuple(dilations)
        self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.patience=patience
        self.rng=np.random.default_rng(seed); self.params=self._init_params()
    def _init_params(self):
        lstm_in=self.input_dim+self.hidden_dim; lstm_scale=1.0/math.sqrt(lstm_in)
        tcn_scale=1.0/math.sqrt(self.input_dim*self.kernel_size)
        def lw(): return self.rng.normal(0.0,lstm_scale,size=(lstm_in,self.hidden_dim))
        tcn_dim=len(self.dilations)*self.tcn_channels; fusion_dim=self.hidden_dim+tcn_dim
        return {"Wf":lw(),"Wi":lw(),"Wo":lw(),"Wg":lw(),
                "bf":np.zeros(self.hidden_dim)+0.5,"bi":np.zeros(self.hidden_dim),
                "bo":np.zeros(self.hidden_dim),"bg":np.zeros(self.hidden_dim),
                "Wtcn":self.rng.normal(0.0,tcn_scale,size=(len(self.dilations),self.kernel_size,self.input_dim,self.tcn_channels)),
                "btcn":np.zeros((len(self.dilations),self.tcn_channels)),
                "Wy":self.rng.normal(0.0,1.0/math.sqrt(fusion_dim),size=(fusion_dim,)),
                "by":np.zeros(1)}
    def _forward(self, x):
        batch,seq_len,_=x.shape; h=np.zeros((batch,self.hidden_dim)); c=np.zeros((batch,self.hidden_dim))
        lc={"concat":[],"f":[],"i":[],"o":[],"g":[],"c_prev":[],"c":[],"h":[]}
        for t in range(seq_len):
            xt=x[:,t,:]; concat=np.concatenate([xt,h],axis=1)
            f=sigmoid(concat@self.params["Wf"]+self.params["bf"])
            i=sigmoid(concat@self.params["Wi"]+self.params["bi"])
            o=sigmoid(concat@self.params["Wo"]+self.params["bo"])
            g=np.tanh(concat@self.params["Wg"]+self.params["bg"])
            c_prev=c; c=f*c_prev+i*g; h=o*np.tanh(c)
            for k,v in zip(["concat","f","i","o","g","c_prev","c","h"],[concat,f,i,o,g,c_prev,c,h]): lc[k].append(v)
        z=np.zeros((batch,len(self.dilations),self.tcn_channels)); selected=[]
        for d_i,dilation in enumerate(self.dilations):
            sd=[]
            for k in range(self.kernel_size):
                idx=max(0,seq_len-1-k*dilation); xk=x[:,idx,:]; sd.append(xk)
                z[:,d_i,:]+=xk@self.params["Wtcn"][d_i,k]
            z[:,d_i,:]+=self.params["btcn"][d_i]; selected.append(sd)
        ht3=np.tanh(z); ht=ht3.reshape(batch,-1); fusion=np.concatenate([h,ht],axis=1)
        pred=fusion@self.params["Wy"]+self.params["by"][0]
        return pred,{"lstm":lc,"selected":selected,"h_lstm":h,"h_tcn_3d":ht3,"fusion":fusion}
    def _loss_and_grads(self, x, y):
        pred,cache=self._forward(x); batch=len(x); diff=pred-y
        loss=float(np.mean(diff**2)); dy=(2.0/batch)*diff
        grads={n:np.zeros_like(v) for n,v in self.params.items()}
        fusion=cache["fusion"]; grads["Wy"]+=fusion.T@dy; grads["by"]+=np.array([np.sum(dy)])
        dfusion=np.outer(dy,self.params["Wy"])
        dhl=dfusion[:,:self.hidden_dim]; dht=dfusion[:,self.hidden_dim:].reshape(batch,len(self.dilations),self.tcn_channels)
        lc=cache["lstm"]; dh_next=dhl; dc_next=np.zeros_like(dh_next)
        for t in reversed(range(x.shape[1])):
            f=lc["f"][t]; i=lc["i"][t]; o=lc["o"][t]; g=lc["g"][t]
            c=lc["c"][t]; c_prev=lc["c_prev"][t]; concat=lc["concat"][t]
            tanh_c=np.tanh(c); do=dh_next*tanh_c; dao=do*o*(1.0-o)
            dc=dh_next*o*(1.0-tanh_c**2)+dc_next
            df=dc*c_prev; daf=df*f*(1.0-f); di=dc*g; dai=di*i*(1.0-i)
            dg=dc*i; dag=dg*(1.0-g**2)
            grads["Wf"]+=concat.T@daf; grads["Wi"]+=concat.T@dai
            grads["Wo"]+=concat.T@dao; grads["Wg"]+=concat.T@dag
            grads["bf"]+=np.sum(daf,axis=0); grads["bi"]+=np.sum(dai,axis=0)
            grads["bo"]+=np.sum(dao,axis=0); grads["bg"]+=np.sum(dag,axis=0)
            dconcat=(daf@self.params["Wf"].T+dai@self.params["Wi"].T+
                     dao@self.params["Wo"].T+dag@self.params["Wg"].T)
            dh_next=dconcat[:,self.input_dim:]; dc_next=dc*f
        ht3=cache["h_tcn_3d"]; dz=dht*(1.0-ht3**2); selected=cache["selected"]
        for d_i in range(len(self.dilations)):
            grads["btcn"][d_i]+=np.sum(dz[:,d_i,:],axis=0)
            for k in range(self.kernel_size): grads["Wtcn"][d_i,k]+=selected[d_i][k].T@dz[:,d_i,:]
        gn=math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
        if gn>5.0: s=5.0/(gn+1e-8); grads={k:v*s for k,v in grads.items()}
        return loss,grads
    def fit(self, x, y):
        split=max(36,int(len(x)*0.85)); split=min(split,len(x)-18)
        xt,yt=x[:split],y[:split]; xv,yv=x[split:],y[split:]
        m={n:np.zeros_like(v) for n,v in self.params.items()}
        v={n:np.zeros_like(v) for n,v in self.params.items()}; b1,b2=0.9,0.999
        best_loss=float("inf"); best={k:v.copy() for k,v in self.params.items()}; stag=0; step=0
        for epoch in range(self.epochs):
            order=self.rng.permutation(len(xt))
            for start in range(0,len(order),self.batch_size):
                idx=order[start:start+self.batch_size]; _,g=self._loss_and_grads(xt[idx],yt[idx]); step+=1
                for n in self.params:
                    m[n]=b1*m[n]+(1.0-b1)*g[n]; v[n]=b2*v[n]+(1.0-b2)*(g[n]**2)
                    mh=m[n]/(1.0-b1**step); vh=v[n]/(1.0-b2**step)
                    self.params[n]-=self.lr*mh/(np.sqrt(vh)+1e-8)
            vl=float(np.mean((self.predict_scaled(xv)-yv)**2))
            if vl<best_loss-1e-5: best_loss=vl; best={k:v.copy() for k,v in self.params.items()}; stag=0
            else: stag+=1
            if stag>=self.patience: break
        self.params=best; self.best_val_loss_=best_loss; self.epochs_run_=epoch+1; return self
    def predict_scaled(self, x):
        preds=[]
        for start in range(0,len(x),256): p,_=self._forward(x[start:start+256]); preds.append(p)
        return np.concatenate(preds,axis=0)

# ================================================================
# Data pipeline — drone target + battle intensity
# ================================================================
def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    for col in df.columns:
        if col not in {"date", "active_campaigns"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["active_campaigns"] = df["active_campaigns"].fillna("无显式战役标签")
    df["drone_loss"] = df["drone_loss"].clip(lower=0.0)
    return df

def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    y = df["drone_loss"].astype(float).clip(lower=0.0)
    tank = df["tank_loss_clean"].astype(float).clip(lower=0.0)
    frame = pd.DataFrame(index=df.index)
    frame["date"] = df["date"]; frame["target"] = y; frame["y_log"] = np.log1p(y)
    ma30 = y.shift(1).rolling(30, min_periods=2).mean()
    ma30 = ma30.fillna(y.shift(1).expanding(min_periods=1).mean()).fillna(y.expanding().mean())
    frame["ma30_target_log"] = np.log1p(ma30)
    frame["resid_ma30_log"] = frame["y_log"] - frame["ma30_target_log"]
    for window in (7, 14, 30):
        roll = y.shift(1).rolling(window, min_periods=2)
        frame[f"drone_roll{window}_mean_log"] = np.log1p(roll.mean().fillna(y.shift(1).expanding().mean()).fillna(y.mean()))
    frame["drone_roll7_std_log"] = np.log1p(y.shift(1).rolling(7, min_periods=2).std().fillna(0.0))
    frame["drone_diff1_log"] = frame["y_log"].diff().fillna(0.0)
    frame["drone_zero_lag1"] = (y.shift(1).fillna(0.0) <= 0).astype(float)
    frame["tank_roll7_mean_log"] = np.log1p(tank.shift(1).rolling(7, min_periods=2).mean().fillna(0.0))
    frame["tank_roll30_mean_log"] = np.log1p(tank.shift(1).rolling(30, min_periods=2).mean().fillna(0.0))
    frame["tank_to_drone_roll14"] = (
        tank.shift(1).rolling(14, min_periods=2).sum()
        / (y.shift(1).rolling(14, min_periods=2).sum() + 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dow = df["date"].dt.dayofweek.to_numpy(); month = df["date"].dt.month.to_numpy()
    ordinal = (df["date"] - df["date"].min()).dt.days.to_numpy(dtype=float)
    frame["dow_sin"] = np.sin(2*np.pi*dow/7); frame["dow_cos"] = np.cos(2*np.pi*dow/7)
    frame["month_sin"] = np.sin(2*np.pi*month/12); frame["month_cos"] = np.cos(2*np.pi*month/12)
    frame["time_index"] = ordinal / max(1.0, ordinal.max())
    battle_I = get_battle_intensity(pd.DatetimeIndex(df["date"]))
    frame["battle_intensity"] = battle_I.values
    frame["active_campaign_count"] = df["active_campaign_count"].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sequence_cols = [
        "y_log", "ma30_target_log", "resid_ma30_log",
        "drone_roll7_mean_log", "drone_roll14_mean_log", "drone_roll30_mean_log",
        "drone_roll7_std_log", "drone_diff1_log", "drone_zero_lag1",
        "tank_roll7_mean_log", "tank_roll30_mean_log", "tank_to_drone_roll14",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "time_index",
        "battle_intensity", "active_campaign_count",
    ]
    groups = {
        "历史与残差": ["y_log", "ma30_target_log", "resid_ma30_log"],
        "无人机滚动动量": [c for c in sequence_cols if c.startswith("drone_")],
        "坦克强度": [c for c in sequence_cols if c.startswith("tank_")],
        "日历节奏": ["dow_sin", "dow_cos", "month_sin", "month_cos", "time_index"],
        "外部变量": ["battle_intensity", "active_campaign_count"],
    }
    return frame, sequence_cols, groups

def make_sequences(frame, feature_cols, x_scaler, y_scaler):
    xv = x_scaler.transform(frame[list(feature_cols)].to_numpy(dtype=float))
    ys = y_scaler.transform(frame[["resid_ma30_log"]].to_numpy(dtype=float)).ravel()
    xs,ys2,dates,raw_y,bl=[],[],[],[],[]
    for end in range(SEQ_LEN, len(frame)):
        xs.append(xv[end-SEQ_LEN:end]); ys2.append(ys[end])
        dates.append(frame.loc[end,"date"]); raw_y.append(frame.loc[end,"target"])
        bl.append(frame.loc[end,"ma30_target_log"])
    return {"X":np.asarray(xs,dtype=float),"y_scaled":np.asarray(ys2,dtype=float),
            "date":np.asarray(dates,dtype="datetime64[ns]"),
            "y_raw":np.asarray(raw_y,dtype=float),"baseline_log":np.asarray(bl,dtype=float)}

def make_tabular(frame, seq_cols):
    rows,dates,raw_y,bl,resid=[],[],[],[],[]
    y_log=frame["y_log"].to_numpy(dtype=float); res=frame["resid_ma30_log"].to_numpy(dtype=float)
    for end in range(SEQ_LEN, len(frame)):
        row={}
        for lag in (1,2,3,7,14,30):
            row[f"lag_y_log_{lag}"]=float(y_log[end-lag])
            row[f"lag_resid_{lag}"]=float(res[end-lag])
        for col in seq_cols: row[col]=float(frame.loc[end,col])
        rows.append(row); dates.append(frame.loc[end,"date"])
        raw_y.append(frame.loc[end,"target"]); bl.append(frame.loc[end,"ma30_target_log"])
        resid.append(frame.loc[end,"resid_ma30_log"])
    tc=[f"lag_y_log_{l}" for l in (1,2,3,7,14,30)]+[f"lag_resid_{l}" for l in (1,2,3,7,14,30)]+list(seq_cols)
    return (pd.DataFrame(rows,columns=tc).replace([np.inf,-np.inf],np.nan).fillna(0.0),
            np.asarray(resid,dtype=float),np.asarray(dates,dtype="datetime64[ns]"),
            np.asarray(raw_y,dtype=float),np.asarray(bl,dtype=float))

def inverse_pred(ps, ys, bl):
    pr=ys.inverse(ps.reshape(-1,1)).ravel(); return np.clip(np.expm1(bl+pr),0.0,None)

def build_xgb_branch(seed):
    return XGBRegressor(objective="reg:squarederror",n_estimators=260,max_depth=3,
        learning_rate=0.035,subsample=0.9,colsample_bytree=0.85,reg_lambda=3.0,
        random_state=seed,n_jobs=1,verbosity=0,eval_metric="rmse"),"xgboost.XGBRegressor"

def fit_bundle(frame, seq_cols, train_end, seed_offset=0):
    tr=(frame["date"]<train_end).to_numpy()
    xs=Scaler.fit(frame.loc[tr,seq_cols].to_numpy(dtype=float))
    ys=Scaler.fit(frame.loc[tr,["resid_ma30_log"]].to_numpy(dtype=float))
    seq=make_sequences(frame,seq_cols,xs,ys); sq_tr=pd.to_datetime(seq["date"])<train_end
    lstm=NumpyLSTMRegressor(input_dim=len(seq_cols),hidden_dim=10,seed=SEED+seed_offset+11)
    tcn=NumpyTCNRegressor(input_dim=len(seq_cols),channels=8,seed=SEED+seed_offset+22)
    print(f"Training LSTM/TCN before {train_end.date()} with {int(sq_tr.sum())} seqs")
    lstm.fit(seq["X"][sq_tr],seq["y_scaled"][sq_tr]); tcn.fit(seq["X"][sq_tr],seq["y_scaled"][sq_tr])
    tx,trsid,td,_,_=make_tabular(frame,seq_cols); ttr=pd.to_datetime(td)<train_end
    ts=StandardScaler(); yts=ys.transform(trsid.reshape(-1,1)).ravel()
    xgb_m,xgb_i=build_xgb_branch(SEED+seed_offset+33)
    xgb_m.fit(ts.fit_transform(tx.loc[ttr].to_numpy(dtype=float)),yts[ttr])
    return {"lstm":lstm,"tcn":tcn,"xgb":xgb_m,"xgb_impl":xgb_i,
            "x_scaler":xs,"y_scaler":ys,"tab_scaler":ts,"tab_columns":list(tx.columns),"sequence_cols":list(seq_cols)}

def predict_bundle(bundle, frame, target_dates):
    td=np.asarray(target_dates,dtype="datetime64[ns]")
    seq=make_sequences(frame,bundle["sequence_cols"],bundle["x_scaler"],bundle["y_scaler"])
    sm=np.isin(seq["date"],td)
    lr=inverse_pred(bundle["lstm"].predict_scaled(seq["X"][sm]),bundle["y_scaler"],seq["baseline_log"][sm])
    tr=inverse_pred(bundle["tcn"].predict_scaled(seq["X"][sm]),bundle["y_scaler"],seq["baseline_log"][sm])
    tx,_,td2,ty,tb=make_tabular(frame,bundle["sequence_cols"]); tm=np.isin(td2,td)
    xt=bundle["tab_scaler"].transform(tx.loc[tm,bundle["tab_columns"]].to_numpy(dtype=float))
    xr=inverse_pred(bundle["xgb"].predict(xt),bundle["y_scaler"],tb[tm])
    return {"date":td2[tm],"actual":ty[tm],"lstm":lr,"tcn":tr,"xgb":xr}

def branch_weights(yt, preds):
    err=np.array([rmse(yt,preds[:,i]) for i in range(preds.shape[1])],dtype=float)
    raw=1.0/np.maximum(err,1e-6); raw=raw/raw.sum(); floor=0.15
    return floor+(1.0-floor*len(raw))*raw

def interval_from_sigma(pred, sl):
    pl=np.log1p(np.clip(pred,0.0,None))
    return np.clip(np.expm1(pl-1.96*sl),0.0,None),np.clip(np.expm1(pl+1.96*sl),0.0,None)

def evaluate(name,yt,yp,lo,hi,wt=None):
    return {"model":name,"MAE":mae(yt,yp),"RMSE":rmse(yt,yp),"sMAPE":smape(yt,yp),
            "PI95_coverage":coverage(yt,lo,hi),"PI95_mean_width":float(np.mean(hi-lo)),
            "ensemble_weight":np.nan if wt is None else float(wt)}

def append_future_row(raw_df, yhat, tank_hat):
    last=raw_df.iloc[-1].copy(); new=last.copy()
    new["date"]=last["date"]+pd.Timedelta(days=1); new["day"]=float(last["day"])+1
    new["drone_loss"]=yhat; new["tank_loss_clean"]=tank_hat
    return pd.concat([raw_df,pd.DataFrame([new])],ignore_index=True)

def recursive_forecast(bundle, raw_df, weights, horizon=30):
    working=raw_df.copy(); rows=[]
    for _ in range(horizon):
        th=float(np.mean(working["tank_loss_clean"].astype(float).tail(30)))
        yp=float(np.mean(working["drone_loss"].astype(float).tail(30)))
        tr=append_future_row(working,yp,th); tf,_,_=build_feature_frame(tr)
        nd=np.asarray([tr["date"].iloc[-1]],dtype="datetime64[ns]")
        p=predict_bundle(bundle,tf,nd)
        br=np.array([p["lstm"][0],p["tcn"][0],p["xgb"][0]]); hy=float(br@weights)
        rows.append({"date":tr["date"].iloc[-1],"lstm_branch":br[0],"tcn_branch":br[1],
                      "xgboost_branch":br[2],"hybrid_ensemble":hy})
        working=append_future_row(working,hy,th)
    return pd.DataFrame(rows)

def feature_importance(bundle):
    v=getattr(bundle["xgb"],"feature_importances_",None)
    if v is None: v=np.zeros(len(bundle["tab_columns"]))
    return pd.DataFrame({"feature":bundle["tab_columns"],"importance":v}).sort_values("importance",ascending=False)

def fit_fusion_base(frame, seq_cols, train_end, seed_offset=0):
    tr=(frame["date"]<train_end).to_numpy()
    xs=Scaler.fit(frame.loc[tr,seq_cols].to_numpy(dtype=float))
    ys=Scaler.fit(frame.loc[tr,["resid_ma30_log"]].to_numpy(dtype=float))
    seq=make_sequences(frame,seq_cols,xs,ys); sq_tr=pd.to_datetime(seq["date"])<train_end
    m=LstmTcnFusionRegressor(input_dim=len(seq_cols),hidden_dim=10,tcn_channels=8,
        lr=0.0075,epochs=135,batch_size=32,patience=20,seed=SEED+seed_offset)
    print(f"Training Fusion before {train_end.date()} with {int(sq_tr.sum())} seqs")
    m.fit(seq["X"][sq_tr],seq["y_scaled"][sq_tr])
    return {"model":m,"x_scaler":xs,"y_scaler":ys,"sequence_cols":list(seq_cols)}

def predict_fusion_base(bundle, frame, target_dates):
    td=np.asarray(target_dates,dtype="datetime64[ns]")
    seq=make_sequences(frame,bundle["sequence_cols"],bundle["x_scaler"],bundle["y_scaler"])
    m=np.isin(seq["date"],td); ps=bundle["model"].predict_scaled(seq["X"][m])
    pr=inverse_pred(ps,bundle["y_scaler"],seq["baseline_log"][m])
    return {"date":seq["date"][m],"actual":seq["y_raw"][m],"base_pred":pr}

def residual_feature_frame(frame, seq_cols, target_dates, base_pred):
    td=np.asarray(target_dates,dtype="datetime64[ns]")
    tx,_,td2,ty,_=make_tabular(frame,seq_cols); m=np.isin(td2,td)
    x=tx.loc[m].reset_index(drop=True).copy(); bp=np.asarray(base_pred,dtype=float)
    x["lstm_tcn_base_pred"]=bp; x["lstm_tcn_base_log"]=np.log1p(np.clip(bp,0.0,None))
    x["base_vs_ma30"]=x["lstm_tcn_base_log"]-x["ma30_target_log"].to_numpy(dtype=float)
    return x,ty[m],td2[m],tx.columns.to_numpy()

def build_residual_booster(seed):
    return XGBRegressor(objective="reg:squarederror",n_estimators=90,max_depth=2,
        learning_rate=0.035,subsample=0.85,colsample_bytree=0.80,reg_lambda=8.0,
        min_child_weight=5.0,random_state=seed,n_jobs=1,verbosity=0,eval_metric="rmse"),"xgboost.XGBRegressor"

def fit_residual_optimizer(x, residual, seed_offset=0):
    fc=list(x.columns); v=x[fc].to_numpy(dtype=float); r=np.asarray(residual,dtype=float)
    sp=max(24,int(len(v)*0.72)); sp=min(sp,len(v)-12)
    pm,impl=build_residual_booster(SEED+seed_offset); pm.fit(v[:sp],r[:sp])
    pp=pm.predict(v[sp:]); grid=np.linspace(0.0,1.0,21)
    alpha=min(grid,key=lambda a:float(np.mean((r[sp:]-a*pp)**2)))
    m,_=build_residual_booster(SEED+seed_offset+17); m.fit(x[fc].to_numpy(dtype=float),r)
    return {"model":m,"implementation":impl,"feature_cols":fc,"alpha":float(alpha)}

def predict_residual_optimizer(bundle, x):
    al=x.reindex(columns=bundle["feature_cols"],fill_value=0.0)
    return float(bundle.get("alpha",1.0))*bundle["model"].predict(al.to_numpy(dtype=float))

def residual_importance(bundle):
    v=getattr(bundle["model"],"feature_importances_",None)
    if v is None: v=np.zeros(len(bundle["feature_cols"]))
    return pd.DataFrame({"feature":bundle["feature_cols"],"importance":v}).sort_values("importance",ascending=False)

def apply_residual_correction(base, correction):
    return np.clip(np.asarray(base,dtype=float)+np.asarray(correction,dtype=float),0.0,None)

def moving_average_predictions(frame, target_dates, window=30):
    y=frame["target"].to_numpy(dtype=float); dates=frame["date"].to_numpy(dtype="datetime64[ns]")
    d2i={d:i for i,d in enumerate(dates)}; preds=[]
    for d in np.asarray(target_dates,dtype="datetime64[ns]"):
        idx=d2i[np.datetime64(d,"ns")]; start=max(0,idx-window)
        preds.append(float(np.mean(y[start:idx])))
    return np.asarray(preds,dtype=float)

def recursive_residual_forecast(base_bundle, residual_bundle, raw_df, seq_cols, horizon=30):
    working=raw_df.copy(); rows=[]
    for _ in range(horizon):
        th=float(np.mean(working["tank_loss_clean"].astype(float).tail(30)))
        ph=float(np.mean(working["drone_loss"].astype(float).tail(30)))
        tr=append_future_row(working,ph,th); tf,_,_=build_feature_frame(tr)
        nd=np.asarray([tr["date"].iloc[-1]],dtype="datetime64[ns]")
        base=predict_fusion_base(base_bundle,tf,nd)["base_pred"]
        xf,_,_,_=residual_feature_frame(tf,seq_cols,nd,base)
        corr=predict_residual_optimizer(residual_bundle,xf)
        final=apply_residual_correction(base,corr)
        rows.append({"date":tr["date"].iloc[-1],"lstm_tcn_base":float(base[0]),
                      "xgboost_residual_correction":float(corr[0]),"final_prediction":float(final[0])})
        working=append_future_row(working,float(final[0]),th)
    return pd.DataFrame(rows)

# ================================================================
# Plotting — original (adapted) + shulitongji PNG
# ================================================================
def pretty_feature_name(name):
    m={"ma30_target_log":"MA30基线","battle_intensity":"战役烈度","active_campaign_count":"活跃战役数",
       "time_index":"时间趋势","drone_roll7_std_log":"7日波动","drone_roll7_mean_log":"7日均值",
       "drone_roll14_mean_log":"14日均值","drone_roll30_mean_log":"30日均值",
       "tank_roll7_mean_log":"坦克7日均值","tank_roll30_mean_log":"坦克30日均值",
       "tank_to_drone_roll14":"坦克/无人机比"}
    if name in m: return m[name]
    if name.startswith("lag_y_log_"): return name.replace("lag_y_log_","损失滞后")
    if name.startswith("lag_resid_"): return name.replace("lag_resid_","残差滞后")
    return name[:28]

def style_time_axis(ax):
    ax.grid(True,linestyle="--",alpha=0.35,color=GRID_GRAY)
    for s in["top","right"]: ax.spines[s].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

def save_figure(fig, path):
    fig.tight_layout(pad=2); fig.savefig(path,dpi=250,facecolor="white",edgecolor="none",bbox_inches="tight"); plt.close(fig)

def add_battle_axis(fig, ax, battle_I):
    ax_r=ax.twinx(); ax_r.set_ylabel('Battle Intensity',fontsize=11,color=DARK_ORANGE)
    ax_r.plot(battle_I.index,battle_I.values,color=DARK_ORANGE,linewidth=1.0,alpha=0.85,zorder=5)
    ax_r.set_ylim(0,1.05); ax_r.tick_params(axis='y',colors=DARK_ORANGE)
    for s in["top","left"]: ax_r.spines[s].set_visible(False); return ax_r

def plot_overview(raw, battle_I, path):
    fig=plt.figure(figsize=(24,14)); ax=fig.add_subplot(111)
    ax.plot(raw["date"],raw["drone_loss"],color=ROYAL_BLUE,alpha=0.4,lw=0.7,label="Actual Data",zorder=2)
    ax.plot(raw["date"],raw["drone_loss"].rolling(30,min_periods=1).mean(),color=DARK_ORANGE,lw=1.3,label="MA30 Smooth",zorder=3)
    ax.set_ylabel("Daily Drone Loss",fontsize=12); ax.set_xlabel("Date",fontsize=12); ax.set_ylim(bottom=-5)
    ax.set_title("LSTM-TCN-XGBoost Input  |  Left=Drone Loss  Right=Intensity",fontsize=13,fontweight="bold")
    ax.legend(loc="upper left",fontsize=8.5,framealpha=0.9,ncol=2); style_time_axis(ax); add_battle_axis(fig,ax,battle_I)
    save_figure(fig,path)

def plot_architecture(weights, path):
    fig,ax=plt.subplots(figsize=(10.8,4.6)); ax.axis("off")
    boxes=[(0.06,0.55,0.22,0.23,"30日历史序列\n滚动动量/日历/战役烈度"),
           (0.39,0.78,0.19,0.15,f"LSTM 分支\n权重 {weights[0]:.2f}"),
           (0.39,0.55,0.19,0.15,f"TCN 分支\n权重 {weights[1]:.2f}"),
           (0.39,0.32,0.19,0.15,f"XGBoost 分支\n权重 {weights[2]:.2f}"),
           (0.72,0.52,0.22,0.22,"联合输出\n未来30日无人机损失")]
    for x,y,w,h,t in boxes:
        r=plt.Rectangle((x,y),w,h,fc="#F8FAFC",ec="#334155",lw=1.5,transform=ax.transAxes); ax.add_patch(r)
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=11,transform=ax.transAxes)
    for y in(0.855,0.625,0.395):
        ax.annotate("",xy=(0.39,y),xytext=(0.28,0.665),arrowprops=dict(arrowstyle="->",lw=1.5),xycoords=ax.transAxes)
        ax.annotate("",xy=(0.72,0.63),xytext=(0.58,y),arrowprops=dict(arrowstyle="->",lw=1.5),xycoords=ax.transAxes)
    ax.set_title("LSTM/TCN/XGBoost 三分支联合建模框架 (Drone)",fontsize=14,pad=10)
    fig.savefig(path,bbox_inches="tight",facecolor="white"); plt.close(fig)

def plot_holdout(pred, path):
    d=pd.to_datetime(pred["date"]); fig,ax=plt.subplots(figsize=(10.8,4.9))
    ax.plot(d,pred["actual"],color="#111827",lw=2.0,label="观测值")
    ax.plot(d,pred["lstm_branch"],color="#60A5FA",lw=1.1,alpha=0.78,label="LSTM分支")
    ax.plot(d,pred["tcn_branch"],color="#34D399",lw=1.1,alpha=0.78,label="TCN分支")
    ax.plot(d,pred["xgboost_branch"],color="#A78BFA",lw=1.1,alpha=0.78,label="XGBoost分支")
    ax.plot(d,pred["hybrid_ensemble"],color="#B45309",lw=2.2,label="联合模型")
    ax.fill_between(d,pred["hybrid_lower95"],pred["hybrid_upper95"],color="#F59E0B",alpha=0.16,label="95%区间")
    ax.set_title("最近8日评估 (Drone)"); ax.set_ylabel("无人机日损失")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.grid(alpha=0.25); ax.legend(ncol=3,fontsize=8.2)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight",facecolor="white"); plt.close(fig)

def plot_weights_importance(weights, imp, path):
    top=imp.head(10).iloc[::-1]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.8,4.7),gridspec_kw={"width_ratios":[0.85,1.45]})
    ax1.bar(["LSTM","TCN","XGBoost"],weights,color=["#60A5FA","#34D399","#A78BFA"])
    ax1.set_ylim(0,max(0.6,float(weights.max())+0.08)); ax1.set_title("分支权重"); ax1.grid(axis="y",alpha=0.22)
    labels=[pretty_feature_name(x) for x in top["feature"]]
    ax2.barh(labels,top["importance"],color="#64748B"); ax2.set_title("XGBoost 主要特征"); ax2.grid(axis="x",alpha=0.22)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight",facecolor="white"); plt.close(fig)

def plot_forecast(recent_frame, forecast, path):
    hist=recent_frame.tail(150); d=pd.to_datetime(forecast["date"]); fig,ax=plt.subplots(figsize=(10.8,4.9))
    ax.plot(hist["date"],hist["target"],color="#111827",lw=1.7,label="历史观测")
    ax.axvline(recent_frame["date"].max(),color="#6B7280",ls="--",lw=1.1,label="预测起点")
    ax.plot(d,forecast["hybrid_ensemble"],color="#B45309",lw=2.2,label="未来30日联合预测")
    ax.fill_between(d,forecast["hybrid_lower95"],forecast["hybrid_upper95"],color="#F59E0B",alpha=0.17,label="95%区间")
    ax.set_title("未来30日无人机日损失联合预测"); ax.set_ylabel("无人机日损失")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.grid(alpha=0.25); ax.legend(fontsize=8.8)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight",facecolor="white"); plt.close(fig)

def plot_residual_architecture(path):
    fig,ax=plt.subplots(figsize=(22,10)); ax.axis("off")
    boxes=[(0.04,0.55,0.20,0.22,"全样本滑动窗口\n过去30日序列+战役烈度"),
           (0.34,0.74,0.18,0.15,"LSTM 记忆分支"),(0.34,0.45,0.18,0.15,"TCN 卷积分支"),
           (0.61,0.58,0.15,0.18,"融合层\n基础预测"),(0.61,0.25,0.15,0.15,"残差\n真实值-基础预测"),
           (0.82,0.25,0.14,0.15,"XGBoost\n残差修正"),(0.82,0.58,0.14,0.18,"最终输出\n未来30日预测")]
    for x,y,w,h,t in boxes:
        r=plt.Rectangle((x,y),w,h,fc="#FFF7ED",ec=GRAY_LINE,lw=1.6,transform=ax.transAxes); ax.add_patch(r)
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=13,transform=ax.transAxes)
    for s,e in[((0.24,0.66),(0.34,0.815)),((0.24,0.66),(0.34,0.525)),((0.52,0.815),(0.61,0.67)),
               ((0.52,0.525),(0.61,0.67)),((0.685,0.58),(0.685,0.40)),((0.76,0.325),(0.82,0.325)),
               ((0.89,0.40),(0.89,0.58))]:
        ax.annotate("",xy=e,xytext=s,arrowprops=dict(arrowstyle="->",lw=1.6,color=GRAY_LINE),xycoords=ax.transAxes)
    ax.set_title("LSTM-TCN Fusion + XGBoost Residual Correction  |  Drone + Battle Intensity",
                 fontsize=14,fontweight="bold",pad=10)
    save_figure(fig,path)

def plot_residual_holdout(pred, path):
    d=pd.to_datetime(pred["date"]); ev=rmse(pred["actual"].to_numpy(dtype=float),pred["final_prediction"].to_numpy(dtype=float))
    fig,ax=plt.subplots(figsize=(22,12))
    ax.plot(d,pred["actual"],color=ROYAL_BLUE,alpha=0.75,lw=1.6,label="Actual",zorder=2)
    ax.plot(d,pred["ma30"],color=DODGER_BLUE,lw=1.2,alpha=0.55,ls="--",label="MA30 Baseline",zorder=2)
    ax.plot(d,pred["lstm_tcn_base"],color=DARK_ORANGE,lw=1.3,label="LSTM-TCN Fit",zorder=3)
    ax.plot(d,pred["final_prediction"],color=CRIMSON,lw=2.2,label="XGBoost Corrected",zorder=4)
    ax.fill_between(d,pred["final_lower95"],pred["final_upper95"],color=CRIMSON,alpha=0.16,label="95% CI",zorder=2)
    ax.set_title(f"Recent 8-day  |  LSTM-TCN+XGBoost  |  RMSE={ev:.2f}",fontsize=13,fontweight="bold")
    ax.set_ylabel("Daily Drone Loss",fontsize=12); ax.set_xlabel("Date",fontsize=12); ax.set_ylim(bottom=-5)
    style_time_axis(ax); ax.legend(loc="upper left",fontsize=8.5,framealpha=0.9,ncol=4); save_figure(fig,path)

def plot_residual_correction(pred, imp, path):
    before=pred["actual"].to_numpy(dtype=float)-pred["lstm_tcn_base"].to_numpy(dtype=float)
    after=pred["actual"].to_numpy(dtype=float)-pred["final_prediction"].to_numpy(dtype=float)
    top=imp.head(10).iloc[::-1]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(22,11),gridspec_kw={"width_ratios":[0.95,1.35]})
    ax1.scatter(before,after,color=ROYAL_BLUE,alpha=0.72,s=48,zorder=3)
    lim=max(1.0,float(np.max(np.abs(np.r_[before,after]))))
    ax1.axhline(0,color=GRAY_LINE,lw=1.0,alpha=0.6); ax1.axvline(0,color=GRAY_LINE,lw=1.0,alpha=0.6)
    ax1.plot([-lim,lim],[-lim,lim],color=GRAY_LINE,lw=1.0,ls="--",alpha=0.6)
    ax1.set_xlim(-lim,lim); ax1.set_ylim(-lim,lim); ax1.set_xlabel("修正前残差"); ax1.set_ylabel("修正后残差")
    ax1.set_title("Residual Correction | Before vs After"); ax1.grid(True,linestyle="--",alpha=0.35,color=GRID_GRAY)
    for s in["top","right"]: ax1.spines[s].set_visible(False)
    labels=[pretty_feature_name(x) for x in top["feature"]]
    ax2.barh(labels,top["importance"],color=[BLUE_TO_RED(i/max(len(labels)-1,1)) for i in range(len(labels))],alpha=0.92)
    ax2.set_title("XGBoost Feature Importance | Residual learner"); ax2.grid(axis="x",linestyle="--",alpha=0.35,color=GRID_GRAY)
    for s in["top","right"]: ax2.spines[s].set_visible(False); save_figure(fig,path)

def plot_residual_forecast(frame, forecast, battle_I, path):
    hist=frame.tail(150); d=pd.to_datetime(forecast["date"]); fh=float(np.mean(forecast["final_prediction"]))
    vs=pd.to_datetime(hist["date"].min()); ve=pd.to_datetime(d.max())
    fig,ax=plt.subplots(figsize=(22,12))
    ax.plot(hist["date"],hist["target"],color=ROYAL_BLUE,alpha=0.4,lw=0.7,label="Actual Data",zorder=2)
    ax.plot(d,forecast["lstm_tcn_base"],color=DARK_ORANGE,lw=1.3,alpha=0.95,label="LSTM-TCN Base",zorder=3)
    ax.fill_between(d,forecast["final_lower95"],forecast["final_upper95"],color=CRIMSON,alpha=0.16,label="95% CI",zorder=2)
    ax.plot(d,forecast["final_prediction"],color=CRIMSON,lw=2.2,label="Forecast",zorder=4)
    ax.axvline(frame["date"].max(),color=GRAY_LINE,ls=":",lw=1.0,alpha=0.6,label="Forecast Start")
    ax.set_title(f"LSTM-TCN+XGBoost Forecast  |  Left=Drone Loss  Right=Intensity  |  30d Mean={fh:.2f}",
                 fontsize=13,fontweight="bold")
    ax.set_ylabel("Daily Drone Loss",fontsize=12); ax.set_xlabel("Date",fontsize=12); ax.set_ylim(bottom=-5)
    ax.set_xlim(vs,ve+pd.Timedelta(days=16)); ax.legend(loc="upper left",fontsize=8.5,framealpha=0.9,ncol=4)
    style_time_axis(ax); add_battle_axis(fig,ax,battle_I.loc[vs:ve+pd.Timedelta(days=16)]); save_figure(fig,path)

# ================================================================
# Shulitongji PNGs
# ================================================================
def plot_shulitongji_regression(frame, forecast, fusion_fit, fusion_dates, final_fit, final_dates, rmse_b, rmse_f, mae_f):
    fig,ax=plt.subplots(figsize=(22,12))
    ad=frame["date"]; ya=frame["target"].values
    ax.plot(ad,ya,color=ROYAL_BLUE,alpha=0.4,lw=0.7,label='Actual Drone Loss',zorder=2)
    ax.plot(pd.to_datetime(fusion_dates),fusion_fit,color=DODGER_BLUE,alpha=0.55,ls='--',lw=1.0,label='LSTM-TCN Base',zorder=2)
    ax.plot(pd.to_datetime(final_dates),final_fit,color=DARK_ORANGE,lw=1.3,label='LSTM-TCN+XGBoost Final Fit',zorder=3)
    d=pd.to_datetime(forecast["date"])
    ax.plot(d,forecast["final_prediction"],color=CRIMSON,lw=2.2,ls='--',label=f'Forecast ({FORECAST_HORIZON}d)',zorder=4)
    ax.axvline(x=ad.iloc[-1],color=GRAY_LINE,ls=':',lw=1.0,alpha=0.6)
    ax.set_ylabel('Daily Drone Loss',fontsize=12)
    ax.set_title(f'LSTM-TCN + XGBoost  |  Drone Daily Loss  |  RMSE: {rmse_b:.1f}→{rmse_f:.1f}  MAE={mae_f:.1f}',
                 fontsize=13,fontweight='bold')
    ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,ncol=2)
    ax.grid(True,linestyle='--',alpha=0.35,color='#cccccc'); ax.set_ylim(bottom=-20)
    for s in['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout(pad=2)
    fig.savefig(BASE_DIR/"drone_ML_Regression.png",dpi=250,facecolor='white',edgecolor='none')
    print(f"  -> {BASE_DIR/'drone_ML_Regression.png'}"); plt.close(fig)

def plot_shulitongji_overlay(frame, forecast, fusion_fit, fusion_dates, final_fit, final_dates, battle_I, rmse_b, rmse_f, mae_f):
    fig=plt.figure(figsize=(24,14)); gs=fig.add_gridspec(2,1,height_ratios=[2.8,1],hspace=0.3)
    ad=frame["date"]; ya=frame["target"].values
    ax=fig.add_subplot(gs[0])
    ax.plot(ad,ya,color=ROYAL_BLUE,alpha=0.4,lw=0.7,label='Actual Drone Loss',zorder=2)
    ax.plot(pd.to_datetime(fusion_dates),fusion_fit,color=DODGER_BLUE,alpha=0.55,ls='--',lw=1.0,label='LSTM-TCN Base',zorder=2)
    ax.plot(pd.to_datetime(final_dates),final_fit,color=DARK_ORANGE,lw=1.3,label='LSTM-TCN+XGBoost Final Fit',zorder=3)
    d=pd.to_datetime(forecast["date"])
    ax.plot(d,forecast["final_prediction"],color=CRIMSON,lw=2.2,ls='--',label=f'Forecast ({FORECAST_HORIZON}d)',zorder=4)
    ax.axvline(x=ad.iloc[-1],color=GRAY_LINE,ls=':',lw=1.0,alpha=0.6)
    ax.set_ylabel('Daily Drone Loss',fontsize=12)
    ax.set_title(f'LSTM-TCN+XGBoost  |  Left=Loss  Right=Intensity  |  RMSE: {rmse_b:.1f}→{rmse_f:.1f}  MAE={mae_f:.1f}',
                 fontsize=13,fontweight='bold')
    ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,ncol=2)
    ax.grid(True,linestyle='--',alpha=0.35,color='#cccccc'); ax.set_ylim(bottom=-20)
    for s in['top','right']: ax.spines[s].set_visible(False)
    add_battle_axis(fig,ax,battle_I)
    ax2=fig.add_subplot(gs[1])
    rf=ya[-len(final_fit):]-final_fit
    ax2.plot(pd.to_datetime(final_dates),rf,color='gray',alpha=0.45,lw=0.5,label=f'Final Residuals (RMSE={rmse_f:.2f})')
    ax2.axhline(y=0,color='black',ls='-',alpha=0.3)
    ax2.set_ylabel('Residual',fontsize=11); ax2.set_xlabel('Date',fontsize=12)
    ax2.set_title('LSTM-TCN+XGBoost Final Residuals',fontsize=12,fontweight='bold')
    ax2.legend(loc='upper left',fontsize=9,framealpha=0.9)
    ax2.grid(True,linestyle='--',alpha=0.35,color='#cccccc')
    for s in['top','right']: ax2.spines[s].set_visible(False)
    fig.tight_layout(pad=2)
    fig.savefig(BASE_DIR/"drone_ML_Overlay.png",dpi=250,facecolor='white',edgecolor='none')
    print(f"  -> {BASE_DIR/'drone_ML_Overlay.png'}"); plt.close(fig)

# ================================================================
# main
# ================================================================
def main():
    print(f"CSV: {CSV_PATH}\nBattles: {BATTLES_CSV}")
    raw=load_csv(); frame,seq_cols,groups=build_feature_frame(raw)
    battle_I=get_battle_intensity(pd.DatetimeIndex(frame["date"]))
    ld=frame["date"].max(); ts=ld-pd.Timedelta(days=EVAL_DAYS-1)
    td=frame.loc[frame["date"]>=ts,"date"].to_numpy(dtype="datetime64[ns]")

    # Pipeline A: 3-branch ensemble
    bundle=fit_bundle(frame,seq_cols,ts,seed_offset=0)
    tp=predict_bundle(bundle,frame,td)
    br=np.column_stack([tp["lstm"],tp["tcn"],tp["xgb"]]); wts=branch_weights(tp["actual"],br); hy=br@wts
    sl=float(np.std(np.log1p(tp["actual"])-np.log1p(np.clip(hy,0.0,None)),ddof=1)); sl=max(sl,0.42)
    lo,hi=interval_from_sigma(hy,sl)
    metrics=pd.DataFrame([
        evaluate("LSTM 分支",tp["actual"],tp["lstm"],*interval_from_sigma(tp["lstm"],sl),wts[0]),
        evaluate("TCN 分支",tp["actual"],tp["tcn"],*interval_from_sigma(tp["tcn"],sl),wts[1]),
        evaluate("XGBoost 分支",tp["actual"],tp["xgb"],*interval_from_sigma(tp["xgb"],sl),wts[2]),
        evaluate("联合模型",tp["actual"],hy,lo,hi,None)])
    preds=pd.DataFrame({"date":pd.to_datetime(tp["date"]),"actual":tp["actual"],
        "lstm_branch":tp["lstm"],"tcn_branch":tp["tcn"],"xgboost_branch":tp["xgb"],
        "hybrid_ensemble":hy,"hybrid_lower95":lo,"hybrid_upper95":hi})
    fe=recursive_forecast(bundle,raw,wts,FORECAST_HORIZON)
    lf,hf=interval_from_sigma(fe["hybrid_ensemble"].to_numpy(dtype=float),sl)
    fe["hybrid_lower95"]=lf; fe["hybrid_upper95"]=hf; xgb_imp=feature_importance(bundle)

    # Pipeline B: Fusion + XGBoost residual
    tbb=fit_fusion_base(frame,seq_cols,ts,seed_offset=300)
    rd=frame.loc[frame["date"]<ts,"date"].to_numpy(dtype="datetime64[ns]")
    rb=predict_fusion_base(tbb,frame,rd); rd=rb["date"]
    rx,ra,_,_=residual_feature_frame(frame,seq_cols,rd,rb["base_pred"]); rt=ra-rb["base_pred"]
    ro=fit_residual_optimizer(rx,rt,seed_offset=200); rc=predict_residual_optimizer(ro,rx)
    rf_val=apply_residual_correction(rb["base_pred"],rc); ma_r=moving_average_predictions(frame,rd)
    sb=float(np.std(np.log1p(ra)-np.log1p(np.clip(rb["base_pred"],0.0,None)),ddof=1))
    sma=float(np.std(np.log1p(ra)-np.log1p(np.clip(ma_r,0.0,None)),ddof=1))
    sf=float(np.std(np.log1p(ra)-np.log1p(np.clip(rf_val,0.0,None)),ddof=1)); sf=max(sf,0.42)
    tb=predict_fusion_base(tbb,frame,td)
    tx,ta,_,_=residual_feature_frame(frame,seq_cols,td,tb["base_pred"])
    tc_val=predict_residual_optimizer(ro,tx); fp=apply_residual_correction(tb["base_pred"],tc_val)
    ma30t=moving_average_predictions(frame,td)
    ml,mu=interval_from_sigma(ma30t,sma); bl,bu=interval_from_sigma(tb["base_pred"],sb)
    fl,fu=interval_from_sigma(fp,sf)
    rm=pd.DataFrame([
        evaluate("MA30 局部基线",ta,ma30t,ml,mu,None),
        evaluate("LSTM-TCN 融合基础模型",ta,tb["base_pred"],bl,bu,None),
        evaluate("LSTM-TCN + XGBoost 残差修正",ta,fp,fl,fu,None)])
    rp=pd.DataFrame({"date":pd.to_datetime(tb["date"]),"actual":ta,"ma30":ma30t,
        "lstm_tcn_base":tb["base_pred"],"xgboost_residual_correction":tc_val,
        "final_prediction":fp,"final_lower95":fl,"final_upper95":fu})
    fbb=fit_fusion_base(frame,seq_cols,ld+pd.Timedelta(days=1),seed_offset=600)
    frd=frame["date"].to_numpy(dtype="datetime64[ns]")
    frb=predict_fusion_base(fbb,frame,frd); frd=frb["date"]
    frx,fra,_,_=residual_feature_frame(frame,seq_cols,frd,frb["base_pred"]); frt=fra-frb["base_pred"]
    fro=fit_residual_optimizer(frx,frt,seed_offset=500)
    rfc=recursive_residual_forecast(fbb,fro,raw,seq_cols,FORECAST_HORIZON)
    fl2,fu2=interval_from_sigma(rfc["final_prediction"].to_numpy(dtype=float),sf)
    rfc["final_lower95"]=fl2; rfc["final_upper95"]=fu2; rxi=residual_importance(fro)

    # Save original outputs
    metrics.to_csv(TABLE_DIR/"ml_residual_model_metrics.csv",index=False,encoding="utf-8-sig")
    preds.to_csv(TABLE_DIR/"ml_residual_holdout_predictions.csv",index=False,encoding="utf-8-sig")
    fe.to_csv(TABLE_DIR/"ml_residual_30day_forecast.csv",index=False,encoding="utf-8-sig")
    xgb_imp.to_csv(TABLE_DIR/"ml_residual_xgb_feature_importance.csv",index=False,encoding="utf-8-sig")
    rm.to_csv(TABLE_DIR/"ml_residual_fusion_metrics.csv",index=False,encoding="utf-8-sig")
    rp.to_csv(TABLE_DIR/"ml_residual_fusion_holdout.csv",index=False,encoding="utf-8-sig")
    rfc.to_csv(TABLE_DIR/"ml_residual_fusion_forecast.csv",index=False,encoding="utf-8-sig")
    rxi.to_csv(TABLE_DIR/"ml_residual_fusion_xgb_importance.csv",index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR/"ml_residual_results_summary.xlsx") as w:
        metrics.to_excel(w,sheet_name="ensemble_metrics",index=False)
        preds.to_excel(w,sheet_name="ensemble_holdout",index=False)
        fe.to_excel(w,sheet_name="ensemble_forecast",index=False)
        xgb_imp.to_excel(w,sheet_name="ensemble_xgb_imp",index=False)
        rm.to_excel(w,sheet_name="fusion_metrics",index=False)
        rp.to_excel(w,sheet_name="fusion_holdout",index=False)
        rfc.to_excel(w,sheet_name="fusion_forecast",index=False)
        rxi.to_excel(w,sheet_name="fusion_xgb_imp",index=False)

    # Original figures
    plot_overview(raw,battle_I,FIG_DIR/"fig01_data_overview.png")
    plot_architecture(wts,FIG_DIR/"fig02_architecture.png")
    plot_holdout(preds,FIG_DIR/"fig03_holdout.png")
    plot_weights_importance(wts,xgb_imp,FIG_DIR/"fig04_weights_importance.png")
    plot_forecast(frame,fe,FIG_DIR/"fig05_forecast.png")
    plot_residual_architecture(FIG_DIR/"fig06_residual_architecture.png")
    plot_residual_holdout(rp,FIG_DIR/"fig07_residual_holdout.png")
    plot_residual_correction(rp,rxi,FIG_DIR/"fig08_residual_correction.png")
    plot_residual_forecast(frame,rfc,battle_I,FIG_DIR/"fig09_residual_forecast.png")

    # ---- Shulitongji PNG additions ----
    print("\n--- Shulitongji PNG additions ---")
    ada=frame["date"].to_numpy(dtype="datetime64[ns]")
    fba=predict_fusion_base(fbb,frame,ada); fd2=fba["date"]; ff=fba["base_pred"]
    rxa,_,rda,_,_=residual_feature_frame(frame,seq_cols,ada[SEQ_LEN:],ff)
    ca=predict_residual_optimizer(fro,rxa); ffa=apply_residual_correction(ff,ca)
    rbv=float(rm.loc[rm["model"].eq("LSTM-TCN 融合基础模型"),"RMSE"].iloc[0])
    rfv=float(rm.loc[rm["model"].eq("LSTM-TCN + XGBoost 残差修正"),"RMSE"].iloc[0])
    mfv=float(rm.loc[rm["model"].eq("LSTM-TCN + XGBoost 残差修正"),"MAE"].iloc[0])
    plot_shulitongji_regression(frame,rfc,ff,fd2,ffa,rda,rbv,rfv,mfv)
    plot_shulitongji_overlay(frame,rfc,ff,fd2,ffa,rda,battle_I,rbv,rfv,mfv)

    # Manifest
    mf={"csv_path":str(CSV_PATH),"battles_csv":str(BATTLES_CSV),"rows":int(len(raw)),
        "target":"drone_loss","external_regressor":"battle_intensity (russia_ukraine_battles.csv)",
        "date_start":raw["date"].min().date().isoformat(),"date_end":raw["date"].max().date().isoformat(),
        "lookback_days":SEQ_LEN,"horizon_days":FORECAST_HORIZON,
        "base_test_rmse":rbv,"final_test_rmse":rfv,
        "forecast_mean_30d":float(rfc["final_prediction"].mean()),
        "forecast_day30":float(rfc["final_prediction"].iloc[-1]),
        "drone_mean":float(raw["drone_loss"].mean()),"drone_max":float(raw["drone_loss"].max()),
        "feature_groups":groups}
    (TABLE_DIR/"ml_residual_run_manifest.json").write_text(json.dumps(mf,ensure_ascii=False,indent=2),encoding="utf-8")
    print(rm.to_string(index=False))
    print(json.dumps(mf,ensure_ascii=False,indent=2))
    print(f"\nDone. Output: {OUT_DIR}")
    print(f"Additional PNGs: {BASE_DIR/'drone_ML_Regression.png'}, {BASE_DIR/'drone_ML_Overlay.png'}")

if __name__=="__main__":
    main()
