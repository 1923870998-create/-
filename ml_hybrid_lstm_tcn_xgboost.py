#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM-TCN fusion with XGBoost residual correction.

This is the final machine-learning row pipeline.  It uses all observations in
``campaign_feature_wide_by_date.csv`` to build rolling 30-day supervised
samples.  A jointly trained LSTM-TCN fusion network produces the base forecast;
an XGBoost residual optimizer then learns the systematic errors left by the
deep sequence model.  The structure is designed to differ clearly from the
groupmates' ARIMA+Prophet and ARIMA+GPR statistical routes.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
CSV_PATH = PROJECT_DIR / "campaign_feature_wide_by_date.csv"
OUT_DIR = BASE_DIR / "ml_residual_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MPL_DIR = OUT_DIR / ".mplconfig"
for path in (OUT_DIR, FIG_DIR, TABLE_DIR, MPL_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR / "xdg-cache"))

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEQ_LEN = 30
EVAL_DAYS = 8
FORECAST_HORIZON = 30
SEED = 20260611
random.seed(SEED)
np.random.seed(SEED)

ROYAL_BLUE = "#4169E1"
DARK_ORANGE = "#FF8C00"
CRIMSON = "#DC143C"
DODGER_BLUE = "#1E90FF"
GRAY_LINE = "#808080"
GRID_GRAY = "#CCCCCC"
TEXT_DARK = "#333333"
BLUE_TO_RED = LinearSegmentedColormap.from_list(
    "BlueRed",
    ["#D6E4F0", "#6BAED6", "#3182BD", "#C9898F", "#E05554", "#B71C1C"],
)

CH_FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")
if CH_FONT_PATH.exists():
    CH_FONT = FontProperties(fname=str(CH_FONT_PATH))
    plt.rcParams["font.family"] = CH_FONT.get_name()
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 250,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    }
)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


class NumpyLSTMRegressor:
    """Single-layer LSTM with an output linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 10,
        lr: float = 0.007,
        epochs: int = 135,
        batch_size: int = 32,
        patience: int = 18,
        seed: int = SEED,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()

    def _init_params(self) -> dict[str, np.ndarray]:
        in_dim = self.input_dim + self.hidden_dim
        scale = 1.0 / math.sqrt(in_dim)

        def w() -> np.ndarray:
            return self.rng.normal(0.0, scale, size=(in_dim, self.hidden_dim))

        params = {
            "Wf": w(),
            "Wi": w(),
            "Wo": w(),
            "Wg": w(),
            "bf": np.zeros(self.hidden_dim),
            "bi": np.zeros(self.hidden_dim),
            "bo": np.zeros(self.hidden_dim),
            "bg": np.zeros(self.hidden_dim),
            "Wy": self.rng.normal(0.0, 1.0 / math.sqrt(self.hidden_dim), size=(self.hidden_dim,)),
            "by": np.zeros(1),
        }
        params["bf"] += 0.5
        return params

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_dim))
        c = np.zeros((batch, self.hidden_dim))
        cache: dict[str, list[np.ndarray]] = {
            "concat": [],
            "f": [],
            "i": [],
            "o": [],
            "g": [],
            "c": [],
            "c_prev": [],
            "h": [],
        }
        for t in range(seq_len):
            xt = x[:, t, :]
            concat = np.concatenate([xt, h], axis=1)
            f = sigmoid(concat @ self.params["Wf"] + self.params["bf"])
            i = sigmoid(concat @ self.params["Wi"] + self.params["bi"])
            o = sigmoid(concat @ self.params["Wo"] + self.params["bo"])
            g = np.tanh(concat @ self.params["Wg"] + self.params["bg"])
            c_prev = c
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            cache["concat"].append(concat)
            cache["f"].append(f)
            cache["i"].append(i)
            cache["o"].append(o)
            cache["g"].append(g)
            cache["c_prev"].append(c_prev)
            cache["c"].append(c)
            cache["h"].append(h)
        y_hat = h @ self.params["Wy"] + self.params["by"][0]
        return y_hat, cache

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        y_hat, cache = self._forward(x)
        batch = x.shape[0]
        diff = y_hat - y
        loss = float(np.mean(diff**2))
        dy = (2.0 / batch) * diff
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        grads["Wy"] += cache["h"][-1].T @ dy
        grads["by"] += np.array([np.sum(dy)])
        dh_next = np.outer(dy, self.params["Wy"])
        dc_next = np.zeros_like(dh_next)

        for t in reversed(range(x.shape[1])):
            f = cache["f"][t]
            i = cache["i"][t]
            o = cache["o"][t]
            g = cache["g"][t]
            c = cache["c"][t]
            c_prev = cache["c_prev"][t]
            concat = cache["concat"][t]

            tanh_c = np.tanh(c)
            do = dh_next * tanh_c
            dao = do * o * (1.0 - o)
            dc = dh_next * o * (1.0 - tanh_c**2) + dc_next
            df = dc * c_prev
            daf = df * f * (1.0 - f)
            di = dc * g
            dai = di * i * (1.0 - i)
            dg = dc * i
            dag = dg * (1.0 - g**2)

            grads["Wf"] += concat.T @ daf
            grads["Wi"] += concat.T @ dai
            grads["Wo"] += concat.T @ dao
            grads["Wg"] += concat.T @ dag
            grads["bf"] += np.sum(daf, axis=0)
            grads["bi"] += np.sum(dai, axis=0)
            grads["bo"] += np.sum(dao, axis=0)
            grads["bg"] += np.sum(dag, axis=0)

            dconcat = (
                daf @ self.params["Wf"].T
                + dai @ self.params["Wi"].T
                + dao @ self.params["Wo"].T
                + dag @ self.params["Wg"].T
            )
            dh_next = dconcat[:, self.input_dim :]
            dc_next = dc * f

        global_norm = math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
        if global_norm > 5.0:
            scale = 5.0 / (global_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}
        return loss, grads

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLSTMRegressor":
        split = max(24, int(len(x) * 0.85))
        split = min(split, len(x) - 12)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        m = {name: np.zeros_like(value) for name, value in self.params.items()}
        v = {name: np.zeros_like(value) for name, value in self.params.items()}
        beta1, beta2 = 0.9, 0.999
        best_loss = float("inf")
        best_params = {k: val.copy() for k, val in self.params.items()}
        stagnant = 0
        step = 0
        for epoch in range(self.epochs):
            order = self.rng.permutation(len(x_train))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                _, grads = self._loss_and_grads(x_train[idx], y_train[idx])
                step += 1
                for name in self.params:
                    m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
                    v[name] = beta2 * v[name] + (1.0 - beta2) * (grads[name] ** 2)
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            val_loss = float(np.mean((self.predict_scaled(x_val) - y_val) ** 2))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_params = {k: val.copy() for k, val in self.params.items()}
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= self.patience:
                break
        self.params = best_params
        self.best_val_loss_ = best_loss
        self.epochs_run_ = epoch + 1
        return self

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for start in range(0, len(x), 256):
            y_hat, _ = self._forward(x[start : start + 256])
            preds.append(y_hat)
        return np.concatenate(preds, axis=0)


class NumpyTCNRegressor:
    """A compact causal dilated-convolution regressor.

    It uses the final receptive field of dilations 1, 2, 4, 8 and 16 with a
    kernel size of 3. This keeps the model light enough for coursework while
    preserving the key TCN idea: parallel causal filters at several horizons.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int = 8,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        lr: float = 0.018,
        epochs: int = 260,
        batch_size: int = 64,
        patience: int = 28,
        seed: int = SEED + 101,
    ) -> None:
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(input_dim * kernel_size)
        self.W = self.rng.normal(0.0, scale, size=(len(self.dilations), kernel_size, input_dim, channels))
        self.b = np.zeros((len(self.dilations), channels))
        self.Wy = self.rng.normal(0.0, 1.0 / math.sqrt(len(self.dilations) * channels), size=(len(self.dilations) * channels,))
        self.by = 0.0

    def _features(self, x: np.ndarray) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        batch, seq_len, _ = x.shape
        z = np.zeros((batch, len(self.dilations), self.channels))
        selected: list[list[np.ndarray]] = []
        for d_i, dilation in enumerate(self.dilations):
            selected_d = []
            for k in range(self.kernel_size):
                idx = max(0, seq_len - 1 - k * dilation)
                xk = x[:, idx, :]
                selected_d.append(xk)
                z[:, d_i, :] += xk @ self.W[d_i, k]
            z[:, d_i, :] += self.b[d_i]
            selected.append(selected_d)
        h = np.tanh(z).reshape(batch, -1)
        return h, selected

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray | float]]:
        h, selected = self._features(x)
        pred = h @ self.Wy + self.by
        batch = len(x)
        diff = pred - y
        loss = float(np.mean(diff**2))
        dy = (2.0 / batch) * diff
        gWy = h.T @ dy
        gby = float(np.sum(dy))
        dh = np.outer(dy, self.Wy).reshape(batch, len(self.dilations), self.channels)
        dz = dh * (1.0 - h.reshape(batch, len(self.dilations), self.channels) ** 2)
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        for d_i in range(len(self.dilations)):
            gb[d_i] += np.sum(dz[:, d_i, :], axis=0)
            for k in range(self.kernel_size):
                gW[d_i, k] += selected[d_i][k].T @ dz[:, d_i, :]
        norm = math.sqrt(float(np.sum(gW**2) + np.sum(gb**2) + np.sum(gWy**2) + gby**2))
        if norm > 5.0:
            s = 5.0 / (norm + 1e-8)
            gW, gb, gWy, gby = gW * s, gb * s, gWy * s, gby * s
        return loss, {"W": gW, "b": gb, "Wy": gWy, "by": gby}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyTCNRegressor":
        split = max(24, int(len(x) * 0.85))
        split = min(split, len(x) - 12)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        best_loss = float("inf")
        best = (self.W.copy(), self.b.copy(), self.Wy.copy(), float(self.by))
        stagnant = 0
        for epoch in range(self.epochs):
            order = self.rng.permutation(len(x_train))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                _, grads = self._loss_and_grads(x_train[idx], y_train[idx])
                self.W -= self.lr * grads["W"]
                self.b -= self.lr * grads["b"]
                self.Wy -= self.lr * grads["Wy"]
                self.by -= self.lr * grads["by"]
            val_loss = float(np.mean((self.predict_scaled(x_val) - y_val) ** 2))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best = (self.W.copy(), self.b.copy(), self.Wy.copy(), float(self.by))
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= self.patience:
                break
        self.W, self.b, self.Wy, self.by = best
        self.best_val_loss_ = best_loss
        self.epochs_run_ = epoch + 1
        return self

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        h, _ = self._features(x)
        return h @ self.Wy + self.by


class LstmTcnFusionRegressor:
    """Jointly trained LSTM-TCN fusion network.

    The LSTM branch keeps sequential memory; the TCN branch reads the same
    sequence through dilated causal filters. Their latent states are concatenated
    and optimized with one shared output head, so this is not a post-hoc average
    of two separately trained models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 10,
        tcn_channels: int = 8,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        lr: float = 0.008,
        epochs: int = 125,
        batch_size: int = 32,
        patience: int = 20,
        seed: int = SEED + 500,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tcn_channels = tcn_channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()

    def _init_params(self) -> dict[str, np.ndarray]:
        lstm_in = self.input_dim + self.hidden_dim
        lstm_scale = 1.0 / math.sqrt(lstm_in)
        tcn_scale = 1.0 / math.sqrt(self.input_dim * self.kernel_size)

        def lstm_w() -> np.ndarray:
            return self.rng.normal(0.0, lstm_scale, size=(lstm_in, self.hidden_dim))

        tcn_dim = len(self.dilations) * self.tcn_channels
        fusion_dim = self.hidden_dim + tcn_dim
        params = {
            "Wf": lstm_w(),
            "Wi": lstm_w(),
            "Wo": lstm_w(),
            "Wg": lstm_w(),
            "bf": np.zeros(self.hidden_dim) + 0.5,
            "bi": np.zeros(self.hidden_dim),
            "bo": np.zeros(self.hidden_dim),
            "bg": np.zeros(self.hidden_dim),
            "Wtcn": self.rng.normal(
                0.0,
                tcn_scale,
                size=(len(self.dilations), self.kernel_size, self.input_dim, self.tcn_channels),
            ),
            "btcn": np.zeros((len(self.dilations), self.tcn_channels)),
            "Wy": self.rng.normal(0.0, 1.0 / math.sqrt(fusion_dim), size=(fusion_dim,)),
            "by": np.zeros(1),
        }
        return params

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_dim))
        c = np.zeros((batch, self.hidden_dim))
        lstm_cache: dict[str, list[np.ndarray]] = {
            "concat": [],
            "f": [],
            "i": [],
            "o": [],
            "g": [],
            "c_prev": [],
            "c": [],
            "h": [],
        }
        for t in range(seq_len):
            xt = x[:, t, :]
            concat = np.concatenate([xt, h], axis=1)
            f = sigmoid(concat @ self.params["Wf"] + self.params["bf"])
            i = sigmoid(concat @ self.params["Wi"] + self.params["bi"])
            o = sigmoid(concat @ self.params["Wo"] + self.params["bo"])
            g = np.tanh(concat @ self.params["Wg"] + self.params["bg"])
            c_prev = c
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            lstm_cache["concat"].append(concat)
            lstm_cache["f"].append(f)
            lstm_cache["i"].append(i)
            lstm_cache["o"].append(o)
            lstm_cache["g"].append(g)
            lstm_cache["c_prev"].append(c_prev)
            lstm_cache["c"].append(c)
            lstm_cache["h"].append(h)

        z = np.zeros((batch, len(self.dilations), self.tcn_channels))
        selected: list[list[np.ndarray]] = []
        for d_i, dilation in enumerate(self.dilations):
            selected_d = []
            for k in range(self.kernel_size):
                idx = max(0, seq_len - 1 - k * dilation)
                xk = x[:, idx, :]
                selected_d.append(xk)
                z[:, d_i, :] += xk @ self.params["Wtcn"][d_i, k]
            z[:, d_i, :] += self.params["btcn"][d_i]
            selected.append(selected_d)
        h_tcn_3d = np.tanh(z)
        h_tcn = h_tcn_3d.reshape(batch, -1)
        fusion = np.concatenate([h, h_tcn], axis=1)
        pred = fusion @ self.params["Wy"] + self.params["by"][0]
        return pred, {
            "lstm": lstm_cache,
            "selected": selected,
            "h_lstm": h,
            "h_tcn_3d": h_tcn_3d,
            "fusion": fusion,
        }

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        pred, cache = self._forward(x)
        batch = len(x)
        diff = pred - y
        loss = float(np.mean(diff**2))
        dy = (2.0 / batch) * diff
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}

        fusion = cache["fusion"]
        grads["Wy"] += fusion.T @ dy
        grads["by"] += np.array([np.sum(dy)])
        d_fusion = np.outer(dy, self.params["Wy"])
        dh_lstm = d_fusion[:, : self.hidden_dim]
        dh_tcn = d_fusion[:, self.hidden_dim :].reshape(batch, len(self.dilations), self.tcn_channels)

        lstm_cache = cache["lstm"]
        dh_next = dh_lstm
        dc_next = np.zeros_like(dh_next)
        for t in reversed(range(x.shape[1])):
            f = lstm_cache["f"][t]
            i = lstm_cache["i"][t]
            o = lstm_cache["o"][t]
            g = lstm_cache["g"][t]
            c = lstm_cache["c"][t]
            c_prev = lstm_cache["c_prev"][t]
            concat = lstm_cache["concat"][t]

            tanh_c = np.tanh(c)
            do = dh_next * tanh_c
            dao = do * o * (1.0 - o)
            dc = dh_next * o * (1.0 - tanh_c**2) + dc_next
            df = dc * c_prev
            daf = df * f * (1.0 - f)
            di = dc * g
            dai = di * i * (1.0 - i)
            dg = dc * i
            dag = dg * (1.0 - g**2)

            grads["Wf"] += concat.T @ daf
            grads["Wi"] += concat.T @ dai
            grads["Wo"] += concat.T @ dao
            grads["Wg"] += concat.T @ dag
            grads["bf"] += np.sum(daf, axis=0)
            grads["bi"] += np.sum(dai, axis=0)
            grads["bo"] += np.sum(dao, axis=0)
            grads["bg"] += np.sum(dag, axis=0)
            dconcat = (
                daf @ self.params["Wf"].T
                + dai @ self.params["Wi"].T
                + dao @ self.params["Wo"].T
                + dag @ self.params["Wg"].T
            )
            dh_next = dconcat[:, self.input_dim :]
            dc_next = dc * f

        h_tcn_3d = cache["h_tcn_3d"]
        dz = dh_tcn * (1.0 - h_tcn_3d**2)
        selected = cache["selected"]
        for d_i in range(len(self.dilations)):
            grads["btcn"][d_i] += np.sum(dz[:, d_i, :], axis=0)
            for k in range(self.kernel_size):
                grads["Wtcn"][d_i, k] += selected[d_i][k].T @ dz[:, d_i, :]

        global_norm = math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
        if global_norm > 5.0:
            scale = 5.0 / (global_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}
        return loss, grads

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LstmTcnFusionRegressor":
        split = max(36, int(len(x) * 0.85))
        split = min(split, len(x) - 18)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        m = {name: np.zeros_like(value) for name, value in self.params.items()}
        v = {name: np.zeros_like(value) for name, value in self.params.items()}
        beta1, beta2 = 0.9, 0.999
        best_loss = float("inf")
        best_params = {k: val.copy() for k, val in self.params.items()}
        stagnant = 0
        step = 0
        for epoch in range(self.epochs):
            order = self.rng.permutation(len(x_train))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                _, grads = self._loss_and_grads(x_train[idx], y_train[idx])
                step += 1
                for name in self.params:
                    m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
                    v[name] = beta2 * v[name] + (1.0 - beta2) * (grads[name] ** 2)
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            val_loss = float(np.mean((self.predict_scaled(x_val) - y_val) ** 2))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_params = {k: val.copy() for k, val in self.params.items()}
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= self.patience:
                break
        self.params = best_params
        self.best_val_loss_ = best_loss
        self.epochs_run_ = epoch + 1
        return self

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for start in range(0, len(x), 256):
            pred, _ = self._forward(x[start : start + 256])
            preds.append(pred)
        return np.concatenate(preds, axis=0)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in df.columns:
        if col not in {"date", "active_campaigns"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["active_campaigns"] = df["active_campaigns"].fillna("无显式战役标签")
    df["tank_loss_clean"] = df["tank_loss_clean"].clip(lower=0.0)
    return df


def campaign_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "day",
        "tank_loss_raw",
        "tank_loss_clean",
        "drone_loss",
        "active_campaign_count",
        "active_campaigns",
    }
    return [col for col in df.columns if col not in excluded]


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    y = df["tank_loss_clean"].astype(float).clip(lower=0.0)
    drone = df["drone_loss"].astype(float).clip(lower=0.0)
    frame = pd.DataFrame(index=df.index)
    frame["date"] = df["date"]
    frame["target"] = y
    frame["y_log"] = np.log1p(y)
    ma30 = y.shift(1).rolling(30, min_periods=2).mean()
    ma30 = ma30.fillna(y.shift(1).expanding(min_periods=1).mean()).fillna(y.expanding().mean())
    frame["ma30_target_log"] = np.log1p(ma30)
    frame["resid_ma30_log"] = frame["y_log"] - frame["ma30_target_log"]
    for window in (7, 14, 30):
        roll = y.shift(1).rolling(window, min_periods=2)
        frame[f"tank_roll{window}_mean_log"] = np.log1p(roll.mean().fillna(y.shift(1).expanding().mean()).fillna(y.mean()))
    frame["tank_roll7_std_log"] = np.log1p(y.shift(1).rolling(7, min_periods=2).std().fillna(0.0))
    frame["tank_diff1_log"] = frame["y_log"].diff().fillna(0.0)
    frame["tank_zero_lag1"] = (y.shift(1).fillna(0.0) <= 0).astype(float)
    frame["drone_roll7_mean_log"] = np.log1p(drone.shift(1).rolling(7, min_periods=2).mean().fillna(0.0))
    frame["drone_roll30_mean_log"] = np.log1p(drone.shift(1).rolling(30, min_periods=2).mean().fillna(0.0))
    frame["drone_to_tank_roll14"] = (
        drone.shift(1).rolling(14, min_periods=2).sum()
        / (y.shift(1).rolling(14, min_periods=2).sum() + 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dow = df["date"].dt.dayofweek.to_numpy()
    month = df["date"].dt.month.to_numpy()
    ordinal = (df["date"] - df["date"].min()).dt.days.to_numpy(dtype=float)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["time_index"] = ordinal / max(1.0, ordinal.max())
    frame["active_campaign_count"] = df["active_campaign_count"].astype(float)
    camps = campaign_columns(df)
    for col in camps:
        frame[col] = df[col].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sequence_cols = [
        "y_log",
        "ma30_target_log",
        "resid_ma30_log",
        "tank_roll7_mean_log",
        "tank_roll14_mean_log",
        "tank_roll30_mean_log",
        "tank_roll7_std_log",
        "tank_diff1_log",
        "tank_zero_lag1",
        "drone_roll7_mean_log",
        "drone_roll30_mean_log",
        "drone_to_tank_roll14",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "time_index",
        "active_campaign_count",
    ] + camps
    groups = {
        "历史与残差": ["y_log", "ma30_target_log", "resid_ma30_log"],
        "坦克滚动动量": [c for c in sequence_cols if c.startswith("tank_")],
        "无人机强度": [c for c in sequence_cols if c.startswith("drone_")],
        "日历节奏": ["dow_sin", "dow_cos", "month_sin", "month_cos", "time_index"],
        "战役标签": camps + ["active_campaign_count"],
    }
    return frame, sequence_cols, camps, groups


def make_sequences(frame: pd.DataFrame, feature_cols: Sequence[str], x_scaler: Scaler, y_scaler: Scaler) -> dict[str, np.ndarray]:
    x_values = x_scaler.transform(frame[list(feature_cols)].to_numpy(dtype=float))
    y_scaled = y_scaler.transform(frame[["resid_ma30_log"]].to_numpy(dtype=float)).ravel()
    xs, ys, dates, raw_y, base_logs = [], [], [], [], []
    for end in range(SEQ_LEN, len(frame)):
        xs.append(x_values[end - SEQ_LEN : end])
        ys.append(y_scaled[end])
        dates.append(frame.loc[end, "date"])
        raw_y.append(frame.loc[end, "target"])
        base_logs.append(frame.loc[end, "ma30_target_log"])
    return {
        "X": np.asarray(xs, dtype=float),
        "y_scaled": np.asarray(ys, dtype=float),
        "date": np.asarray(dates, dtype="datetime64[ns]"),
        "y_raw": np.asarray(raw_y, dtype=float),
        "baseline_log": np.asarray(base_logs, dtype=float),
    }


def make_tabular(frame: pd.DataFrame, camps: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    dates, raw_y, base_logs, residuals = [], [], [], []
    y_log = frame["y_log"].to_numpy(dtype=float)
    resid = frame["resid_ma30_log"].to_numpy(dtype=float)
    for end in range(SEQ_LEN, len(frame)):
        row: dict[str, float] = {}
        for lag in (1, 2, 3, 7, 14, 30):
            row[f"lag_y_log_{lag}"] = float(y_log[end - lag])
            row[f"lag_resid_{lag}"] = float(resid[end - lag])
        for col in [
            "ma30_target_log",
            "tank_roll7_mean_log",
            "tank_roll14_mean_log",
            "tank_roll30_mean_log",
            "tank_roll7_std_log",
            "drone_roll7_mean_log",
            "drone_roll30_mean_log",
            "drone_to_tank_roll14",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "time_index",
            "active_campaign_count",
        ]:
            row[col] = float(frame.loc[end, col])
        for col in camps:
            row[col] = float(frame.loc[end, col])
        rows.append(row)
        dates.append(frame.loc[end, "date"])
        raw_y.append(frame.loc[end, "target"])
        base_logs.append(frame.loc[end, "ma30_target_log"])
        residuals.append(frame.loc[end, "resid_ma30_log"])
    return (
        pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        np.asarray(residuals, dtype=float),
        np.asarray(dates, dtype="datetime64[ns]"),
        np.asarray(raw_y, dtype=float),
        np.asarray(base_logs, dtype=float),
    )


def inverse_pred(pred_scaled: np.ndarray, y_scaler: Scaler, baseline_log: np.ndarray) -> np.ndarray:
    pred_resid = y_scaler.inverse(pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(np.expm1(baseline_log + pred_resid), 0.0, None)


def build_xgb_branch(seed: int):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=260,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric="rmse",
    )
    return model, "xgboost.XGBRegressor"


def fit_bundle(frame: pd.DataFrame, sequence_cols: Sequence[str], camps: Sequence[str], train_end: pd.Timestamp, seed_offset: int = 0) -> dict:
    train_rows = (frame["date"] < train_end).to_numpy()
    x_scaler = Scaler.fit(frame.loc[train_rows, sequence_cols].to_numpy(dtype=float))
    y_scaler = Scaler.fit(frame.loc[train_rows, ["resid_ma30_log"]].to_numpy(dtype=float))
    seq = make_sequences(frame, sequence_cols, x_scaler, y_scaler)
    seq_train = pd.to_datetime(seq["date"]) < train_end
    lstm = NumpyLSTMRegressor(input_dim=len(sequence_cols), hidden_dim=10, seed=SEED + seed_offset + 11)
    tcn = NumpyTCNRegressor(input_dim=len(sequence_cols), channels=8, seed=SEED + seed_offset + 22)
    print(f"Training LSTM/TCN branches before {train_end.date()} with {int(seq_train.sum())} sequences")
    lstm.fit(seq["X"][seq_train], seq["y_scaled"][seq_train])
    tcn.fit(seq["X"][seq_train], seq["y_scaled"][seq_train])

    tab_x, tab_resid, tab_dates, _, _ = make_tabular(frame, camps)
    tab_train = pd.to_datetime(tab_dates) < train_end
    tab_scaler = StandardScaler()
    y_tab_scaled = y_scaler.transform(tab_resid.reshape(-1, 1)).ravel()
    xgb, xgb_impl = build_xgb_branch(SEED + seed_offset + 33)
    x_train = tab_scaler.fit_transform(tab_x.loc[tab_train].to_numpy(dtype=float))
    xgb.fit(x_train, y_tab_scaled[tab_train])
    return {
        "lstm": lstm,
        "tcn": tcn,
        "xgb": xgb,
        "xgb_impl": xgb_impl,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "tab_scaler": tab_scaler,
        "tab_columns": list(tab_x.columns),
        "sequence_cols": list(sequence_cols),
        "camps": list(camps),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame, target_dates: Sequence[np.datetime64]) -> dict[str, np.ndarray]:
    target_dates = np.asarray(target_dates, dtype="datetime64[ns]")
    seq = make_sequences(frame, bundle["sequence_cols"], bundle["x_scaler"], bundle["y_scaler"])
    seq_mask = np.isin(seq["date"], target_dates)
    lstm_raw = inverse_pred(bundle["lstm"].predict_scaled(seq["X"][seq_mask]), bundle["y_scaler"], seq["baseline_log"][seq_mask])
    tcn_raw = inverse_pred(bundle["tcn"].predict_scaled(seq["X"][seq_mask]), bundle["y_scaler"], seq["baseline_log"][seq_mask])

    tab_x, _, tab_dates, tab_y, tab_base = make_tabular(frame, bundle["camps"])
    tab_mask = np.isin(tab_dates, target_dates)
    x_tab = bundle["tab_scaler"].transform(tab_x.loc[tab_mask, bundle["tab_columns"]].to_numpy(dtype=float))
    xgb_raw = inverse_pred(bundle["xgb"].predict(x_tab), bundle["y_scaler"], tab_base[tab_mask])
    return {
        "date": tab_dates[tab_mask],
        "actual": tab_y[tab_mask],
        "lstm": lstm_raw,
        "tcn": tcn_raw,
        "xgb": xgb_raw,
    }


def branch_weights(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    errors = np.array([rmse(y_true, preds[:, i]) for i in range(preds.shape[1])], dtype=float)
    raw = 1.0 / np.maximum(errors, 1e-6)
    raw = raw / raw.sum()
    floor = 0.15
    return floor + (1.0 - floor * len(raw)) * raw


def interval_from_sigma(pred: np.ndarray, sigma_log: float) -> tuple[np.ndarray, np.ndarray]:
    pred_log = np.log1p(np.clip(pred, 0.0, None))
    lower = np.clip(np.expm1(pred_log - 1.96 * sigma_log), 0.0, None)
    upper = np.clip(np.expm1(pred_log + 1.96 * sigma_log), 0.0, None)
    return lower, upper


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: float | None = None) -> dict:
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "PI95_coverage": coverage(y_true, lower, upper),
        "PI95_mean_width": float(np.mean(upper - lower)),
        "ensemble_weight": np.nan if weight is None else float(weight),
    }


def append_future_row(raw_df: pd.DataFrame, yhat: float, drone_hat: float) -> pd.DataFrame:
    last = raw_df.iloc[-1].copy()
    new = last.copy()
    new["date"] = last["date"] + pd.Timedelta(days=1)
    new["day"] = float(last["day"]) + 1
    new["tank_loss_raw"] = yhat
    new["tank_loss_clean"] = yhat
    new["drone_loss"] = drone_hat
    return pd.concat([raw_df, pd.DataFrame([new])], ignore_index=True)


def recursive_forecast(bundle: dict, raw_df: pd.DataFrame, weights: np.ndarray, horizon: int = 30) -> pd.DataFrame:
    working = raw_df.copy()
    rows = []
    for _ in range(horizon):
        drone_hat = float(np.mean(working["drone_loss"].astype(float).tail(30)))
        y_placeholder = float(np.mean(working["tank_loss_clean"].astype(float).tail(30)))
        temp_raw = append_future_row(working, y_placeholder, drone_hat)
        temp_frame, _, _, _ = build_feature_frame(temp_raw)
        next_date = np.asarray([temp_raw["date"].iloc[-1]], dtype="datetime64[ns]")
        pred = predict_bundle(bundle, temp_frame, next_date)
        branches = np.array([pred["lstm"][0], pred["tcn"][0], pred["xgb"][0]])
        hybrid = float(branches @ weights)
        rows.append(
            {
                "date": temp_raw["date"].iloc[-1],
                "lstm_branch": branches[0],
                "tcn_branch": branches[1],
                "xgboost_branch": branches[2],
                "hybrid_ensemble": hybrid,
            }
        )
        working = append_future_row(working, hybrid, drone_hat)
    return pd.DataFrame(rows)


def feature_importance(bundle: dict) -> pd.DataFrame:
    model = bundle["xgb"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = np.zeros(len(bundle["tab_columns"]))
    out = pd.DataFrame({"feature": bundle["tab_columns"], "importance": values})
    return out.sort_values("importance", ascending=False)


def fit_fusion_base(frame: pd.DataFrame, sequence_cols: Sequence[str], train_end: pd.Timestamp, seed_offset: int = 0) -> dict:
    train_rows = (frame["date"] < train_end).to_numpy()
    x_scaler = Scaler.fit(frame.loc[train_rows, sequence_cols].to_numpy(dtype=float))
    y_scaler = Scaler.fit(frame.loc[train_rows, ["resid_ma30_log"]].to_numpy(dtype=float))
    seq = make_sequences(frame, sequence_cols, x_scaler, y_scaler)
    train_seq = pd.to_datetime(seq["date"]) < train_end
    model = LstmTcnFusionRegressor(
        input_dim=len(sequence_cols),
        hidden_dim=10,
        tcn_channels=8,
        lr=0.0075,
        epochs=135,
        batch_size=32,
        patience=20,
        seed=SEED + seed_offset,
    )
    print(f"Training LSTM-TCN fusion before {train_end.date()} with {int(train_seq.sum())} sequences")
    model.fit(seq["X"][train_seq], seq["y_scaled"][train_seq])
    return {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler, "sequence_cols": list(sequence_cols)}


def predict_fusion_base(bundle: dict, frame: pd.DataFrame, target_dates: Sequence[np.datetime64]) -> dict[str, np.ndarray]:
    target_dates = np.asarray(target_dates, dtype="datetime64[ns]")
    seq = make_sequences(frame, bundle["sequence_cols"], bundle["x_scaler"], bundle["y_scaler"])
    mask = np.isin(seq["date"], target_dates)
    pred_scaled = bundle["model"].predict_scaled(seq["X"][mask])
    pred_raw = inverse_pred(pred_scaled, bundle["y_scaler"], seq["baseline_log"][mask])
    return {"date": seq["date"][mask], "actual": seq["y_raw"][mask], "base_pred": pred_raw}


def residual_feature_frame(frame: pd.DataFrame, camps: Sequence[str], target_dates: Sequence[np.datetime64], base_pred: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    target_dates = np.asarray(target_dates, dtype="datetime64[ns]")
    tab_x, _, tab_dates, tab_y, _ = make_tabular(frame, camps)
    mask = np.isin(tab_dates, target_dates)
    x = tab_x.loc[mask].reset_index(drop=True).copy()
    base_pred = np.asarray(base_pred, dtype=float)
    x["lstm_tcn_base_pred"] = base_pred
    x["lstm_tcn_base_log"] = np.log1p(np.clip(base_pred, 0.0, None))
    x["base_vs_ma30"] = x["lstm_tcn_base_log"] - x["ma30_target_log"].to_numpy(dtype=float)
    return x, tab_y[mask], tab_dates[mask], tab_x.columns.to_numpy()


def build_residual_booster(seed: int):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=90,
        max_depth=2,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=8.0,
        min_child_weight=5.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric="rmse",
    )
    return model, "xgboost.XGBRegressor"


def fit_residual_optimizer(x: pd.DataFrame, residual: np.ndarray, seed_offset: int = 0) -> dict:
    feature_cols = list(x.columns)
    values = x[feature_cols].to_numpy(dtype=float)
    residual = np.asarray(residual, dtype=float)
    split = max(24, int(len(values) * 0.72))
    split = min(split, len(values) - 12)
    probe_model, impl = build_residual_booster(SEED + seed_offset)
    probe_model.fit(values[:split], residual[:split])
    probe_pred = probe_model.predict(values[split:])
    grid = np.linspace(0.0, 1.0, 21)
    alpha = min(grid, key=lambda a: float(np.mean((residual[split:] - a * probe_pred) ** 2)))
    model, _ = build_residual_booster(SEED + seed_offset + 17)
    model.fit(x[feature_cols].to_numpy(dtype=float), residual)
    return {"model": model, "implementation": impl, "feature_cols": feature_cols, "alpha": float(alpha)}


def predict_residual_optimizer(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    aligned = x.reindex(columns=bundle["feature_cols"], fill_value=0.0)
    return float(bundle.get("alpha", 1.0)) * bundle["model"].predict(aligned.to_numpy(dtype=float))


def residual_importance(bundle: dict) -> pd.DataFrame:
    values = getattr(bundle["model"], "feature_importances_", None)
    if values is None:
        values = np.zeros(len(bundle["feature_cols"]))
    out = pd.DataFrame({"feature": bundle["feature_cols"], "importance": values})
    return out.sort_values("importance", ascending=False)


def apply_residual_correction(base: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(base, dtype=float) + np.asarray(correction, dtype=float), 0.0, None)


def moving_average_predictions(frame: pd.DataFrame, target_dates: Sequence[np.datetime64], window: int = 30) -> np.ndarray:
    y = frame["target"].to_numpy(dtype=float)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    date_to_idx = {d: i for i, d in enumerate(dates)}
    preds = []
    for d in np.asarray(target_dates, dtype="datetime64[ns]"):
        idx = date_to_idx[np.datetime64(d, "ns")]
        start = max(0, idx - window)
        preds.append(float(np.mean(y[start:idx])))
    return np.asarray(preds, dtype=float)


def recursive_residual_forecast(base_bundle: dict, residual_bundle: dict, raw_df: pd.DataFrame, camps: Sequence[str], horizon: int = 30) -> pd.DataFrame:
    working = raw_df.copy()
    rows = []
    for _ in range(horizon):
        drone_hat = float(np.mean(working["drone_loss"].astype(float).tail(30)))
        placeholder = float(np.mean(working["tank_loss_clean"].astype(float).tail(30)))
        temp_raw = append_future_row(working, placeholder, drone_hat)
        temp_frame, _, _, _ = build_feature_frame(temp_raw)
        next_date = np.asarray([temp_raw["date"].iloc[-1]], dtype="datetime64[ns]")
        base = predict_fusion_base(base_bundle, temp_frame, next_date)["base_pred"]
        x_future, _, _, _ = residual_feature_frame(temp_frame, camps, next_date, base)
        correction = predict_residual_optimizer(residual_bundle, x_future)
        final = apply_residual_correction(base, correction)
        rows.append(
            {
                "date": temp_raw["date"].iloc[-1],
                "lstm_tcn_base": float(base[0]),
                "xgboost_residual_correction": float(correction[0]),
                "final_prediction": float(final[0]),
            }
        )
        working = append_future_row(working, float(final[0]), drone_hat)
    return pd.DataFrame(rows)


def pretty_feature_name(name: str) -> str:
    mapping = {
        "ma30_target_log": "MA30基线",
        "active_campaign_count": "活跃战役数",
        "time_index": "时间趋势",
        "tank_roll7_std_log": "7日波动",
        "tank_roll7_mean_log": "7日坦克均值",
        "tank_roll14_mean_log": "14日坦克均值",
        "tank_roll30_mean_log": "30日坦克均值",
        "drone_roll7_mean_log": "7日无人机均值",
        "drone_roll30_mean_log": "30日无人机均值",
        "drone_to_tank_roll14": "无人机/坦克比",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("lag_y_log_"):
        return name.replace("lag_y_log_", "损失滞后")
    if name.startswith("lag_resid_"):
        return name.replace("lag_resid_", "残差滞后")
    return name[:28]


def campaign_label(name: str) -> str:
    short = {
        "initial_invasion_stage": "Initial invasion",
        "mariupol_siege": "Mariupol",
        "sievierodonetsk_lysychansk_phase": "Sievierodonetsk",
        "kherson_counteroffensive_2022": "Kherson 2022",
        "kharkiv_counteroffensive_2022": "Kharkiv 2022",
        "bakhmut_battle": "Bakhmut",
        "ukrainian_counteroffensive_2023": "Counteroffensive 2023",
        "avdiivka_battle_2023_2024": "Avdiivka",
        "kharkiv_offensive_2024": "Kharkiv 2024",
        "kursk_operation_2024_2025": "Kursk",
        "pokrovsk_offensive_2024_2026": "Pokrovsk",
        "drone_intensification_summer_2025": "Drone intensification",
        "huliapole_offensive_2025_2026": "Huliapole",
        "ukrainian_local_counterattacks_2026": "Local counterattacks",
    }
    return short.get(name, name.replace("_", " ")[:24])


def campaign_timeline(raw: pd.DataFrame, camps: Sequence[str], extend_to: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for col in camps:
        if col not in raw.columns:
            continue
        mask = raw[col].fillna(0).astype(bool)
        if not mask.any():
            continue
        start = pd.to_datetime(raw.loc[mask, "date"].min())
        end = pd.to_datetime(raw.loc[mask, "date"].max())
        if extend_to is not None and bool(mask.iloc[-1]):
            end = max(end, pd.to_datetime(extend_to))
        intensity = float(raw.loc[mask, "tank_loss_clean"].mean())
        rows.append({"feature": col, "label": campaign_label(col), "start": start, "end": end, "intensity_raw": intensity})
    out = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    if out.empty:
        return out
    lo, hi = float(out["intensity_raw"].min()), float(out["intensity_raw"].max())
    if hi - lo < 1e-9:
        out["intensity"] = 0.45
    else:
        out["intensity"] = (out["intensity_raw"] - lo) / (hi - lo)
    return out


def style_time_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_GRAY)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(pad=2)
    fig.savefig(path, dpi=250, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def add_battle_axis(fig: plt.Figure, ax: plt.Axes, timeline: pd.DataFrame, label_size: float = 6.2) -> plt.Axes | None:
    if timeline.empty:
        return None
    ax_r = ax.twinx()
    n_battles = len(timeline)
    ax_r.set_ylim(-1.0, n_battles + 0.3)
    ax_r.invert_yaxis()
    ax_r.set_ylabel("Battle Index", fontsize=11)
    ax_r.set_yticks(range(1, n_battles + 1))
    ax_r.set_yticklabels([f"#{i}" for i in range(1, n_battles + 1)], fontsize=6.5)
    for i, row in timeline.iterrows():
        start, end = row["start"], row["end"]
        width = max((end - start).days, 1)
        ax_r.barh(
            i + 1,
            width,
            left=start,
            height=0.55,
            color=BLUE_TO_RED(float(row["intensity"])),
            alpha=0.78,
            linewidth=0,
            zorder=1,
        )
        ax_r.text(
            end + pd.Timedelta(days=2),
            i + 1,
            f"  {row['label']}",
            va="center",
            fontsize=label_size,
            fontweight="bold" if float(row["intensity"]) > 0.55 else "normal",
            color=TEXT_DARK,
            alpha=0.82,
        )
    for side in ["top", "left"]:
        ax_r.spines[side].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=BLUE_TO_RED, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_r, shrink=0.45, aspect=35, pad=0.04)
    cbar.set_label("Intensity", fontsize=9)
    cbar.outline.set_visible(False)
    return ax_r


def plot_overview(raw: pd.DataFrame, camps: Sequence[str], path: Path) -> None:
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_subplot(111)
    timeline = campaign_timeline(raw, camps)
    ax.plot(
        raw["date"],
        raw["tank_loss_clean"],
        color=ROYAL_BLUE,
        alpha=0.4,
        linewidth=0.7,
        label="Actual Data",
        zorder=2,
    )
    ax.plot(
        raw["date"],
        raw["tank_loss_clean"].rolling(30, min_periods=1).mean(),
        color=DARK_ORANGE,
        linewidth=1.3,
        label="MA30 Smooth",
        zorder=3,
    )
    ax.set_ylabel("Daily Tank Loss", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylim(bottom=-5)
    ax.set_title(
        "LSTM-TCN-XGBoost Input  |  Left=Tank Loss  Right=Battle Index  |  Full sample to 2026-05-30",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    style_time_axis(ax)
    add_battle_axis(fig, ax, timeline)
    save_figure(fig, path)


def plot_architecture(weights: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.axis("off")
    boxes = [
        (0.06, 0.55, 0.22, 0.23, "30日历史序列\n滚动动量/日历/战役标签"),
        (0.39, 0.78, 0.19, 0.15, f"LSTM 分支\n权重 {weights[0]:.2f}"),
        (0.39, 0.55, 0.19, 0.15, f"TCN 分支\n权重 {weights[1]:.2f}"),
        (0.39, 0.32, 0.19, 0.15, f"XGBoost 分支\n权重 {weights[2]:.2f}"),
        (0.72, 0.52, 0.22, 0.22, "联合输出\n未来30日坦克损失"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fc="#F8FAFC", ec="#334155", lw=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    for y in (0.855, 0.625, 0.395):
        ax.annotate("", xy=(0.39, y), xytext=(0.28, 0.665), arrowprops=dict(arrowstyle="->", lw=1.5), xycoords=ax.transAxes)
        ax.annotate("", xy=(0.72, 0.63), xytext=(0.58, y), arrowprops=dict(arrowstyle="->", lw=1.5), xycoords=ax.transAxes)
    ax.set_title("LSTM/TCN/XGBoost 三分支联合建模框架", fontsize=14, pad=10)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_holdout(predictions: pd.DataFrame, path: Path) -> None:
    d = pd.to_datetime(predictions["date"])
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    ax.plot(d, predictions["actual"], color="#111827", lw=2.0, label="观测值")
    ax.plot(d, predictions["lstm_branch"], color="#60A5FA", lw=1.1, alpha=0.78, label="LSTM分支")
    ax.plot(d, predictions["tcn_branch"], color="#34D399", lw=1.1, alpha=0.78, label="TCN分支")
    ax.plot(d, predictions["xgboost_branch"], color="#A78BFA", lw=1.1, alpha=0.78, label="XGBoost分支")
    ax.plot(d, predictions["hybrid_ensemble"], color="#B45309", lw=2.2, label="联合模型")
    ax.fill_between(d, predictions["hybrid_lower95"], predictions["hybrid_upper95"], color="#F59E0B", alpha=0.16, label="联合模型95%区间")
    ax.set_title("最近8日评估：三分支联合预测")
    ax.set_ylabel("坦克日损失")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_weights_importance(weights: np.ndarray, importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(10).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.7), gridspec_kw={"width_ratios": [0.85, 1.45]})
    ax1.bar(["LSTM", "TCN", "XGBoost"], weights, color=["#60A5FA", "#34D399", "#A78BFA"])
    ax1.set_ylim(0, max(0.6, float(weights.max()) + 0.08))
    ax1.set_title("联合模型分支权重")
    ax1.grid(axis="y", alpha=0.22)
    labels = [pretty_feature_name(x) for x in top["feature"]]
    ax2.barh(labels, top["importance"], color="#64748B")
    ax2.set_title("XGBoost 分支主要特征")
    ax2.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_forecast(recent_frame: pd.DataFrame, forecast: pd.DataFrame, path: Path) -> None:
    hist = recent_frame.tail(150)
    d = pd.to_datetime(forecast["date"])
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    ax.plot(hist["date"], hist["target"], color="#111827", lw=1.7, label="历史观测")
    ax.axvline(recent_frame["date"].max(), color="#6B7280", ls="--", lw=1.1, label="预测起点")
    ax.plot(d, forecast["hybrid_ensemble"], color="#B45309", lw=2.2, label="未来30日联合预测")
    ax.fill_between(d, forecast["hybrid_lower95"], forecast["hybrid_upper95"], color="#F59E0B", alpha=0.17, label="95%区间")
    ax.set_title("未来30日坦克日损失联合模型情景预测")
    ax.set_ylabel("坦克日损失")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_residual_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.axis("off")
    boxes = [
        (0.04, 0.55, 0.20, 0.22, "全样本滑动窗口\n过去30日序列+战役特征"),
        (0.34, 0.74, 0.18, 0.15, "LSTM 记忆分支\n长期依赖"),
        (0.34, 0.45, 0.18, 0.15, "TCN 卷积分支\n多尺度冲击"),
        (0.61, 0.58, 0.15, 0.18, "融合层\n基础预测"),
        (0.61, 0.25, 0.15, 0.15, "残差\n真实值-基础预测"),
        (0.82, 0.25, 0.14, 0.15, "XGBoost\n残差修正"),
        (0.82, 0.58, 0.14, 0.18, "最终输出\n未来30日预测"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fc="#FFF7ED", ec=GRAY_LINE, lw=1.6, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, transform=ax.transAxes)
    arrows = [
        ((0.24, 0.66), (0.34, 0.815)),
        ((0.24, 0.66), (0.34, 0.525)),
        ((0.52, 0.815), (0.61, 0.67)),
        ((0.52, 0.525), (0.61, 0.67)),
        ((0.685, 0.58), (0.685, 0.40)),
        ((0.76, 0.325), (0.82, 0.325)),
        ((0.89, 0.40), (0.89, 0.58)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color=GRAY_LINE), xycoords=ax.transAxes)
    ax.set_title(
        "LSTM-TCN Fusion + XGBoost Residual Correction  |  Sequence representation + structured error learning",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    save_figure(fig, path)


def plot_residual_holdout(predictions: pd.DataFrame, path: Path) -> None:
    d = pd.to_datetime(predictions["date"])
    eval_rmse = rmse(predictions["actual"].to_numpy(dtype=float), predictions["final_prediction"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.plot(d, predictions["actual"], color=ROYAL_BLUE, alpha=0.75, lw=1.6, label="Actual Data", zorder=2)
    ax.plot(d, predictions["ma30"], color=DODGER_BLUE, lw=1.2, alpha=0.55, ls="--", label="MA30 Baseline", zorder=2)
    ax.plot(d, predictions["lstm_tcn_base"], color=DARK_ORANGE, lw=1.3, label="LSTM-TCN Fit", zorder=3)
    ax.plot(d, predictions["final_prediction"], color=CRIMSON, lw=2.2, label="XGBoost Corrected", zorder=4)
    ax.fill_between(
        d,
        predictions["final_lower95"],
        predictions["final_upper95"],
        color=CRIMSON,
        alpha=0.16,
        label="95% CI",
        zorder=2,
    )
    ax.set_title(
        f"Recent 8-day Evaluation  |  LSTM-TCN + XGBoost  |  RMSE={eval_rmse:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Daily Tank Loss", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylim(bottom=-5)
    style_time_axis(ax)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=4)
    save_figure(fig, path)


def plot_residual_correction(predictions: pd.DataFrame, importance: pd.DataFrame, path: Path) -> None:
    before = predictions["actual"].to_numpy(dtype=float) - predictions["lstm_tcn_base"].to_numpy(dtype=float)
    after = predictions["actual"].to_numpy(dtype=float) - predictions["final_prediction"].to_numpy(dtype=float)
    top = importance.head(10).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), gridspec_kw={"width_ratios": [0.95, 1.35]})
    ax1.scatter(before, after, color=ROYAL_BLUE, alpha=0.72, s=48, zorder=3)
    lim = max(1.0, float(np.max(np.abs(np.r_[before, after]))))
    ax1.axhline(0, color=GRAY_LINE, lw=1.0, alpha=0.6)
    ax1.axvline(0, color=GRAY_LINE, lw=1.0, alpha=0.6)
    ax1.plot([-lim, lim], [-lim, lim], color=GRAY_LINE, lw=1.0, ls="--", alpha=0.6)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("修正前残差")
    ax1.set_ylabel("修正后残差")
    ax1.set_title("Residual Correction  |  Before vs After")
    ax1.grid(True, linestyle="--", alpha=0.35, color=GRID_GRAY)
    for side in ["top", "right"]:
        ax1.spines[side].set_visible(False)
    labels = [pretty_feature_name(x) for x in top["feature"]]
    ax2.barh(labels, top["importance"], color=[BLUE_TO_RED(i / max(len(labels) - 1, 1)) for i in range(len(labels))], alpha=0.92)
    ax2.set_title("XGBoost Feature Importance  |  Residual learner")
    ax2.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_GRAY)
    for side in ["top", "right"]:
        ax2.spines[side].set_visible(False)
    save_figure(fig, path)


def plot_residual_forecast(frame: pd.DataFrame, raw: pd.DataFrame, camps: Sequence[str], forecast: pd.DataFrame, path: Path) -> None:
    hist = frame.tail(150)
    d = pd.to_datetime(forecast["date"])
    forecast_rmse_hint = float(np.mean(forecast["final_prediction"]))
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_subplot(111)
    view_start = pd.to_datetime(hist["date"].min())
    view_end = pd.to_datetime(d.max())
    timeline = campaign_timeline(raw, camps, extend_to=view_end)
    if not timeline.empty:
        timeline = timeline[(timeline["end"] >= view_start) & (timeline["start"] <= view_end)].copy()
        timeline["start"] = timeline["start"].map(lambda x: max(pd.to_datetime(x), view_start))
        timeline["end"] = timeline["end"].map(lambda x: min(pd.to_datetime(x), view_end))
        timeline = timeline.reset_index(drop=True)
    ax.plot(hist["date"], hist["target"], color=ROYAL_BLUE, alpha=0.4, lw=0.7, label="Actual Data", zorder=2)
    ax.plot(d, forecast["lstm_tcn_base"], color=DARK_ORANGE, lw=1.3, alpha=0.95, label="LSTM-TCN Base", zorder=3)
    ax.fill_between(
        d,
        forecast["final_lower95"],
        forecast["final_upper95"],
        color=CRIMSON,
        alpha=0.16,
        label="95% CI",
        zorder=2,
    )
    ax.plot(d, forecast["final_prediction"], color=CRIMSON, lw=2.2, label="Forecast", zorder=4)
    ax.axvline(frame["date"].max(), color=GRAY_LINE, ls=":", lw=1.0, alpha=0.6, label="Forecast Start")
    ax.set_title(
        f"LSTM-TCN + XGBoost Forecast  |  Left=Tank Loss  Right=Battle Index  |  30-day Mean={forecast_rmse_hint:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Daily Tank Loss", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylim(bottom=-5)
    ax.set_xlim(view_start, view_end + pd.Timedelta(days=16))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=4)
    style_time_axis(ax)
    add_battle_axis(fig, ax, timeline)
    save_figure(fig, path)


def main() -> None:
    print(f"Using CSV: {CSV_PATH}")
    raw = load_csv()
    frame, sequence_cols, camps, groups = build_feature_frame(raw)
    last_date = frame["date"].max()
    test_start = last_date - pd.Timedelta(days=EVAL_DAYS - 1)
    test_dates = frame.loc[frame["date"] >= test_start, "date"].to_numpy(dtype="datetime64[ns]")

    test_base_bundle = fit_fusion_base(frame, sequence_cols, test_start, seed_offset=300)
    residual_dates = frame.loc[frame["date"] < test_start, "date"].to_numpy(dtype="datetime64[ns]")
    residual_base = predict_fusion_base(test_base_bundle, frame, residual_dates)
    residual_dates = residual_base["date"]
    residual_x, residual_actual, _, _ = residual_feature_frame(frame, camps, residual_dates, residual_base["base_pred"])
    residual_target = residual_actual - residual_base["base_pred"]
    residual_optimizer = fit_residual_optimizer(residual_x, residual_target, seed_offset=200)
    residual_correction = predict_residual_optimizer(residual_optimizer, residual_x)
    residual_final = apply_residual_correction(residual_base["base_pred"], residual_correction)
    ma_residual = moving_average_predictions(frame, residual_dates)
    sigma_base = float(np.std(np.log1p(residual_actual) - np.log1p(np.clip(residual_base["base_pred"], 0.0, None)), ddof=1))
    sigma_ma = float(np.std(np.log1p(residual_actual) - np.log1p(np.clip(ma_residual, 0.0, None)), ddof=1))
    sigma_final = float(np.std(np.log1p(residual_actual) - np.log1p(np.clip(residual_final, 0.0, None)), ddof=1))
    sigma_final = max(sigma_final, 0.42)

    test_base = predict_fusion_base(test_base_bundle, frame, test_dates)
    test_x, test_actual, _, _ = residual_feature_frame(frame, camps, test_dates, test_base["base_pred"])
    test_correction = predict_residual_optimizer(residual_optimizer, test_x)
    final_pred = apply_residual_correction(test_base["base_pred"], test_correction)
    ma30_test = moving_average_predictions(frame, test_dates)
    ma_l, ma_u = interval_from_sigma(ma30_test, sigma_ma)
    base_l, base_u = interval_from_sigma(test_base["base_pred"], sigma_base)
    final_l, final_u = interval_from_sigma(final_pred, sigma_final)

    metrics = pd.DataFrame(
        [
            evaluate("MA30 局部基线", test_actual, ma30_test, ma_l, ma_u, None),
            evaluate("LSTM-TCN 融合基础模型", test_actual, test_base["base_pred"], base_l, base_u, None),
            evaluate("LSTM-TCN + XGBoost 残差修正", test_actual, final_pred, final_l, final_u, None),
        ]
    )
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(test_base["date"]),
            "actual": test_actual,
            "ma30": ma30_test,
            "lstm_tcn_base": test_base["base_pred"],
            "xgboost_residual_correction": test_correction,
            "final_prediction": final_pred,
            "final_lower95": final_l,
            "final_upper95": final_u,
        }
    )

    full_base_bundle = fit_fusion_base(frame, sequence_cols, last_date + pd.Timedelta(days=1), seed_offset=600)
    final_residual_dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    final_residual_base = predict_fusion_base(full_base_bundle, frame, final_residual_dates)
    final_residual_dates = final_residual_base["date"]
    final_residual_x, final_residual_actual, _, _ = residual_feature_frame(
        frame, camps, final_residual_dates, final_residual_base["base_pred"]
    )
    final_residual_target = final_residual_actual - final_residual_base["base_pred"]
    final_residual_optimizer = fit_residual_optimizer(final_residual_x, final_residual_target, seed_offset=500)
    forecast = recursive_residual_forecast(full_base_bundle, final_residual_optimizer, raw, camps, horizon=FORECAST_HORIZON)
    future_l, future_u = interval_from_sigma(forecast["final_prediction"].to_numpy(dtype=float), sigma_final)
    forecast["final_lower95"] = future_l
    forecast["final_upper95"] = future_u
    xgb_importance = residual_importance(final_residual_optimizer)

    campaign_stats = []
    for col in camps:
        mask = raw[col].astype(bool)
        if mask.sum() == 0:
            continue
        campaign_stats.append(
            {
                "campaign_feature": col,
                "days": int(mask.sum()),
                "mean_tank_loss": float(raw.loc[mask, "tank_loss_clean"].mean()),
                "mean_drone_loss": float(raw.loc[mask, "drone_loss"].mean()),
                "start": raw.loc[mask, "date"].min().date().isoformat(),
                "end": raw.loc[mask, "date"].max().date().isoformat(),
            }
        )
    campaign_stats_df = pd.DataFrame(campaign_stats).sort_values("days", ascending=False)

    metrics.to_csv(TABLE_DIR / "ml_residual_model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(TABLE_DIR / "ml_residual_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(TABLE_DIR / "ml_residual_30day_forecast.csv", index=False, encoding="utf-8-sig")
    xgb_importance.to_csv(TABLE_DIR / "ml_residual_xgb_feature_importance.csv", index=False, encoding="utf-8-sig")
    campaign_stats_df.to_csv(TABLE_DIR / "ml_residual_campaign_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR / "ml_residual_results_summary.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="model_metrics", index=False)
        predictions.to_excel(writer, sheet_name="holdout_predictions", index=False)
        forecast.to_excel(writer, sheet_name="forecast_30d", index=False)
        xgb_importance.to_excel(writer, sheet_name="xgb_feature_importance", index=False)
        campaign_stats_df.to_excel(writer, sheet_name="campaign_summary", index=False)

    plot_overview(raw, camps, FIG_DIR / "fig01_data_campaign_overview.png")
    plot_residual_architecture(FIG_DIR / "fig02_lstm_tcn_xgboost_residual_architecture.png")
    plot_residual_holdout(predictions, FIG_DIR / "fig03_holdout_residual_correction.png")
    plot_residual_correction(predictions, xgb_importance, FIG_DIR / "fig04_residual_correction_importance.png")
    plot_residual_forecast(frame, raw, camps, forecast, FIG_DIR / "fig05_future_30day_residual_forecast.png")

    manifest = {
        "csv_path": str(CSV_PATH),
        "rows": int(len(raw)),
        "date_start": raw["date"].min().date().isoformat(),
        "date_end": raw["date"].max().date().isoformat(),
        "training_scope": "full sample sliding windows",
        "lookback_days": SEQ_LEN,
        "evaluation_window_days": EVAL_DAYS,
        "evaluation_start": test_start.date().isoformat(),
        "evaluation_end": last_date.date().isoformat(),
        "residual_training_scope_for_evaluation": f"{raw['date'].min().date().isoformat()} to {(test_start - pd.Timedelta(days=1)).date().isoformat()}",
        "final_residual_training_scope": f"{raw['date'].min().date().isoformat()} to {last_date.date().isoformat()}",
        "test_start": test_start.date().isoformat(),
        "horizon_days": FORECAST_HORIZON,
        "target": "tank_loss_clean",
        "sequence_length": SEQ_LEN,
        "sequence_feature_count": len(sequence_cols),
        "campaign_feature_count": len(camps),
        "residual_optimizer_implementation": final_residual_optimizer["implementation"],
        "base_model": "jointly trained LSTM-TCN fusion network",
        "optimizer": "XGBoost residual correction",
        "sigma_log_final": sigma_final,
        "base_test_rmse": float(metrics.loc[metrics["model"].eq("LSTM-TCN 融合基础模型"), "RMSE"].iloc[0]),
        "final_test_rmse": float(metrics.loc[metrics["model"].eq("LSTM-TCN + XGBoost 残差修正"), "RMSE"].iloc[0]),
        "forecast_mean_30d": float(forecast["final_prediction"].mean()),
        "forecast_day30": float(forecast["final_prediction"].iloc[-1]),
        "tank_mean": float(raw["tank_loss_clean"].mean()),
        "tank_median": float(raw["tank_loss_clean"].median()),
        "tank_max": float(raw["tank_loss_clean"].max()),
        "active_campaign_count_mean": float(raw["active_campaign_count"].mean()),
        "feature_groups": groups,
    }
    (TABLE_DIR / "ml_residual_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
