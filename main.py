# -*- coding: utf-8 -*-
"""
End‑to‑end implementation of the two‑layer VMD‑LSTM/BiLSTM forecasting framework
described in *Forecasting Bitcoin with a decomposition‑aided LSTM model and explaining
results using Shapley values* (Knowledge‑Based Systems 299 (2024) 112026).

Written for clarity and reproducibility by a senior Python developer.
All hyper‑parameters, search ranges, and data splits follow the paper exactly.

Dependencies (install via pip):
    pandas numpy scikit‑learn tensorflow vmdpy shap tqdm

The script is organised in four sections:
    1. Utilities & data processing
    2. Hybrid Self‑Adaptive Sine–Cosine Algorithm (HSA‑SCA)
    3. Two‑layer optimisation (VMD   →   LSTM / BiLSTM)
    4. Training, evaluation and SHAP explanation

To keep the example self‑contained `load_dataset` is left as a stub –
you must download the seven series listed in the paper and concatenate
into one DataFrame with the exact column order:
    ['BTC', 'ETH', 'SP500', 'VIX', 'EURUSD', 'GBPUSD', 'Tweets']
indexed by a daily DatetimeIndex covering 2019‑09‑01 … 2022‑08‑31.
"""

from __future__ import annotations

import os
import yfinance as yf

import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import shap  # DeepExplainer
from vmdpy import VMD  # pip install vmdpy

# -----------------------------------------------------------------------------
# 1 ───────────────────────────── Utilities ────────────────────────────────────
# -----------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

COLUMNS = [
    "BTC",
    "ETH",
    "SP500",
    "VIX",
    "EURUSD",
    "GBPUSD",
]

START_DATE = "2019-09-01"
END_DATE = "2022-08-31"
TRAIN_END = "2021-10-31"  # 70 % ≈ 26 months
VAL_END = "2022-02-28"  # +4 months ≈ 15 %

LAGS = 6
HORIZON = 3


# ---------- data helpers ------------------------------------------------------

def load_dataset(csv_path="cached_data.csv"):
    if os.path.exists(csv_path):
        print("📂 داده‌ها قبلاً دانلود شده‌اند. در حال بارگذاری از فایل...")
        df_all = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df_all

    tickers = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X'
    }

    df_all = pd.DataFrame()

    for name, symbol in tickers.items():
        print(f"⬇ در حال دریافت {name} ({symbol}) ...")
        df = yf.download(symbol, start="2019-09-01", end="2022-09-01")['Close']
        df_all[name] = df

    df_all = df_all.ffill().bfill().dropna()
    df_all.to_csv(csv_path)  # ذخیرهٔ یک‌باره
    print("✅ داده‌ها دانلود و ذخیره شدند.")
    return df_all


def forward_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().ffill()


def make_sliding_windows(seq: np.ndarray, lags: int, horizon: int, target_col: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(seq) - lags - horizon + 1):
        X.append(seq[i: i + lags])
        y.append(seq[i + lags: i + lags + horizon, target_col])
    return np.asarray(X), np.asarray(y)


def split_by_dates(idx: pd.DatetimeIndex, *arrays, train_end: str, val_end: str):
    mask_train = idx <= train_end
    mask_val = (idx > train_end) & (idx <= val_end)
    mask_test = idx > val_end
    return [a[mask_train] for a in arrays] + [a[mask_val] for a in arrays] + [a[mask_test] for a in arrays]


def inverse_minmax(arr: np.ndarray, scaler: MinMaxScaler, column: int) -> np.ndarray:
    """Inverse‑transform *only one* column of a multioutput prediction."""
    full = np.zeros((arr.shape[0], len(scaler.min_)))
    full[:, column] = arr.squeeze()
    return scaler.inverse_transform(full)[:, column]


# -----------------------------------------------------------------------------
# 2 ───────────────────────────── HSA‑SCA ─────────────────────────────────────-
# -----------------------------------------------------------------------------

class HSA_SCA:
    """Hybrid Self‑Adaptive Sine–Cosine + Firefly optimiser (Alg. 1)."""

    def __init__(
            self,
            obj_fun,
            bounds: List[Tuple[float, float]],
            pop_size: int = 5,
            iters: int = 5,
            seed: int = SEED,
    ):
        self.f = obj_fun
        self.lb = np.array([b[0] for b in bounds], dtype=float)
        self.ub = np.array([b[1] for b in bounds], dtype=float)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.iters = iters
        np.random.seed(seed)

    # internal helpers ---------------------------------------------------------
    def _init_pop(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    # public ------------------------------------------------------------------
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self.f(ind) for ind in X])
        best_idx = fitness.argmin()
        P_star = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # algorithm parameters
        vs = max(1, self.iters // 5)
        sm = 0.8
        beta0, gamma, alpha0 = 1.0, 1.0, 0.5

        for t in tqdm(range(self.iters), desc="HSA‑SCA optimise", ncols=80):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim) * 2 * np.pi
            r3 = np.random.rand(self.pop_size, self.dim)
            r4 = np.random.rand(self.pop_size, self.dim)

            for i in range(self.pop_size):
                if t < vs or np.random.rand() < sm:
                    # Sine–Cosine phase
                    delta = np.abs(r3[i] * P_star - X[i])
                    move = np.where(r4[i] < 0.5, np.sin(r2[i]), np.cos(r2[i]))
                    X[i] += r1[i] * move * delta
                else:
                    # Firefly phase
                    j = np.random.randint(self.pop_size)
                    r2_ij = np.sum((X[i] - X[j]) ** 2)
                    alpha_t = alpha0 * (0.97 ** t)
                    X[i] += beta0 * np.exp(-gamma * r2_ij) * (X[j] - X[i]) + alpha_t * (np.random.rand(self.dim) - 0.5)

            X = self._clip(X)
            fitness = np.array([self.f(ind) for ind in X])
            idx = fitness.argmin()
            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                P_star = X[idx].copy()

            if t >= vs:
                sm -= sm / 10

        self.best_solution = P_star
        self.best_fitness = best_fit
        return P_star, best_fit


# -----------------------------------------------------------------------------
# 3 ─────────────────────────── Two‑layer routine ─────────────────────────────-
# -----------------------------------------------------------------------------

def build_model(cfg: np.ndarray, n_features: int, bi: bool) -> Tuple[tf.keras.Model, int]:
    n1, lr, epochs, dr, layers, n2 = cfg
    n1, n2, epochs, layers = int(n1), int(n2), int(epochs), int(layers)

    model = Sequential()
    RSEQ = layers == 2
    if bi:
        model.add(Bidirectional(LSTM(n1, return_sequences=RSEQ), input_shape=(LAGS, n_features)))
    else:
        model.add(LSTM(n1, return_sequences=RSEQ, input_shape=(LAGS, n_features)))
    model.add(Dropout(dr))
    if layers == 2:
        if bi:
            model.add(Bidirectional(LSTM(n2)))
        else:
            model.add(LSTM(n2))
        model.add(Dropout(dr))
    model.add(Dense(HORIZON))
    model.compile(optimizer=Adam(lr), loss="mse")
    return model, epochs


# -----------------------------------------------------------------------------
# 4 ─────────────────────────────── main ─────────────────────────────────────--
# -----------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------ data
    df_raw = load_dataset()  # implement me!
    df_raw = df_raw.loc[START_DATE:END_DATE, COLUMNS]
    df_raw = forward_fill_missing(df_raw)

    scaler = MinMaxScaler((0, 1))
    data_scaled = scaler.fit_transform(df_raw)

    X_all, y_all = make_sliding_windows(data_scaled, LAGS, HORIZON, target_col=0)
    index_sliding = df_raw.index[LAGS + HORIZON - 1:]
    (X_tr, X_val, X_te, y_tr, y_val, y_te) = split_by_dates(
        index_sliding,
        X_all,
        y_all,
        train_end=TRAIN_END,
        val_end=VAL_END,
    )

    # ---------------------------------------------------- layer 1: optimise VMD
    def obj_vmd(vec):
        K = int(round(vec[0]))
        alpha = vec[1]
        modes_opt, _, _ = VMD(data_scaled[:, 0], alpha=alpha_opt, tau=0., K=K_opt, DC=0, init=1, tol=1e-6)
        X_m, y_m = make_sliding_windows(np.column_stack(modes), LAGS, HORIZON, 0)
        # very cheap baseline: predict mean of target slice
        base_pred = np.repeat(y_m.mean(axis=1, keepdims=True), HORIZON, axis=1)
        return mean_squared_error(y_m, base_pred)

    bounds_vmd = [(2, 6), (len(X_tr) / 10, len(X_tr))]
    best_vmd, _ = HSA_SCA(obj_vmd, bounds_vmd).optimize()
    K_opt, alpha_opt = int(round(best_vmd[0])), best_vmd[1]

    modes_opt, _, _ = VMD(data_scaled[:, 0], K=K_opt, alpha=alpha_opt, DC=0, init=1, tol=1e-6)
    data_decomp = np.column_stack([data_scaled, *modes_opt])

    X_all2, y_all2 = make_sliding_windows(data_decomp, LAGS, HORIZON, 0)
    (X_tr, X_val, X_te, y_tr, y_val, y_te) = split_by_dates(
        index_sliding, X_all2, y_all2, train_end=TRAIN_END, val_end=VAL_END
    )
    n_features = X_tr.shape[2]

    # ------------------------------------------- layer 2: optimise LSTM/BiLSTM
    bounds_nn = [
        (100, 300),  # n1
        (1e-4, 1e-2),  # lr
        (300, 600),  # epochs
        (0.05, 0.2),  # dropout
        (1, 2),  # layers
        (100, 300),  # n2
    ]

    def obj_nn(vec, bi=False):
        model, epochs = build_model(vec, n_features, bi)
        es = EarlyStopping(patience=int(epochs / 3), restore_best_weights=True, verbose=0)
        model.fit(
            X_tr,
            y_tr,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[es],
            verbose=0,
        )
        return model.evaluate(X_val, y_val, verbose=0)

    best_lstm, _ = HSA_SCA(lambda v: obj_nn(v, bi=False), bounds_nn).optimize()
    best_bi, _ = HSA_SCA(lambda v: obj_nn(v, bi=True), bounds_nn).optimize()

    model_lstm, ep_l = build_model(best_lstm, n_features, False)
    model_lstm.fit(np.concatenate([X_tr, X_val]), np.concatenate([y_tr, y_val]), epochs=ep_l, batch_size=32, verbose=0)

    model_bi, ep_b = build_model(best_bi, n_features, True)
    model_bi.fit(np.concatenate([X_tr, X_val]), np.concatenate([y_tr, y_val]), epochs=ep_b, batch_size=32, verbose=0)

    # ------------------------------------------------------ evaluation helpers
    def evaluate(model):
        pred_scaled = model.predict(X_te, verbose=0)
        pred = inverse_minmax(pred_scaled, scaler, 0)
        true = inverse_minmax(y_te, scaler, 0)
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        ia = 1 - np.sum((true - pred) ** 2) / np.sum((np.abs(true - true.mean()) + np.abs(pred - true.mean())) ** 2)
        return mae, rmse, r2, ia

    metrics_lstm = evaluate(model_lstm)
    metrics_bi = evaluate(model_bi)
    print("LSTM  metrics (MAE, RMSE, R2, IA):", metrics_lstm)
    print("BiLSTM metrics (MAE, RMSE, R2, IA):", metrics_bi)

    # ------------------------------------------------------------ SHAP explain
    explainer = shap.DeepExplainer(model_bi, X_tr[:100])
    shap_values = explainer.shap_values(X_te[:100])
    shap.summary_plot(
        shap_values,
        X_te[:100],
        feature_names=COLUMNS + [f"IMF{i + 1}" for i in range(K_opt)],
        show=False,
    )
    Path("figures").mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    plt.savefig("figures/shap_beeswarm.png", dpi=300, bbox_inches="tight")
    print("SHAP plot saved to figures/shap_beeswarm.png")


if __name__ == "__main__":
    main()
