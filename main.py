import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------
# تنظیم لاگ   (INFO برای مراحل اصلی / DEBUG برای جزئیات)
# ------------------------------------------------------
logging.basicConfig(
    format='%(levelname)s | %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- ۱) دریافت داده --------------------

def load_dataset(csv_path: str = "cached_data.csv") -> pd.DataFrame:
    """دانلود شش شاخص مالی یا بارگذاری از کش محلی."""
    if os.path.exists(csv_path):
        logger.info("📂 داده‌ها قبلاً دانلود شده‌اند؛ بارگذاری از فایل …")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    tickers = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SP500": "^GSPC",
        "VIX": "^VIX",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
    }
    df = pd.DataFrame()
    for name, symbol in tickers.items():
        logger.info(f"⬇ در حال دریافت {name} ({symbol}) …")
        df[name] = yf.download(symbol, start="2019-09-01", end="2022-09-01")[
            "Close"
        ]
    df = df.ffill().bfill().dropna()
    df.to_csv(csv_path)
    logger.info("✅ دانلود کامل و ذخیره شد.")
    return df

# --------------- ۲) پنجره‌های زمانی ------------------

def make_sliding_windows(arr: np.ndarray, lags: int = 6, horizon: int = 3):
    """ساخت X(نمونه،زمان،ویژگی) و y(نمونه،horizon)"""
    X, y = [], []
    for i in range(len(arr) - lags - horizon + 1):
        X.append(arr[i : i + lags])
        y.append(arr[i + lags : i + lags + horizon, 0])  # فقط BTC هدف است
    return np.array(X), np.array(y)

# ---------- ۳) تقسیم train/val/test بر اساس تاریخ -------

def split_by_ratio(X, y, train: float = 0.7, val: float = 0.15):
    n = len(X)
    tr, va = int(n * train), int(n * (train + val))
    return (
        X[:tr],
        X[tr:va],
        X[va:],
        y[:tr],
        y[tr:va],
        y[va:],
    )

# -------- ۴) نسخه سادهٔ HSA‑SCA برای مثال ----------------

class HSA_SCA:
    def __init__(self, obj_fun, bounds, pop_size=5, iters=5, seed=42):
        self.f = obj_fun
        self.bounds = bounds
        self.pop_size = pop_size
        self.iters = iters
        np.random.seed(seed)
        self.dim = len(bounds)
        self.lb = np.array([b[0] for b in bounds], dtype=float)
        self.ub = np.array([b[1] for b in bounds], dtype=float)

    def _init_pop(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)

    def optimize(self):
        X = self._init_pop()
        fit = np.array([self.f(ind) for ind in X])
        best_idx = fit.argmin(); best = X[best_idx]
        logger.info(f"▶ شروع بهینه‌سازی؛ مقدار شروع: {fit[best_idx]:.6f}")
        for t in range(self.iters):
            for i in range(self.pop_size):
                r = np.random.rand(self.dim)
                X[i] = np.clip(X[i] + r * (best - X[i]), self.lb, self.ub)
                fit[i] = self.f(X[i])
                if fit[i] < fit[best_idx]:
                    best_idx, best = i, X[i]
            logger.info(f"  دور {t+1}/{self.iters} → بهترین MSE: {fit[best_idx]:.6f}")
        return best, fit[best_idx]

# ---------------- ۵) تابع هدف VMD -----------------------

def build_baseline_lstm(input_shape, horizon=3):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.1),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

def obj_vmd(params):
    K = int(round(params[0]))
    alpha = params[1]
    logger.debug(f"ارزیابی VMD با K={K}, alpha={alpha:.2f}")

    modes, _, _ = VMD(
        data_scaled[:, 0], alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=1e-6
    )
    modes = np.stack(modes, axis=-1)  # (نمونه، K)
    feat_matrix = np.concatenate([data_scaled, modes], axis=1)

    X_seq, y_seq = make_sliding_windows(feat_matrix)
    X_tr, X_val, _, y_tr, y_val, _ = split_by_ratio(X_seq, y_seq)

    model = build_baseline_lstm(input_shape=X_tr.shape[1:])
    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_tr,
        y_tr,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=0,
    )
    mse = model.evaluate(X_val, y_val, verbose=0)
    logger.debug(f"MSE={mse:.6f} برای K={K}, alpha={alpha:.2f}")
    return mse

# ---------------- ۶) اجرای اصلی ------------------------

def main():
    global data_scaled  # در obj_vmd نیاز داریم

    df = load_dataset()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    bounds_vmd = [(2, 6), (len(data_scaled) / 10, len(data_scaled))]
    best_params, best_mse = HSA_SCA(obj_vmd, bounds_vmd, iters=3).optimize()
    K_opt = int(round(best_params[0])); alpha_opt = best_params[1]
    logger.info(f"✔ بهترین پارامترها: K={K_opt}, alpha={alpha_opt:.2f}, MSE={best_mse:.6f}")

    # اجرای نهایی VMD و ساخت دیتاست نهایی
    modes_opt, _, _ = VMD(
        data_scaled[:, 0], alpha=alpha_opt, tau=0.0, K=K_opt, DC=0, init=1, tol=1e-6
    )
    feat_final = np.concatenate([data_scaled, np.stack(modes_opt, axis=-1)], axis=1)
    logger.info("✅  دادهٔ نهایی با مدهای VMD ساخته شد؛ آمادهٔ آموزش شبکهٔ اصلی هستیم.")

if __name__ == "__main__":
    main()
