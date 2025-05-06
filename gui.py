import warnings
warnings.filterwarnings("ignore", message="No supported index is available")

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, timedelta
import threading
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ─── Load artifacts ──────────────────────────────────────────────────────────────
lstm_model   = load_model("LSTM/lstm_model.keras", compile=False)
prep         = joblib.load("LSTM/preprocessing.pkl")
scaler       = prep["scaler"]
feature_cols = prep["feature_cols"]

rnn_model    = load_model("RNN/rnn_model.keras", compile=False)
arima_model  = joblib.load("ARIMA/arima_model.pkl")
es_model     = joblib.load("ExponentialSmoothing/es_model.pkl")

# ─── Feature helper (force 1-D) ─────────────────────────────────────────────────
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume   import OnBalanceVolumeIndicator

look_back = 60

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    feats = pd.DataFrame(index=df.index)
    feats["Close"]    = close
    feats["SMA10"]    = pd.Series(SMAIndicator(close, window=10).sma_indicator().values.flatten(), index=df.index)
    feats["EMA20"]    = pd.Series(EMAIndicator(close, window=20).ema_indicator().values.flatten(), index=df.index)
    feats["RSI14"]    = pd.Series(RSIIndicator(close, window=14).rsi().values.flatten(), index=df.index)
    macd               = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    feats["MACD"]      = pd.Series(macd.macd().values.flatten(), index=df.index)
    feats["MACD_SIG"]  = pd.Series(macd.macd_signal().values.flatten(), index=df.index)
    obv = OnBalanceVolumeIndicator(close, pd.Series(df["Volume"].values.flatten(), index=df.index))
    feats["OBV"]       = pd.Series(obv.on_balance_volume().values.flatten(), index=df.index)
    feats.dropna(inplace=True)
    return feats

# ─── Forecast functions with robust download & manual inversion ────────────────
def forecast_lstm():
    df = yf.download(
        "7010.SR",
        period=f"{look_back*3}d",
        progress=False
    )
    feats = compute_features(df)[feature_cols]
    arr   = scaler.transform(feats.values)
    if arr.shape[0] < look_back:
        raise ValueError(f"Not enough rows ({arr.shape[0]}) for look_back={look_back}")
    window   = arr[-look_back:][np.newaxis, :, :]
    p_scaled = lstm_model.predict(window, verbose=0).item()
    c_min, c_max = scaler.data_min_[0], scaler.data_max_[0]
    return float(p_scaled * (c_max - c_min) + c_min)

def forecast_rnn():
    df = yf.download(
        "7010.SR",
        period=f"{look_back*3}d",
        progress=False
    )
    feats = compute_features(df)[feature_cols]
    arr   = scaler.transform(feats.values)
    if arr.shape[0] < look_back:
        raise ValueError(f"Not enough rows ({arr.shape[0]}) for look_back={look_back}")
    window   = arr[-look_back:][np.newaxis, :, :]
    p_scaled = rnn_model.predict(window, verbose=0).item()
    c_min, c_max = scaler.data_min_[0], scaler.data_max_[0]
    return float(p_scaled * (c_max - c_min) + c_min)

def forecast_arima():
    return float(arima_model.forecast(steps=1)[0])

def forecast_es():
    return float(es_model.forecast(1)[0])

# ─── GUI callbacks ──────────────────────────────────────────────────────────────
def fetch_latest():
    try:
        data   = yf.download("7010.SR", period="7d", progress=False)
        latest = float(data["Close"].dropna().iloc[-1])
        lbl_latest_val.config(text=f"{latest:.2f} SAR")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch latest: {e}")

def run_forecasts():
    def worker():
        btn_forecast.config(state=tk.DISABLED)
        try:
            v1 = forecast_lstm()
            v2 = forecast_rnn()
            v3 = forecast_arima()
            v4 = forecast_es()
            lbl_lstm_val.config(text=f"{v1:.2f} SAR")
            lbl_rnn_val.config(text=f"{v2:.2f} SAR")
            lbl_arima_val.config(text=f"{v3:.2f} SAR")
            lbl_es_val.config(text=f"{v4:.2f} SAR")
        except Exception as e:
            messagebox.showerror("Error", f"Forecast failed: {e}")
        finally:
            btn_forecast.config(state=tk.NORMAL)

    threading.Thread(target=worker, daemon=True).start()

# ─── Build GUI ───────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("STC Stock Forecast")

frm = ttk.Frame(root, padding=20)
frm.grid()

ttk.Label(frm, text="STC (7010.SR) Next-Day Forecast", font=("Helvetica", 16))\
    .grid(columnspan=2, pady=5)

ttk.Label(frm, text="Latest Close (SAR):").grid(row=1, column=0, sticky=tk.W)
lbl_latest_val = ttk.Label(frm, text="—")
lbl_latest_val.grid(row=1, column=1, sticky=tk.E)

btn_fetch    = ttk.Button(frm, text="Fetch Latest",      command=fetch_latest)
btn_forecast = ttk.Button(frm, text="Forecast Next Day", command=run_forecasts)
btn_fetch.grid(row=2, column=0, pady=10)
btn_forecast.grid(row=2, column=1, pady=10)

labels = ["LSTM", "Simple RNN", "ARIMA", "Exp. Smoothing"]
vals   = []
for i, lbl in enumerate(labels, start=3):
    ttk.Label(frm, text=f"{lbl}:").grid(row=i, column=0, sticky=tk.W)
    val_lbl = ttk.Label(frm, text="—")
    val_lbl.grid(row=i, column=1, sticky=tk.E)
    vals.append(val_lbl)

lbl_lstm_val, lbl_rnn_val, lbl_arima_val, lbl_es_val = vals

root.mainloop()
