import streamlit as st
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model




# ─── 1) Load models & artifacts ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    lstm   = load_model("LSTM/lstm_model.keras")
    scaler = joblib.load("LSTM/scaler.pkl")
    rnn    = load_model("RNN/rnn_model.keras")
    arima  = joblib.load("ARIMA/arima_model.pkl")
    es     = joblib.load("ExponentialSmoothing/es_model.pkl")
    return lstm, scaler, rnn, arima, es

lstm_model, scaler, rnn_model, arima_model, es_model = load_artifacts()

# ─── 2) Feature‐engineering helper ───────────────────────────────────────────────
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume   import OnBalanceVolumeIndicator

def compute_features(df):
    feats = pd.DataFrame(index=df.index)
    feats["Close"]    = df["Close"]
    feats["SMA10"]    = SMAIndicator(df["Close"], window=10).sma_indicator()
    feats["EMA20"]    = EMAIndicator(df["Close"], window=20).ema_indicator()
    feats["RSI14"]    = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    feats["MACD"]     = macd.macd()
    feats["MACD_SIG"] = macd.macd_signal()
    feats["OBV"]      = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    feats.dropna(inplace=True)
    return feats

# ─── 3) Forecasting functions ────────────────────────────────────────────────────
look_back = 60

def forecast_lstm(forecast_date):
    start = forecast_date - timedelta(days=look_back*3)
    df    = yf.download("7010.SR", start=start, end=forecast_date)
    feats = compute_features(df)
    arr   = scaler.transform(feats.values)
    window= arr[-look_back:]
    X     = window.reshape(1, look_back, arr.shape[1])
    pred  = lstm_model.predict(X, verbose=0)[0,0]
    # inverse‐scale only the Close channel
    pad = np.zeros((1, arr.shape[1]-1))
    inv = scaler.inverse_transform(np.hstack([[[pred]], pad]))
    return float(inv[0,0])

def forecast_rnn(forecast_date):
    start = forecast_date - timedelta(days=look_back*3)
    df    = yf.download("7010.SR", start=start, end=forecast_date)
    feats = compute_features(df)
    arr   = scaler.transform(feats.values)
    window= arr[-look_back:]
    X     = window.reshape(1, look_back, arr.shape[1])
    pred  = rnn_model.predict(X, verbose=0).flatten()[0]
    pad = np.zeros((1, arr.shape[1]-1))
    inv = scaler.inverse_transform(np.hstack([[[pred]], pad]))
    return float(inv[0,0])

def forecast_arima():
    return float(arima_model.predict(n_periods=1)[0])

def forecast_es():
    return float(es_model.forecast(1)[0])

# ─── 4) Streamlit page ────────────────────────────────────────────────────────────
st.set_page_config(page_title="STC Stock Forecasts", layout="wide")

st.title("📈 STC (7010.SR) Next-Day Forecast Comparison")
st.markdown(
    "**Actual** close price from Yahoo Finance, plus **predicted** close for "
    "tomorrow from four different models."
)

# show latest close
data         = yf.download("7010.SR", period="7d", interval="1d")
latest_close = data["Close"].iloc[-1]
st.metric("Latest Close (SAR)", f"{latest_close:.2f}")

st.markdown("---")

# when the user clicks, run all forecasts
if st.button("Forecast Next Day"):
    target_date = date.today() + timedelta(days=1)
    with st.spinner("Running all models…"):
        lstm_pred  = forecast_lstm(target_date)
        rnn_pred   = forecast_rnn(target_date)
        arima_pred = forecast_arima()
        es_pred    = forecast_es()

    # display as metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LSTM",           f"{lstm_pred:.2f} SAR")
    c2.metric("Simple RNN",     f"{rnn_pred:.2f} SAR")
    c3.metric("ARIMA",          f"{arima_pred:.2f} SAR")
    c4.metric("Exp. Smoothing", f"{es_pred:.2f} SAR")

    st.markdown("**Comparison Table**")
    df_preds = pd.DataFrame({
        "Model":          ["LSTM", "Simple RNN", "ARIMA", "Exp. Smoothing"],
        "Forecast (SAR)": [lstm_pred, rnn_pred, arima_pred, es_pred]
    }).set_index("Model")
    st.table(df_preds)

else:
    st.info("Click **Forecast Next Day** to see each model’s prediction.")
