import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Stock Predictor (LSTM)", layout="wide")

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not API_KEY:
    st.error("API key missing. Set ALPHA_VANTAGE_API_KEY.")
    st.stop()

# ---------------- CACHED FETCH ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY
    }
    response = requests.get(url, params=params, timeout=10)
    return response.json()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
sequence_length = st.sidebar.slider("Sequence Length (Days)", 30, 90, 60)

# ---------------- MAIN ----------------
st.title("📈 AI Stock Market Prediction Tool (LSTM)")
st.write("Deep Learning based time-series forecasting")

if st.sidebar.button("Run LSTM Prediction"):

    raw = fetch_stock_data(symbol)

    if "Note" in raw:
        st.warning("⏳ API rate limit reached. Please wait.")
        st.stop()

    if "Time Series (Daily)" not in raw:
        st.error("❌ Invalid symbol or no data.")
        st.stop()

    df = pd.DataFrame.from_dict(
        raw["Time Series (Daily)"], orient="index"
    )

    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    df = df.sort_index()

    prices = df["Close"].values.reshape(-1, 1)

    # ---------------- SCALING ----------------
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    # ---------------- SEQUENCE CREATION ----------------
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i, 0])
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ---------------- LSTM MODEL ----------------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # ---------------- NEXT DAY PREDICTION ----------------
    last_sequence = scaled_prices[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

    next_day_scaled = model.predict(last_sequence, verbose=0)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

    last_close = df["Close"].iloc[-1]

    # ---------------- 7-DAY FORECAST ----------------
    future_prices = []
    current_seq = last_sequence.copy()

    for _ in range(7):
        pred = model.predict(current_seq, verbose=0)
        price = scaler.inverse_transform(pred)[0][0]
        future_prices.append(round(price, 2))

        pred_scaled = pred.reshape(1, 1, 1)
        current_seq = np.append(
            current_seq[:, 1:, :], pred_scaled, axis=1
        )

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)
    col1.metric("Last Close", f"${last_close:.2f}")
    col2.metric("Next Day Prediction", f"${next_day_price:.2f}")

    st.subheader("📅 7-Day LSTM Forecast")
    st.table(pd.DataFrame({
        "Day": range(1, 8),
        "Predicted Price ($)": future_prices
    }))

    st.subheader("📊 Historical Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Close Price")
    ax.legend()
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
---
### ⚠️ Disclaimer
This project is for **educational purposes only**.

### 🧠 Model
LSTM (Long Short-Term Memory) Neural Network  
Used for sequential time-series forecasting.
""")
