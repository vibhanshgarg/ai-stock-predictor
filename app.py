import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Stock Market Predictor", layout="wide")

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not API_KEY:
    st.error("API key missing. Set ALPHA_VANTAGE_API_KEY in environment.")
    st.stop()

# ---------------- CACHED DATA FETCH ----------------
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
history_days = st.sidebar.slider("Historical Days", 30, 300, 120)

# ---------------- MAIN UI ----------------
st.title("📈 AI Stock Market Prediction Tool")
st.write("Advanced ML-based stock analysis & prediction system")

if st.sidebar.button("Run Prediction"):

    data = fetch_stock_data(symbol)

    if "Note" in data:
        st.warning("⏳ API rate limit reached. Please wait 1 minute.")
        st.stop()

    if "Error Message" in data or "Time Series (Daily)" not in data:
        st.error("❌ Invalid stock symbol or no data.")
        st.stop()

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    )

    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    df = df.sort_index()
    df = df.tail(history_days)

    # ---------------- FEATURE ENGINEERING ----------------
    df["Prev_Close"] = df["Close"].shift(1)
    df["SMA"] = df["Close"].rolling(5).mean()
    df["Volatility"] = df["Close"].rolling(5).std()
    df["EMA"] = df["Close"].ewm(span=10).mean()
    df.dropna(inplace=True)

    # ---------------- MODEL ----------------
    X = df[["Prev_Close", "SMA", "Volatility"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    last = df.iloc[-1]
    next_day_price = model.predict([[
        last["Close"],
        last["SMA"],
        last["Volatility"]
    ]])[0]

    # ---------------- BUY / SELL SIGNAL ----------------
    if next_day_price > last["Close"] * 1.01:
        signal = "BUY 🟢"
    elif next_day_price < last["Close"] * 0.99:
        signal = "SELL 🔴"
    else:
        signal = "HOLD ⚪"

    # ---------------- DISPLAY RESULTS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Last Close", f"${last['Close']:.2f}")
    col2.metric("Predicted Price", f"${next_day_price:.2f}")
    col3.metric("Signal", signal)

    # ---------------- 7-DAY FORECAST ----------------
    future_prices = []
    temp_row = last.copy()

    for _ in range(7):
        pred = model.predict([[
            temp_row["Close"],
            temp_row["SMA"],
            temp_row["Volatility"]
        ]])[0]
        future_prices.append(round(pred, 2))
        temp_row["Close"] = pred

    st.subheader("📅 7-Day Price Forecast")
    st.table(pd.DataFrame({
        "Day": range(1, 8),
        "Predicted Price ($)": future_prices
    }))

    # ---------------- CHART ----------------
    st.subheader("📊 Stock Price Analysis")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Close")
    ax.plot(df["SMA"], label="SMA (5)")
    ax.plot(df["EMA"], label="EMA (10)")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
---
### ⚠️ Disclaimer
This application is for **educational purposes only**.  
It does **not** provide financial or investment advice.

### ℹ️ About
Built using **Streamlit**, **Machine Learning**, and **Alpha Vantage API**.  
Deployed on **Render** with production-grade handling.
""")

