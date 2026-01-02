import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

st.set_page_config(page_title="AI Stock Predictor")
st.title("📈 AI Stock Market Prediction Tool")

if not API_KEY:
    st.error("API key not found. Please set ALPHA_VANTAGE_API_KEY.")
    st.stop()

symbol = st.text_input("Enter Stock Symbol (AAPL, TSLA, MSFT)", "AAPL")

if st.button("Predict"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if "Time Series (Daily)" not in data:
        st.error("❌ No data returned (API limit / invalid symbol).")
        st.code(data)
        st.stop()

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    )

    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    df = df.sort_index()

    df["Prev_Close"] = df["Close"].shift(1)
    df.dropna(inplace=True)

    X = df[["Prev_Close"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    last_close = df["Close"].iloc[-1]
    predicted_price = model.predict([[last_close]])[0]

    trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

    st.subheader("Prediction Result")
    st.write(f"Last Close: $ {round(last_close, 2)}")
    st.write(f"Predicted Price: $ {round(predicted_price, 2)}")
    st.write(f"Trend: {trend}")

    fig, ax = plt.subplots()
    ax.plot(df["Close"])
    ax.set_title("Stock Price (Last ~100 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.caption("⚠️ Educational purpose only. Not financial advice.")
