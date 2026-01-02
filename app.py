import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Market Predictor")

st.title("📈 AI Stock Market Prediction Tool")
st.write("AI-based stock price prediction using Machine Learning")

symbol = st.text_input(
    "Enter Stock Symbol (AAPL, TSLA, INFY.NS)",
    "AAPL"
)

if st.button("Predict"):
    data = yf.download(
        tickers=symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        threads=False,
        progress=False
    )

    if data is None or len(data) == 0:
        st.error("❌ No data available. Try again in a few seconds.")
        st.stop()

    df = data[['Close']].copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_close = df['Close'].iloc[-1]
    predicted_price = model.predict(np.array([[last_close]]))[0]

    trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

    st.subheader("Prediction Result")
    st.write(f"Previous Close: ₹ {round(float(last_close), 2)}")
    st.write(f"Predicted Price: ₹ {round(float(predicted_price), 2)}")
    st.write(f"Trend: {trend}")

    st.subheader("Stock Price Trend (Last 1 Year)")
    fig, ax = plt.subplots()
    ax.plot(df['Close'])
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.caption("⚠️ Educational purpose only. Not financial advice.")
