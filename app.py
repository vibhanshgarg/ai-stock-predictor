import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor")

st.title("📈 AI Stock Market Prediction Tool")
st.write("Educational project using Machine Learning")

stock = st.text_input(
    "Enter Stock Symbol",
    "AAPL (US) or INFY.NS (India)"
)

if st.button("Predict"):
    try:
        # Download stock data (more reliable)
        data = yf.download(stock, period="2y", progress=False)

        if data.empty:
            st.error("❌ No data found. Please check stock symbol.")
            st.stop()

        data = data[['Close']].dropna()
        data['Prev_Close'] = data['Close'].shift(1)
        data.dropna(inplace=True)

        X = data[['Prev_Close']]
        y = data['Close']

        model = LinearRegression()
        model.fit(X, y)

        last_close = float(data['Close'].iloc[-1])
        predicted_price = float(model.predict([[last_close]])[0])

        trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

        st.success(f"Previous Close: ₹ {round(last_close,2)}")
        st.success(f"Predicted Price: ₹ {round(predicted_price,2)}")
        st.write("Trend:", trend)

        st.subheader("Stock Price Trend (Last 2 Years)")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label="Close Price")
        ax.legend()
        st.pyplot(fig)

        st.caption("⚠️ For educational purposes only")

    except Exception as e:
        st.error("⚠️ Error occurred:")
        st.code(e)
