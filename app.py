import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor")

st.title("📈 AI Stock Market Prediction Tool")

symbol = st.text_input(
    "Enter Stock Symbol (AAPL, TSLA, MSFT, INFY.NS)",
    "AAPL"
)

if st.button("Predict"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y", interval="1d")

        if data is None or data.empty:
            st.error("❌ No data received from Yahoo Finance.")
            st.stop()

        df = data[['Close']].copy()
        df['Prev_Close'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        X = df[['Prev_Close']]
        y = df['Close']

        model = LinearRegression()
        model.fit(X, y)

        last_close = df['Close'].iloc[-1]
        predicted_price = model.predict([[last_close]])[0]

        trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

        st.subheader("Prediction Result")
        st.write(f"Previous Close: $ {round(float(last_close), 2)}")
        st.write(f"Predicted Price: $ {round(float(predicted_price), 2)}")
        st.write(f"Trend: {trend}")

        fig, ax = plt.subplots()
        ax.plot(df['Close'])
        ax.set_title("Last 1 Year Closing Prices")
        st.pyplot(fig)

    except Exception as e:
        st.error("Unexpected error occurred")
        st.code(str(e))
