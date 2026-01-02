import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Market Predictor")

st.title("📈 AI Stock Market Prediction Tool")
st.write("AI-based stock price prediction using Machine Learning")

# User input
symbol = st.text_input(
    "Enter Stock Symbol (AAPL, TSLA, INFY.NS)",
    "AAPL"
)

if st.button("Predict"):
    data = yf.download(symbol, period="1y", progress=False)

    if data.empty:
        st.error("❌ No data found. Check the stock symbol.")
        st.stop()

    # Prepare data
    df = data[['Close']].copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['Close']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    last_close = df['Close'].iloc[-1]

    predicted_price = model.predict(
        np.array([[last_close]], dtype=float)
    )[0]

    trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

    # Results
    st.subheader("Prediction Result")
    st.write(f"Previous Close: ₹ {round(float(last_close), 2)}")
    st.write(f"Predicted Price: ₹ {round(float(predicted_price), 2)}")
    st.write(f"Trend: {trend}")

    # Plot
    st.subheader("Stock Price Trend (Last 1 Year)")
    fig, ax = plt.subplots()
    ax.plot(df['Close'])
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.caption("⚠️ For educational purposes only. Not financial advice.")
