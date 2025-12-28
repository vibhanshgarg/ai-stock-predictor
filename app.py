import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor")

st.title("📈 AI Stock Market Prediction Tool")
st.write("Educational project using Machine Learning")

stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY)", "AAPL")

if st.button("Predict"):
    try:
        data = yf.download(stock, start="2018-01-01")

        data = data[['Close']]
        data['Prev_Close'] = data['Close'].shift(1)
        data.dropna(inplace=True)

        X = data[['Prev_Close']]
        y = data['Close']

        model = LinearRegression()
        model.fit(X, y)

        last_close = data['Close'].iloc[-1]
        predicted_price = model.predict([[last_close]])[0]

        trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

        st.success(f"Previous Close: ₹ {round(last_close,2)}")
        st.success(f"Predicted Price: ₹ {round(predicted_price,2)}")
        st.write("Trend:", trend)

        st.subheader("Actual Stock Price Trend")
        fig, ax = plt.subplots()
        ax.plot(data['Close'])
        st.pyplot(fig)

        st.caption("⚠️ For educational purposes only")

    except:
        st.error("Invalid stock symbol or data not available")
