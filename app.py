import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor")

st.title("📈 AI Stock Market Prediction Tool")
st.write("AI-based stock price prediction using Machine Learning")

symbol = st.text_input("Enter Stock Symbol (AAPL / TSLA / INFY.NS)", "AAPL")

if st.button("Predict"):
    # -----------------------------
    # 1️⃣ FETCH DATA (WORKING PART)
    # -----------------------------
    data = yf.download(
        symbol,
        period="1y",
        progress=False,
        threads=False
    )

    if data.empty:
        st.error("❌ No data found for this symbol")
        st.stop()

    st.success("✅ Stock data fetched successfully")

    # -----------------------------
    # 2️⃣ DATA PREPARATION
    # -----------------------------
    df = data[['Close']].copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    if len(df) < 10:
        st.error("❌ Not enough data for prediction")
        st.stop()

    X = df[['Prev_Close']]
    y = df['Close']

    # -----------------------------
    # 3️⃣ TRAIN MODEL
    # -----------------------------
    model = LinearRegression()
    model.fit(X, y)

    # -----------------------------
    # 4️⃣ PREDICTION (FIXED)
    # -----------------------------
    last_close = float(df['Close'].iloc[-1])
    predicted_price = float(model.predict([[last_close]])[0])

    trend = "UP 📈" if predicted_price > last_close else "DOWN 📉"

    # -----------------------------
    # 5️⃣ DISPLAY RESULTS
    # -----------------------------
    st.subheader("Prediction Result")
    st.write(f"**Previous Close:** ₹ {round(last_close, 2)}")
    st.write(f"**Predicted Price:** ₹ {round(predicted_price, 2)}")
    st.write(f"**Trend:** {trend}")

    # -----------------------------
    # 6️⃣ VISUALIZATION
    # -----------------------------
    st.subheader("Stock Price Trend (Last 1 Year)")
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label="Closing Price")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.caption("⚠️ Educational purpose only. Not financial advice.")
