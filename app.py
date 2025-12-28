import streamlit as st
import yfinance as yf

st.title("Stock Data Test")

symbol = st.text_input("Enter stock symbol", "AAPL")

if st.button("Fetch Data"):
    st.write("Fetching data...")

    data = yf.download(
        symbol,
        period="1mo",
        progress=False,
        threads=False
    )

    st.write("Raw data:")
    st.write(data)

    if data.empty:
        st.error("❌ Data is EMPTY")
    else:
        st.success("✅ Data fetched successfully")
