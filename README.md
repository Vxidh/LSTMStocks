# Stock Price Prediction using LSTM

This Streamlit app retrieves historical stock data, trains an LSTM model with PyTorch to predict future prices, and visualizes actual vs. predicted values.

---

## Workflow

1. **Fetch Data:** Enter a stock symbol and click "Fetch Data" to retrieve historical closing prices.
2. **Preprocess:** Normalize data and convert it into sequences for the LSTM.
3. **Train Model:** Train the LSTM model using MSE loss and the Adam optimizer.
4. **Prdiction:** Buy/Sell/Hold will be generated.
---

## Dependencies
pip install requirements.txt