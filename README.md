# Stock Price Prediction using LSTMs

This Streamlit app retrieves historical stock data, trains an LSTM model with PyTorch to predict future prices, and visualizes actual vs. predicted values.

## Setup

1. Make sure you have Python 3.7+ installed
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Enter your Alpha Vantage API key (or use the default one provided)
4. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
5. Adjust the model parameters as desired
6. Click "Predict" to generate predictions

## Features

- LSTM-based stock price prediction
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Interactive visualizations
- Performance metrics
- Risk assessment
- Buy/Sell signals with confidence threshold

## Disclaimer

This application is for educational purposes only. The predictions made by this model should not be used as the sole basis for making investment decisions. Always conduct your own research and consider consulting financial advisors before making investment decisions.

---

## Workflow

1. **Fetch Data:** Enter a stock symbol and click "Fetch Data" to retrieve historical closing prices.
2. **Preprocess:** Normalize data and convert it into sequences for the LSTM.
3. **Train Model:** Train the LSTM model using MSE loss and the Adam optimizer.
4. **Prdiction:** Buy/Sell/Hold based on whatever the model thinks.
---

## Dependencies
pip install requirements.txt