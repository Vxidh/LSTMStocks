import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
from data import fetch_stock_data, add_technical_indicators
from train import prepare_data, train_model
from utils import clear_memory
import seaborn as sns

st.title('Stock Price Prediction with LSTM')
st.sidebar.header('User Input Parameters')

api_key = st.sidebar.text_input('Alpha Vantage API Key:', '53M5TREJYNTTTL91', type='password')
symbol = st.sidebar.text_input('Stock Symbol:', 'AAPL').upper()

st.sidebar.subheader('Model Parameters')
look_back = st.sidebar.slider('Look Back Period (days):', min_value=5, max_value=100, value=60)
hidden_size = st.sidebar.slider('Hidden Layer Size:', min_value=50, max_value=200, value=100)
num_layers = st.sidebar.slider('Number of LSTM Layers:', min_value=1, max_value=5, value=3)
dropout_rate = st.sidebar.slider('Dropout Rate:', min_value=0.0, max_value=0.5, value=0.3, step=0.1)
learning_rate = st.sidebar.slider('Learning Rate:', min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
epochs = st.sidebar.slider('Number of Epochs:', min_value=50, max_value=200, value=100)

st.sidebar.subheader('Technical Indicators')
use_rsi = st.sidebar.checkbox('Use RSI', value=True)
use_macd = st.sidebar.checkbox('Use MACD', value=True)
use_bb = st.sidebar.checkbox('Use Bollinger Bands', value=True)
use_atr = st.sidebar.checkbox('Use ATR', value=True)

st.sidebar.subheader('Visualization Theme')
theme = st.sidebar.selectbox(
    'Select Theme',
    ['Default', 'Dark', 'Light']
)

st.sidebar.subheader('Signal Parameters')
confidence_threshold = st.sidebar.slider(
    'Confidence Threshold (%)',
    min_value=0,
    max_value=100,
    value=70
)

if st.sidebar.button('Predict'):
    with st.spinner('Fetching stock data...'):
        try:
            # Fetch and preprocess data
            data = fetch_stock_data(api_key, symbol)
            data = add_technical_indicators(data, use_rsi=use_rsi, use_macd=use_macd, use_bb=use_bb, use_atr=use_atr)
            if data.isna().any().any():
                st.error("Warning: Data still contains NaN values after preprocessing")
                data = data.dropna()
            st.subheader(f'Recent {symbol} Stock Price Data')
            st.dataframe(data.tail())
            st.subheader(f'{symbol} Historical Prices')
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data['Close'][-365:], label='Close Price')
            ax1.set_title(f'{symbol} Stock Price (Last Year)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            st.pyplot(fig1)
            input_size = len(data.columns)
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_prices = data['Close'].values.reshape(-1, 1)
            close_scaler.fit(close_prices)
            # Prepare data for LSTM
            X, y, scaler = prepare_data(data, look_back)
            prediction_dates = data.index[look_back:].copy()
            actual_closes = data.loc[prediction_dates, 'Close'].values
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
            test_dates = prediction_dates[train_size:]
            test_actuals = actual_closes[train_size:]
            clear_memory()
            # Train model
            with st.spinner('Training the model... This may take a few minutes.'):
                model, train_losses, val_losses = train_model(
                    X_train, y_train, X_test, y_test,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    dropout=dropout_rate
                )
                st.subheader('Training Progress')
                fig_train, ax_train = plt.subplots(figsize=(10, 6))
                ax_train.plot(train_losses, label='Training Loss')
                ax_train.plot(val_losses, label='Validation Loss')
                ax_train.set_title('Model Training Progress')
                ax_train.set_xlabel('Epoch')
                ax_train.set_ylabel('Loss')
                ax_train.legend()
                st.pyplot(fig_train)
                model.eval()
                with torch.no_grad():
                    predicted_scaled = model(X_test)
                predicted_scaled_np = predicted_scaled.numpy()
                predicted_prices = close_scaler.inverse_transform(predicted_scaled_np)
                clear_memory()
                signals = []
                for i in range(1, len(predicted_prices)):
                    price_change = ((predicted_prices[i] - predicted_prices[i-1]) / predicted_prices[i-1]) * 100
                    if abs(price_change) < confidence_threshold:
                        signals.append('Hold')
                    elif price_change > 0:
                        signals.append('Buy')
                    else:
                        signals.append('Sell')
                signals.insert(0, 'Hold')
                signal_df = pd.DataFrame({
                    'Date': test_dates,
                    'Real Price': test_actuals,
                    'Predicted Price': predicted_prices.flatten(),
                    'Signal': signals
                })
                st.subheader('Prediction Results')
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(signal_df['Date'], signal_df['Real Price'], color='red', label='Real Stock Price')
                ax2.plot(signal_df['Date'], signal_df['Predicted Price'], color='blue', label='Predicted Stock Price')
                buy_signals = signal_df[signal_df['Signal'] == 'Buy']
                sell_signals = signal_df[signal_df['Signal'] == 'Sell']
                ax2.scatter(buy_signals['Date'], buy_signals['Predicted Price'], marker='^', color='green', label='Buy Signal', alpha=1)
                ax2.scatter(sell_signals['Date'], sell_signals['Predicted Price'], marker='v', color='red', label='Sell Signal', alpha=1)
                ax2.set_title(f'{symbol} Buy/Sell Signals')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Price')
                ax2.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)
                st.subheader('Most Recent Signal')
                latest_date = signal_df['Date'].iloc[-1]
                latest_signal = signal_df['Signal'].iloc[-1]
                latest_price = signal_df['Real Price'].iloc[-1]
                if latest_signal == 'Buy':
                    st.success(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                elif latest_signal == 'Sell':
                    st.error(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                else:
                    st.info(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                st.subheader('Signal History (Last 10 Days)')
                st.dataframe(signal_df.tail(10)[['Date', 'Real Price', 'Predicted Price', 'Signal']].style.format({
                    'Real Price': '${:.2f}',
                    'Predicted Price': '${:.2f}'
                }))
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(test_actuals, predicted_prices)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_actuals, predicted_prices)
                r2 = r2_score(test_actuals, predicted_prices)
                st.subheader('Model Performance Metrics')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                col4.metric("R² Score", f"{r2:.4f}")
                st.subheader("Today's Prediction")
                recent_data = scaler.transform(data.iloc[-look_back:].values)
                recent_data = np.reshape(recent_data, (1, look_back, input_size))
                recent_tensor = torch.tensor(recent_data, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    today_pred_scaled = model(recent_tensor)
                today_pred_price = close_scaler.inverse_transform(today_pred_scaled.numpy())[0, 0]
                yesterday_price = data['Close'].iloc[-1]
                if today_pred_price > yesterday_price:
                    today_signal = "Buy"
                    st.success(f"**TODAY'S RECOMMENDATION: {today_signal} {symbol}**")
                    st.write(f"Predicted price: ${today_pred_price:.2f}")
                    st.write(f"Previous closing price: ${yesterday_price:.2f}")
                    st.write(f"Predicted change: +${(today_pred_price - yesterday_price):.2f} ({((today_pred_price - yesterday_price)/yesterday_price)*100:.2f}%)")
                elif today_pred_price < yesterday_price:
                    today_signal = "Sell"
                    st.error(f"**TODAY'S RECOMMENDATION: {today_signal} {symbol}**")
                    st.write(f"Predicted price: ${today_pred_price:.2f}")
                    st.write(f"Previous closing price: ${yesterday_price:.2f}")
                    st.write(f"Predicted change: -${(yesterday_price - today_pred_price):.2f} ({((yesterday_price - today_pred_price)/yesterday_price)*100:.2f}%)")
                else:
                    today_signal = "Hold"
                    st.info(f"**TODAY'S RECOMMENDATION: {today_signal} {symbol}**")
                    st.write(f"Predicted price: ${today_pred_price:.2f}")
                    st.write(f"Previous closing price: ${yesterday_price:.2f}")
                    st.write("No significant price change predicted.")
                st.subheader('Summary Statistics')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Average Price', f"${data['Close'].mean():.2f}")
                with col2:
                    st.metric('Price Volatility', f"{data['Close'].std():.2f}")
                with col3:
                    st.metric('Trading Volume', f"{data['Volume'].mean():,.0f}")
                st.subheader('Model Performance Comparison')
                performance_df = pd.DataFrame({
                    'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
                    'Value': [mse, rmse, mae, r2]
                })
                st.bar_chart(performance_df.set_index('Metric'))
                st.subheader('Risk Assessment')
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
                sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() != 0 else 0
                max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
                var_95 = np.percentile(returns, 5) * np.sqrt(252)
                cvar_95 = returns[returns <= var_95].mean() * np.sqrt(252)
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                with risk_col1:
                    st.metric('Annualized Volatility', f"{volatility:.2%}")
                    st.metric('Sharpe Ratio', f"{sharpe_ratio:.2f}")
                with risk_col2:
                    st.metric('Sortino Ratio', f"{sortino_ratio:.2f}")
                    st.metric('Max Drawdown', f"{max_drawdown:.2%}")
                with risk_col3:
                    st.metric('VaR (95%)', f"{var_95:.2%}")
                    st.metric('CVaR (95%)', f"{cvar_95:.2%}")
                risk_rating = "High"
                if volatility < 0.2:
                    risk_rating = "Low"
                elif volatility < 0.5:
                    risk_rating = "Medium"
                st.subheader('Risk Rating')
                if risk_rating == "High":
                    st.error(f"**Risk Rating: {risk_rating}** - This stock shows high volatility and risk. Consider smaller position sizes and ensure proper risk management.")
                elif risk_rating == "Medium":
                    st.warning(f"**Risk Rating: {risk_rating}** - This stock shows moderate volatility. Monitor position sizes and market conditions.")
                else:
                    st.success(f"**Risk Rating: {risk_rating}** - This stock shows lower volatility. Still, maintain proper risk management practices.")
                st.subheader('Risk Analysis')
                risk_analysis = f"""
                - **Volatility**: {volatility:.2%} annualized volatility indicates {'high' if volatility > 0.5 else 'moderate' if volatility > 0.2 else 'low'} price fluctuations
                - **Risk-Adjusted Returns**: 
                    - Sharpe Ratio of {sharpe_ratio:.2f} suggests {'poor' if sharpe_ratio < 0.5 else 'moderate' if sharpe_ratio < 1 else 'good'} risk-adjusted returns
                    - Sortino Ratio of {sortino_ratio:.2f} indicates {'poor' if sortino_ratio < 0.5 else 'moderate' if sortino_ratio < 1 else 'good'} downside risk management
                - **Downside Risk**:
                    - Maximum Drawdown of {max_drawdown:.2%} shows the worst historical price decline
                    - Value at Risk (VaR) of {var_95:.2%} indicates potential loss at 95% confidence
                    - Conditional VaR of {cvar_95:.2%} shows average loss in worst scenarios
                """
                st.info(risk_analysis)
                st.subheader('Risk Visualization')
                fig_risk, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                returns.hist(bins=50, ax=ax1)
                ax1.set_title('Returns Distribution')
                ax1.set_xlabel('Returns')
                ax1.set_ylabel('Frequency')
                data['Close'].plot(ax=ax2)
                ax2.set_title('Price History')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Price')
                plt.tight_layout()
                st.pyplot(fig_risk)
                st.sidebar.markdown("---")
                st.sidebar.subheader("Risk Disclaimer")
                st.sidebar.warning(
                    """
                    **Important Risk Information:**
                    1. This model is for educational purposes only
                    2. Past performance does not guarantee future results
                    3. The model's accuracy is limited by:
                       - Market conditions
                       - Available historical data
                       - Technical limitations
                    4. Always conduct your own research
                    5. Consider consulting financial advisors
                    6. Never invest more than you can afford to lose
                    """
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check the stock symbol and API key, or try again later.")
else:
    st.write("""
    ## How to use this app:
    1. Enter your Alpha Vantage API key in the sidebar (default key provided but has request limits)
    2. Enter the stock symbol you want to analyze (default: AAPL)
    3. Adjust the look-back period if desired (default: 60 days)
    4. Click the "Predict" button to generate predictions
    
    The app will fetch historical data, train an LSTM model, and provide a buy/sell recommendation for today.
    """)
    st.info("Note: The initial prediction may take a few minutes to process due to model training.")
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses LSTM (Long Short-Term Memory) neural networks to predict stock prices 
    and generate buy/sell signals based on predicted price movements.
    """
)
