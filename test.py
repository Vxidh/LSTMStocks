import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import time

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_layer_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out

# App title
st.title('Stock Price Prediction with LSTM')

# Sidebar for inputs
st.sidebar.header('User Input Parameters')

# Input for API key
api_key = st.sidebar.text_input('Alpha Vantage API Key:', '53M5TREJYNTTTL91')

# Input for stock symbol
symbol = st.sidebar.text_input('Stock Symbol:', 'AAPL')

# Input for lookback period
look_back = st.sidebar.slider('Look Back Period (days):', min_value=5, max_value=100, value=60)

# Button to trigger predictions
if st.sidebar.button('Predict'):
    with st.spinner('Fetching stock data...'):
        try:
            # Fetch the stock data from Alpha Vantage
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            
            # Process the data
            data = data.reset_index()
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Display recent stock data
            st.subheader(f'Recent {symbol} Stock Price Data')
            st.dataframe(data.tail())
            
            # Plot historical prices
            st.subheader(f'{symbol} Historical Prices')
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data['Close'][-365:], label='Close Price')
            ax1.set_title(f'{symbol} Stock Price (Last Year)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            st.pyplot(fig1)
            
            # Create a separate scaler just for Close prices
            # This simplifies the inverse transformation process
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_prices = data['Close'].values.reshape(-1, 1)
            close_scaler.fit(close_prices)
            
            # Normalize all data for the LSTM model
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
            
            # Create the dataset for LSTM model
            X, y = [], []
            
            for i in range(look_back, len(data_scaled)):
                X.append(data_scaled[i-look_back:i].values)
                y.append(data_scaled.iloc[i]['Close'])  # Using 'Close' directly from dataframe
            
            X, y = np.array(X), np.array(y)
            
            # Store the corresponding dates and actual close prices for later use
            prediction_dates = data.index[look_back:].copy()
            actual_closes = data.loc[prediction_dates, 'Close'].values
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            
            # Train-test split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
            test_dates = prediction_dates[train_size:]
            test_actuals = actual_closes[train_size:]
            
            with st.spinner('Training the model... This may take a few minutes.'):
                # Initialize and train the model
                model = LSTMModel(input_size=5)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Progress bar for training
                progress_bar = st.progress(0)
                
                epochs = 50
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress
                    progress_bar.progress((epoch + 1) / epochs)
                    
                # Make predictions
                model.eval()
                with torch.no_grad():
                    predicted_scaled = model(X_test)
                
                # Convert predictions back to original scale
                predicted_scaled_np = predicted_scaled.numpy()
                predicted_prices = close_scaler.inverse_transform(predicted_scaled_np)
                
                # Generate Buy/Sell Signals
                signals = []
                for i in range(1, len(predicted_prices)):
                    if predicted_prices[i] > predicted_prices[i - 1]:
                        signals.append('Buy')
                    elif predicted_prices[i] < predicted_prices[i - 1]:
                        signals.append('Sell')
                    else:
                        signals.append('Hold')
                
                # Adjust the first element to 'Hold' since there is no previous price for comparison
                signals.insert(0, 'Hold')
                
                # Create a dataframe for plotting signals along with the real prices
                signal_df = pd.DataFrame({
                    'Date': test_dates,
                    'Real Price': test_actuals,
                    'Predicted Price': predicted_prices.flatten(),
                    'Signal': signals
                })
                
                # Plot the results with signals
                st.subheader('Prediction Results')
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(signal_df['Date'], signal_df['Real Price'], color='red', label='Real Stock Price')
                ax2.plot(signal_df['Date'], signal_df['Predicted Price'], color='blue', label='Predicted Stock Price')
                
                # Highlight Buy/Sell signals
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
                
                # Display the most recent recommendation
                st.subheader('Most Recent Signal')
                latest_date = signal_df['Date'].iloc[-1]
                latest_signal = signal_df['Signal'].iloc[-1]
                latest_price = signal_df['Real Price'].iloc[-1]
                
                # Color-coded recommendation
                if latest_signal == 'Buy':
                    st.success(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                elif latest_signal == 'Sell':
                    st.error(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                else:
                    st.info(f"**Recommendation for {symbol} on {latest_date.date()}: {latest_signal} at ${latest_price:.2f}**")
                
                # Display signal history in a table
                st.subheader('Signal History (Last 10 Days)')
                st.dataframe(signal_df.tail(10)[['Date', 'Real Price', 'Predicted Price', 'Signal']].style.format({
                    'Real Price': '${:.2f}',
                    'Predicted Price': '${:.2f}'
                }))
                
                # Calculate model performance metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(test_actuals, predicted_prices)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_actuals, predicted_prices)
                r2 = r2_score(test_actuals, predicted_prices)
                
                # Display metrics in columns
                st.subheader('Model Performance Metrics')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                col4.metric("R² Score", f"{r2:.4f}")
                
                # Feature for today's prediction
                st.subheader("Today's Prediction")
                
                # Get the most recent look_back days of data for prediction
                recent_data = data_scaled.iloc[-look_back:].values
                recent_data = np.reshape(recent_data, (1, look_back, 5))
                recent_tensor = torch.tensor(recent_data, dtype=torch.float32)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    today_pred_scaled = model(recent_tensor)
                
                # Transform back to original scale using the close price scaler
                today_pred_price = close_scaler.inverse_transform(today_pred_scaled.numpy())[0, 0]
                
                # Get yesterday's price for comparison
                yesterday_price = data['Close'].iloc[-1]
                
                # Generate signal based on prediction
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

# Add additional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses LSTM (Long Short-Term Memory) neural networks to predict stock prices 
    and generate buy/sell signals based on predicted price movements.
    
    The model is trained on historical data from Alpha Vantage.
    """
)
st.sidebar.warning(
    """
    **Disclaimer**: This app is for educational purposes only. 
    Trading stocks based solely on these predictions is risky. 
    Always do your own research before making investment decisions.
    """
)