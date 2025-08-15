import pandas as pd
import ta
from alpha_vantage.timeseries import TimeSeries

def fetch_stock_data(api_key, symbol, max_data_points=1000):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data.tail(max_data_points)
    return data

def add_technical_indicators(data, use_rsi=True, use_macd=True, use_bb=True, use_atr=True):
    if use_rsi:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    if use_macd:
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    if use_bb:
        data['BB_high'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['BB_low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    if use_atr:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return data
