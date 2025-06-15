from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf

import joblib


#* load a pre-trained model and its corresponding scaler
model = load_model("./lstm_nn/06 Jun - 00 34.h5")
scaler = joblib.load("./scalers/06 Jun - 00 34.save")

# create a prediction pipeline
def prepare_input_data(stock_data, look_back=60):
    """Convert raw stock data to model input format"""
    scaled_data = scaler.transform(stock_data[['Close']].values)
    sequence = scaled_data[-look_back:]  # Take most recent {look_back} days
    return np.reshape(sequence, (1, look_back, 1))  # Reshape for LSTM

def generate_signal(stock_data):
    """Generate buy/sell/hold recommendation"""
    X = prepare_input_data(stock_data)
    predicted_price = scaler.inverse_transform(model.predict(X))[0][0]
    current_price = stock_data['Close'].iloc[-1]
    
    # Simple strategy - buy if predicted to rise >2%
    threshold = 0.02  # 2%
    if predicted_price > current_price * (1 + threshold):
        return "BUY", predicted_price
    elif predicted_price < current_price * (1 - threshold):
        return "SELL", predicted_price
    else:
        return "HOLD", predicted_price

#* analyze a stock and return a recommendation
def analyze_stock(ticker):
    """Analyze a stock and return recommendation"""
    data = yf.download(ticker, period="3mo", interval="1d")
    if data.empty:
        return None
    
    signal, predicted_price = generate_signal(data)
    return {
        'ticker': ticker,
        'current_price': data['Close'].iloc[-1],
        'predicted_price': predicted_price,
        'signal': signal,
        'confidence': abs(predicted_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    }