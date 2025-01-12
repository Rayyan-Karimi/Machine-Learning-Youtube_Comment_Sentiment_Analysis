import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_data(file_path, window_size=10):
    # Load preprocessed data
    data = pd.read_csv(file_path)
    
    # Extract 'Close' prices
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data to 0 to 1 range so that the NN can perform better, as this
    # avoids large difference in magnitude between features
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    normalized_prices = scaler.fit_transform(close_prices)
    
    # Create sliding windows (input-output pairs from our sequential data)
    X, y = [], []
    for i in range(len(normalized_prices) - window_size):
        X.append(normalized_prices[i:i+window_size])  # Input: n days
        y.append(normalized_prices[i+window_size])   # Output: n+1th day
    
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets - maintaining order for prediction to make sense
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Path to preprocessed data
    # file_path = "data/AAPL_preprocessed.csv"
    file_path = "data/MSFT_preprocessed.csv"
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_path)
