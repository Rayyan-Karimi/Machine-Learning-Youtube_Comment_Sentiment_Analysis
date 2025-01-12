import yfinance as yf
import pandas as pd
import os

def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        stock_symbol (str): The ticker symbol of the stock (e.g., AAPL for Apple).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical stock data as a DataFrame.
    """
    try:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data fetched for {stock_symbol}. Check the symbol or date range.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

def save_to_csv(data, stock_symbol):
    """
    Save stock data to a CSV file.
    
    Args:
        data (pd.DataFrame): Data to save.
        stock_symbol (str): Stock ticker symbol (used in the file name).
    """
    if data.empty:
        print(f"No data to save for {stock_symbol}.")
        return
    
    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    file_path = f"data/{stock_symbol}_stock_data.csv"
    data.to_csv(file_path, index=True)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Example: Fetch Apple's stock data
    stock_symbol = "MSFT"
    start_date = "2020-01-01"
    end_date = "2023-12-31"

    # Fetch data
    data = fetch_data(stock_symbol, start_date, end_date)

    # Display the first few rows of the data
    print(data.head())

    # Save data to CSV
    save_to_csv(data, stock_symbol)
