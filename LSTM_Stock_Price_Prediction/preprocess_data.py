import pandas as pd

def preprocess_data(input_file, output_file):
    try:
        # Load the CSV file with the correct header and index
        data = pd.read_csv(input_file, skiprows=1)  # Skip the extra header row

        # Check the first few rows to verify the structure
        print("Data preview before preprocessing:")
        print(data.head())

        # Rename columns to meaningful names
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Convert 'Date' column to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Drop rows where 'Date' couldn't be parsed
        data.dropna(subset=['Date'], inplace=True)

        # Sort data by date
        data.sort_values(by='Date', inplace=True)

        # Save the cleaned data to a new CSV file
        data.to_csv(output_file, index=False)

        print("Data preprocessing completed successfully. Cleaned data saved.")
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        print("Data preprocessing failed.")

if __name__ == "__main__": # executes the script when run directly
    input_file = "data/MSFT_stock_data.csv"
    output_file = "data/MSFT_preprocessed.csv"
    # input_file = "data/AAPL_stock_data.csv"
    # output_file = "data/AAPL_preprocessed.csv"
    preprocess_data(input_file, output_file)
