import pandas as pd

csv_path = "data/AAPL_stock_data.csv"
data = pd.read_csv(csv_path)
print(data.head())
