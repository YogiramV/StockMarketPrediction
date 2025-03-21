import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    stock_data = None
    symbol = None

    def __init__(self, symbol, predict_date):
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(7300)).strftime('%Y-%m-%d')
        self.symbol = symbol
        # Download stock data from Yahoo Finance
        self.stock_data = yf.download(symbol, start=start_date, end=end_date)

        # Flatten the MultiIndex columns
        self.stock_data.columns = [
            f'{col[0]}_{col[1]}' for col in self.stock_data.columns]

        # Reset the index so that 'Date' becomes a column
        self.stock_data.reset_index(inplace=True)

        # Ensure 'Date' column is of type datetime (if it's not already)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])

        # Set the Date column as the index if it's not already
        self.stock_data.set_index('Date', inplace=True)

    def getStockData(self):
        return self.stock_data

    def getCloseData(self):
        # Extract only the 'Close' price for the given stock (e.g., 'AAPL')
        return self.stock_data[['Date', 'Close_'+self.symbol]]
