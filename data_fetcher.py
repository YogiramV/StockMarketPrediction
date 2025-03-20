import yfinance as yf


class DataFetcher:
    stock_data = None

    def __init__(self, symbol, start_date, end_date):
        self.stock_data = yf.download(symbol, start_date, end_date)
        return

    def getStockData(self):
        return self.stock_data
