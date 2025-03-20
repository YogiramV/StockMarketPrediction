from data_fetcher import DataFetcher
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


class Predictor:
    def __init__(self, symbol, start_date, end_date):
        with open('model_pipeline.pkl', 'rb') as f:
            self.symbol = symbol
            self.pipeline = pickle.load(f)
            stock_data = DataFetcher(
                symbol, start_date, end_date).getStockData()
            X = stock_data.drop('Close', axis=1)
            y = stock_data['Close']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            self.pipeline.fit(X_train, y_train)

    def predict(self, date):
        date = pd.to_datetime(date)
        next_date = date + pd.Timedelta(days=1)
        data = DataFetcher(self.symbol, date, next_date).getStockData()
        prediction = self.pipeline.predict(data.drop('Close', axis=1))
        return prediction
