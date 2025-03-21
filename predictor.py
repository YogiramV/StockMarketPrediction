import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_fetcher import DataFetcher


def prepare_data(data, company):
    # Create lag features (previous day price)
    data['Lag1'] = data['Close_'+company].shift(1)
    data['Lag2'] = data['Close_'+company].shift(2)
    data['Lag3'] = data['Close_'+company].shift(3)

    # Create rolling mean (7-day moving average)
    data['Rolling7'] = data['Close_'+company].rolling(window=7).mean()

    # Remove missing values created by lag and rolling mean
    data = data.dropna()

    return data


def predict(company, predict_date):
    # Fetch the stock data using the DataFetcher class
    fetcher = DataFetcher(company, predict_date)
    data = fetcher.getStockData()

    if f'Close_{company}' not in data.columns:
        raise ValueError(
            f"Data does not contain 'Close' column for {company}.")

    # Prepare data by creating features
    data = prepare_data(data, company)

    # Define X (features) and y (target)
    X = data[['Lag1', 'Lag2', 'Lag3', 'Rolling7']]  # Features
    y = data['Close_'+company]  # Target variable

    # **Time-based split** (Use last 20% of data for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices on the test set
    y_pred = model.predict(X_test)

    # Print the mean squared error to evaluate the model
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    # Predict future stock prices (e.g., next 'forecast_days' days)
    # Use the last few rows for future predictions
    latest_data = X.iloc[-1].values.reshape(1, -1)
    future_prediction = model.predict(latest_data)[0]

    return future_prediction.round(2)
