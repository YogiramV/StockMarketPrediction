import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_fetcher import DataFetcher
from datetime import timedelta

def prepare_data(data, company):
    # Create lag features (previous day price)
    for i in range(0,30):
        data['Lag'+str(i+1)] = data['Close_'+company].shift(i+1)

    # Define weights for WMA7 (e.g., linearly increasing weights: 1, 2, ..., 7)
    weights = np.arange(1, 8)

    # Create rolling averages for the given company
    data['SMA7'] = data['Close_' + company].rolling(window=7).mean()
    data['EMA7'] = data['Close_' + company].ewm(span=7, adjust=False).mean()

    # Weighted Moving Average (WMA7) with linear weights
    data['WMA7'] = data['Close_' + company].rolling(window=7).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

    # Remove missing values created by lag and rolling mean
    data = data.dropna()
    return data


def predict(company, target_date):
    # Fetch the stock data using the DataFetcher class (up to target_date)
    fetcher = DataFetcher(company, target_date)
    data = fetcher.getStockData()

    close_col = f'Close_{company}'

    if close_col not in data.columns:
        raise ValueError(f"Data does not contain 'Close' column for {company}.")

    # Prepare data by creating features (lags and moving averages)
    data = prepare_data(data, company)

    # Define feature columns: 30 lags + SMA7, EMA7, WMA7
    lag_cols = [f'Lag{i}' for i in range(1, 31)]
    ma_cols = ['SMA7', 'EMA7', 'WMA7']
    feature_cols = lag_cols + ma_cols

    # Feature matrix X and target vector y
    X = data[feature_cols]
    y = data[close_col]

    test_days = 20
    X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
    y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]


    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Initialize predictions list and last known date
    predictions = []
    last_known_date = data.index[-1]
    target_date = pd.to_datetime(target_date)

    if target_date <= pd.to_datetime(last_known_date):
        raise ValueError("Target date should be after the last known date.")

    # Define weights for WMA7 (linear weights 1 to 7)
    weights = np.arange(1, 8)

    # Initialize lag features list from last available data
    last_lags = [data[close_col].iloc[-i] for i in range(1, 31)]  # Lag1 to Lag30

    # Predict iteratively day-by-day until target_date
    for i in range((target_date - last_known_date).days + 1):
        predicted_date = last_known_date + timedelta(days=i)

        # Build feature row dictionary
        feature_row = {}

        # Add lag features from last_lags list
        for lag_i in range(30):
            feature_row[f'Lag{lag_i+1}'] = last_lags[lag_i]

        # Calculate moving averages from last_lags (most recent first)
        last_7_prices = last_lags[:7][::-1]  # reverse to get chronological order
        feature_row['SMA7'] = np.mean(last_7_prices)
        feature_row['EMA7'] = pd.Series(last_7_prices).ewm(span=7, adjust=False).mean().iloc[-1]
        feature_row['WMA7'] = np.dot(last_7_prices, weights) / weights.sum()

        # Convert to DataFrame for prediction
        feature_df = pd.DataFrame([feature_row])

        # Predict price
        predicted_price = model.predict(feature_df)[0]

        # Store prediction
        predictions.append((predicted_date.strftime('%Y-%m-%d'), predicted_price))

        # Update last_lags rolling list: new predicted price + drop oldest lag
        last_lags = [predicted_price] + last_lags[:-1]

        # Optionally, append predicted close price to data for tracking
        new_row = pd.DataFrame({close_col: predicted_price}, index=[predicted_date])
        data = pd.concat([data, new_row])

    # Evaluate model on test set (unchanged)
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    print(f"Test Mean Squared Error: {mse:.4f}")

    return predictions
