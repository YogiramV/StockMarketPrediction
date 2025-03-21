import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_fetcher import DataFetcher
from datetime import timedelta


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


def predict(company, target_date):
    # Fetch the stock data using the DataFetcher class (up to target_date)
    fetcher = DataFetcher(company, target_date)
    data = fetcher.getStockData()

    if f'Close_{company}' not in data.columns:
        raise ValueError(
            f"Data does not contain 'Close' column for {company}.")

    # Prepare data by creating features
    data = prepare_data(data, company)

    # Define X (features) and y (target)
    X = data[['Lag1', 'Lag2', 'Lag3', 'Rolling7']]  # Features
    y = data['Close_'+company]  # Target variable

    # **Time-based split** (Use all the data until the target date)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the stock prices iteratively for each day until the target_date
    predictions = []
    last_known_date = data.index[-1]  # Last date in the data

    target_date = pd.to_datetime(target_date)

    if target_date <= pd.to_datetime(last_known_date):
        raise ValueError("Target date should be after the last known date.")

    for i in range((target_date - last_known_date).days + 1):
        # Use the last row for prediction
        last_row = X.iloc[-1:]
        predicted_price = model.predict(last_row)[0]

        # Store the predicted price for the day
        predicted_date = last_known_date + timedelta(days=i)
        predictions.append((predicted_date.strftime(
            '%Y-%m-%d'), predicted_price))

        # Append this predicted price to the data to use it for predicting the next day
        new_data = {'Lag1': predictions[-1][1],  # Use the predicted price as Lag1
                    'Lag2': data['Lag1'].iloc[-1],
                    'Lag3': data['Lag2'].iloc[-1],
                    'Rolling7': data['Rolling7'].iloc[-1]}
        new_row = pd.DataFrame([new_data], index=[predicted_date])
        data = pd.concat([data, new_row], axis=0)
        X = data[['Lag1', 'Lag2', 'Lag3', 'Rolling7']]

    return predictions
