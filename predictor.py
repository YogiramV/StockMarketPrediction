import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from data_fetcher import DataFetcher
from datetime import timedelta

def prepare_data(data, company):
    for i in range(0,30):
        data['Lag'+str(i+1)] = data['Close_'+company].shift(i+1)

    weights = np.arange(1, 8)

    data['SMA7'] = data['Close_' + company].rolling(window=7).mean()
    data['EMA7'] = data['Close_' + company].ewm(span=7, adjust=False).mean()

    data['WMA7'] = data['Close_' + company].rolling(window=7).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

    data = data.dropna()
    return data


def predict(company, target_date):
    fetcher = DataFetcher(company, target_date)
    data = fetcher.getStockData()

    close_col = f'Close_{company}'

    if close_col not in data.columns:
        raise ValueError(f"Data does not contain 'Close' column for {company}.")

    data = prepare_data(data, company)

    lag_cols = [f'Lag{i}' for i in range(1, 31)]
    ma_cols = ['SMA7', 'EMA7', 'WMA7']
    feature_cols = lag_cols + ma_cols

    X = data[feature_cols]
    y = data[close_col]

    tscv = TimeSeriesSplit(n_splits=5)
    
    '''Linear Models
    #Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    Train ridge regression
    ridge = Ridge(alpha=1.0) 
    
    #Train Lasso regression
    lasso = Lasso(alpha=0.001, max_iter=10000)  # smaller alpha = less regularization

    #Train ElasticNet
    elastic_net = ElasticNet(alpha=0.0001,l1_ratio=0.7,max_iter=10000,random_state=42)
    
    #ridge = Ridge(alpha=1.0)'''
    
    #Train RandomForestRegressor
    rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
    )
    
    mse_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    avg_mse = np.mean(mse_scores)
    print(f"Average Mean Squared Error (Cross-Validation): {avg_mse:.4f}")

    predictions = []
    last_known_date = data.index[-1]
    target_date = pd.to_datetime(target_date)

    if target_date <= pd.to_datetime(last_known_date):
        raise ValueError("Target date should be after the last known date.")

    weights = np.arange(1, 8)

    last_lags = [data[close_col].iloc[-i] for i in range(1, 31)]  # Lag1 to Lag30

    for i in range((target_date - last_known_date).days + 1):
        predicted_date = last_known_date + timedelta(days=i)

        feature_row = {}

        for lag_i in range(30):
            feature_row[f'Lag{lag_i+1}'] = last_lags[lag_i]

        last_7_prices = last_lags[:7][::-1]
        feature_row['SMA7'] = np.mean(last_7_prices)
        feature_row['EMA7'] = pd.Series(last_7_prices).ewm(span=7, adjust=False).mean().iloc[-1]
        feature_row['WMA7'] = np.dot(last_7_prices, weights) / weights.sum()

        feature_df = pd.DataFrame([feature_row])

        predicted_price = rf.predict(feature_df)[0]

        predictions.append((predicted_date.strftime('%Y-%m-%d'), predicted_price))

        last_lags = [predicted_price] + last_lags[:-1]

        new_row = pd.DataFrame({close_col: predicted_price}, index=[predicted_date])
        data = pd.concat([data, new_row])

    return predictions
