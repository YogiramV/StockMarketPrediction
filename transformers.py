from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ColumnRenamerAndResetIndex(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reset_index(drop=True)
        X.columns = ['_'.join(map(str, col)) for col in X.columns]
        return X
