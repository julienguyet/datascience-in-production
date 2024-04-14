# functions for cleansing and feature engineering
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def features_selection_and_cleaning(data: pd.DataFrame,
                                    columns: list) -> pd.DataFrame:

    ''' Fills missing values in numerical columns
    of a DataFrame with the median.'''

    df = data[columns]

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)

    return df


def data_split(df: pd.DataFrame) -> pd.DataFrame:

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def train_data_encoder(df: pd.DataFrame, path: str) -> pd.DataFrame:

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df)
    X_train_encoded = encoder.transform(df)
    encoder_path = os.path.join(path, 'one-hot-encoder.joblib')
    joblib.dump(encoder, encoder_path)

    return X_train_encoded


def test_data_encoder(df: pd.DataFrame, path: str) -> pd.DataFrame:

    encoder = joblib.load(os.path.join(path, 'one-hot-encoder.joblib'))
    X_test_encoded = encoder.transform(df)

    return X_test_encoded


def train_data_scaler(data: np.array, path: str) -> np.array:

    scaler = MinMaxScaler()
    scaler.fit(data)
    X_train_scaled = scaler.transform(data)
    scaler_path = os.path.join(path, 'min-max-scaler.joblib')
    joblib.dump(scaler, scaler_path)

    return X_train_scaled


def test_data_scaler(data: np.array, path: str) -> np.array:

    scaler = joblib.load(os.path.join(path, 'min-max-scaler.joblib'))
    X_test_scaled = scaler.transform(data)

    return X_test_scaled
