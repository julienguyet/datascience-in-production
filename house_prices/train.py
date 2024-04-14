# function to train the model and return its performance
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import ElasticNet
from house_prices.preprocess import data_split, train_data_encoder
from house_prices.preprocess import test_data_encoder, train_data_scaler
from house_prices.preprocess import test_data_scaler
from house_prices.__init__ import MODEL_BASE_PATH
import sys
sys.path.append('./house_prices')


def build_model(data: pd.DataFrame) -> dict[str, str]:

    X_train, X_test, y_train, y_test = data_split(data)
    X_train_encoded = train_data_encoder(df=X_train, path=MODEL_BASE_PATH)
    X_train_scaled = train_data_scaler(data=X_train_encoded,
                                       path=MODEL_BASE_PATH)
    X_test_encoded = test_data_encoder(df=X_test, path=MODEL_BASE_PATH)
    X_test_scaled = test_data_scaler(data=X_test_encoded, path=MODEL_BASE_PATH)

    elastic = ElasticNet(alpha=0.1, random_state=42)
    elastic.fit(X_train_scaled, y_train)

    model_path = os.path.join(MODEL_BASE_PATH, 'elastic-net.joblib')
    joblib.dump(elastic, model_path)

    y_pred_train = elastic.predict(X_train_scaled)
    y_pred_test = elastic.predict(X_test_scaled)

    rmsle_train = round(np.sqrt(mean_squared_log_error(y_pred_train,
                                                       y_train)), 2)
    rmsle_test = round(np.sqrt(mean_squared_log_error(y_pred_test, y_test)), 2)

    error = {"RMSLE Train": rmsle_train, "RMSLE Test": rmsle_test}

    return error
