# function to run the trained model on test data and return the predictions
import joblib
import numpy as np
import pandas as pd
from house_prices.preprocess import test_data_encoder, test_data_scaler
from house_prices.__init__ import MODEL_BASE_PATH
import sys
sys.path.append('./house_prices')


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:

    X_test_encoded = test_data_encoder(df=input_data, path=MODEL_BASE_PATH)
    X_test_scaled = test_data_scaler(data=X_test_encoded,
                                    path=MODEL_BASE_PATH)
    elastic = joblib.load('../models/elastic-net.joblib')
    Y = elastic.predict(X_test_scaled)

    return {"Here are our predictions using Elastic Net": Y}
