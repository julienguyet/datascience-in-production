# Data Science in Production

## 1. Environment set up

Start by cloning the repository and then at the root, execute the below command:

```
mkdir data && cd data
```

Then, download the dataset using the link below and save the files in your data folder. 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

This dataset was compiled by Dean De Cock and contains information about accomodations in the US. Our goal is to predict the sales price for each house (SalePrice variable).

## 2. Repository Overview

This repository is organized as follow:

------------

    ├── LICENSE
    ├── README.md          <- The top-level README.
    ├── data
    │   ├── train.csv       <- Data to use for training your model.
        ├── test.csv        <- Data to use for running predictions.
    │
    ├── house_prices        <- folder with .py files for model deployment
    │
    ├── models             <- Trained and serialized models: encoder and model joblib files
                              needed to run the predictions on fresh data.
    │
    ├── notebooks          <- Jupyter notebooks. Execute model-industrialization-final 
                              to run the prediction job.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment. You can
                              install them with `pip install -r requirements.txt`


## 3. Code Explanation

## 4. To go further
