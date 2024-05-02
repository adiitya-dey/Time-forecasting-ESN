from ..ESN.tf_ESN import ESN


from darts.datasets import AirPassengersDataset, AusBeerDataset, ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ExchangeRateDataset, TrafficDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from einops import rearrange, repeat, reduce



uni_to_uni_datasets = [{"name": "AirPassengers",
             "dataset": AirPassengersDataset(),
             "input": 1,
             "output": 1},
            {"name":"AusBeer",
             "dataset": AusBeerDataset(),
             "input": 1,
             "output": 1}
            ]


multi_to_uni_datasets = [{"name": "ETTh1",
                   "dataset": ETTh1Dataset(),
                   "input": 6,
                   "output": 1},
                   {"name": "ETTh2",
                   "dataset": ETTh2Dataset(),
                   "input": 6,
                   "output": 1},
                   {"name": "ETTm1",
                   "dataset": ETTm1Dataset(),
                   "input": 6,
                   "output": 1},
                   {"name": "ETTm2",
                   "dataset": ETTm2Dataset(),
                   "input": 6,
                   "output": 1}]

multi_to_multi_datasets = [{"name": "ETTh1",
                   "dataset": ExchangeRateDataset(),
                   "input": 8,
                   "output": 8}]


for i in uni_to_uni_datasets:
    data = i["dataset"]
    input_size = i["input"]
    output_size = i["output"]
    name = i["name"]

    time_series = data.load()
    X = time_series.values()
    X_train, X_test = train_test_split(X, test_size=test_size, shuffle=False)

    sc = MinMaxScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)