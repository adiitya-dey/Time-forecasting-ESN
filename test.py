from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import scipy.signal as sig
import scipy.integrate as sint

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim

from einops import rearrange, repeat, reduce

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.tools import diff

# torch.manual_seed(123)
np.random.seed(123)

class ESN(BaseEstimator):

    def __init__(self, reservoir_size=100, input_size=1, output_size=1,  spectral_radius=1.0, connectivity_rate=1.0, epochs=1, lr=0.01, leaky_parameter=1.0, washout=1):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.epochs = epochs
        self.connectivity_rate = connectivity_rate
        self.lr = lr
        self.spectral_radius = spectral_radius
        self.washout = washout
        self.leaky_parameter = leaky_parameter
        self.output_size = output_size


        self.state = np.zeros((self.reservoir_size, 1))
        self.W_in = np.random.randn(reservoir_size, input_size) * 2 - 1
        
        self.W_out = None

        ## Initialize a random matrix and induce sparsity.
        self.W_res = np.random.randn(reservoir_size, reservoir_size)
        self.W_res[np.random.rand(*self.W_res.shape) > self.connectivity_rate] = 0

        ##  Scale the matrix based on user defined spectral radius.
        current_spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res = self.W_res * (self.spectral_radius / current_spectral_radius)


    def fit(self, X_train, y_train):
        
        ## Epochs are useless. Keep it as 1 always.
        for _ in range(self.epochs): 

            ## Reset states.
            concat_matrix = np.zeros((self.input_size + self.reservoir_size, 1))
            # self.state = np.zeros((self.reservoir_size, 1))

            ## Calculate state of reservoirs per time step
            for i in range(X_train.shape[0]-1):
                input = X_train[i].reshape(-1,1)
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = np.tanh(input_product + state_product)
                concat_matrix= np.hstack((concat_matrix, np.concatenate((self.state, input))))
            

            ## Update W_out
            mat1 = concat_matrix.T[1 + self.washout + self.output_size:,:]
            ridge_regressor= Ridge(alpha=self.lr)
            ridge_regressor.fit(mat1, y_train[self.washout:,:])
            self.W_out = ridge_regressor.coef_
            # self.W_out = np.dot(np.linalg.pinv(mat1), y_train)


    def predict(self, X_test):
            prediction = []
            for i in range(X_test.shape[0]):
                input = X_test[i].reshape(-1,1)
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = np.tanh(input_product + state_product)
                concat_matrix= np.concatenate((self.state, input))
                pred =  self.W_out@concat_matrix
                prediction.append(pred)
            
            prediction = np.array(prediction)
            prediction = rearrange(prediction, 'c r 1-> r c')
            return prediction.T





# Sinusoidal Curves - Seasonal stationary data.
def func1(x):
    return np.sin(x) + np.cos(2*x)


## The function will create Non-Overlapping sequences.
def create_sequences(data, pred_len):
    size = data.shape[0]
   
    X = []

    # Fill X and y with non-overlapping sequences
    for i in range(size - pred_len):
        start_idx = i + 1
        end_idx = start_idx + pred_len

        
        X.append(data[start_idx:end_idx, :])
            

    return np.array(X)


points = np.linspace(1,50, 500)
aug_series1 = func1(points)

## Train and Test Splitting of Time Series Data
X = aug_series1
X = X.reshape(-1,1)
X_train, X_test = train_test_split(X, test_size=0.1, shuffle=False)
print(X_train.shape)
print(X_test.shape)

sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# param_grid = {'reservoir_size': [10, 20, 50, 100],
#               'spectral_radius': [0.1, 0.2, 0.5, 0.7, 0.9, 1.0],
#             #   'connectivity_rate': [0.1, 0.2, 0.5, 0.7, 0.9, 1.0],
#               'leaky_parameter': [0.0, 0.2, 0.5, 0.8, 1.0]}

# grid = GridSearchCV(ESN(input_size=1, output_size=1), param_grid=param_grid, refit=True, cv=3, scoring="neg_mean_squared_error")


# grid.fit(X_train_std, X_train_std)


esn = ESN(reservoir_size=20, input_size=1, output_size=1, spectral_radius=0.8, connectivity_rate=0.8, washout=3)

y_train_std = create_sequences(X_train_std, 2)
y_train_std = rearrange(y_train_std, 'r c 1 -> r c')

esn.fit(X_train_std, y_train_std)

y_pred = esn.predict(X_test_std[:-1,:])

test_values= create_sequences(X_test_std[1:], 2)
test_values = rearrange(test_values, 'r c 1 -> r c')

print(np.sqrt(mean_squared_error(test_values[:,1], y_pred[-2:, 1])))

# y_pred = []
# for i in range(len(X_test_std)-1):
#     pred = esn.predict1(X_test_std[i].reshape(-1,1))
#     y_pred.append(pred)

# predictions = np.array(y_pred).reshape(-1, 1)
# test_values= X_test_std[1:]

# print(np.sqrt(mean_squared_error(test_values, predictions)))

# predictions_scaled = sc.inverse_transform(predictions)

# print(np.sqrt(mean_squared_error(predictions_scaled, X_test[1:])))