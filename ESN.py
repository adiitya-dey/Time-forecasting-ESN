from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import scipy.signal as sig
import scipy.integrate as sint

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim

from einops import rearrange, repeat, reduce

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.tools import diff

# torch.manual_seed(123)
np.random.seed(123)

class ESN:

    def __init__(self, reservoir_size, input_size, output_size, spectral_radius=1.0, connectivity_rate=1.0, leaky_parameter=1.0):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.state = np.zeros((self.reservoir_size, 1))
        self.W_in = np.random.randn(reservoir_size, input_size) * 2 - 1
        
        self.W_out = None # np.random.randn(output_size, reservoir_size + input_size) # This will not work because it will be replaced with new shape in ridge

        ## Create an all positive valued Reservoir Weight matrix with sparsity induced based on connectivity rate.
        self.W_res = np.random.randn(reservoir_size, reservoir_size)
        self.W_res[np.random.rand(*self.W_res.shape) > connectivity_rate] = 0

        ##  Scale the matrix based on user defined spectral radius.
        current_spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res = self.W_res * (spectral_radius / current_spectral_radius)


    def fit1(self, X, epochs=10, lr=0.01):
        for epoch in range(epochs): 

            # Reset states.
            concat_matrix = np.zeros((self.input_size + self.reservoir_size, 1))
            self.state = np.zeros((self.reservoir_size, 1))

            for i in range(X.shape[0]-1):
                input = X[i].reshape(-1,1)
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = np.tanh(input_product + state_product)
                concat_matrix= np.hstack((concat_matrix, np.concatenate((self.state, input))))
            

            # Update W_out
            y = X[1:,:]
            mat1 = concat_matrix.T[1:,:]
            ridge_regressor= Ridge(alpha=lr)
            ridge_regressor.fit(mat1, y)
            self.W_out = ridge_regressor.coef_


    def predict1(self, X):
            input_product = self.W_in@X
            state_product = self.W_res@self.state
            self.state = np.tanh(input_product + state_product)
            concat_matrix= np.concatenate((self.state, X))
            pred =  self.W_out@concat_matrix
            return pred









# Sinusoidal Curves - Seasonal stationary data.
def func1(x):
    return np.sin(x) + np.cos(2*x)


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


esn = ESN(reservoir_size=10, input_size=1, output_size=1, spectral_radius=0.8, connectivity_rate=0.8)

esn.fit1(X_train_std)

y_pred = []
for i in range(len(X_test_std)-1):
    pred = esn.predict1(X_test_std[i].reshape(-1,1))
    y_pred.append(pred)

predictions = np.array(y_pred).reshape(-1, 1)
test_values= X_test_std[1:]

print(np.sqrt(mean_squared_error(test_values, predictions)))

predictions_scaled = sc.inverse_transform(predictions)

print(np.sqrt(mean_squared_error(predictions_scaled, X_test[1:])))