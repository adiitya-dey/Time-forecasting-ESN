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

        # ## Initializing Reservoir Weights according to origianl paper(2001).
        # ##
        # ## Initialize a random matrix and induce sparsity 
        # self.W_res = np.random.randn(reservoir_size, reservoir_size)
        # self.W_res[np.random.rand(*self.W_res.shape) > self.connectivity_rate] = 0

        # ##  Scale the matrix based on user defined spectral radius.
        # current_spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        # self.W_res = self.W_res * (self.spectral_radius / current_spectral_radius)


        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        self.W_res = np.random.rand(reservoir_size, reservoir_size)
        self.W_res[np.random.rand(*self.W_res.shape) > self.connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res = self.W_res * (self.spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = self.W_res.size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = self.W_res.flatten()
        W_flat[negative_indices] *= -1
        self.W_res = W_flat.reshape(self.W_res.shape)


        self.all_states = [self.state]

    @staticmethod
    def activation(x):
         
        ## Hyperbolic Tangent Function
         return np.tanh(x)

        # ## ReLU Fuction
        # return np.clip(x, 0, np.inf)
    
        # ## Sigmoid Function
        # return 1 / (1 + np.exp(-x))


    def fit(self, X_train, y_train=None):
        
        ## Epochs are useless. Keep it as 1 always.
        for _ in range(self.epochs): 

            ## Reset states.
            state_collection_matrix = np.zeros((self.input_size + self.reservoir_size, 1))
            # self.state = np.zeros((self.reservoir_size, 1))

            ## Calculate state of reservoirs per time step
            for i in range(X_train.shape[0]-1):

               

                input = X_train[i].reshape(-1,1)
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = self.activation(input_product + state_product)
                state_collection_matrix= np.hstack((state_collection_matrix, np.concatenate((self.state, input))))

                self.all_states.append(self.state)

            ## Update W_out
            mat1 = state_collection_matrix.T[self.washout:,:]
            ridge_regressor= Ridge(alpha=self.lr)
            ridge_regressor.fit(mat1, y_train[self.washout:,:])
            self.W_out = ridge_regressor.coef_
            # self.W_out = np.dot(np.linalg.pinv(mat1), y_train)


    # def predict(self, X_test):
    #         input_product = self.W_in@X_test
    #         state_product = self.W_res@self.state
    #         self.state = np.tanh(input_product + state_product)
    #         concat_matrix= np.concatenate((self.state, X_test))
    #         pred =  self.W_out@concat_matrix
    #         return pred
    
    def predict(self, X_test):
            prediction = np.zeros((1,1))
            for i in range(X_test.shape[0]- 1):
                input = X_test[i].reshape(-1,1)
                input_product = self.W_in@input
                state_product = self.W_res@self.state
                self.state = self.activation(input_product + state_product)
                concat_matrix= np.concatenate((self.state, input))
                pred =  self.W_out@concat_matrix
                prediction = np.hstack([prediction, pred])

                self.all_states.append(self.state)
            
            prediction = rearrange(prediction, 'c r -> r c')
            return prediction[1:,:]
    
