import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


torch.set_default_dtype(torch.float64)

class ESN(nn.Module):

    def __init__(self, reservoir_size=100, input_size=1, output_size=1,  spectral_radius=1.0, connectivity_rate=1.0, learning_rate = 0.1, epochs=1, washout=1, activation="tanh"):
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.epochs = epochs
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.washout = washout
        self.output_size = output_size
        self.lr = learning_rate
        self.activation = self.activation_fn(activation)
        

        self.state = torch.zeros(self.reservoir_size, 1)
        self.W_in = torch.rand((reservoir_size, input_size)) * 2 - 1
        
        
        self.W_out = None

        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        self.W_res = torch.rand((reservoir_size, reservoir_size))
        self.W_res.data[torch.rand(*self.W_res.shape) > self.connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(self.W_res.data)))
        self.W_res.data = self.W_res * (self.spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = self.reservoir_size * self.reservoir_size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = self.W_res.data.flatten()
        W_flat[negative_indices] *= -1
        self.W_res = W_flat.reshape(*self.W_res.shape)


        self.all_states = [self.state]

    @staticmethod
    def activation_fn(x):
         
        activation_keys = ["sigmoid", "relu", "tanh"]

        if x in activation_keys:
              if x == "tanh":
                   return nn.Tanh()
              elif x == "relu":
                   return nn.ReLU()
              elif x == "sigmoid":
                   return nn.Sigmoid()
            
        else:
            raise ValueError(f"Activation {x} does not exists")
    



    def fit(self, X_train, y_train):
        

        state_collection_matrix = torch.zeros((self.input_size + self.reservoir_size, 1), dtype=torch.FloatTensor)
        # self.state = np.zeros((self.reservoir_size, 1))

        ## Calculate state of reservoirs per time step
        for i in range(X_train.shape[0]-1):

            

            input = X_train[i].reshape(-1,1)
            input_product = self.W_in@input
            state_product = self.W_res@self.state
            self.state = self.activation(input_product + state_product)
            state_collection_matrix= torch.hstack((state_collection_matrix, torch.concatenate((self.state, input))))

            self.all_states.append(self.state)

        ## Update W_out
        mat1 = state_collection_matrix.T[self.washout:,:]
        self.W_out = torch.matmul(torch.linalg.inv(torch.matmul(mat1, mat1.T) + self.lr * torch.eye(self.reservoir_size), torch.matmul(mat1, y_train[self.washout:,:])))

    
    def predict(self, X_test):
            prediction = np.zeros((self.output_size,1))
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
            if self.output_size == self.input_size:
                return prediction[1:,:]
            else:
                return prediction
            