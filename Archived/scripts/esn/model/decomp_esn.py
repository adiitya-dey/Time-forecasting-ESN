import torch
import torch.nn as nn
from ..layers.esn import ESN
from ..layers.decompose import series_decomp



class DecompESN(nn.Module):
    def __init__(self, reservoir_size=100, input_size=1, output_size=1, spectral_radius=1.0, connectivity_rate=1.0, washout=1, activation="tanh"):
        super(DecompESN, self).__init__()

        self.washout = washout


        self.esn = ESN(reservoir_size=reservoir_size, input_size=input_size, spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, activation=activation)
        self.state_collection_matrix = torch.zeros(input_size + reservoir_size, 1)
        # self.W_out = None
        self.W_out = nn.Parameter(torch.empty(output_size, reservoir_size+input_size), requires_grad=True)



    def train(self, X, y):
        for i in range(X.shape[0]):
            out = self.esn(X[i])
            self.state_collection_matrix =  torch.hstack((self.state_collection_matrix, out))

        
        A_matrix = self.state_collection_matrix.T[1 + self.washout:, :]
        B_matrix = y[self.washout:, :]
        # print(A_matrix.shape, B_matrix.shape)
        self.W_out.data = torch.linalg.lstsq(A_matrix,B_matrix).solution.T
        # print(self.W_out.shape)
        

    def predict(self, X):
        predictions = []

        for i in range(X.shape[0]):
            out = self.esn(X[i])
            y_pred = self.W_out@out
            predictions.append(y_pred)


        return torch.stack(predictions, dim=0)