import torch
import torch.nn as nn
from ..layers.esn import ESN
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

class DenseESN(nn.Module):
    def __init__(self, epochs= 5, batch_size=10, reservoir_size=100, input_size=1, output_size=1, spectral_radius=1.0, connectivity_rate=1.0, washout=1, activation=nn.Tanh()):
        super(DenseESN, self).__init__()

        self.washout = washout
        self.epochs = epochs
        self.batch_size = batch_size

        self.state_collection_matrix = torch.zeros(input_size + reservoir_size, 1)
        # self.W_out = nn.Parameter(torch.empty(output_size, reservoir_size+input_size), requires_grad=True)

        self.esn = ESN(reservoir_size=reservoir_size, input_size=input_size, spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, activation=activation)
        self.dense1 = nn.Linear(in_features=reservoir_size+input_size, out_features=output_size)

    
    def reset_collection_matrix(self):
        self.state_collection_matrix = self.state_collection_matrix[:,[0]]


    def train(self, X, y):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.dense1.parameters(), lr=0.01)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                for i in range(batch_X.shape[0]):
                    out = self.esn(batch_X[i])
                    self.state_collection_matrix =  torch.hstack((self.state_collection_matrix, out))

            
                X_matrix = self.state_collection_matrix.T[1 + self.washout:, :]
                # X_matrix = rearrange(X_matrix, 'b c -> b 1 c')
                y_matrix = batch_y[self.washout:, :]
                # y_matrix = rearrange(y_matrix, 'b c -> b 1 c')

        
                y_pred = self.dense1(X_matrix)

                loss = loss_fn(y_pred, y_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(f'Finished epoch {epoch}, latest loss {loss}')

                self.reset_collection_matrix()


    def predict(self, X):
        predictions = []

        for i in range(X.shape[0]):
            out = self.esn(X[i])
            y_pred = self.dense1(out.T)
           
            predictions.append(y_pred)
        output = torch.stack(predictions, dim=0)

        return output.permute(0,2,1)