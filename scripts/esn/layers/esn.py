import torch
import torch.nn as nn
import numpy as np


class ESN(nn.Module):
    def __init__(self, reservoir_size=100, input_size=1, spectral_radius=1.0, connectivity_rate=1.0, activation=nn.Tanh()):
        super(ESN, self).__init__()

        nn.Parameter.default_dtype = torch.float32

        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.activation = activation

        self.state = nn.Parameter(torch.zeros(self.reservoir_size, 1), requires_grad=False)
        self.W_in = nn.Parameter(torch.rand((reservoir_size, input_size)) * 2 - 1, requires_grad=False)
        
        # self.W_res = self.simple_reservoir_matrix(self.reservoir_size, self.connectivity_rate, self.spectral_radius)
        self.W_res = self.cosine_transform_matrix(self.reservoir_size, connectivity_rate)
        


        self.all_states = [self.state]


    def forward(self, input):
        input_product = self.W_in@input
        state_product = self.W_res@self.state
        self.state.data = self.activation(input_product + state_product)

        self.all_states.append(self.state.data)
        return torch.cat((self.state.data, input))
    

    def simple_reservoir_matrix(self, size, connectivity_rate, spectral_radius):
        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        W_res = nn.Parameter(torch.rand((size, size)), requires_grad=False)
        W_res.data[torch.rand(*W_res.data.shape) > connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res.data)))
        W_res.data = W_res.data * (spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = size * size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = W_res.data.flatten()
        W_flat[negative_indices] *= -1
        W_res.data = W_flat.reshape(*W_res.shape)

        return W_res
    

    def cosine_transform_matrix(self, size, connectivity_rate):
        W_res = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        for k in range(size):
            for n in range(size):
                W_res.data[k, n] = torch.sqrt(2.0 / torch.tensor(size)) * torch.cos(torch.pi * (2 * n + 1) * k / (2 * torch.tensor(size)))
        W_res.data[0, :] *= 1 / torch.sqrt(torch.tensor(2.0))
        W_res.data[torch.rand(*W_res.data.shape) > connectivity_rate] = 0
        return W_res

