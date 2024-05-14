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
        

        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        self.W_res = nn.Parameter(torch.rand((reservoir_size, reservoir_size)), requires_grad=False)
        self.W_res.data[torch.rand(*self.W_res.data.shape) > self.connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(self.W_res.data)))
        self.W_res.data = self.W_res.data * (self.spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = self.reservoir_size * self.reservoir_size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = self.W_res.data.flatten()
        W_flat[negative_indices] *= -1
        self.W_res.data = W_flat.reshape(*self.W_res.shape)


        self.all_states = [self.state]


#     @staticmethod
#     def activation_fn(x):
         
#         activation_keys = ["sigmoid", "relu", "tanh", "leakyrelu", "selu", "relu", "softplus", "celu"]

#         if x in activation_keys:
#               if x.lower() == "tanh":
#                    return nn.Tanh()
#               elif x.lower() == "relu":
#                    return nn.ReLU()
#               elif x.lower() == "sigmoid":
#                    return nn.Sigmoid()
#               elif x.lower() == "leakyrelu":
#                    return nn.LeakyReLU()
#               elif x.lower() == "selu":
#                    return nn.SELU()
#               elif x.lower() == "celu":
#                    return nn.CELU()
#               elif x.lower() == "softplus":
#                    return nn.Softplus()
            
#         else:
#             raise ValueError(f"Activation {x} does not exists")
        



    def forward(self, input):
        input_product = self.W_in@input
        state_product = self.W_res@self.state
        self.state.data = self.activation(input_product + state_product)

        self.all_states.append(self.state.data)
        return torch.cat((self.state.data, input))
