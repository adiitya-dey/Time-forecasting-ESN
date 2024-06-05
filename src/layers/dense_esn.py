# Configure logging to write to a file
import logging


import torch
import torch.nn as nn
import numpy as np



class DenseESN(nn.Module):

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 connectivity_rate=1.0,
                 spectral_radius=1.0,
                 *args, **kwargs) -> None:
        super(DenseESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius

        self.last_input = None
        self.last_state = None


        ## Initialize initial state as a Zero Vector.
        self.state = torch.zeros(self.reservoir_size, 1)

        ## Initialize Input Weights as randomly distributed [-1,1].
        w = torch.empty(self.reservoir_size, self.input_size)
        self.W_in = nn.init.uniform_(w, a=-1.0, b=1.0)

        ##Initialize Reservoir Weight matrix.
        self.W_res = self.create_reservoir_weights(self.reservoir_size,
                                                   self.connectivity_rate,
                                                   self.spectral_radius)
        logging.info(f"W_res is initalized: {self.W_res}")
        
        
        self.dense1 = nn.Linear(in_features=self.reservoir_size,
                                out_features=self.output_size,
                                bias=False)

        ## To capture all states for abalation study.
        self.plot_states = [self.state]

        # self.all_states = self.state.clone().detach().squeeze(-1)      
          

    @staticmethod
    def create_reservoir_weights(size, connectivity_rate, spectral_radius):
        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        
        # w = torch.empty(size,size)
        # W_res = nn.init.eye_(w)

        W_res = torch.rand((size, size))
        W_res[torch.rand(*W_res.shape) > connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
        W_res = W_res * (spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = size * size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = W_res.flatten()
        W_flat[negative_indices] *= -1
        W_res = W_flat.reshape(*W_res.shape)

        return W_res.to_sparse()
    

    ## Using dropouts to create zeros in matrix.
    @staticmethod
    def create_reservoir_weights2(size, connectivity_rate, spectral_radius):
        w = torch.empty(size, size)
        W_res = nn.init.uniform_(w, a=-1.0, b=1.0)
        dropout = nn.Dropout(p=connectivity_rate)
        W_res = dropout(W_res) / (1 - connectivity_rate) # Since dropout scales other values by 1/(1-p).
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
        W_res = W_res * (spectral_radius / current_spectral_radius)

        return W_res


    ## Calculate states for all inputs.
    def update_state(self, X):
        all_states = torch.empty(self.reservoir_size)
        for i in range(X.shape[0]):
            self.state = self.get_state(X[i])
            logging.info(f"State for {i}th input is during training is: {self.state}")
            all_states = torch.vstack([all_states, self.state.clone().detach().squeeze(-1)])
        return all_states


    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = self.last_state
        # self.all_states = self.last_state
        logging.warning(f"Resetting all states to zero.")
                    
    ## Calculate state.
    def get_state(self, input, output=None):
        return self.activation(self.W_in@input + self.W_res@self.state)
        
    ## Collect last values of state, input and output.
    def collect(self, input):
        self.last_state = self.state.clone().detach()
        self.last_input = input.clone().detach()
        logging.info(f"Collected last values. last state: {self.last_state}, last_input: {self.last_input}")
    

    def forward(self, X):
        
       
        states = self.update_state(X)
        
        logging.warning(f"shape of all_states = {states.shape}")

        out = self.dense1(states[1:]) #All state excepts first state which is at zero.
        
        if self.training:
            self.collect(X[-1])
            # self.all_states = self.all_states[-1]
        else:

            self.reset_states()
        
        return out

