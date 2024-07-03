# Configure logging to write to a file
import logging


import torch
import torch.nn as nn
import numpy as np



class DRNN(nn.Module):

    def __init__(self, 
                 input_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 spectral_radius=1.0,
                 *args, **kwargs) -> None:
        super(DRNN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.spectral_radius = spectral_radius

        self.last_input = None
        self.last_state = None


        ## Initialize initial state as a Zero Vector.
        self.state = torch.zeros(self.reservoir_size, 1)

        ## Intialize Input weight matrix as Uniform.
        w_b = torch.empty(self.reservoir_size, self.input_size)
        self.W_in = nn.init.uniform_(w_b, a=-1.0, b=1.0)


        ##Initialize Reservoir Weight matrix as diagonal matrix.
        w_d =  torch.empty(self.reservoir_size)
        d = nn.init.uniform_(w_d, a=-self.spectral_radius, b=self.spectral_radius)
        self.W_res = torch.diag(d)

        ## To capture all states for abalation study.
        self.plot_states = self.state.clone().detach().squeeze(1)

        # self.all_states = self.state.clone().detach().squeeze(-1)      
          

    


    ## Calculate states for all inputs.
    def update_state(self, X):
        all_states = torch.empty(X.shape[0], self.reservoir_size)
        for i in range(X.shape[0]):
            self.state = self.get_state(X[i])
            all_states[i, :] = self.state.detach().clone().squeeze(1)
            self.plot_states = torch.vstack([self.plot_states, self.state.detach().clone().squeeze(1)])
        return all_states


    ## Reset all states for next batch operation.
    def reset_states(self):
         self.state = torch.zeros(self.reservoir_size, 1)
                    
    ## Calculate state.
    def get_state(self, input):
        return self.activation(self.W_in@input + self.W_res@self.state)  


       
    def forward(self, x):
       states = self.update_state(x)
       return torch.unsqueeze(states, -1)