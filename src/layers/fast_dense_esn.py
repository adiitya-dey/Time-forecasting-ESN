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
        w_p = torch.empty(self.reservoir_size, self.reservoir_size)
        P = nn.init.orthogonal_(w_p)

        w_b = torch.empty(self.reservoir_size, self.input_size)
        B = nn.init.uniform_(w_b, a=-1.0, b=1.0)

        self.W_in = torch.matmul(torch.linalg.inv(P), B)



        ##Initialize Reservoir Weight matrix.
        w_d =  torch.empty(self.reservoir_size)
        d = nn.init.uniform_(w_d, a=-0.5, b=0.5)
        self.W_res = torch.diag(d)
        
        
        self.dense1 = nn.Linear(in_features=self.reservoir_size,
                                out_features=self.output_size,
                                bias=False)

        ## To capture all states for abalation study.
        self.plot_states = [self.state]

        # self.all_states = self.state.clone().detach().squeeze(-1)      
          

    


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

