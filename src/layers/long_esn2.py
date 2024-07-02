# Configure logging to write to a file
import logging


import torch
import torch.nn as nn
import numpy as np



class ESN(nn.Module):

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 connectivity_rate=1.0,
                 spectral_radius=1.0,
                 window_len = 24,
                 pred_len = 24,
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.window_len = window_len
        self.pred_len = pred_len

        self.last_input = None
        self.last_state = None


        ## Initialize initial state as a Zero Vector.
        self.state = torch.zeros(self.reservoir_size, 1)

        ## Initialize Input Weights as randomly distributed [-1,1].
        # w_p = torch.empty(self.reservoir_size, self.reservoir_size)
        # P = nn.init.orthogonal_(w_p)

        w_b = torch.empty(self.reservoir_size, self.input_size)
        B = nn.init.uniform_(w_b, a=-1.0, b=1.0)

        self.W_in = B
        # self.W_in = torch.matmul(torch.linalg.inv(P), B)



        ##Initialize Reservoir Weight matrix.
        w_d =  torch.empty(self.reservoir_size)
        d = nn.init.uniform_(w_d, a=-self.spectral_radius, b=self.spectral_radius)
        self.W_res = torch.diag(d)
        
        
        # self.input_linear_layer = nn.Linear(in_features=self.window_len,
        #                                            out_features=self.window_len, 
        #                                            bias=False)
        # self.input_linear_layer.weight.requires_grad_ = False
        
        self.output_linear_layer = nn.Linear(in_features=self.reservoir_size,
                                                 out_features=self.pred_len,
                                                 bias=False)

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
    def get_state(self, input, output=None):
        return self.activation(self.W_in@input + self.W_res@self.state)  

    


       
    def forward(self, x):
        # x: [Batch, Input length]
        seq_len = x.shape[0]
        
        # x: [Batch, Channel, Input_length]
        # x = x.permute(0,2,1)

        # Calculate total input segments.
        segments = seq_len // self.window_len

        # x: [Batch, Channel, Segments, window_length]
        x = x.view(segments, self.window_len)

        # Calculate prediction segments.
        m = self.pred_len // self.window_len

        # x = torch.unsqueeze(x, -1)
        # x = self.input_linear_layer(x)
        x = torch.unsqueeze(x, -1)

        # Caapture last of every segment inside the batch.
        states = self.update_state(x)[-1, :]
        x = self.output_linear_layer(states)
        x = torch.unsqueeze(x, 1)

        return x
        # return x.permute(0,2,1) # to [Batch, Output length, Channel]
