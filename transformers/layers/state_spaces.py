import torch
import torch.nn as nn
import numpy as np

import logging

class StateSpace(nn.Module):

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 spectral_radius=1.0,
                 *args, **kwargs) -> None:
        super(StateSpace, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.spectral_radius = spectral_radius

        # self.last_input = None
        # self.last_state = None


        # ## Initialize initial state as a Zero Vector.
        # self.state = torch.zeros(self.reservoir_size, 1)

        ## Initialize Input Weights as randomly distributed [-1,1].
        w = torch.empty(self.reservoir_size, self.input_size)
        B = nn.init.uniform_(w, a=-1.0, b=1.0)

        ##Initialize Reservoir Weight matrix.
        # self.W_res = self.create_reservoir_weights(self.reservoir_size,
        #                                            self.connectivity_rate,
        #                                            self.spectral_radius)

        

        ## The idea is from S4 model where A = PDP^-1
        A = self.create_uniform_reservoir_weights(self.reservoir_size,
                                                  self.spectral_radius)
        
        eigens = torch.linalg.eig(A)

        self.W_res = torch.diag(eigens.eigenvalues)
        self.W_in = torch.matmul(torch.linalg.inv(eigens.eigenvectors), B)
        


        ## To capture all states for abalation study.
        # self.plot_states = [self.state]

        # self.all_states = self.state.clone().detach().squeeze(-1)   
        # 

    @staticmethod
    def create_uniform_reservoir_weights(size, spectral_radius):
        w = torch.empty(size, size)
        W_res = nn.init.uniform_(w, a=-0.05, b =0.05)
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
        W_res = W_res * (spectral_radius / current_spectral_radius) 
        return W_res

    # @staticmethod
    # def create_reservoir_weights(size, connectivity_rate, spectral_radius):
    #     ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
    #     ##
    #     ## Initialize a random matrix and induce sparsity.
        
    #     # w = torch.empty(size,size)
    #     # W_res = nn.init.eye_(w)

    #     W_res = torch.rand((size, size))
    #     W_res[torch.rand(*W_res.shape) > connectivity_rate] = 0

    #     ## Scale the matrix based on user defined spectral radius.
    #     current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
    #     W_res = W_res * (spectral_radius / current_spectral_radius)

    #     ## Induce half of the weights as negative weights.
    #     total_entries = size * size
    #     num_negative_entries = total_entries//2
    #     negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
    #     W_flat = W_res.flatten()
    #     W_flat[negative_indices] *= -1
    #     W_res = W_flat.reshape(*W_res.shape)

    #     return W_res.to_sparse()

    ## Calculate states for all inputs.
    def update_state(self, X):

        size = X.shape[0]
        all_f_states = torch.zeros(self.reservoir_size, 1)
        all_b_states = torch.zeros(self.reservoir_size, 1)
        for i in range(size):
            # Calculate forward states.
            f_state = self.get_state(X[i], all_f_states[i])
            all_f_states = torch.hstack([all_f_states, f_state.clone.detach()])

            if self.training:
            # Calculate backward states.
                b_state = self.get_state(X[size - i], all_b_states[i])
                all_b_states = torch.hstack([all_b_states, b_state.clone().detach()])




        all_f_states = all_f_states[1:].T

        if self.training:
            all_b_states = all_b_states[1:].T
            all_b_states = torch.flip(all_b_states, [0])

            all_states = torch.max(all_f_states, all_b_states)
            states = all_states.clone().detach()
            
        else:
            states = all_f_states.clone().detach()


        return states
    



    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = self.last_state
        # self.all_states = self.last_state

                    
    ## Calculate state.
    def get_state(self, input, state):
        return self.activation(self.W_in@input + self.W_res@state)
        
    # ## Collect last values of state, input and output.
    # def collect(self, input):
    #     self.last_state = self.state.clone().detach()
    #     self.last_input = input.clone().detach()

    

    def forward(self, x):
 
        out = self.update_state(x)
        self.collect(x[-1])
        self.reset_states()
        return out[1:]