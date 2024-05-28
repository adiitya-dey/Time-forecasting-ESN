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
                 teacher_forcing=False,
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.teacher_forcing =teacher_forcing
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius

        self.last_input = None
        self.last_output= None
        self.last_state = None


        ## Initialize initial state as a Zero Vector.
        self.state = torch.zeros(self.reservoir_size, 1)

        ## Initialize Input Weights as randomly distributed [-1,1].
        self.W_in = torch.rand((self.reservoir_size, self.input_size)) * 2 -1

        ##Initialize Reservoir Weight matrix.
        self.W_res = self.create_reservoir_weights(self.reservoir_size,
                                                   self.connectivity_rate,
                                                   self.spectral_radius)
        
        self.W_out = None
                             
        ## To capture all states for abalation study.
        self.all_states = [self.state]                                    
        
        ## Initialize Feedback weights as gaussian randomly distributed.
        if self.teacher_forcing:
            self.W_fb = nn.Parameter(torch.rand((self.reservoir_size, self.output_size)), 
                                     requires_grad=False)
            
            self.W = torch.hstack([self.W_in, self.W_res, self.W_fb])
            

        else:

            self.W = torch.hstack([self.W_in, self.W_res])


    @staticmethod
    def create_reservoir_weights(size, connectivity_rate, spectral_radius):
        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.

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


    def update_state(self, X, y=None):
        for i in range(X.shape[0]):

            if self.teacher_forcing:
                self.state = self.get_state(X[i], y[i])
                
            else:
                self.state = self.get_state(X[i])
            
            self.all_states.append(self.state)


    def calc_output_weight(self, y):
        A_matrix = torch.hstack(self.all_states[1:]).T
        B_matrix = y
        self.W_out = torch.linalg.lstsq(A_matrix, B_matrix).solution.T

    def reset_states(self):
        self.state = torch.zeros(self.reservoir_size, 1)
        self.all_states = []
                    
    
    def get_state(self, input, output=None):
        if self.teacher_forcing:
            return self.activation(torch.matmul(self.W, torch.vstack([input, self.state])))
        else:
            return self.activation(torch.matmul(self.W, torch.vstack([input, self.state, output])))
        

    def collect(self, input, output):
        self.last_state = self.state
        self.last_input = input
        self.last_output = output
    
    def forward(self, X, y=None):
        

        if self.training:
            self.update_state(X, y)
            self.calc_output_weight(y)
            self.reset_states()
            self.collect(X[-1], y[-1])
        
        out = [self.last_output]
        

        for i in range(X.shape[0]):
            if self.teacher_forcing:
                
                self.state = self.get_state(X[i], out[i])

            else:
                self.state = self.get_state(X[i])

            pred = torch.matmul(self.W_out, self.state)
            out.append(pred)

        self.state = self.last_state

        
        return torch.stack(out[1:], dim=0)

