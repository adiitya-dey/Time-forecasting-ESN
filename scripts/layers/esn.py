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
        # w = torch.empty(size,size)
        self.W_in = torch.rand((self.reservoir_size, self.input_size)) * 2 -1

        ##Initialize Reservoir Weight matrix.
        self.W_res = self.create_reservoir_weights(self.reservoir_size,
                                                   self.connectivity_rate,
                                                   self.spectral_radius)
        
        self.W_out = None
                             
        ## To capture all states for abalation study.
        self.plot_states = [self.state]

        self.all_states = [self.state]                                    
        
        ## Initialize Feedback weights as gaussian randomly distributed.
        if self.teacher_forcing:
            self.W_fb = torch.rand((self.reservoir_size, self.output_size))
          


    @staticmethod
    def create_reservoir_weights(size, connectivity_rate, spectral_radius):
        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        
        # w = torch.empty(size,size)
        # W_res = nn.init.orthogonal_(w)
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


    ## Calculate states for all inputs.
    def update_state(self, X, y=None):

        # Change dimension of y from (B, D) to (B, D, 1)
        y = y.unsqueeze(2)

        # Add initial y value as 0.
        y = torch.vstack([torch.zeros(1, y.shape[1], y.shape[2]), y])
        # print(y.shape)
        for i in range(X.shape[0]):

            if self.teacher_forcing:
                self.state = self.get_state(X[i], y[i])
                
            else:
                self.state = self.get_state(X[i])
            
            self.all_states.append(self.state)

    ## Calculate Output Weights using ordinary least squares.
    def calc_output_weight(self, y):
        A_matrix = torch.hstack(self.all_states[1:]).T
        B_matrix = y
        # print(A_matrix.shape, B_matrix.shape)
        self.W_out = torch.linalg.lstsq(A_matrix, B_matrix).solution.T


    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = torch.zeros(self.reservoir_size, 1)
        self.all_states = [self.state]  
                    
    ## Calculate state.
    def get_state(self, input, output=None):
        if self.teacher_forcing:
            return self.activation(self.W_in@input + self.W_res@self.state + self.W_fb@output)
        else:
            return self.activation(self.W_in@input + self.W_res@self.state )
        
    ## Collect last values of state, input and output.
    def collect(self, input, output):
        self.last_state = self.state
        self.last_input = input
        self.last_output = output
    

    def forward(self, X, y=None):
        
        ## Perform the below steps only during training to update the output weights.
        if self.training:
            self.update_state(X, y)     # 1. Calculate all states.
            self.calc_output_weight(y)  # 2. Calculate W_out.
            self.reset_states()         # 3. Reset batch for next batch.
            self.collect(X[-1], y[-1].view(y.shape[1], 1))  # 4. Collect last values of last batch to predict future values.
        
        out = [self.last_output]
        

        for i in range(X.shape[0]):
            if self.teacher_forcing:
                
                self.state = self.get_state(X[i], out[i])

            else:
                self.state = self.get_state(X[i])

            self.plot_states.append(self.state)
            pred = torch.matmul(self.W_out, self.state)
            out.append(pred)

        self.state = self.last_state

        
        return torch.stack(out[1:], dim=0)

