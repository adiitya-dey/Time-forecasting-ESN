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
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius

        self.last_input = None
        self.last_output= None
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
        
        # w = torch.empty(self.output_size, self.reservoir_size)
        # self.W_out = nn.Parameter(w, requires_grad=True)
        self.W_out =None

        ## To capture all states for abalation study.
        self.plot_states = self.state.clone().detach().squeeze(1)

        # self.all_states = [self.state]          

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
    

    # ## Using dropouts to create zeros in matrix.
    # @staticmethod
    # def create_reservoir_weights2(size, connectivity_rate, spectral_radius):
    #     w = torch.empty(size, size)
    #     W_res = nn.init.uniform_(w, a=-1.0, b=1.0)
    #     dropout = nn.Dropout(p=connectivity_rate)
    #     W_res = dropout(W_res) / (1 - connectivity_rate) # Since dropout scales other values by 1/(1-p).
    #     current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
    #     W_res = W_res * (spectral_radius / current_spectral_radius)

    #     return W_res


    ## Calculate states for all inputs.
    def update_state(self, X):
        all_states = torch.empty(X.shape[0], self.reservoir_size)
        for i in range(X.shape[0]):
            self.state = self.get_state(X[i])
            all_states[i, :] = self.state.detach().clone().squeeze(1)
            self.plot_states = torch.vstack([self.plot_states, self.state.detach().clone().squeeze(1)])
        return all_states

    ## Calculate Output Weights using ordinary least squares.
    def calc_output_weight(self, states, y):
        A_matrix = states
        B_matrix = y
        # print(A_matrix.shape, B_matrix.shape)
        self.W_out = torch.linalg.lstsq(A_matrix, B_matrix).solution.T


    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = torch.zeros(self.reservoir_size, 1)
                    
    ## Calculate state.
    def get_state(self, input):
        return self.activation(self.W_in@input + self.W_res@self.state)
    

    def fit(self, X, y):
            states = self.update_state(X)     # 1. Calculate all states.
            self.calc_output_weight(states, y)  # 2. Calculate W_out.
        

        
    def predict(self, X):
        out = torch.empty(X.shape)
        for i in range(X.shape[0]):
            self.state = self.get_state(X[i])
            # self.plot_states.append(self.state)
            out[i] = self.W_out@self.state
           

        
        return out.squeeze(1)

