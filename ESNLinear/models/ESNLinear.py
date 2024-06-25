from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ESN(nn.Module):

    def __init__(self, 
                 input_size: int,
                #  output_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 connectivity_rate=1.0,
                 spectral_radius=1.0,
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius

        # self.last_input = None
        

        # self.batch_norm = nn.BatchNorm1d(self.reservoir_size, affine=False)

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
        d = nn.init.uniform_(w_d, a=-0.95, b=0.95)
        self.W_res = torch.diag(d)


        # ## Initialize Input Weights as randomly distributed [-1,1].
        # w = torch.empty(self.reservoir_size, self.input_size)
        # self.W_in = nn.init.uniform_(w, a=-1.0, b=1.0)

        # ##Initialize Reservoir Weight matrix.
        # self.W_res = self.create_reservoir_weights(self.reservoir_size,
        #                                            self.connectivity_rate,
        #                                            self.spectral_radius)
          

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
        segments, seg_len  = X.shape
        all_states = torch.empty(self.reservoir_size, segments)

        for t in range(segments):
            self.state = self.get_state(X[t, :])
            all_states[:, t] = self.state.clone().detach().view(-1)
        return all_states

        # for i in range(X.shape[0]):
        #     self.state = self.get_state(X[i])
        #     all_states = torch.vstack([all_states, self.state.clone().detach().squeeze(-1)])
        # return all_states


    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = torch.zeros(self.reservoir_size, 1)
        # self.all_states = self.last_state
                    
    ## Calculate state.
    def get_state(self, input):
        input = torch.unsqueeze(input, 1)
        return self.activation(self.W_in@input + self.W_res@self.state)
        
    ## Collect last values of state, input and output.
    # def collect(self, input):
    #     self.last_state = self.state.clone().detach()
    #     self.last_input = input.clone().detach()
    

    def forward(self, X):
        states = self.update_state(X)
        return states[:,-1]



class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.linear_projection = nn.Linear(in_features=48, out_features=24, bias=False)

    
        self.linear_projection.requires_grad_ = False

        self.esn = ESN(reservoir_size=100,
                             activation= nn.SELU(),
                             spectral_radius=0.95,
                             connectivity_rate=0.7,
                             input_size=24)


        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len, bias=False)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len, bias=False)
            self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        X = x.flatten()


        # seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        # seasonal_init = seasonal_init.permute(1, 0, 2).contiguous().view(seasonal_init.shape[0], -1)
        # trend_init = trend_init.permute(1, 0, 2).contiguous().view(trend_init.shape[0], -1)

        # Breaking into segments
        total_segments = X.shape[0] // 48
        X_w = X.view(total_segments, 48)

        X_d = self.linear_projection(X_w)

        state = self.esn(X_d)

        # seasonal_init = seasonal_init.view(total_segments, 48)
        # trend_init = trend_init.view(total_segments, 48)

        # # Passing through a linear projection
        # seasonal_init = self.seasonal_linear_projection(seasonal_init)
        # trend_init = self.trend_linear_projection(trend_init)


        # if self.individual:
        #     seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
        #     trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
        #     for i in range(self.channels):
        #         seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
        #         trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        # else:
        #     seasonal_states = self.seasonal_esn(seasonal_init)
        #     trend_states = self.trend_esn(trend_init)
        #     seasonal_output = self.Linear_Seasonal(seasonal_states)
        #     trend_output = self.Linear_Trend(trend_states)

        # x = seasonal_output
        # x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    def reset(self):
        self.esn.reset_states()