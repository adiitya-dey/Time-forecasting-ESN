from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from ..layers.Embed import PositionalEmbedding

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
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.activation = activation


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


    ## Calculate states for all inputs.
    def update_state(self, X):
        batch, segments, windown_len  = X.shape
        all_states = torch.empty(batch, segments, self.reservoir_size,)

        for b in range(batch):
            for t in range(segments):
                self.state = self.get_state(X[b, t, :])
                all_states[b,t, :] = self.state.clone().detach().squeeze(1)
        return all_states

        # for i in range(X.shape[0]):
        #     self.state = self.get_state(X[i])
        #     all_states = torch.vstack([all_states, self.state.clone().detach().squeeze(-1)])
        # return all_states


    ## Reset all states for next batch operation.
    def reset_states(self):
        self.state = torch.zeros(self.reservoir_size, 1)

                    
    ## Calculate state.
    def get_state(self, input):
        input = torch.unsqueeze(input, 1)
        return self.activation(self.W_in@input + self.W_res@self.state)

    def forward(self, X):
        states = self.update_state(X)
        return states



class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_len = 48
        self.reservoir_size = 50

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in


        if self.individual:
            self.input_linear_layer = nn.ModuleList()
            self.esn_layer = nn.ModuleList()
            self.output_linear_layer = nn.ModuleList()
            for i in range(self.channels):
                self.input_linear_layer.append(nn.Linear(in_features=self.window_len,
                                                   out_features=self.window_len, 
                                                   bias=False))
                self.esn_layer.append(ESN(reservoir_size=self.reservoir_size,
                                             activation=nn.LeakyReLU(1.0),
                                             input_size=self.window_len))
                self.output_linear_layer.append(in_features=self.window_len,
                                                 out_features=self.window_len, 
                                                 bias=False)
        else:
            self.input_linear_layer = nn.Linear(in_features=self.window_len,
                                                   out_features=self.window_len, 
                                                   bias=False)
            self.esn_layer = ESN(reservoir_size=self.reservoir_size,
                                             activation=nn.LeakyReLU(1.0),
                                             input_size=self.window_len)
            self.output_linear_layer = nn.Linear(in_features=self.reservoir_size,
                                                 out_features=self.pred_len,
                                                 bias=False)


       
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch, seq_len, channel = x.shape
        
        # x: [Batch, Channel, Input_length]
        x = x.permute(0,2,1)

        # Calculate total input segments.
        segments = seq_len // self.window_len

        # x: [Batch, Channel, Segments, window_length]
        x = x.view(batch, channel, segments, self.window_len)

        # Calculate prediction segments.
        m = self.pred_len // self.window_len

        if self.individual:
            for i in range(channel):
                x = self.input_linear_layer[i](x[:,i,:,:])
                states = self.esn_layer[i](x)
        else:
            x = x.squeeze(1)
            x = self.input_linear_layer(x)

            # Caapture last of every segment inside the batch.
            states = self.esn_layer(x)[:,-1, :]
            x = self.output_linear_layer(states)
            x = torch.unsqueeze(x, 1)

        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    def reset(self):
        self.esn_layer.reset_states()