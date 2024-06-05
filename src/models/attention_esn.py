import torch
import torch.nn as nn

import logging

from src.layers.state_spaces import StateSpace
from src.layers.positional_encoding import StatePositionalEncoding

class AttentiveESN(nn.Module):

    def __init__(self, 
                 input_size: int = 1,
                 output_size: int = 1,
                 reservoir_size: int = 50,
                 activation = nn.Tanh(),
                 connectivity_rate: float = 1.0,
                 spectral_radius: float = 1.0,
                 max_len : int = 5000,
                 *args, **kwargs) -> None:
        super(AttentiveESN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.max_len = max_len

        self.state_space = StateSpace(input_size=self.input_size,
                                      output_size=self.output_size,
                                      reservoir_size=self.reservoir_size,
                                      activation=self.activation,
                                      connectivity_rate=self.connectivity_rate,
                                      spectral_radius=self.spectral_radius)
        
        self.pos_encoder = StatePositionalEncoding(d_model=self.reservoir_size, max_len=self.max_len)

        self.dense = nn.Linear(in_features=self.reservoir_size,
                               out_features=self.output_size, bias=False)
        
        self.D = nn.Parameter(torch.rand(self.output_size, self.input_size), requires_grad=True)
    
    def forward(self, x):
        states = self.state_space(x)
        logging.warning(f"Shape of states : {states.shape}")

        # states = torch.fft.fft(states)
    #    states = self.pos_encoder(states)
    #    logging.warning(f"Shape of states after Positional Encoding = {states.shape}")

        out = self.dense(states) + self.D@x
        
        return out

