import torch
import torch.nn as nn

from einops import rearrange

import logging

from src2.layers.state_spaces import StateSpace
from src2.layers.positional_encoding import PositionalEncoding

class AttentiveESN(nn.Module):

    def __init__(self, 
                 input_size: int = 1,
                 output_size: int = 1,
                 reservoir_size: int = 50,
                 activation = nn.Tanh(),
                 connectivity_rate: float = 1.0,
                 spectral_radius: float = 1.0,
                 max_len : int = 10000,
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
        
        self.pos_encoder = PositionalEncoding(d_model=self.reservoir_size,
                                              max_len=self.max_len)
        
        self.W_q = nn.Parameter(torch.rand(self.reservoir_size, self.reservoir_size),
                                requires_grad=True)
        
        self.W_k = nn.Parameter(torch.rand(self.reservoir_size, self.reservoir_size),
                                requires_grad=True)
        
        self.W_v = nn.Parameter(torch.rand(self.reservoir_size, self.reservoir_size),
                                requires_grad=True)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.reservoir_size,
                                                         num_heads=5,
                                                         batch_first=True)



        # self.layernorm = nn.LayerNorm()
        
    
    def forward(self, x):
        states = self.state_space(x)
        

        states = rearrange(states, ' b s -> b 1 s')
        logging.warning(f"Shape of states : {states.shape}")

        pos_states = self.pos_encoder(states)
        logging.warning(f"Shape of Pos_states: {pos_states.shape}")

        query = pos_states@self.W_q
        key = pos_states@self.W_k
        value = pos_states@self.W_v
        logging.warning(f"Query: {query.shape}, Value: {value.shape}, key: {key.shape}")


        attn_output = self.multihead_attention(query, key, value, need_weights=False)
        logging.warning(f"Shape of Attention: {len(attn_output)}, ")

 
        
        return pos_states

