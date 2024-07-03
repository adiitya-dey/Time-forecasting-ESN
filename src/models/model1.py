import torch
import torch.nn as nn

from src.layers.drnn import DRNN
from src.layers.embedding import PositionalEmbedding


class Model(nn.Module):

    def __init__(self, input_size: int, output_size: int, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)


        self.window_len = input_size
        self.output_size = output_size

        self.batch_norm = nn.BatchNorm1d(num_features=2)

        self.drnn1 = DRNN(input_size=2, 
                    spectral_radius=0.5,
                    reservoir_size=50,
                    activation=nn.LeakyReLU(1.0))
        
        self.drnn2 = DRNN(input_size=50,
                    reservoir_size=50,
                    spectral_radius=0.5,
                    activation=nn.LeakyReLU(1.0))
        
        self.pos_embed = PositionalEmbedding(d_model=50)

        self.linear = nn.Linear(in_features=50,
                                out_features=2)
        


    def forward(self, x):
        ## x: 
        seq_len = x.shape[0]

        # # ## Calculate total input segments.
        input_segments = seq_len // self.window_len

        # # ## x: [Segments, Window Length]
        x = x.view(input_segments, self.window_len)


        ## Frequency conversion to amplitude and phase.
        ## x: [ sequence, amplitude, phase]
        # x = self.calc_amp_phase(x.squeeze(1))

        ## x : [Segments, Window Length, 1]
        x = torch.unsqueeze(x, -1)

        # x = self.batch_norm(x)
        ## states : [segments, reservoir size]
        states = self.drnn1(x)
        
        
        ## states: [segments, reservoir_size, 1]
        states = torch.unsqueeze(states, -1)

        ## states: [reservoir size]
        ## Only Last state is captured.
        states = self.drnn2(states)[-1, :]

        ## Copy states according to the output size.
        states = torch.tile(states, (self.output_size, 1))

        ## Add position embedding to the states.
        states = self.pos_embed(states)

        ## Trainable Linear Layer.
        x = self.linear(states)

        # x = self.inv_calc_amp_phase(x)

        return torch.unsqueeze(x, 1)
    

    def reset_states(self):
        self.drnn1.reset_states()
        self.drnn2.reset_states()


    def calc_amp_phase(self, x):
        x = torch.fft.fft(x)
        fft_amp = torch.sqrt(x.real**2 + x.imag**2)
        fft_phase = torch.atan(x.imag / x.real)

        return torch.vstack([fft_amp.T, fft_phase.T]).mT

        
    def inv_calc_amp_phase (self,x):
        x = x.permute(1,0)
        a = x[0] * torch.cos(x[1])
        b = x[0] * torch.sin(x[1])
        c = torch.complex(a, b)
        x = torch.fft.ifft(c)
        return x.real
        
