import torch
import torch.nn as nn

from src.layers.drnn import DRNN
from src.layers.embedding import PositionalEmbedding


class Model(nn.Module):

    def __init__(self, input_size: int, output_size: int, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)


        self.window_len = input_size
        self.output_size = output_size


        self.drnn1 = DRNN(input_size=1, 
                    spectral_radius=0.01,
                    reservoir_size=25,
                    activation=nn.LeakyReLU(1.0))
        
        self.drnn2 = DRNN(input_size=25,
                    reservoir_size=50,
                    spectral_radius=0.01,
                    activation=nn.LeakyReLU(1.0))
        
        self.drnn3 = DRNN(input_size=1,
                    reservoir_size=50,
                    spectral_radius=0.01,
                    activation=nn.LeakyReLU(1.0))
        
        self.drnn4 = DRNN(input_size=50,
                    reservoir_size=25,
                    spectral_radius=0.01,
                    activation=nn.LeakyReLU(1.0))
        
        self.pos_embed = PositionalEmbedding(d_model=50)

        self.linear1 = nn.Linear(in_features=50,
                                out_features=self.output_size)
        
        self.linear2 = nn.Linear(in_features=25,
                                 out_features=self.output_size)
        


    def forward(self, x):
        ## x: 
        # seq_len = x.shape[0]

        # # # ## Calculate total input segments.
        # input_segments = seq_len // self.window_len

        # # # ## x: [Segments, Window Length]
        # x = x.view(input_segments, self.window_len)

        ## x : [Segments, Window Length, 1]
        x = torch.unsqueeze(x, -1)

        ## states : [segments, reservoir size, 1]
        x = self.drnn1(x)

        ## states: [segments, reservoir size, 1]
        x = self.drnn2(x)

        ## states: [segments, reservoir size, 1]
        x = self.linear1(torch.squeeze(x[-1,:,:]))

        x = self.drnn3(torch.unsqueeze(torch.unsqueeze(x, 1), 1))

        x= self.drnn4(x)

        x = self.linear2(torch.squeeze(x[-1,:,:]))

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
        
