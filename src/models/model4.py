import torch
import torch.nn as nn

from src.layers.drnn import DRNN
from src.layers.embedding import PositionalEmbedding


class Model(nn.Module):

    def __init__(self, input_size: int, output_size: int, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)


        self.window_len = input_size
        self.output_size = output_size


        self.drnn = DRNN(input_size=2, 
                    spectral_radius=0.5,
                    reservoir_size=50,
                    activation=nn.LeakyReLU(1.0))
        
        self.pos_embed = PositionalEmbedding(d_model=50)

        self.linear1 = nn.Linear(in_features=400*50,
                                out_features=self.output_size,
                                bias=False)
        
        

    ## x: [Seq_len,  1]
    def forward(self, x):

        ## Convert to frequency domain using Fast Fourier Transform
        x = torch.fft.fft(x.squeeze(1))
        
        ## Convert to Vector format of [real part, imaginary part, amplitude, phase]
        x = torch.vstack([x.real.t(), x.imag.t()]) # , torch.sqrt(x.real**2 + x.imag**2).t(), torch.atan(x.imag / x.real).t()
       
        x = x.permute(1, 0)

        norm = torch.linalg.norm(x, dim=0, keepdim=True)

        x = x / norm

        ## Use ESN to convert to high dimensional non-linear mapping.
        x = self.drnn(torch.unsqueeze(x, -1))

        ## Add position embedding.
        # x = self.pos_embed(x)

        ## Calculate cosine similarity.
        # x = torch.mm(x, x.T)

        # norm = torch.linalg.norm(x, dim=0, keepdim=True)

        # x = x /norm

        x = x.flatten() 

        x = self.linear1(x)

        

        return torch.unsqueeze(x, 1)
    

    def reset_states(self):
        self.drnn.reset_states()
        # self.drnn2.reset_states()


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
        
