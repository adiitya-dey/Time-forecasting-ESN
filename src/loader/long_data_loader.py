


import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from einops import rearrange
from sklearn.model_selection import train_test_split


class AugmentedDataset:

    def __init__(self):
        self.points = np.linspace(1,100, 2000)
        self.noise = np.random.randn(len(self.points))

    def __call__(self, data_name):
        data_dict = {"linear": self.points,
                     "seasonal": self.seasonal(self.points),
                     "noise": self.noise,
                     "linear seasonal": self.linear_seasonal(self.points),
                     "linear noise": self.points + self.noise,
                     "seasonal noise": self.seasonal(self.points) + self.noise,
                     "linear seasonal noise": self.linear_seasonal(self.points) + self.noise

        }
        return data_dict[data_name]

            


    def seasonal(self,x):
        return np.sin(2 * np.pi * 10 * x)
    
    def linear_seasonal(self,x, frequency=10, amplitude=1, trend_slope=0.1):
        
        trend = trend_slope * x
        oscillations = amplitude * np.cos(2 * np.pi * frequency * x)
        y = trend + oscillations
        return y