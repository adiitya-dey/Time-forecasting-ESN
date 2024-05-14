
from darts.datasets import AirPassengersDataset, AusBeerDataset, ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ExchangeRateDataset, TrafficDataset

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from einops import rearrange
from sklearn.model_selection import train_test_split


class AugmentedDataset:

    def __init__(self):
        self.points = np.linspace(1,50, 500)
        self.noise = np.random.randn(len(self.points))

    def __call__(self, data):
        self.data = data
        match self.data.lower():
            case "linear":
               return self.points
            case "seasonal":
                return self.seasonal(self.points)
            case "noise":
                return self.noise
            case "linear seasonal":
                return self.linear_seasonal(self.points)
            case "linear noise":
                return self.points + self.noise
            case "seasonal noise":
                return self.seasonal(self.points) + self.noise
            case "linear seasonal noise":
                return self.linear_seasonal(self.points) + self.noise

            


    def seasonal(self,x):
        return np.sin(2 * np.pi * 10 * x)
    
    def linear_seasonal(self,x, frequency=10, amplitude=1, trend_slope=0.1):
        
        trend = trend_slope * x
        oscillations = amplitude * np.cos(2 * np.pi * frequency * x)
        y = trend + oscillations
        return y
    


class DartsDataset:

    def get_details(self, data):
        match data.lower():
            case "airpassengers":
                return {"input": 1,
                        "output": 1}
            case "ausbeer":
                return {"input": 1,
                        "output": 1}
            case "etth1":
                return {"input": 6,
                        "output": 1}
            case "etth2":
               return {"input": 6,
                        "output": 1}
            case "ettm1":
                return {"input": 6,
                        "output": 1}
            case "ettm2":
               return {"input": 6,
                        "output": 1}
            case "exchangerate":
                return {"input": 8,
                        "output": 8}


    def uni_uni(self, data):
        X_train, X_test = train_test_split(data, test_size=0.2, shuffle=False)
        sc = MinMaxScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        X_train_std = torch.Tensor(X_train_std)
        X_test_std = torch.Tensor(X_test_std)

        X_train_std = rearrange(X_train_std, 'b 1 -> b 1 1')
        X_test_std = rearrange(X_test_std, 'b 1 -> b 1 1')

        X_input = X_train_std[:-1,:,:]
        y_input = X_train_std[1:, :, :]
        y_input = rearrange(y_input, 'b 1 1 -> b 1')

        X_test = X_test_std[:-1,:,:]
        y_test = X_test_std[1:,:,:]
        y_test = rearrange(y_test, 'b 1 1 -> b 1')
        return X_input, X_test, y_input, y_test
    
    def multi_multi(self, data):
        X_train, X_test = train_test_split(data, test_size=0.2, shuffle=False)
        sc = MinMaxScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        X_train_std = torch.Tensor(X_train_std)
        X_test_std = torch.Tensor(X_test_std)

        X_train_std = rearrange(X_train_std, 'b c -> b c 1')
        X_test_std = rearrange(X_test_std, 'b c -> b c 1')

        X_input = X_train_std[:-1,:,:]
        y_input = X_train_std[1:, :, :]
        y_input = rearrange(y_input, 'b c 1 -> b c')

        X_test = X_test_std[:-1,:,:]
        y_test = X_test_std[1:,:,:]
        y_test = rearrange(y_test, 'b c 1 -> b c')
        # print(X_input.shape, X_test.shape, y_input.shape, y_test.shape)
        return X_input, X_test, y_input, y_test


    def multi_uni(self, data):
        X = data[:,:-1]
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
        sc1 = MinMaxScaler()
        sc2 = MinMaxScaler()
    
        y_train = rearrange(y_train, 'r -> r 1')
        y_test = rearrange(y_test, 'r -> r 1')
    

        X_train_std = sc1.fit_transform(X_train)
        X_test_std = sc1.transform(X_test)

        y_train_std = sc2.fit_transform(y_train)
        y_test_std = sc2.transform(y_test)

        X_train_std = torch.Tensor(X_train_std)
        X_test_std = torch.Tensor(X_test_std)

        y_train_std = torch.Tensor(y_train_std)
        y_test_std = torch.Tensor(y_test_std)

        X_train_std = rearrange(X_train_std, 'b c -> b c 1')
        X_test_std = rearrange(X_test_std, 'b c -> b c 1')

        return X_train_std, X_test_std, y_train_std, y_test_std


    def __call__(self, data):

        match data.lower():
            case "airpassengers":
                dataset = AirPassengersDataset().load().values()
                return self.uni_uni(dataset)

            case "ausbeer":
                dataset = AusBeerDataset().load().values()
                return self.uni_uni(dataset)
            
            case "etth1":
                dataset = ETTh1Dataset().load().values()
                return self.multi_uni(dataset)

            
            case "etth2":
                dataset = ETTh2Dataset().load().values()
                return self.multi_uni(dataset)
            
            case "ettm1":
                dataset = ETTm1Dataset().load().values()
                return self.multi_uni(dataset)
            
            case "ettm2":
                dataset = ETTm2Dataset().load().values()
                return self.multi_uni(dataset)
            
            case "exchangerate":
                dataset = ExchangeRateDataset().load().values()
                return self.multi_multi(dataset)
            

            

