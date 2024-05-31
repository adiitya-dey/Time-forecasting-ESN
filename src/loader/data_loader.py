


import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from einops import rearrange
from sklearn.model_selection import train_test_split

from darts.datasets import ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset,ExchangeRateDataset, TrafficDataset, WeatherDataset


class AugmentedDataset:

    def __init__(self):
        self.points = np.linspace(1,100, 500)
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
    


class DartsDataset:

    def get_details(self, data_name):
        dataset_dict = {"ETTh1_M_U": {"input": 6,
                                      "output": 1,
                                      "target": "OT",
                                      "dataset": ETTh1Dataset().load().pd_dataframe()},
                        "ETTh2_M_U": {"input": 6,
                                      "output": 1,
                                      "target": "OT",
                                      "dataset": ETTh2Dataset().load().pd_dataframe()},
                        "ETTm1_M_U": {"input": 6,
                                      "output": 1,
                                      "target": "OT",
                                      "dataset": ETTm1Dataset().load().pd_dataframe()},
                        "ETTm2_M_U": {"input": 6,
                                      "output": 1,
                                      "target": "OT",
                                      "dataset": ETTm2Dataset().load().pd_dataframe()},
                        "ExRate_M_U": {"input": 7,
                                      "output": 1,
                                      "target": "7",
                                      "dataset": ExchangeRateDataset().load().pd_dataframe()},
                        "Weather_M_U": {"input": 20,
                                      "output": 1,
                                      "target": "T (degC)",
                                      "dataset": WeatherDataset().load().pd_dataframe()},
                        "ExRate_U": {"input": 1,
                                      "output": 1,
                                      "target": "7",
                                      "dataset": ExchangeRateDataset().load().pd_dataframe()},
                        # "Traffic_U": {"input": 1,
                        #               "output": 1,
                        #               "target": "1",
                        #               "dataset": TrafficDataset().load().pd_dataframe()},                                      
        }       

        return dataset_dict[data_name]


      


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


    
            

            

