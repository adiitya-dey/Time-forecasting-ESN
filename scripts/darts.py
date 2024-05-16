import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from einops import rearrange

import torch
import torch.nn as nn

from .esn.model.simple_esn import SimpleESN
from .esn.model.dense_esn import DenseESN
from .esn.model.batch_esn import BatchESN
from .esn.data_loader import DartsDataset
from .esn.model.hd_esn import HDESN

###############################
## Set Seed
###############################
torch.manual_seed(256)
np.random.seed(256)


###############################
## Set Hyperparameters
###############################

reservoir_size = 50
spectral_radius = 0.7
connectivity_rate = 0.8
washout=1
activation=nn.SiLU()
batch_size = 5000
epochs = 20
dimension=13


datas = ["airpassengers", "ausbeer","etth1", "etth2", "ettm1" , "ettm2", "exchangerate"]
# datas = ["ettm1"]

dataset = DartsDataset()

for i in datas:
    

    details = dataset.get_details(i)
    X_input, X_test, y_input, y_test = dataset(i)

    ###############################
    ## Train and Fit Model
    ###############################
    # model = SimpleESN(reservoir_size=reservoir_size, 
    #                   input_size=details["input"], 
    #                   spectral_radius=spectral_radius, 
    #                   connectivity_rate=connectivity_rate, 
    #                   washout=1, 
    #                   activation =activation)
    
    # model = DenseESN(batch_size= batch_size, 
    #                  epochs=epochs, 
    #                  reservoir_size=reservoir_size, 
    #                  input_size=details["input"], 
    #                  output_size=details["output"], 
    #                  spectral_radius=spectral_radius, 
    #                  connectivity_rate=connectivity_rate, 
    #                  washout=1, 
    #                  activation =activation)
    
    # model = BatchESN(reservoir_size=reservoir_size, 
    #                  input_size=details["input"], 
    #                  spectral_radius=spectral_radius, 
    #                  connectivity_rate=connectivity_rate, 
    #                  washout=1, activation =activation, 
    #                  batch_size=batch_size)
    
    # model = HDESN(batch_size= batch_size, 
                #   epochs=epochs,
                #   dimensions=dimension, 
                #   reservoir_size=reservoir_size, 
                #   input_size=details["input"], 
                #   output_size=details["output"], 
                #   spectral_radius=spectral_radius, 
                #   connectivity_rate=connectivity_rate, 
                #   washout=1, 
                #   activation =activation)
    
    model.train(X_input, y_input)

    ###############################
    ## Predict and Calculate scores
    ###############################
    y_pred = model.predict(X_test)
    y_pred = y_pred.squeeze(-1)
    y_pred = y_pred.detach().numpy()

    y_test = y_test.detach().numpy()
    ###############################
    ## Print Scores
    ###############################
    # print(y_pred.shape, y_test.shape)
    print(f"{i},{mean_squared_error(y_pred, y_test)},{mean_absolute_error(y_pred, y_test)}")

    ##############################
    # Plot prediction
    ##############################
    # plt.figure(figsize=(10,5))
    # plt.plot(y_test)
    # plt.plot(y_pred,linestyle="--")
    # # plt.legend()
    # plt.title(f"Prediction Plot of {i} for standardized data")
    
    # plt.savefig(f'plots/{i.lower().replace(" ","")}.png', dpi=300, bbox_inches="tight")

    ##############################
    # Get Model Summary
    ##############################
    # total_params = sum(p.numel() for p in model.parameters())
    # learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    # nonlearnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
    # print(f"Total Parameters: {total_params}, Learnable Parameters: {learnable_params},Non-Learnable Parameters: {nonlearnable_params}, ")
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('Model Size: {:.3f}MB'.format(size_all_mb))