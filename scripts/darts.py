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
from .esn.data_loader import DartsDataset


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
activation=nn.SELU()
batch_size = 100
epochs = 50


datas = ["airpassengers", "ausbeer","exchangerate", "etth1", "etth2", "ettm1"]

dataset = DartsDataset()

for i in datas:
    

    details = dataset.get_details(i)
    X_input, X_test, y_input, y_test = dataset(i)

    ###############################
    ## Train and Fit Model
    ###############################
    model = SimpleESN(reservoir_size=reservoir_size, input_size=details["input"], spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, washout=1, activation =activation)
    # model = DenseESN(batch_size= batch_size, epochs=epochs, reservoir_size=reservoir_size, input_size=details["input"], output_size=details["output"], spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, washout=1, activation =activation)
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