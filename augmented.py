import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(['.'])

from src.layers.esn import ESN
from src.data.data_loader import AugmentedDataset


###############################
## Set Seed
###############################
torch.manual_seed(256)
np.random.seed(256)



###############################
## Set Hyperparameters
###############################

reservoir_size = 7
input_size = 1
output_size = 1
spectral_radius = 0.95
connectivity_rate = 0.5
activation=nn.Tanh()
batch_size = 100
epochs = 50


datas = ["linear", "seasonal", "noise", "linear seasonal", "linear noise", "seasonal noise", "linear seasonal noise"]

dataset = AugmentedDataset()

for i in datas:
    X = dataset(i)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(f"{i.upper()} Data")


    ###############################
    ## Train Test Split and Standardization.
    ###############################

    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    X_train = rearrange(X_train, 'r -> r 1')
    X_test = rearrange(X_test, 'r -> r 1')

    ###############################
    ## Train-Test plot
    ##############################
    ax1.plot(np.arange(len(X_train)), X_train, c='blue', label='Train')
    ax1.plot(np.arange(len(X_train), len(X_train) + len(X_test)), X_test, c="red", label='Test')
    ax1.legend()
    ax1.set_title(f"Train-Test Plot")

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

    data_train = TensorDataset(X_input, y_input)
    dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=False)

    ###############################
    ## Train and Fit Model
    ###############################
    model = ESN(reservoir_size=reservoir_size, 
                input_size=input_size,
                output_size=output_size,
                spectral_radius=spectral_radius, 
                connectivity_rate=connectivity_rate, 
                activation =activation,
                teacher_forcing=True)
    # model = DenseESN(batch_size= batch_size, epochs=epochs, reservoir_size=reservoir_size, input_size=input_size, spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, washout=1, activation =activation)
    model.train()
    for batch_X, batch_y in dataloader:

        out = model(batch_X, batch_y)

    

    ###############################
    ## Predict and Calculate scores
    ###############################
    model.eval()

    y_pred = model(torch.ones(X_test.shape))
    y_pred = rearrange(y_pred, 'c 1 1 -> c 1')
    y_pred = y_pred.detach().numpy()


    ###############################
    ## Validate States using plot
    ###############################
    all_states = np.array([x.detach().clone().numpy() for x in model.plot_states])
    all_states = rearrange(all_states, 'r c 1-> r c')
    ax2.plot(all_states)
    ax2.set_title(f"Reservoir States Plot")

    ###############################
    ## Plot prediction
    ###############################
    ax3.plot(y_test[:15], label="Ground Truth", c="blue")
    ax3.plot(y_pred[:15], label="Predicted", c="red", linestyle="--")
    ax3.legend()
    ax3.set_title(f"Prediction Plot for standardized data")

    ###############################
    ## Print Scores
    ###############################
    print(f"{i},{mean_squared_error(y_pred, y_test)},{mean_absolute_error(y_pred, y_test)}")
    plt.savefig(f'plots/{i.lower().replace(" ","")}.png', dpi=300, bbox_inches="tight")