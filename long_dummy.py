import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(['.'])

# from src.layers.esn import ESN
# from src.layers.dense_esn import ESN
# from src.layers.fast_dense_esn import ESN
from src.layers.long_esn import ESN
from src.loader.long_data_loader import AugmentedDataset


###############################
## Set Seed
###############################
torch.manual_seed(256)
np.random.seed(256)



###############################
## Set Hyperparameters
###############################

reservoir_size = 50
input_size = 1
output_size = 1
spectral_radius = 0.3
# connectivity_rate = 0.5
activation=nn.LeakyReLU(1.0)
batch_size = 100
epochs = 20
window_len = 20
pred_len = 20


datas = ["linear", "seasonal", "linear seasonal"]

dataset = AugmentedDataset()

for i in datas:
    X = dataset(i)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(f"{i.upper()} Data")


    ###############################
    ## Train Test Split and Standardization.
    ###############################

    X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)
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

    # X_train_std = rearrange(X_train_std, 'b 1 -> b 1 1')
    # X_test_std = rearrange(X_test_std, 'b 1 -> b 1 1')

    X_input = X_train_std.detach().clone()
    y_input = X_train_std.detach().clone()
    # y_input = rearrange(y_input, 'b 1 1 -> b 1')

    X_test = X_train_std.detach().clone()
    y_test = X_test_std.detach().clone()
    # y_test = rearrange(y_test, 'b 1 1 -> b 1')

    data_train = TensorDataset(X_input, y_input)
    train_dataloader = DataLoader(data_train, batch_size=100, shuffle=False, drop_last=True)

    # data_test = TensorDataset(X_test)
    # test_dataloader = DataLoader(data_test, batch_size=100, shuffle=False)


    

    ###############################
    ## Train and Fit Model
    ###############################
    model = ESN(reservoir_size=reservoir_size, 
                input_size=window_len,
                output_size=output_size,
                spectral_radius=spectral_radius, 
                # connectivity_rate=connectivity_rate, 
                activation =activation,
                window_len = window_len,
                pred_len=pred_len
                )
    # model = DenseESN(batch_size= batch_size, epochs=epochs, reservoir_size=reservoir_size, input_size=input_size, spectral_radius=spectral_radius, connectivity_rate=connectivity_rate, washout=1, activation =activation)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
       
    model.train()
    for _ in range(epochs):
        train_loss = []
        for batch_X, batch_y in train_dataloader:

            # Use this only for Pure ESN method.
            # out = model.fit(batch_X, batch_y)

            # Use the below for ESN with dense layers.
            optimizer.zero_grad()
            out = model(batch_X[:-pred_len,:])
            

            
            
            loss = criterion(out, batch_y[-pred_len:,:])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # model.reset_states()
        print(f"Epoch {_} Train Loss: {np.average(train_loss)}")
    
    ###############################
    ## Validate States using plot
    ###############################
    all_states = model.plot_states.numpy()
    # all_states = rearrange(all_states, 'r c 1-> r c')
    ax2.plot(all_states)
    ax2.set_title(f"Reservoir States Plot")

    ###############################
    ## Predict and Calculate scores
    ###############################
    model.eval()

   
    # Use the below for ESN with dense layers.
    y_pred = model(X_test)

    y_pred = y_pred.detach().numpy()
    y_test = y_test.detach().numpy()

   

    ###############################
    ## Plot prediction
    ###############################
    ax3.plot(y_test[:pred_len+25], label="Ground Truth", c="blue")
    ax3.plot(y_pred, label="Predicted", c="red", linestyle="--")
    ax3.legend()
    ax3.set_title(f"Prediction Plot for standardized data")

    

    ###############################
    ## Print Scores
    ###############################
    print(f"{i},{mean_squared_error(y_pred, y_test[:pred_len,:])},{mean_absolute_error(y_pred, y_test[:pred_len,:])}")
    plt.savefig(f'plots/{i.lower().replace(" ","")}.png', dpi=300, bbox_inches="tight")

    ###############################
    ## Plot Weights
    ###############################
    plt.figure(figsize=(5,5))
    plt.imshow(model.output_linear_layer.weight.detach().numpy(), cmap='Oranges', interpolation='bilinear')
    plt.xlabel("Reservoir Size")
    plt.ylabel("Prediction Size")
    # plt.colorbar()
    plt.savefig(f'plots/{i.lower().replace(" ","")}_weights.png', dpi=300, bbox_inches="tight")