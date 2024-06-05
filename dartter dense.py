from datetime import datetime
now = datetime.now()

import logging
filename = now.strftime("dartter_%d_%m_%H_%M")

logging.basicConfig(
    filename=f"log/dartter.log",
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

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


from src.layers.dense_esn import DenseESN
from src.loader.data_loader import DartsDataset

###############################
## Set Seed
###############################
torch.manual_seed(256)
np.random.seed(256)


###############################
## Set Hyperparameters
###############################

reservoir_size = 50
spectral_radius = 0.95
connectivity_rate = 0.8
washout=1
activation=nn.Tanh()
batch_size = 500
epochs = 1
max_len = 5000


## Multivariate to univariate datasets.
datas = ["ETTh1_M_U", "ETTh2_M_U", "ETTm1_M_U", "ETTm2_M_U", "ExRate_M_U"]

## Univariate datasets.
# datas = ["ExRate_U"]

dataset = DartsDataset()

for i in datas:
    

    data_dict = dataset.get_details(i)

    # ## Multviariate to Univariate Processing.
    # ## Rearrange the dataset to have target as last column.
    columns = [col for col in data_dict["dataset"].columns.tolist() if col!= data_dict["target"]]
    columns_reordered = columns + [data_dict["target"]]
    df_reordered = data_dict["dataset"][columns_reordered]

    # print(df_reordered.head(3).to_numpy())
    X_input, X_test, y_input, y_test = dataset.multi_uni(df_reordered.to_numpy())

    ## Univariate Processing
    # df = data_dict["dataset"][data_dict["target"]]
    # X_input, X_test, y_input, y_test = dataset.uni_uni(df.to_numpy().reshape(-1,1))

    data_train = TensorDataset(X_input, y_input)
    dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=False)

    ###############################
    ## Train and Fit Model
    ###############################
       
    model = DenseESN(reservoir_size=reservoir_size, 
                input_size=data_dict["input"],
                output_size=data_dict["output"],
                spectral_radius=spectral_radius, 
                connectivity_rate=connectivity_rate, 
                activation =activation,
                )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
        
    model.train()
    for _ in range(epochs):
        for m, (batch_X, batch_y) in enumerate(dataloader):
            logging.warning(f"Batch#{m} running.")
            out = model(batch_X)
            loss = criterion(batch_y, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    ###############################
    ## Predict and Calculate scores
    ###############################
    model.eval()
    logging.warning(f"Prediction begins.")
    y_pred = model(X_test)
    # y_pred = rearrange(y_pred, 'c 1 1 -> c 1')
    y_pred = y_pred.detach().numpy()

    ###############################
    ## Print Scores
    ###############################
    # print(y_pred.shape, y_test.shape)
    print(f"{i},{mean_squared_error(y_pred, y_test)},{mean_absolute_error(y_pred, y_test)}")

    # #############################
    # Plot prediction
    # #############################
    plt.figure(figsize=(10,5))
    plt.plot(y_test)
    plt.plot(y_pred,linestyle="--")
    # plt.legend()
    plt.title(f"Prediction Plot of {i} for standardized data")
    
    plt.savefig(f'plots/{i.lower().replace(" ","")}.png', dpi=300, bbox_inches="tight")

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


    ##############################
    # Get CSV
    ##############################
    df = pd.DataFrame({"y_test": y_test.flatten(),
                       "y_pred": y_pred.flatten()})
    df.to_csv(f'csv/{i}.csv', index=False)