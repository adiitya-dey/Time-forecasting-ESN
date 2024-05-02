from ESN.tf_ESN import ESN
from ..ESN.data_loader import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from einops import rearrange

data = Dataset("airpassengers")
print(data["name"])

