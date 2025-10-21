import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
warnings.filterwarnings('ignore')



class LSTMRiskPredictor (nn.Module):
    def __init__(self, input_size, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super.__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(
            input_size= input_size,
            hidden_size= self.hidden_size,
            num_layers= self.num_layers,
            batch_first= True,
            dropout = dropout if num_layers > 1 else 0
        )

        self.fc_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):

        lstm_output, _ =  self.LSTM(x)
        last_time = lstm_output[:, -1, :]

        output, _ = self.fc_network(last_time)
        
        return output.squeeze()
        

