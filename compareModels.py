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
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


def compare_models(regression_results, LSTM_results):
    print("MODEL COMPARISON SUMARRY HEADER")

    comparison_data = []

    for model_name, results in regression_results:
        model_results = dict(
            Model = model_name,
            Type = "Regression",
            train_RMSE = results['train']['rmse'],
            test_RMSE = results['test']['rmse'],
            train_R = results['train']['r2'],
            test_R = results['test']['r2'],
            Overfitting = results['train']['r2']  - results['test']['r2']
        )

    comparison_data.append(model_results)

    if LSTM_results and torch.cuda.is_available():
        LSTM_model_results = dict(
            Model = "LSTM",
            Type = "Deep Dearning",
            train_RMSE = results['train']['rmse'],
            test_RMSE = results['test']['rmse'],
            train_R = results['train']['r2'],
            test_R = results['test']['r2'],
            Overfitting = results['train']['r2']  - results['test']['r2']
        )

    comparison_data.append(LSTM_model_results)

    
    