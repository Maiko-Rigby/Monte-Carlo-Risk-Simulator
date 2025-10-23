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
    comparison_df = pd.DataFrame(comparison_df)
    comparison_df = comparison_df.sort_values("test_R", ascending= False)

    print(comparison_df.head)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(comparison_df))
    axes[0].bar([i-0.2 for i in x], comparison_df['Train R²'], width=0.4, 
               label='Train', alpha=0.8)
    axes[0].bar([i+0.2 for i in x], comparison_df['Test R²'], width=0.4, 
               label='Test', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Performance Comparison (R²)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar([i-0.2 for i in x], comparison_df['Train RMSE'], width=0.4, 
               label='Train', alpha=0.8)
    axes[1].bar([i+0.2 for i in x], comparison_df['Test RMSE'], width=0.4, 
               label='Test', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Model Performance Comparison (RMSE)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
