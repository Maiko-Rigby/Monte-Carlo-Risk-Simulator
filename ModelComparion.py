import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from CompareModels.py import compare_models

df = pd.readcsv("simulation_results_progressive.csv")



