import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RegressionModels:
    
    def __init__(self, data_dict):
        self.X_train = data_dict['X_train']
        self.X_test = data_dict['X_test']
        self.y_train = data_dict['y_train']
        self.y_test = data_dict['y_test']
        self.feature_names = data_dict['feature_names']
        self.models = {}
        self.results = {}

    def train_models(self):
        print("Training regression models")
        
        models = { 
            "Ridge Regression" : Ridge(),
            "Lasso Regression" : Lasso(),
            "Random Forest" : RandomForestRegressor(),
            "Gradient Boosting" : GradientBoostingRegressor()
        }

        for name, model in models.items():
            print(f"Training {name} model....")
            print(f"Training on {self.X_train.shape[0]} samples, testing on {self.X_test.shape[0]} samples")

            model.fit(self.X_train, self.y_train)

            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            train_metrics = self._calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self._calculate_metrics(self.y_test, y_test_pred)

            self.models[name] = model
            self.results[name] = {
                'train': train_metrics,
                'test': test_metrics,
                'predictions': y_test_pred
            }

            print(f"  Train R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Test R²:  {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return self.results

