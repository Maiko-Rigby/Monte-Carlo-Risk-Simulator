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
        

class LSTMTrainer:
    def __init__(self, data_dict, hidden_size = 64, num_layers = 2, dropout = 0.2):
        self.X_train = torch.tensor(data_dict['X_train'], dtype=torch.float32)
        self.X_test = torch.tensor(data_dict['X_test'], dtype=torch.float32)
        self.y_train = torch.tensor(data_dict['y_train'], dtype=torch.float32)
        self.y_test = torch.tensor(data_dict['y_test'], dtype=torch.float32)

        self.input_size = self.X_train.shape[2]

        self.model = LSTMRiskPredictor(
            input_size= self.input_size,
            hidden_size= hidden_size,
            num_layers= num_layers,
            dropout = dropout if num_layers > 1 else 0
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        print(f"LSTM Model initialised on {self.device}")
        print(f"Input size: {self.input_size},Hidden size: {hidden_size},Layers: {num_layers}")

    def train(self, epochs = 50, batch_size = 64, learning_rate = 0.001):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)

        loss_function = nn.MSELoss()
        optimiser = optim.Adam(self.model.parameters(), lr = learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")

        for epoch in range(epochs):
            
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimiser.zero_grad()
                output = self.model(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(epoch_loss)
            train_losses.append(avg_train_loss)

            self.model.eval()
            with torch.no_grad():
                X_test_device = self.X_test.to(self.device)
                y_test_device = self.y_test.to(self.device)
                val_pred = self.model(X_test_device)
                val_loss = loss_function(val_pred, y_test_device).item()
                val_losses.append(val_loss)

            lr_scheduler.step(val_loss)

            if (epoch + 1) % 10:
                print(f"Epoch {epoch+1}/{epochs}, Train loss: {avg_train_loss}, Validation loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_lstm_model.pth")

        self.model.load_state_dict(torch.load("best_lstm_model.pth"))

        return train_losses, val_losses
    
    def evaluate(self):

        self.model.eval()
            
        with torch.no_grad():
            # Predictions
            X_train_device = self.X_train.to(self.device)
            X_test_device = self.X_test.to(self.device)
            
            y_train_pred = self.model(X_train_device).cpu().numpy()
            y_test_pred = self.model(X_test_device).cpu().numpy()
            
            y_train_true = self.y_train.numpy()
            y_test_true = self.y_test.numpy()
        
        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'mae': mean_absolute_error(y_train_true, y_train_pred),
            'r2': r2_score(y_train_true, y_train_pred)
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
            'mae': mean_absolute_error(y_test_true, y_test_pred),
            'r2': r2_score(y_test_true, y_test_pred)
        }
        
        print(f"\nLSTM Results:")
        print(f"  Train R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(f"  Test R²:  {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_test_pred,
            'actuals': y_test_true
        }


        
        
