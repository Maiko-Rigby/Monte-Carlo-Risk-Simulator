import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PortfolioDataPreprocessor():

    def __init__(self, csv_path: str = None) -> pd.DataFrame:
        self.df = pd.read_csv(csv_path)
        print(f"Dataset loaded!")
        print(f"Number of rows: {len(self.df)}")
        print(self.df.head)

    def engineer_features(self):
        print("Engineering features...")

        if "day" in self.df.columns:
            self.df.sort_values(['simulation_id','day']).reset_index(drop=True)

        # Lag features 
        lag_columns = ['daily_return','volatility','sharpe_ratio']
        for column in lag_columns:
            if column in self.df.columns:
                self.df['col_lag1'] = self.df.groupby('simulation_id')[column].shift(1)
                self.df['col_lag2'] = self.df.groupby('simulation_id')[column].shift(2)

        # Rolling statistic (short-term trends)
        if 'daily_return' in self.df.columns:
            self.df['return_ma5'] = self.df.groupby('simulation_id')['daily_return'].transform(
                lambda x: x.rolling(window = 5, min_periods= 1).mean()
            )
            self.df['return_std5'] = self.df.groupby('simulation_id')['daily_return'].transform(
                lambda x: x.rolling(window = 5, min_periods= 1).std()
            )
                
        # Momentum indicators
        if 'portfolio_value' in self.df.columns:
            self.df['momentum_5'] = self.df.groupby('simulation_id')['portfolio_value'].pct_change(5)
            self.df['momentum_10'] = self.df.groupby('simulation_id')['portfolio_value'].pct_change(10)

        # Risk-adjusted returns
        if 'total_return' in self.df.columns and 'volatility' in self.df.columns:
            self.df['risk_adjusted_returns'] = self.df['total_return'] / (self.df['volatility'] + 1e-6)
        
        if 'total_return' and 'max_drawdown' in self.df.columns:
            self.df['calmar_ratio'] = self.df['total_return'] / abs(self.df['max_drawdown'] + 1e-6)
        
        # Cumulative time progress
        if 'day' in self.df.columns:
            self.df['time_progress'] = self.df['day'] / self.df['day'].max()

        self.df.dropna()

        print(f'Feature engineering complete: {self.df.shape()}')
        return self.df
    
    def prepare_ml_data(self, target_col='sharpe_ratio', test_size=0.2) -> dict:
        print(f'Preparing ML data with the target value: {target_col}')

        exclude_cols = [
            'simulation_id', 'day', 'sharpe_ratio', 
            'risk_adjusted_return', 'calmar_ratio', 
            'total_return', 'final_value'
        ]

        feature_cols = []
        for col in self.df:
            if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
        
        X = self.df[feature_cols]
        y = self.df[target_col]

        print(f'Feature columns, {feature_cols}\nShape of X : {X.shape}\nShape of Y : {y.shape}')

        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size= test_size, random_state=42, shuffle= True)

        scaler = StandardScaler()

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f'Shape of train set: {X_train_scaled.shape}\nShape of test set: {X_test_scaled.shape}')

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': scaler
        }
    
    def prepare_lstm_data(self, target_col='sharpe_ratio', sequence_length=10, test_size=0.2, forecast_horizon=1):
        print("preparing LSTM data")

        feature_cols = ['daily_return', 'volatility', 'portfolio_value', 'max_drawdown', 'momentum_5']
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        x_sequences = []
        y_targets = []

        for sim_id in self.df['simulation_id'].unique():
            sim_id = self.df[self.df['simulation_id'] == sim_id]

            if len(sim_id) < sequence_length + forecast_horizon:
                continue

            features = sim_id[feature_cols].values
            target = sim_id[target_col].values

            for i in range(0, len(features)):
                x_sequences.append(features[i : i + sequence_length])
                y_targets.append(target[i + sequence_length + forecast_horizon - 1])

        x_sequences = np.array(x_sequences)
        y_targets = np.array(y_targets)

        print(f"Shape of X_sequences : {x_sequences.shape}")
        print(f"Shape of Y_target : {y_targets.shape}")

        split_index = int(len(x_sequences) * (1 - test_size))
        X_train = x_sequences[split_index:]
        X_test = x_sequences[:split_index]
        y_train = y_targets[split_index:]
        y_test = y_targets[:split_index]

        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_flat)
        X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        print(f"Train sequences: {X_train_scaled.shape}, Test sequences: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': scaler,
            'sequence_length': sequence_length
        }
    

