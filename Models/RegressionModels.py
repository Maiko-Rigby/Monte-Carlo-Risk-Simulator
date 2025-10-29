import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib as plt
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

    def _calculate_metrics(self, y_true, y_pred):
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        }
    
    def feature_importance(self, model_name: str = "Random Forest"):

        model = self.models[model_name]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
                
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), 
                      [self.feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            return dict(zip(self.feature_names, importances))
        
        else:
            print(f"{model_name} does not have feature_importances_")
            return None

    def plot_predictions(self, model_name = "Random Forest"):

        y_pred = self.result[model_name]["predictions"]

        fig, axes = plt.subplots(1, 2, figsize = (14,5))

        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=10)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Sharpe Ratio')
        axes[0].set_ylabel('Predicted Sharpe Ratio')
        axes[0].set_title(f'{model_name} - Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name} - Residual Plot')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    