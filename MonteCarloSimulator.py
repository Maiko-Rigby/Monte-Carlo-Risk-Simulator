import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

class Stock:
    """
    ---------------------------------------------------------------
                                Stock parameters
    ---------------------------------------------------------------
    """
    ticker : str
    mean_annual_return : float
    annual_volatility : float

class MonteCarloSimulator:
    """
    Monte Carlo Simulator for a multi-asset portfolio risk analysis
    """

    def __init__(self, stocks : List[Stock], weights: List[float],correlation_matrix: np.ndarray, initial_investment = 100000):
        """
        ---------------------------------------------------------------
                                    Initialise
        ---------------------------------------------------------------
        stocks : List[Stock]
            Stock objects (for example, ['AAPL','GOOGL','MSFT'])
        weights: List[float]
            Portfolio weights (must sum up to 1.0)
        initial_investments: float
            Starting portfolio investment
        correlation_matrix : np.ndarray
            Correlation matrix between between stocks (n x n)
        """
        self.stocks = stocks
        self.weights = np.array(weights)
        self.correlation_matrix = correlation_matrix
        self.initial_investment = initial_investment

        # Validate the inputs
        assert len(stocks) == len(weights), "Number of stocks must match weights"
        assert abs(sum(self.weights) - 1) < 1e-6, "Weights must sum to 1.0"
        assert correlation_matrix.shape == (len(stocks), len(stocks)), "Correlation matrix dimensions must match the number of stocks"

        # Conver the annual parameters to daily
        self.daily_returns = np.array([s.means_annual_return / 252 for s in stocks])
        self.daily_volatilities = np.array([s.annual_volatility / 252 for s in stocks])

        # Create a covariance matrix for correlation and volatilities
        self.cov_matrix = self._correlation_to_covariance(correlation_matrix, self.daily_volatilities)

    @staticmethod
    def _correlation_to_covariance(corr_matrix, volatilities):
        """
        ---------------------------------------------------------------
                Convert correlation matrix to covariance matrix
        ---------------------------------------------------------------
        """
        vol_matrix = np.diag(volatilities)
        return vol_matrix @ corr_matrix @ vol_matrix

    def simulate_single_path(self, days: int, seed: int = None) -> Tuple[np.ndarray, dict]:
        """
        ---------------------------------------------------------------
            Simulate a single portfolio path using Brownian Motion
        ---------------------------------------------------------------
        days : int
            The number of trading days to simulate
        seed : int
            Random seed for reproducibility

        Returns 
        portfolio_values 
            Array of portfolio values over time
        metrics
            Dictionary of performance metrics
        """
        if seed is not None:
            np.random.seed(seed)

        n_stocks = len(self.stocks)

        # Generate correlated random returns using Cholesky decomposition
        L = np.linalg.cholesky(self.cov_matrix)
        random_normals = np.random.normal(0,1, (days, n_stocks))
        correlated_returns = random_normals @ L.T

        # Add drift 
        returns = self.daily_returns + correlated_returns

        # Calculate the individual stock prices
        stock_prices = np.zeros((days + 1, n_stocks))
        stock_prices[0] = self.initial_investment * self.weights

        for i in range(days):
            stock_prices[i + 1] = stock_prices[i] * (1 + returns[i])
        
        # Portfolio value is sum of all stock positions
        portfolio_values = stock_prices.sum(axis=1)

        # Calculate metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = {
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / self.initial_investment) - 1,
            'mean_daily_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()            
        }
        
        return portfolio_values, metrics
    
    @staticmethod
    def _calculate_max_drawdown(values):
        """
        ---------------------------------------------------------------
                            Calculate the maximum drawdown
        ---------------------------------------------------------------
        """
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)
    
    def run_simulations(self, n_simulations: int, years: int = 5, parallel : bool = False, n_jobs: int = -1) -> pd.DataFrame:
        """
        ---------------------------------------------------------------
                            Run monte carlo simulation
        ---------------------------------------------------------------
        n_simulations: int
            The number of simulation paths
        years : int
            Number of years to simulate
        Parallel : boolean
            Whether to use parallel processing
        n_jobs : int
            The number of parallel jobs

        returns : Dataframe with the results
        """
        days = years * 252 # number of trading days per year

        print(f"Running {n_simulations:,} simulations for {years} years ({days} days)...")
        print(f"Parallel processing: {parallel}")

        start_time = time.time()

        if parallel:
            results = self._run_parallel(n_simulations, days, n_jobs)
        else:
            results = self._run_serial(n_simulations, days)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds ({n_simulations/elapsed:0f} sims/sec)")

        return results
    
    def _run_serial(self, n_simulations: int, days: int) -> pd.DataFrame:
        """
        ---------------------------------------------------------------
                            Run simulation serially
        ---------------------------------------------------------------
        """
        results = []

        for i in range(n_simulations):
            _, metrics = self.simulate_single_path(days, seed = i)
            results.append(metrics)

            if (i+1) % 100 == 0:
                print(f" Progress: {i+1:,}/{n_simulations:,}")

        return pd.DataFrame(results)
    
    def _run_parallel(self, n_simulations: int, days: int, n_jobs: int = -1) -> pd.DataFrame:
        """
        ---------------------------------------------------------------
                            Run simulation in parallel
        ---------------------------------------------------------------
        """
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        print(f"Using {n_jobs} CPU cores")
        
        sim_func = partial(self._simulate_worker, days = days)

        with mp.pool(processes = n_jobs) as pool:
            results = pool.map(sim_func, range(n_simulations))
        
        return pd.DataFrame(results)

    def _simulate_worker(self, sim_id: int, days: int) -> dict:
        """
        ---------------------------------------------------------------
                    Worker function for the parallel execution
        ---------------------------------------------------------------
        """
        _, metrics = self.simulate_single_path(days, seed= sim_id)
        metrics['simulation_id'] = sim_id

        return metrics
    
    def save_results(self, results_df: pd.DataFrame, filename: str = 'simulation_results.csv'):
        """
        ---------------------------------------------------------------
                            Save the results as a CSV
        ---------------------------------------------------------------
        """
        output_path = Path(filename)
        results_df.to_csv(output_path, index=False)
        
