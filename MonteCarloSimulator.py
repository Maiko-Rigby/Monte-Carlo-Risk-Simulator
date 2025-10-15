import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MonteCarloSimulator:
    """
    Monte Carlo Simulator for a multi-asset portfolio risk analysis
    """

    def __init__(self, tickers, weights, initial_investment = 100000):
        """
        ---------------------------------------------------------------
                                    Initialise
        ---------------------------------------------------------------
        tickers : list
            Stock ticker symbols (for example, ['AAPL','GOOGL','MSFT'])
        weights: list
            Portfolio weights (must sum up to 1.0)
        initial_investments: float
            Starting portfolio investment
        """
        self.tickers = tickers
        self.weights = np.array(weights)
        self.initial_investment = initial_investment

    def set_parameters(self, mean_returns, volatilities, correlation_matrix):
        """
        ---------------------------------------------------------------
                            Set the portfolio parameters
        ---------------------------------------------------------------
        mean_returns : array
            The annual expected return for each of the assets
        volatilities : array
            The annual volatility (std dev) for each asset
        correlation_matrix : 2D array
            Corrrelation matrix of all the assets
        """
        self.mean_returns = np.array(mean_returns)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)

        # Calculate the covariance matrix from correlation and volatilities

        vol_matrix = np.diag(self.volatilities)
        self.covarience_matrix = vol_matrix @ self.volatilities @  vol_matrix

    def simulate(self, years = 5, simulations = 10000, trading_days_per_years = 252):
        """
        ---------------------------------------------------------------
                                Run the simulation
        ---------------------------------------------------------------
        years : int
            Simulation time
        simulations :
            Number of Simulations
        trading_days_per_year : int
            Trading days per year (default value = 252)

        Returns :
            Dict containing the simulation results
        """ 
        print(f"Starting {simulations} simulations over {years} years....")
        start_date = datetime.now()

        # Time parameters
        dt = 1 / trading_days_per_years # Time step
        total_steps = years * trading_days_per_years

        # Convert the annual parameters to per-step
        drift = (self.mean_returns - 0.5*self.volatilities**2) * dt
        diffusion = self.volatilities * np.sqrt(dt)

        # Generate correlated random shocks
        # Use Cholesky decomposition for correlation
        L = np.linalg.cholesky(self.correlation_matrix)

        # Initialise the results array
        portfolio_values = np.zeros((simulations, total_steps + len(self.tickers)))
        portfolio_values[:,0] = self.initial_investment

        # Run the simulations
        for t in range(1 , total_steps):
            # Generate the correlated random normal variables
            Z = np.random.standard_normal((simulations, len(self.tickers)))
            Correlated_Z = Z @ L.T

            