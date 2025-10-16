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

@dataclass
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
        self.daily_returns = np.array([s.mean_annual_return / 252 for s in stocks])
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

        with mp.Pool(processes = n_jobs) as pool:
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
        print(f"Results saved at {output_path.absolute()}")
        return output_path
    
    def visualise_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        ---------------------------------------------------------------
                        Create visualsations of the results
        ---------------------------------------------------------------
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of final portfolio values
        ax1 = axes[0, 0]
        ax1.hist(results_df['final_value'], bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(results_df['final_value'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(results_df['final_value'].median(), color='green', 
                   linestyle='--', linewidth=2, label='Median')
        ax1.set_xlabel('Final Portfolio Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Portfolio Values')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Total returns distribution
        ax2 = axes[0, 1]
        ax2.hist(results_df['total_return'] * 100, bins=50, alpha=0.7, 
                edgecolor='black', color='green')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Total Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Total Returns')
        ax2.grid(alpha=0.3)
        
        # 3. Risk-Return scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(results_df['volatility'] * 100, 
                             results_df['total_return'] * 100,
                             c=results_df['sharpe_ratio'], 
                             cmap='RdYlGn', alpha=0.5, s=20)
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Total Return (%)')
        ax3.set_title('Risk-Return Profile')
        plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
        ax3.grid(alpha=0.3)
        
        # 4. Percentile outcomes
        ax4 = axes[1, 1]
        percentiles = [5, 25, 50, 75, 95]
        values = [np.percentile(results_df['final_value'], p) for p in percentiles]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        bars = ax4.barh(percentiles, values, color=colors, edgecolor='black')
        ax4.set_xlabel('Portfolio Value ($)')
        ax4.set_ylabel('Percentile')
        ax4.set_title('Portfolio Value by Percentile')
        ax4.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax4.text(val, bar.get_y() + bar.get_height()/2, 
                    f'${val:,.0f}', va='center', ha='left', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

    def analyse_results(self, results_df: pd.DataFrame) -> dict:
        """
        ---------------------------------------------------------------
            Analyse simulation results and return summary statistics
        ---------------------------------------------------------------
        """
        summary = {
            'n_simulations': len(results_df),
            'expected_final_value': results_df['final_value'].mean(),
            'median_final_value': results_df['final_value'].median(),
            'expected_total_return': results_df['total_return'].mean(),
            'return_std': results_df['total_return'].std(),
            'probability_profit': (results_df['total_return'] > 0).mean(),
            'var_95_portfolio': np.percentile(results_df['final_value'], 5),
            'cvar_95_portfolio': results_df[results_df['final_value'] <= np.percentile(results_df['final_value'], 5)]['final_value'].mean(),
            'best_case': results_df['final_value'].max(),
            'worst_case': results_df['final_value'].min(),
            'mean_sharpe_ratio': results_df['sharpe_ratio'].mean(),
            'mean_max_drawdown': results_df['max_drawdown'].mean(),
        }
        
        return summary


def main():

    stocks = [
        Stock(ticker='AAPL', mean_annual_return=0.12, annual_volatility=0.25),
        Stock(ticker='MSFT', mean_annual_return=0.15, annual_volatility=0.28),
        Stock(ticker='GOOGL', mean_annual_return=0.10, annual_volatility=0.22),
    ]

    weights = [0.4, 0.3, 0.3]

    correlation_matrix = np.array([
        [1.0, 0.7, 0.6],   # AAPL correlations
        [0.7, 1.0, 0.65],  # MSFT correlations
        [0.6, 0.65, 1.0]   # GOOGL correlations
    ])

    portfolio = MonteCarloSimulator(
        stocks=stocks,
        weights=weights,
        correlation_matrix=correlation_matrix,
        initial_investment=10000
    )

    results = portfolio.run_simulations(
        n_simulations=10000,
        years=5,
        parallel=True
    )
    summary = portfolio.analyse_results(results)

    print(f"\nExpected Final Value: ${summary['expected_final_value']:,.2f}")
    print(f"Median Final Value: ${summary['median_final_value']:,.2f}")
    print(f"Expected Total Return: {summary['expected_total_return']*100:.2f}%")
    print(f"Return Std Dev: {summary['return_std']*100:.2f}%")
    print(f"Probability of Profit: {summary['probability_profit']*100:.1f}%")
    print(f"\nRisk Metrics:")
    print(f"  95% VaR: ${summary['var_95_portfolio']:,.2f}")
    print(f"  95% CVaR: ${summary['cvar_95_portfolio']:,.2f}")
    print(f"  Best Case: ${summary['best_case']:,.2f}")
    print(f"  Worst Case: ${summary['worst_case']:,.2f}")
    print(f"  Mean Sharpe Ratio: {summary['mean_sharpe_ratio']:.3f}")
    print(f"  Mean Max Drawdown: {summary['mean_max_drawdown']*100:.2f}%")
    
    # Save results
    output_file = portfolio.save_results(results)
    
    # Visualize
    print("\nGenerating visualizations...")
    portfolio.visualise_results(results, save_path='portfolio_analysis.png')

if __name__ == "__main__":
    mp.freeze_support()  # for Windows compatibility
    main()
