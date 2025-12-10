# -----------------------------
# Data utilities
# -----------------------------

import numpy as np
import pandas as pd
import yfinance as yf

def download_price_data(tickers, start, end):
    """
    Download adjusted daily close prices for the given tickers.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # Handle single vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        # Single ticker case: make it a DataFrame
        prices = data["Adj Close"].to_frame(name=tickers[0])

    # Drop rows with any missing data to keep logic simple
    prices = prices.dropna()

    return prices



# -----------------------------
# Performance metrics
# -----------------------------

def compute_performance_metrics(daily_returns, initial_capital=1.0):
    """
    Given a Series of daily returns, compute common performance stats.
    """
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {
            "total_profit": 0.0,
            "final_equity": initial_capital,
            "num_days": 0,
            "sharpe_annual": 0.0,
            "max_drawdown": 0.0,
            "cagr": 0.0,
        }

    equity = (1 + daily_returns).cumprod() * initial_capital

    # Total profit in currency units
    total_profit = equity.iloc[-1] - equity.iloc[0]

    # Annualized Sharpe with risk-free ~ 0
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Maximum drawdown
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    # CAGR
    num_days = len(daily_returns)
    years = num_days / 252.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    return {
        "total_profit": float(total_profit),
        "final_equity": float(equity.iloc[-1]),
        "num_days": int(num_days),
        "sharpe_annual": float(sharpe),
        "max_drawdown": float(max_dd),
        "cagr": float(cagr),
    }

def compute_equal_weighted_returns(prices):
    """
    Compute equal-weighted portfolio returns from a price DataFrame.
    
    Each day, computes the average return across all stocks in the universe,
    effectively creating an equal-weighted portfolio that rebalances daily.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with datetime index and ticker columns
        
    Returns:
    --------
    returns : pd.Series
        Daily returns with datetime index, named "equal_weighted_returns"
    """
    # Compute daily returns for each stock
    daily_returns = prices.pct_change()
    
    # Drop first row (NaN from pct_change)
    daily_returns = daily_returns.iloc[1:]
    
    # Equal-weighted: average return across all stocks each day
    # This handles missing data by only averaging available stocks
    equal_weighted_returns = daily_returns.mean(axis=1)
    
    # Fill any remaining NaN with 0 (shouldn't happen if data is clean)
    equal_weighted_returns = equal_weighted_returns.fillna(0.0)
    
    # Set name for consistency with other backtest returns
    equal_weighted_returns.name = "equal_weighted_returns"
    
    return equal_weighted_returns

def print_metrics(name, metrics, num_trades):
    """
    Pretty-print summary statistics for a strategy.
    """
    print(f"\n=== {name} ===")
    print(f"Trading days        : {metrics['num_days']}")
    print(f"Number of trades    : {num_trades}")
    print(f"Total profit        : {metrics['total_profit']:.4f}")
    print(f"Final equity        : {metrics['final_equity']:.4f}")
    print(f"CAGR                : {metrics['cagr']:.2%}")
    print(f"Sharpe (annual)     : {metrics['sharpe_annual']:.2f}")
    print(f"Max drawdown        : {metrics['max_drawdown']:.2%}")
