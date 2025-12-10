# -----------------------------
# Plotting utilities
# -----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_equity_curve(returns_series, title="Equity Curve", ax=None):
    """
    Plot cumulative equity curve from daily returns.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with datetime index
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    equity = (1 + returns_series).cumprod()
    ax.plot(equity.index, equity.values, linewidth=2, label=title)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return ax

def plot_drawdown(returns_series, title="Drawdown", ax=None):
    """
    Plot drawdown (underwater) chart.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with datetime index
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    equity = (1 + returns_series).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100  # Convert to percentage
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return ax

def plot_utilization(utilization_series, title="Utilization Over Time", ax=None):
    """
    Plot utilization (fraction of active pairs) over time.
    
    Parameters:
    -----------
    utilization_series : pd.Series
        Daily utilization values (0-1) with datetime index
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(utilization_series.index, utilization_series.values * 100, 
                    0, alpha=0.3, color='blue')
    ax.plot(utilization_series.index, utilization_series.values * 100, 
            linewidth=1.5, color='darkblue', label='Utilization')
    ax.set_xlabel('Date')
    ax.set_ylabel('Utilization (%)')
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return ax

def plot_spread_with_signals(prices, pair_info, trading_window, 
                              strategy_type='distance', entry_z=2.0, exit_z=0.0,
                              title="Spread with Entry/Exit Signals", ax=None):
    """
    Plot spread time series with entry/exit signals for a single pair.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data for the trading window
    pair_info : dict
        Dictionary containing pair information (stock1/stock2 or x/y, spread_mean, spread_std, beta)
    trading_window : pd.DatetimeIndex
        Dates for the trading window
    strategy_type : str
        'distance' or 'cointegration'
    entry_z : float
        Entry z-score threshold
    exit_z : float
        Exit z-score threshold
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get pair tickers
    if strategy_type == 'distance':
        ticker1 = pair_info['stock1']
        ticker2 = pair_info['stock2']
        normalized = prices[[ticker1, ticker2]] / prices[[ticker1, ticker2]].iloc[0]
        spread = normalized[ticker1] - normalized[ticker2]
        mu = pair_info['spread_mean']
        sigma = pair_info['spread_std']
    else:  # cointegration
        ticker1 = pair_info['x']
        ticker2 = pair_info['y']
        log_prices = np.log(prices[[ticker1, ticker2]])
        beta = pair_info['beta']
        spread = log_prices[ticker2] - beta * log_prices[ticker1]
        mu = pair_info['spread_mean']
        sigma = pair_info['spread_std']
    
    # Filter to trading window
    spread = spread.loc[trading_window]
    
    # Calculate z-scores
    z_scores = (spread - mu) / sigma if sigma > 0 else pd.Series(0, index=spread.index)
    
    # Plot spread
    ax.plot(spread.index, spread.values, linewidth=1.5, color='black', label='Spread', alpha=0.7)
    ax.axhline(y=mu, color='gray', linestyle='--', alpha=0.5, label='Mean')
    ax.axhline(y=mu + entry_z * sigma, color='red', linestyle='--', alpha=0.7, label=f'Entry (+{entry_z}σ)')
    ax.axhline(y=mu - entry_z * sigma, color='green', linestyle='--', alpha=0.7, label=f'Entry (-{entry_z}σ)')
    ax.axhline(y=mu + exit_z * sigma, color='orange', linestyle=':', alpha=0.5, label=f'Exit (±{exit_z}σ)')
    ax.axhline(y=mu - exit_z * sigma, color='orange', linestyle=':', alpha=0.5)
    
    # Mark entry/exit points
    positions = []
    current_pos = 0
    
    for date in spread.index:
        z = z_scores.loc[date]
        prev_pos = current_pos
        
        if current_pos == 0:
            if z > entry_z:
                current_pos = -1  # short spread
            elif z < -entry_z:
                current_pos = 1   # long spread
        else:
            if abs(z) < exit_z:
                current_pos = 0
        
        if current_pos != prev_pos:
            positions.append((date, current_pos))
    
    # Plot entry/exit markers
    for date, pos in positions:
        if pos == 1:
            ax.scatter(date, spread.loc[date], color='green', marker='^', s=100, zorder=5, label='Long Entry' if date == positions[0][0] else '')
        elif pos == -1:
            ax.scatter(date, spread.loc[date], color='red', marker='v', s=100, zorder=5, label='Short Entry' if date == positions[0][0] else '')
        elif pos == 0:
            ax.scatter(date, spread.loc[date], color='orange', marker='o', s=80, zorder=5, label='Exit' if date == positions[0][0] else '')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread')
    ax.set_title(f"{title} ({ticker1} vs {ticker2})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return ax

def plot_strategy_comparison(*returns_series, names=None, utilizations=None):
    """
    Create a comprehensive comparison plot of multiple strategies.
    
    Parameters:
    -----------
    *returns_series : pd.Series
        Variable number of daily returns series for different strategies
    names : list of str, optional
        Strategy names/labels. If None, defaults to "Strategy 1", "Strategy 2", etc.
    utilizations : list of pd.Series, optional
        Optional utilization series for each strategy. Must match length of returns_series.
        If provided, utilization plots will be included.
    
    Examples:
    --------
    # Compare two strategies
    plot_strategy_comparison(dist_returns, coint_returns, 
                            names=['Distance', 'Cointegration'],
                            utilizations=[dist_util, coint_util])
    
    # Compare multiple strategies
    plot_strategy_comparison(dist_returns, coint_returns, ou_returns, benchmark_returns,
                            names=['Distance', 'Cointegration', 'OU', 'Benchmark'])
    """
    num_strategies = len(returns_series)
    if num_strategies == 0:
        raise ValueError("At least one returns series must be provided")
    
    # Default names if not provided
    if names is None:
        names = [f"Strategy {i+1}" for i in range(num_strategies)]
    elif len(names) != num_strategies:
        raise ValueError(f"Number of names ({len(names)}) must match number of strategies ({num_strategies})")
    
    # Validate utilizations if provided
    if utilizations is not None:
        if len(utilizations) != num_strategies:
            raise ValueError(f"Number of utilizations ({len(utilizations)}) must match number of strategies ({num_strategies})")
        has_utilization = any(util is not None for util in utilizations)
    else:
        has_utilization = False
        utilizations = [None] * num_strategies
    
    # Determine grid layout based on number of strategies
    if num_strategies == 1:
        rows = 2 if has_utilization else 2
        cols = 1
    elif num_strategies == 2:
        rows = 3 if has_utilization else 2
        cols = 2
    else:
        # For 3+ strategies, use a more compact layout
        rows = 3 if has_utilization else 2
        cols = 2
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)
    
    # Equity curves - single plot with all strategies
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    for i, (returns, name, color) in enumerate(zip(returns_series, names, colors)):
        equity = (1 + returns).cumprod()
        ax1.plot(equity.index, equity.values, linewidth=2, label=name, color=color)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Equity')
    ax1.set_title("Equity Curves Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Drawdowns - plot up to 2 strategies side by side, rest below
    drawdown_axes = []
    if num_strategies <= 2:
        # Plot all strategies side by side
        for i, (returns, name) in enumerate(zip(returns_series[:2], names[:2])):
            ax = fig.add_subplot(gs[1, i])
            plot_drawdown(returns, title=f"{name} Drawdown", ax=ax)
            drawdown_axes.append(ax)
    else:
        # Plot first 2 strategies side by side
        for i in range(2):
            ax = fig.add_subplot(gs[1, i])
            plot_drawdown(returns_series[i], title=f"{names[i]} Drawdown", ax=ax)
            drawdown_axes.append(ax)
        
        # Plot remaining strategies in additional rows if needed
        remaining = num_strategies - 2
        if remaining > 0:
            # Add more rows for remaining strategies
            extra_rows = (remaining + 1) // 2
            for idx in range(remaining):
                row = 2 + (idx // 2)
                col = idx % 2
                if row < rows - (1 if has_utilization else 0):
                    ax = fig.add_subplot(gs[row, col])
                    strategy_idx = idx + 2
                    plot_drawdown(returns_series[strategy_idx], 
                                title=f"{names[strategy_idx]} Drawdown", ax=ax)
                    drawdown_axes.append(ax)
    
    # Utilization (if provided)
    if has_utilization:
        ax_util = fig.add_subplot(gs[-1, :])
        # Find first non-None utilization or use first returns series index
        util_index = None
        for util in utilizations:
            if util is not None:
                util_index = util.index
                break
        if util_index is None:
            util_index = returns_series[0].index
        
        for i, (util, name, color) in enumerate(zip(utilizations, names, colors)):
            if util is not None:
                ax_util.plot(util.index, util.values * 100, 
                           linewidth=1.5, label=name, color=color)
        
        # Add subtle background fill
        ax_util.axhspan(0, 100, alpha=0.05, color='gray', zorder=0)
        ax_util.set_xlabel('Date')
        ax_util.set_ylabel('Utilization (%)')
        ax_util.set_title("Utilization Comparison")
        ax_util.set_ylim(0, 100)
        ax_util.grid(True, alpha=0.3)
        ax_util.legend()
        
        # Format x-axis dates
        ax_util.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_util.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax_util.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
