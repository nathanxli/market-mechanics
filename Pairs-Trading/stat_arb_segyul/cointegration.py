# -----------------------------
# Cointegration approach (Engle–Granger / Vidyamurthy)
# -----------------------------

import itertools
import numpy as np
import pandas as pd
import os
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def select_pairs_cointegration(price_window, top_n_pairs=20, pvalue_threshold=0.05, save=False):
    """
    Formation step for cointegration approach.
    - Use log prices
    - Run Engle–Granger cointegration test (statsmodels.coint) on each pair
    - Keep pairs with p-value below threshold
    - Rank by p-value (strongest cointegration first)
    - For each, estimate hedge ratio via OLS and compute spread mean/std
    
    Parameters:
    -----------
    price_window : pd.DataFrame
        Price data for the formation period
    top_n_pairs : int
        Number of top pairs to select
    pvalue_threshold : float
        Maximum p-value threshold for cointegration test
    save : bool
        If True, save selected pairs to CSV file
    
    Returns:
    --------
    pair_info : list
        List of dictionaries with pair information
    """
    log_prices = np.log(price_window)
    tickers = log_prices.columns

    coint_list = []

    for i, j in itertools.combinations(range(len(tickers)), 2):
        x = log_prices.iloc[:, i]
        y = log_prices.iloc[:, j]

        # Engle–Granger cointegration test
        score, pvalue, _ = coint(y, x)  # y ~ x
        if np.isnan(pvalue):
            continue
        if pvalue < pvalue_threshold:
            coint_list.append((pvalue, tickers[i], tickers[j]))

    # Sort by ascending p-value (strongest cointegration first)
    coint_list.sort(key=lambda x: x[0])
    selected = coint_list[:top_n_pairs]

    pair_info = []
    for pval, x_ticker, y_ticker in selected:
        x = log_prices[x_ticker]
        y = log_prices[y_ticker]

        # OLS to estimate hedge ratio: y = a + beta * x
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[x_ticker]

        spread = y - beta * x
        mu = spread.mean()
        sigma = spread.std()

        pair_info.append(
            {
                "x": x_ticker,          # hedge stock
                "y": y_ticker,          # dependent stock
                "beta": beta,
                "spread_mean": mu,
                "spread_std": sigma,
                "pvalue": pval,
            }
        )
    
    # Save to CSV if requested
    if save:
        # Get time period from price_window
        start_date = price_window.index[0]
        end_date = price_window.index[-1]
        timestamp = datetime.now()
        
        # Prepare data for CSV
        csv_data = []
        for pval, x_ticker, y_ticker in selected:
            csv_data.append({
                "formation_start": start_date,
                "formation_end": end_date,
                "stock1": x_ticker,
                "stock2": y_ticker,
                "pvalue": pval,
                "saved_timestamp": timestamp,
            })
        
        # Create DataFrame
        df_save = pd.DataFrame(csv_data)
        
        # Ensure csv directory exists
        csv_dir = "csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        # File path
        csv_path = os.path.join(csv_dir, "cointegration_pairs.csv")
        
        # Append to CSV (create if doesn't exist)
        if os.path.exists(csv_path):
            df_save.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_save.to_csv(csv_path, mode='w', header=True, index=False)

    return pair_info

def trade_pairs_cointegration(price_window, pair_info, entry_z=2.0, exit_z=0.0, capital_per_pair=1.0):
    """
    Trading step for cointegration approach over a given trading window.
    - Spread = log(y) - beta * log(x)
    - Open long spread when z < -entry_z, short spread when z > entry_z
    - Long spread: long y, short x * beta (scaled to gross = 1)
    - Short spread: short y, long x * beta (scaled to gross = 1)
    - Returns daily portfolio returns, daily utilization, and number of trades.
    """
    prices = price_window
    dates = prices.index
    log_prices = np.log(prices)

    pairs_state = {}
    for info in pair_info:
        key = (info["x"], info["y"])
        beta = info["beta"]

        # raw weights: long spread -> +1*y, -beta*x
        w_y_long_raw, w_x_long_raw = 1.0, -beta
        denom_long = abs(w_y_long_raw) + abs(w_x_long_raw)
        scale_long = 1.0 / denom_long if denom_long > 0 else 0.0
        w_y_long = w_y_long_raw * scale_long
        w_x_long = w_x_long_raw * scale_long

        # short spread -> -1*y, +beta*x
        w_y_short_raw, w_x_short_raw = -1.0, beta
        denom_short = abs(w_y_short_raw) + abs(w_x_short_raw)
        scale_short = 1.0 / denom_short if denom_short > 0 else 0.0
        w_y_short = w_y_short_raw * scale_short
        w_x_short = w_x_short_raw * scale_short

        pairs_state[key] = {
            "info": info,
            "position": 0,  # 0 = flat, +1 = long spread, -1 = short spread
            "weights": {
                "long": (w_x_long, w_y_long),
                "short": (w_x_short, w_y_short),
            },
        }

    daily_returns = []
    daily_utilization = []
    num_trades = 0
    num_pairs = len(pair_info)
    total_capital = num_pairs * capital_per_pair

    for t in range(1, len(dates)):
        date = dates[t]
        prev_date = dates[t - 1]

        pnl_today = 0.0

        # 1) P&L from positions
        for (x_ticker, y_ticker), state in pairs_state.items():
            pos = state["position"]
            if pos == 0:
                continue

            if pos == 1:
                w_x, w_y = state["weights"]["long"]
            else:
                w_x, w_y = state["weights"]["short"]

            ret_x = prices[x_ticker].loc[date] / prices[x_ticker].loc[prev_date] - 1.0
            ret_y = prices[y_ticker].loc[date] / prices[y_ticker].loc[prev_date] - 1.0

            pair_capital = capital_per_pair
            pair_pnl = pair_capital * (w_x * ret_x + w_y * ret_y)
            pnl_today += pair_pnl

        # 2) Update positions based on today's z-score
        for (x_ticker, y_ticker), state in pairs_state.items():
            info = state["info"]
            beta = info["beta"]
            spread = log_prices[y_ticker].loc[date] - beta * log_prices[x_ticker].loc[date]
            z = (spread - info["spread_mean"]) / info["spread_std"] if info["spread_std"] > 0 else 0.0

            pos = state["position"]
            if pos == 0:
                if z > entry_z:
                    state["position"] = -1  # short spread
                    num_trades += 1
                elif z < -entry_z:
                    state["position"] = 1   # long spread
                    num_trades += 1
            else:
                if abs(z) < exit_z:
                    state["position"] = 0

        # 3) Portfolio return and utilization
        daily_return = pnl_today / total_capital if total_capital > 0 else 0.0
        daily_returns.append((date, daily_return))

        active_pairs = sum(1 for state in pairs_state.values() if state["position"] != 0)
        utilization = active_pairs / num_pairs if num_pairs > 0 else 0.0
        daily_utilization.append((date, utilization))

    daily_returns = pd.Series(
        [r for _, r in daily_returns],
        index=[d for d, _ in daily_returns],
        name="cointegration_returns",
    )

    utilization_series = pd.Series(
        [u for _, u in daily_utilization],
        index=[d for d, _ in daily_utilization],
        name="cointegration_utilization",
    )

    return daily_returns, utilization_series, num_trades

def backtest_cointegration(prices, formation_days=252, trading_days=126,
                           top_n_pairs=20, pvalue_threshold=0.05):
    """
    Full backtest loop for cointegration-based pairs trading.
    Returns:
        combined_returns  : Series of daily portfolio returns
        combined_util     : Series of daily utilization (0–1)
        total_trades      : total number of entries
    """
    all_returns = []
    all_utilization = []
    total_trades = 0

    num_days = len(prices)
    step = trading_days
    for start in range(0, num_days - formation_days - trading_days, step):
        formation = prices.iloc[start : start + formation_days]
        trading = prices.iloc[start + formation_days : start + formation_days + trading_days]

        pair_info = select_pairs_cointegration(
            formation,
            top_n_pairs=top_n_pairs,
            pvalue_threshold=pvalue_threshold,
        )

        if not pair_info:
            continue

        window_returns, window_utilization, n_trades = trade_pairs_cointegration(
            trading,
            pair_info,
        )
        all_returns.append(window_returns)
        all_utilization.append(window_utilization)
        total_trades += n_trades

    if not all_returns:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0

    combined_ret = pd.concat(all_returns, axis=0)
    combined_ret = combined_ret[~combined_ret.index.duplicated(keep="first")]
    combined_ret = combined_ret.sort_index()
    combined_ret = combined_ret.asfreq("B", method=None).fillna(0.0)

    combined_util = pd.concat(all_utilization, axis=0)
    combined_util = combined_util[~combined_util.index.duplicated(keep="first")]
    combined_util = combined_util.sort_index()
    combined_util = combined_util.asfreq("B", method=None).fillna(0.0)

    return combined_ret, combined_util, total_trades
