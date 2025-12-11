# -----------------------------
# Distance approach (Gatev et al.)
# -----------------------------

import itertools
import numpy as np
import pandas as pd
import os
from datetime import datetime

def select_pairs_distance(price_window, top_n_pairs=20, save=True):
    """
    Formation step for distance approach.
    - Normalize prices to 1 at start
    - Compute sum of squared deviations (SSD) for each pair
    - Select top N pairs with smallest SSD
    - Precompute spread mean/std from formation period
    
    Parameters:
    -----------
    price_window : pd.DataFrame
        Price data for the formation period
    top_n_pairs : int
        Number of top pairs to select
    save : bool
        If True, save selected pairs to CSV file
    
    Returns:
    --------
    pair_info : list
        List of dictionaries with pair information
    """
    normalized = price_window / price_window.iloc[0]
    tickers = normalized.columns
    ssd_list = []

    for i, j in itertools.combinations(range(len(tickers)), 2):
        a = normalized.iloc[:, i]
        b = normalized.iloc[:, j]
        ssd = np.sum((a - b) ** 2)
        ssd_list.append((ssd, tickers[i], tickers[j]))

    # Sort by distance and take best pairs
    ssd_list.sort(key=lambda x: x[0])
    selected = ssd_list[:top_n_pairs]

    pair_info = []
    for ssd, t1, t2 in selected:
        spread = normalized[t1] - normalized[t2]
        mu = spread.mean()
        sigma = spread.std()
        pair_info.append(
            {
                "stock1": t1,
                "stock2": t2,
                "spread_mean": mu,
                "spread_std": sigma,
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
        for ssd, t1, t2 in selected:
            csv_data.append({
                "formation_start": start_date,
                "formation_end": end_date,
                "stock1": t1,
                "stock2": t2,
                "ssd": ssd,
                "saved_timestamp": timestamp,
            })
        
        # Create DataFrame
        df_save = pd.DataFrame(csv_data)
        
        # Ensure csv directory exists
        csv_dir = "csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        # File path
        csv_path = os.path.join(csv_dir, "distance_pairs.csv")
        
        # Append to CSV (create if doesn't exist)
        if os.path.exists(csv_path):
            df_save.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_save.to_csv(csv_path, mode='w', header=True, index=False)

    return pair_info

def trade_pairs_distance(price_window, pair_info, entry_z=2.0, exit_z=0.0, capital_per_pair=1.0):
    prices = price_window
    dates = prices.index

    # Normalize again on first trading day
    normalized = prices / prices.iloc[0]

    # Initialize state per pair
    pairs_state = {}
    for info in pair_info:
        key = (info["stock1"], info["stock2"])
        pairs_state[key] = {
            "info": info,
            "position": 0,  # 0 = flat, +1 = long spread, -1 = short spread
        }

    daily_returns = []
    daily_utilization = []  # fraction of pairs with non-zero position
    num_trades = 0
    num_pairs = len(pair_info)
    total_capital = num_pairs * capital_per_pair

    for t in range(1, len(dates)):
        date = dates[t]
        prev_date = dates[t - 1]

        pnl_today = 0.0

        # 1) P&L from yesterday's positions
        for (s1, s2), state in pairs_state.items():
            pos = state["position"]
            if pos == 0:
                continue

            if pos == 1:  # long spread (long s1, short s2)
                w1, w2 = 0.5, -0.5
            else:         # short spread (short s1, long s2)
                w1, w2 = -0.5, 0.5

            ret1 = prices[s1].loc[date] / prices[s1].loc[prev_date] - 1.0
            ret2 = prices[s2].loc[date] / prices[s2].loc[prev_date] - 1.0

            pair_capital = capital_per_pair
            pair_pnl = pair_capital * (w1 * ret1 + w2 * ret2)
            pnl_today += pair_pnl

        # 2) Update positions based on today's z-score
        for (s1, s2), state in pairs_state.items():
            info = state["info"]
            spread = normalized[s1].loc[date] - normalized[s2].loc[date]
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

        # 3) Compute portfolio return and utilization for this day
        daily_return = pnl_today / total_capital if total_capital > 0 else 0.0
        daily_returns.append((date, daily_return))

        active_pairs = sum(1 for state in pairs_state.values() if state["position"] != 0)
        utilization = active_pairs / num_pairs if num_pairs > 0 else 0.0
        daily_utilization.append((date, utilization))

    daily_returns = pd.Series(
        [r for _, r in daily_returns],
        index=[d for d, _ in daily_returns],
        name="distance_returns",
    )

    utilization_series = pd.Series(
        [u for _, u in daily_utilization],
        index=[d for d, _ in daily_utilization],
        name="distance_utilization",
    )

    return daily_returns, utilization_series, num_trades

def backtest_distance(prices, formation_days=252, trading_days=126, top_n_pairs=20):
    all_returns = []
    all_utilization = []
    total_trades = 0

    num_days = len(prices)
    step = trading_days
    for start in range(0, num_days - formation_days - trading_days, step):
        formation = prices.iloc[start : start + formation_days]
        trading = prices.iloc[start + formation_days : start + formation_days + trading_days]

        pair_info = select_pairs_distance(formation, top_n_pairs=top_n_pairs)
        window_returns, window_utilization, n_trades = trade_pairs_distance(trading, pair_info)
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
