# ============================================================
# Time-series OU approach (Bertram-style, discretized)
# ============================================================
# Idea:
# 1) For a given pair spread, assume it follows a zero-mean OU process.
# 2) Estimate OU parameters via AR(1) on the centered spread.
# 3) Transform spread to dimensionless z_t = X_t / sigma_stat.
# 4) On the formation window, search over entry/exit thresholds (a, m)
#    and pick those maximizing realized Sharpe (or return per unit time).
# 5) In the trading window, use pair-specific thresholds and the same
#    long/short spread trading rule as for distance/cointegration.


# ------------------------------------------------------------
# OU parameter estimation
# ------------------------------------------------------------

import numpy as np
import pandas as pd

def estimate_ou_params(spread_series):
    """
    Estimate OU parameters for a (mean-reverting) spread via AR(1).

    We assume a zero-mean OU process:
        dX_t = -rho * X_t dt + sigma dW_t

    Discrete approximation:
        X_{t+1} = phi * X_t + eps_t
    where phi = e^{-rho}.

    Parameters
    ----------
    spread_series : pd.Series
        Time series of the spread (e.g., price difference or log spread).

    Returns
    -------
    params : dict
        {
            "phi": AR(1) coefficient,
            "rho": speed of mean reversion (>0 for OU),
            "sigma_eps": std of AR residuals,
            "sigma_stat": stationary std of X_t,
            "half_life": half-life in time steps
        }
        If mean reversion is not detected (phi <= 0 or phi >= 1),
        rho, half_life, sigma_stat may be np.nan.
    """
    # Center spread to approximate zero-mean OU
    x = spread_series - spread_series.mean()
    x = x.dropna()

    x_lag = x.shift(1).dropna()
    x_curr = x.loc[x_lag.index]

    # AR(1) coefficient phi
    denom = np.sum(x_lag ** 2)
    if denom == 0 or len(x_lag) < 2:
        return {
            "phi": np.nan,
            "rho": np.nan,
            "sigma_eps": np.nan,
            "sigma_stat": np.nan,
            "half_life": np.nan,
        }

    phi = np.sum(x_lag * x_curr) / denom

    # Residuals and their std
    eps = x_curr - phi * x_lag
    sigma_eps = eps.std(ddof=1)

    # Stationary variance of AR(1): sigma_eps^2 / (1 - phi^2)
    if abs(phi) < 1:
        var_stat = sigma_eps ** 2 / (1.0 - phi ** 2)
        sigma_stat = np.sqrt(var_stat)
    else:
        sigma_stat = np.nan

    # OU speed and half-life
    if 0 < phi < 1:
        rho = -np.log(phi)
        half_life = np.log(2.0) / rho
    else:
        rho = np.nan
        half_life = np.nan

    return {
        "phi": float(phi),
        "rho": float(rho),
        "sigma_eps": float(sigma_eps),
        "sigma_stat": float(sigma_stat),
        "half_life": float(half_life),
    }


# ------------------------------------------------------------
# Utility: compute z-score series from spread and OU params
# ------------------------------------------------------------

def compute_ou_zscores(spread_series, ou_params):
    """
    Compute dimensionless OU z-scores:
        z_t = (X_t - mean(X)) / sigma_stat

    Here X_t is the centered spread; mean(X) is already removed in
    the OU estimation step, so we just divide by sigma_stat.

    Parameters
    ----------
    spread_series : pd.Series
        Spread time series.
    ou_params : dict
        Output of estimate_ou_params().

    Returns
    -------
    z : pd.Series
        Standardized OU z-scores.
    """
    sigma_stat = ou_params.get("sigma_stat", np.nan)
    if not np.isfinite(sigma_stat) or sigma_stat <= 0:
        # Fall back to simple standard deviation
        sigma_stat = spread_series.std(ddof=1)
        if sigma_stat == 0 or not np.isfinite(sigma_stat):
            return pd.Series(0.0, index=spread_series.index, name="z")

    x_centered = spread_series - spread_series.mean()
    z = x_centered / sigma_stat
    z.name = "z"
    return z


# ------------------------------------------------------------
# Simulate OU trading for a single pair and fixed (a, m)
# ------------------------------------------------------------

def simulate_ou_pair_trading(
    prices_pair,
    z_scores,
    entry_a,
    exit_m,
    capital_per_pair=1.0,
    transaction_cost=0.0,
    max_holding_days=None,
):
    """
    Simulate OU-based trading for a single pair with thresholds (a, m).

    Trading rule (Bertram-style, discretized):
    - z_t is the standardized spread (z-scores).
    - Outer band a > 0, inner band 0 <= m < a.
    - Long spread:
        Enter when z_t <= -a
        Exit when z_t >= -m
    - Short spread:
        Enter when z_t >= a
        Exit when z_t <= m

    We assume:
    - prices_pair has two columns: [stock1, stock2]
    - For simplicity, we use 50/50 capital allocation for long/short legs.

    Parameters
    ----------
    prices_pair : pd.DataFrame
        Two-column DataFrame with daily prices for the pair.
    z_scores : pd.Series
        OU z-scores for the spread, indexed identically to prices_pair.
    entry_a : float
        Outer entry threshold in z-units (> 0).
    exit_m : float
        Inner exit threshold in z-units, 0 <= exit_m < entry_a.
    capital_per_pair : float
        Notional allocated to this pair.
    transaction_cost : float
        Relative transaction cost per completed round-trip (per capital 1).
        Applied as a one-off hit at each entry (approximation).
    max_holding_days : int or None
        Optional maximum holding period; if not None, close any open
        position after this many days.

    Returns
    -------
    daily_returns : pd.Series
        Daily returns series for this pair, given thresholds (a, m).
    num_trades : int
        Number of entries (round-trips started) for this pair.
    """
    dates = prices_pair.index
    s1, s2 = prices_pair.columns[0], prices_pair.columns[1]

    # Align z-scores to prices
    z = z_scores.reindex(dates).ffill().bfill()

    # Position state
    position = 0       # 0 = flat, +1 = long spread, -1 = short spread
    days_in_position = 0
    daily_pnl = []
    num_trades = 0

    for t in range(1, len(dates)):
        date = dates[t]
        prev_date = dates[t - 1]

        # Compute P&L from yesterday's position
        pnl_today = 0.0
        if position != 0:
            # Simple 50/50 weights in each leg
            if position == 1:
                w1, w2 = 0.5, -0.5  # long s1, short s2
            else:
                w1, w2 = -0.5, 0.5  # short s1, long s2

            ret1 = prices_pair[s1].loc[date] / prices_pair[s1].loc[prev_date] - 1.0
            ret2 = prices_pair[s2].loc[date] / prices_pair[s2].loc[prev_date] - 1.0
            pnl_today = capital_per_pair * (w1 * ret1 + w2 * ret2)
            days_in_position += 1

        # Today's z-score
        z_t = z.loc[date]

        # Exit conditions
        if position == 1:
            # Long spread: close if z >= -m or max holding exceeded
            if z_t >= -exit_m or (max_holding_days is not None and days_in_position >= max_holding_days):
                position = 0
                days_in_position = 0

        elif position == -1:
            # Short spread: close if z <= m or max holding exceeded
            if z_t <= exit_m or (max_holding_days is not None and days_in_position >= max_holding_days):
                position = 0
                days_in_position = 0

        # Entry conditions (only if flat)
        if position == 0:
            if z_t >= entry_a:
                # Enter short spread
                position = -1
                days_in_position = 0
                num_trades += 1
                # Apply transaction cost at entry (approximate)
                pnl_today -= capital_per_pair * transaction_cost
            elif z_t <= -entry_a:
                # Enter long spread
                position = 1
                days_in_position = 0
                num_trades += 1
                pnl_today -= capital_per_pair * transaction_cost

        # Normalize by capital_per_pair to get return
        daily_ret = pnl_today / capital_per_pair
        daily_pnl.append((date, daily_ret))

    daily_returns = pd.Series(
        [r for _, r in daily_pnl],
        index=[d for d, _ in daily_pnl],
        name="ou_pair_returns",
    )
    return daily_returns, num_trades


# ------------------------------------------------------------
# Threshold optimization for a single pair on formation window
# ------------------------------------------------------------

def optimize_ou_thresholds_for_pair(
    prices_pair,
    spread_series,
    a_grid=None,
    m_grid=None,
    transaction_cost=0.0,
    min_trades=5,
):
    """
    Search over (entry_a, exit_m) thresholds for a single pair and pick the best
    according to realized Sharpe ratio on the formation window.

    Parameters
    ----------
    prices_pair : pd.DataFrame
        Two-column DataFrame with prices for the pair (formation window).
    spread_series : pd.Series
        Spread time series for the pair (same index as prices_pair).
    a_grid : list or np.array, optional
        Grid of candidate entry thresholds in z-units (>0).
        If None, defaults to [1.0, 1.5, 2.0, 2.5, 3.0].
    m_grid : list or np.array, optional
        Grid of candidate exit thresholds in z-units (>=0).
        If None, defaults to [0.0, 0.5, 1.0, 1.5, 2.0].
    transaction_cost : float
        Relative cost applied at each entry.
    min_trades : int
        Discard threshold combinations that generate fewer than this
        number of trades in the formation window.

    Returns
    -------
    best_params : dict
        {
            "entry_a": best_entry_a,
            "exit_m": best_exit_m,
            "ou_params": ou_params,
            "sharpe_annual": best_sharpe,
            "num_trades": trades_at_optimum
        }
        If OU estimation fails, returns default thresholds (e.g., 2.0, 0.0)
        and Sharpe = 0.
    """
    # Default grids
    if a_grid is None:
        a_grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    if m_grid is None:
        m_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    # Estimate OU parameters
    ou_params = estimate_ou_params(spread_series)
    if not np.isfinite(ou_params.get("rho", np.nan)) or ou_params["rho"] <= 0:
        # Fallback: OU not meaningful
        return {
            "entry_a": 2.0,
            "exit_m": 0.0,
            "ou_params": ou_params,
            "sharpe_annual": 0.0,
            "num_trades": 0,
        }

    z = compute_ou_zscores(spread_series, ou_params)

    best_sharpe = -np.inf
    best_a = None
    best_m = None
    best_trades = 0

    # Search over grid of (a, m) with 0 <= m < a
    for a in a_grid:
        if a <= 0:
            continue
        for m in m_grid:
            if m < 0 or m >= a:
                continue

            returns, n_trades = simulate_ou_pair_trading(
                prices_pair,
                z,
                entry_a=a,
                exit_m=m,
                capital_per_pair=1.0,
                transaction_cost=transaction_cost,
                max_holding_days=None,
            )

            if n_trades < min_trades:
                continue

            r = returns.dropna()
            if r.empty:
                continue

            mean_ret = r.mean()
            std_ret = r.std()
            if std_ret == 0:
                continue

            sharpe = (mean_ret / std_ret) * np.sqrt(252.0)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_a = a
                best_m = m
                best_trades = n_trades

    if best_a is None:
        # No viable combination found; fall back to defaults
        return {
            "entry_a": 2.0,
            "exit_m": 0.0,
            "ou_params": ou_params,
            "sharpe_annual": 0.0,
            "num_trades": 0,
        }

    return {
        "entry_a": float(best_a),
        "exit_m": float(best_m),
        "ou_params": ou_params,
        "sharpe_annual": float(best_sharpe),
        "num_trades": int(best_trades),
    }


# ------------------------------------------------------------
# High-level: OU-based time-series backtest for multiple pairs
# ------------------------------------------------------------

def backtest_ou_time_series(
    prices,
    pair_info,
    formation_days=252,
    trading_days=126,
    transaction_cost=0.0,
    top_n_pairs=None,
):
    """
    OU-based time-series backtest over rolling formation/trading windows.

    For each window:
    1) We start from a list of candidate pairs (pair_info) defined on the
       full universe. Typically this is the output of select_pairs_distance()
       or select_pairs_cointegration() on the formation window.
    2) For each pair, we:
       - compute its spread in the formation window,
       - estimate OU parameters,
       - optimize thresholds (a, m) on the formation window.
    3) In the trading window, we trade using these pair-specific thresholds.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel (universe of stocks), indexed by date, columns = tickers.
    pair_info : list of dict
        Candidate pairs to be considered in each window. Each dict should at least
        contain either:
        - {"stock1": ..., "stock2": ...} for distance-like spreads, or
        - {"x": ..., "y": ..., "beta": ...} for cointegration-like spreads.
        The same list is used structurally; actual spreads are recalculated
        on each rolling window.
    formation_days : int
        Length of formation window in trading days.
    trading_days : int
        Length of trading window in trading days.
    transaction_cost : float
        Relative transaction cost per entry (per unit capital).
    top_n_pairs : int or None
        Maximum number of pairs to trade per window (use first N of pair_info).
        If None, use all pairs in pair_info.

    Returns
    -------
    ou_returns : pd.Series
        OU portfolio daily returns across all windows.
    total_trades : int
        Total number of entries across all pairs/windows.
    """
    if top_n_pairs is not None:
        base_pairs = pair_info[:top_n_pairs]
    else:
        base_pairs = pair_info

    all_returns = []
    total_trades = 0

    num_days = len(prices)
    step = trading_days

    for start in range(0, num_days - formation_days - trading_days, step):
        formation = prices.iloc[start : start + formation_days]
        trading = prices.iloc[start + formation_days : start + formation_days + trading_days]

        # Per-window results aggregated across pairs
        window_pnl = pd.Series(0.0, index=trading.index)
        window_trades = 0
        num_pairs_used = 0

        for pair in base_pairs:
            # Determine pair tickers and spread definition
            if "stock1" in pair and "stock2" in pair:
                # Distance-style: normalized price difference
                s1, s2 = pair["stock1"], pair["stock2"]
                if s1 not in formation.columns or s2 not in formation.columns:
                    continue
                pair_prices_form = formation[[s1, s2]].dropna()
                if len(pair_prices_form) < 50:
                    continue
                # Use normalized price difference as spread
                norm_form = pair_prices_form / pair_prices_form.iloc[0]
                spread_form = norm_form[s1] - norm_form[s2]

            elif "x" in pair and "y" in pair and "beta" in pair:
                # Cointegration-style: log spread y - beta * x
                x, y, beta = pair["x"], pair["y"], pair["beta"]
                if x not in formation.columns or y not in formation.columns:
                    continue
                pair_prices_form = formation[[x, y]].dropna()
                if len(pair_prices_form) < 50:
                    continue
                log_form = np.log(pair_prices_form)
                spread_form = log_form[y] - beta * log_form[x]
                s1, s2 = x, y  # for trading P&L, we still need tickers
            else:
                # Unknown pair structure; skip
                continue

            # Optimize OU thresholds on formation window for this pair
            opt = optimize_ou_thresholds_for_pair(
                pair_prices_form,
                spread_form,
                a_grid=None,
                m_grid=None,
                transaction_cost=transaction_cost,
                min_trades=3,
            )

            entry_a = opt["entry_a"]
            exit_m = opt["exit_m"]

            # Now trade this pair in the trading window with fixed (a, m)
            if s1 not in trading.columns or s2 not in trading.columns:
                continue
            pair_prices_trade = trading[[s1, s2]].dropna()
            if len(pair_prices_trade) < 10:
                continue

            # Recompute OU z-scores on formation+trading to ensure continuity
            # (here we just compute on formation window and reuse mean/std;
            # for simplicity, use the formation OU params on trading window.)
            ou_params = opt["ou_params"]
            spread_all = pd.concat([spread_form, spread_form.iloc[[-1]]])  # placeholder to reuse params
            z_dummy = compute_ou_zscores(spread_all, ou_params)
            # Recompute z on trading prices using same mean and sigma_stat:
            # Spread definition must be consistent between formation and trading:
            if "stock1" in pair and "stock2" in pair:
                norm_all_trade = pair_prices_trade / pair_prices_trade.iloc[0]
                spread_trade = norm_all_trade[s1] - norm_all_trade[s2]
            else:
                log_trade = np.log(pair_prices_trade)
                spread_trade = log_trade[s2] - pair["beta"] * log_trade[s1]

            # Use the same sigma_stat and center as in formation:
            sigma_stat = ou_params.get("sigma_stat", np.nan)
            if not np.isfinite(sigma_stat) or sigma_stat <= 0:
                # Fall back to simple std
                sigma_stat = spread_form.std(ddof=1) if spread_form.std(ddof=1) > 0 else 1.0
            spread_center = spread_form.mean()
            z_trade = (spread_trade - spread_center) / sigma_stat
            z_trade.name = "z"

            returns_pair, n_trades_pair = simulate_ou_pair_trading(
                pair_prices_trade,
                z_trade,
                entry_a=entry_a,
                exit_m=exit_m,
                capital_per_pair=1.0,
                transaction_cost=transaction_cost,
                max_holding_days=None,
            )

            # Align to window index and aggregate P&L equally across pairs
            returns_pair = returns_pair.reindex(window_pnl.index).fillna(0.0)
            window_pnl += returns_pair
            window_trades += n_trades_pair
            num_pairs_used += 1

        # If we traded at least one pair in this window, average across pairs
        if num_pairs_used > 0:
            window_returns = window_pnl / float(num_pairs_used)
            all_returns.append(window_returns)
            total_trades += window_trades

    if not all_returns:
        return pd.Series(dtype=float), 0

    ou_returns = pd.concat(all_returns, axis=0)
    ou_returns = ou_returns[~ou_returns.index.duplicated(keep="first")]
    ou_returns = ou_returns.sort_index()
    ou_returns = ou_returns.asfreq("B", method=None).fillna(0.0)
    ou_returns.name = "ou_time_series_returns"

    return ou_returns, total_trades
