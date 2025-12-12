from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from utils import *
from get_data_usa import *

def standardize_returns(mat: pd.DataFrame, ddof: int = 1):
    means = mat.mean(axis=0) # Mean of all returns R_i (i=1,2,...,N) for each ticker ; N = all stocks in the universe.
    stds = mat.std(axis=0, ddof=ddof) # STD of returns R_i (i=1,2,...,N) for each ticker ; N = all stocks in the universe.

    # Drop columns with zero or NaN std dev (no variation or bad data)
    valid = stds > 0
    if not valid.all():
        dropped = list(stds.index[~valid])
        print(f"Dropping zero-variance or invalid columns: {dropped}")

    mat_valid = mat.loc[:, valid] # Keeps only valid rows in the returns matrix
    means_valid = means[valid] # Keeps only valid means
    stds_valid = stds[valid] # Keeps only valid std

    Y = (mat_valid - means_valid) / stds_valid # Standardize

    return Y, means_valid, stds_valid

def correlation_from_standardized(Y: pd.DataFrame) -> pd.DataFrame:
    M = Y.shape[0]
    Y_values = Y.values # Numpy array view
    Sigma = (Y_values.T @ Y_values) / (M - 1) # Sigma_{ij} = (1 / (M-1)) sum_t Y_{ti} Y_{tj}
    return pd.DataFrame(Sigma, index=Y.columns, columns=Y.columns)

def preprocess_for_pca(mat: pd.DataFrame):
    Y, means, stds = standardize_returns(mat)
    Sigma = correlation_from_standardized(Y)
    return Y, Sigma, means, stds

def pca_from_sigma(
    Sigma: pd.DataFrame,
    max_factors: int | None = None,
):
    # Ensure symmetric numeric matrix
    Sigma_vals = Sigma.values
    tickers = Sigma.index.to_list()
    N = Sigma_vals.shape[0]

    evals, evecs = np.linalg.eigh(Sigma_vals) # Gets the eigen-decomposition of the symmetric matrix

    idx = np.argsort(evals)[::-1] # Sorts eigenvalues/vectors in descending order
    evals = evals[idx]
    evecs = evecs[:, idx]

    if max_factors is not None and max_factors < N:
        evals = evals[:max_factors]
        evecs = evecs[:, :max_factors]
    
    # Converts the eigenvalues into vectors
    factor_names = [f"f{k+1}" for k in range(len(evals))]
    eigenvalues = pd.Series(evals, index=factor_names, name="eigenvalue")
    eigenvectors = pd.DataFrame(
        evecs,
        index=tickers,
        columns=factor_names
    )

    # Since it's correlation matrix sum(evals) = total variance = N
    explained_var_ratio = eigenvalues / eigenvalues.sum()
    explained_var_ratio.name = "explained_variance_ratio"

    return eigenvalues, eigenvectors, explained_var_ratio

# Equation 9 in the paper
# Note: Remember this equation converts factor loadings into a timeseries
def compute_eigenportfolio_returns(
    mat: pd.DataFrame, # M x N Matrix (rows = Days ; cols = Tickers)
    stds: pd.Series, # N sized Vector
    eigenvectors: pd.DataFrame, # N x m PCA eigenvectors (rows = tickers, cols = factors f1..fm)
):
    # Aligning stds and eigenvectors to the return matrix (M x N)
    tickers = mat.columns.to_list()
    stds_vec = stds.reindex(tickers)
    if stds_vec.isna().any():
        missing = list(stds_vec[stds_vec.isna()].index)
        raise ValueError(f"stds missing for tickers: {missing}")
    V = eigenvectors.reindex(index=tickers)
    if V.isna().any().any():
        raise ValueError("eigenvectors contain NaNs after reindexing to mat's columns.")

    # Convert eigenvectors → eigenportfolio weights
    sigma_vals = stds_vec.values[:, None] # shape: (N, 1)
    W = V.values / sigma_vals # shape: (N, m)

    # The equation 9
    R_vals = mat.values
    P_vals = R_vals @ W

    # Wrapping our results back in pandas df
    factor_names = V.columns.to_list()
    dates = mat.index

    ep_returns = pd.DataFrame(P_vals, index=dates, columns=factor_names) # Rows dates ; columns factors
    ep_weights = pd.DataFrame(W, index=tickers, columns=factor_names) # Rows dates ; columns factors

    return ep_returns, ep_weights

# Equation 10
def compute_factor_neutral_residuals(
    mat: pd.DataFrame, # Raw Returns ; Do not use the normalized returns (aka Y)
    ep_returns: pd.DataFrame,
    n_factors: int | None = None,
    lookback: int = 60
):
    mat = mat.tail(lookback)
    ep_returns = ep_returns.tail(lookback)

    common_idx = mat.index.intersection(ep_returns.index) # Finds the intersection of dates and then aligns dates (rows) between mat and factor returns
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between mat and ep_returns.")

    # Actually gets intersect and aligns the DFs
    mat_aligned = mat.loc[common_idx]
    F_aligned = ep_returns.loc[common_idx]

    # Retains the top n_factors (based on variance explained)
    if n_factors is not None:
        F_aligned = F_aligned.iloc[:, :n_factors]

    mat_values = mat_aligned.values # M x N
    F_mat = F_aligned.values # M x K_used

    M, N = mat_values.shape
    M_F, K_used = F_mat.shape
    if M_F != M:
        raise ValueError("Row mismatch after alignment between mat and ep_returns.")

    # TODO: Add Intercept?
    # Compute betas via OLS: B = (F^T * F)^{-1} * F^T * mat
    FtF = F_mat.T @ F_mat # = F^T * F ; Returns a K_used x K_used matrix
    FtMat = F_mat.T @ mat_values # = F^T * mat ; Returns a K_used x N matrix

    # Use pseudo-inverse for numerical stability in case of near-singular FtF
    FtF_inv = np.linalg.pinv(FtF) # ~ = (F^T * F)^{-1}
    B_mat = FtF_inv @ FtMat #  = (F^T * F)^{-1} * F^T * mat (The linear regression formula); Returns a K_used x N matrix

    # Residuals: E = mat - F B
    E_mat = mat_values - F_mat @ B_mat # M x N

    factor_names = F_aligned.columns.to_list()
    tickers = mat_aligned.columns.to_list()

    betas = pd.DataFrame(B_mat, index=factor_names, columns=tickers)
    residuals = pd.DataFrame(E_mat, index=common_idx, columns=tickers)

    return residuals, betas

def estimate_ou_from_residuals(
    residuals: pd.DataFrame,
    window: int = 60,
    min_obs: int = 60,
    annualization: int = 252,
):
    """
    NOTE:
        - phi = b  (AR(1) coefficient)
        - eps_var = Var(η)  (variance of AR(1))
        - sigma_stat = sigma_eq = sqrt(Var(η) / (1 - b^2))
        - half_life = ln(2) / κ   where κ = -ln(b) * 252
        - ou_z = s-score = (X_last - μ_centered) / sigma_eq
    """
    residuals = residuals.sort_index() # Orders by date (rows = date ; cols = tickers)
    residuals_win = residuals.tail(window) # Gets the latest window residuals
    tickers = residuals_win.columns

    phi_dict = {}
    eps_var_dict = {}
    sigma_stat_dict = {}
    half_life_dict = {}
    ou_z_dict = {}
    kappa_dict = {}

    # Temporary storage to allow μ-centering later
    mu_temp = {}
    X_last_temp = {}
    sigma_eq_temp = {}

    # ---- First pass: estimate AR(1) params -------------------------------
    for t in tickers:
        eps = residuals_win[t].dropna().values

        if len(eps) < min_obs:
            continue

        # X_k = cumulative residual (Appendix A)
        X = np.cumsum(eps)

        X_n = X[:-1] # X_{n}
        X_np1 = X[1:] # X_{n+1} ; X shifted by 1 for the regression (becomes the Y for linear regression)

        # OLS with intercept: X_{n+1} = a + b X_n + η_{n+1}
        Z = np.column_stack([np.ones_like(X_n), X_n]) # Builds the X matrix with coeeficient padding [[1, X_1], [1, X_2], [1, X_3]]
        theta, *_ = np.linalg.lstsq(Z, X_np1, rcond=None) # Theta = regression coefficients
        a, b = theta

        eta = X_np1 - (a + b * X_n) # Compute residuals of this regression
        var_eta = np.mean(eta ** 2) # Get their variance

        # Storing data
        phi_dict[t] = b
        eps_var_dict[t] = var_eta

        # If b is not stationary, set NaNs for OU parameters
        # b = e^{-k * dt} ; 0 < e^{-k * dt} < 1 ; k > 0 & dt > 0 ; Thus, e^{-k * dt} = e^{some negative} and 0 < e^{some negative} < 1
        # When 0 < b < 1: OU valid ; b >= 1: non-stationary and OU invalid ; b <= 0: Oscillating? OU invalid
        if not (0 < b < 1):
            sigma_stat_dict[t] = np.nan
            half_life_dict[t] = np.nan
            ou_z_dict[t] = np.nan
            continue

        # OU parameters as in Appendix A
        sigma_eq_sq = var_eta / (1 - b**2)
        sigma_eq = np.sqrt(sigma_eq_sq) if sigma_eq_sq > 0 else np.nan # Remember: The denominator for s-scores
        kappa = -np.log(b) * annualization
        half_life = np.log(2.0) / kappa # In Years
        mu = a / (1 - b)

        # store temporarily for μ-centering
        mu_temp[t] = mu
        X_last_temp[t] = X[-1]
        sigma_eq_temp[t] = sigma_eq
        kappa_dict[t] = kappa

        sigma_stat_dict[t] = sigma_eq
        half_life_dict[t] = half_life

    # ---- Second pass: μ-centering (Eq. A2) s-score generation -------------------------------
    if len(mu_temp) > 0:
        mu_mean = np.mean(list(mu_temp.values()))
    else:
        mu_mean = np.nan

    for t in tickers:
        b = phi_dict.get(t, np.nan)
        sigma_eq = sigma_eq_temp.get(t, np.nan)
        X_last = X_last_temp.get(t, np.nan)
        mu_raw = mu_temp.get(t, np.nan)

        if np.isnan(b) or np.isnan(sigma_eq) or sigma_eq <= 0 or np.isnan(mu_raw) or np.isnan(X_last):
            ou_z_dict[t] = np.nan
            continue

        # centered μ (Eq. A2)
        mu_centered = mu_raw - mu_mean

        # final Z-score: (X_last – μ_centered) / σ_eq # Equation 15 from section 4
        # ou_z = (X_last - mu_centered) / sigma_eq 
        ou_z =  -mu_centered / sigma_eq
        ou_z_dict[t] = ou_z

    # ---- Convert to Series and return with legacy keys -------------------
    phi_ser = pd.Series(phi_dict, name="phi")
    eps_var_ser = pd.Series(eps_var_dict, name="eps_var")
    sigma_stat_ser = pd.Series(sigma_stat_dict, name="sigma_stat")
    half_life_ser = pd.Series(half_life_dict, name="half_life")
    ou_z_ser = pd.Series(ou_z_dict, name="ou_z")
    kappa_ser = pd.Series(kappa_dict, name="kappa")

    return {
        "phi": phi_ser,
        "eps_var": eps_var_ser,
        "sigma_stat": sigma_stat_ser,
        "half_life": half_life_ser,
        "ou_z": ou_z_ser,
        "kappa_ser": kappa_ser,
    }

# Applied Equation 16
def apply_bang_bang_rule(
    s_score: pd.Series,
    phi: pd.Series,
    mean_reversion_period_in_days: int = 30,
    annualization: int = 252,
    sbo: float = 1.25,
    sso: float = 1.25,
    sbc: float = 0.75,
    ssc: float = 0.50,
) -> pd.DataFrame:
    
    tau = 1.0 / (-np.log(phi) * annualization)

    common_tickers = s_score.index.intersection(tau.index).intersection(phi.index)
    s_score = s_score.loc[common_tickers]
    tau = tau.loc[common_tickers]

    valid_mask = (
        (~s_score.isna()) &
        (~tau.isna())
    )

    s_score = s_score[valid_mask]
    tau = tau[valid_mask]

    fast_mask = (tau > 0) & (tau < mean_reversion_period_in_days/annualization)
    s_score = s_score[fast_mask]

    potential_positions = defaultdict(set)
    tickers = s_score.index
    for ticker in tickers:
        s = s_score[ticker]

        # Buy
        # s < -1.25
        if s < -sbo:
            potential_positions[ticker].add(2)
        # Sell
        # s > 1.25
        if s > +sso:
            potential_positions[ticker].add(1)
        
        # close Short
        # s < 0.75
        if s < +sbc:
            potential_positions[ticker].add(-1)
        
        # close Long
        # s > 0.5
        if s > -ssc:
            potential_positions[ticker].add(-2)

    return potential_positions

def build_return_matrix_for_date(
    trade_date: pd.Timestamp,
    components_df: pd.DataFrame, # DF of the market's index's components for each month
    lookback_returns: int = 252, # M from the paper
    ticker_col: str = "yfinance_ticker",
    price_dir: str = "./data/stocks_csv/",
    min_fraction: float = 0.9, # minimum fraction of lookback to accept [To handle missing data from yFinance]
) -> pd.DataFrame:
    trade_date = pd.to_datetime(trade_date).normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00

    comp = components_df.copy()
    comp["date"] = pd.to_datetime(comp["date"]).dt.normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00

    mask = (
        (comp["date"].dt.year == trade_date.year)
        & (comp["date"].dt.month == trade_date.month)
    )
    month_df = comp.loc[mask] # Gets all tickers in the market index for specified day month and year

    if month_df.empty:
        raise ValueError(f"No index composition found for {trade_date.date()}")

    tickers = (
        month_df[ticker_col] # Note: Gets the yfinance ticker as thats how the csv file is stored
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    ) # Cleans and gets all tickers in the market index for specified day month and year

    aligned_returns: dict[str, pd.Series] = {}
    for t in tickers:
        csv_path = Path(price_dir) / f"{t}-data.csv"

        if not csv_path.exists():
            print(f"No data for: {t} ; Path checked: {csv_path} ; Date: {trade_date.date()}")
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"]) # Reads all stock data for the ticker

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce") # Converts colun to actual datetime objects. If error then returns Not-a-Time (NaT)
        df = df.dropna(subset=["Date"]) # Drops all NaTs
        df["Date"] = df["Date"].dt.normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00
        df = df.sort_values("Date") # Sorts the data for the ticker by date (We need to sort for window and rets)

        past = df[df["Date"] < trade_date] # Get data for ticker only before the current trade date
        if len(past) < lookback_returns + 1: # Checks if the number of days is below the lookback returns, if it is, skip ticker
            continue

        window = past.iloc[-(lookback_returns + 1):] # get last M+1 prices
        rets = window["AdjClose"].pct_change().dropna() # Gets all (AdjClose_{t} / AdjClose_{t-1}) - 1 (ex: 0.03, 0.05, etc.) prices and drop NA values
        rets.index = window["Date"].iloc[1:] # align dates to returns

        aligned_returns[t] = rets

    if not aligned_returns:
        raise ValueError(f"No usable return series for {trade_date.date()}")

    # VALIDATION CHECKS
    per_ticker_index = {t: s.index for t, s in aligned_returns.items()} # Get union and intersection of dates [to handle missing data from yFinance]

    common_index = None
    union_index = None
    for idx in per_ticker_index.values():
        common_index = idx if common_index is None else common_index.intersection(idx)
        union_index = idx if union_index is None else union_index.union(idx)

    if common_index is None:
        raise ValueError(f"Could not compute a common index for {trade_date.date()}")

    M_actual = len(common_index)

    # To handle missing data from yFinance (when M_actual < lookback_returns)
    if M_actual < lookback_returns:
        bad_dates = union_index.difference(common_index) # Gets problematic dates

        print(
            f"{trade_date.date()}: common_index has only {M_actual} days,"
            f" requested {lookback_returns}."
        )
        print(f"   Problematic dates (in union but not in intersection):")
        for d in bad_dates:
            print(f"      - {d.date()}")

        print("\n   Checking which tickers are missing which problematic dates:")

        # For each ticker: check which bad dates it is missing
        missing_counter = 0
        for t, idx in per_ticker_index.items():
            missing_dates = sorted([d for d in bad_dates if d not in idx])

            if missing_dates:  # this ticker contributes to the problem
                missing_counter += 1
                print(f"      {t} is missing {len(missing_dates)} of the problematic dates:")
                for md in missing_dates:
                    print(f"         • {md.date()}")
        
        print(f"Total tickers missing problematic dates: {missing_counter} out of {len(aligned_returns)}")

        # Decide whether to accept shorter M based on min_fraction
        if M_actual < int(min_fraction * lookback_returns):
            raise ValueError(f"Too few common days ({M_actual}) for requested M={lookback_returns} on {trade_date.date()}")
        else:
            print(
                f"\n   ➜ Proceeding with reduced M = {M_actual} for {trade_date.date()} "
                f"(acceptable: >= {min_fraction * 100:.1f}% of lookback)"
            )

    common_index = common_index.sort_values()
    common_index = common_index[-M_actual:]

    # Build final matrix (drop tickers with NaNs on common_index)
    final: dict[str, pd.Series] = {}
    for t, s in aligned_returns.items():
        s_aligned = s.reindex(common_index)
        if s_aligned.isna().any():
            print(f"Dropping {t} due to NaNs on final common_index for {trade_date.date()}.")
            continue
        final[t] = s_aligned

    if not final:
        raise ValueError(
            f"No valid tickers with complete returns on {trade_date.date()} after filtering."
        )

    return pd.DataFrame(final, index=common_index)

def get_day_returns(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker: str,
    trade_type: str,
    price_dir: str = "./data/stocks_csv/",
) -> float:
    start_date = pd.to_datetime(start_date).normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00
    end_date = pd.to_datetime(end_date).normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00

    csv_path = Path(price_dir) / f"{ticker}-data.csv"

    if not csv_path.exists():
        return np.nan

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00
    df = df.set_index("Date").sort_index()

    def resolve_date(date, index):
        # If exact date exists, return it
        if date in index:
            return date

        # Otherwise find next date >= given date
        loc = index.searchsorted(date)
        if loc >= len(index):
            return None  # No forward date exists
        return index[loc]

    actual_start = resolve_date(start_date, df.index)
    actual_end = resolve_date(end_date, df.index)
    
    if actual_end is None or actual_start is None:
        return np.nan

    p0 = df.loc[actual_start, "Open"]
    p1 = df.loc[actual_end, "Open"]

    if trade_type == 'long':
        return p1 / p0 - 1.0
    elif trade_type == 'short':
        return -(p1 / p0 - 1.0)
    
    return np.nan

def backtest_pure_mean_reversion(
    start_date: str,
    end_date: str,
    components_df: pd.DataFrame,
    lookback_returns: int = 252,
    ticker_col: str = "yfinance_ticker",
    price_dir: str = "./data/stocks_csv/",
    equity: float = 1_000_000.0,
    target_gross_leverage: float = 1.0,

    is_select_dynamic_factors: bool = True,
    target_evr: float = 0.50,
    static_factors: int = 10,
):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    trade_days = pd.date_range(start=start_date, end=end_date, freq="B")

    # State
    portfolio_records = [] # For equity curve
    results_by_day = {} # Different stats of strat by day
    new_positions = None # New Positions

    active_positions = {}
    # TODO: closs all positions on last day
    # TODO: Calculate the unrealized equity too.
    for i, td in enumerate(trade_days):
        td = td.normalize() # Removes the time component 2025-03-15 14:37:50 -> 2025-03-15 00:00:00
        # --- 1. Implement Trades at Market Open and Realize Positions---
        if new_positions is not None and any(new_positions.values()):
            old_equity = equity
            entries_today = []
            for ticker, pos in new_positions.items():
                # new entry only if ticker not already active
                if ticker not in active_positions:
                    if 2 in pos or 1 in pos:
                        entries_today.append((ticker, pos))

            num_entries = len(entries_today)
            cap_per_trade = (equity * target_gross_leverage / num_entries) if num_entries > 0 else 0.0
            
            for ticker, pos in new_positions.items():
                if ticker in active_positions:
                    entry_det = active_positions[ticker]
                    if -2 in pos or -1 in pos:
                        ret = get_day_returns(
                            start_date=entry_det['date'],
                            end_date=td,
                            ticker=ticker,
                            trade_type=entry_det['type'],
                            price_dir=price_dir,
                        )

                        pnl = entry_det['cap'] * ret
                        equity += pnl

                        del active_positions[ticker]
                else:
                    if 2 in pos:
                        active_positions[ticker] = {'date': td, 'type': 'long', 'cap': cap_per_trade}
                    elif 1 in pos:
                        active_positions[ticker] = {'date': td, 'type': 'short', 'cap': cap_per_trade}
            
            port_ret = (equity - old_equity) / old_equity
        else:
            port_ret = 0.0

        # Record portfolio state for this calendar day (before new trades)
        portfolio_records.append(
            {
                "date": td,
                "equity": equity,
                "daily_return": port_ret,
            }
        )

        # --- 2. Build PCA / factors / residuals using data up to td ---
        try:
            # Section 2
            mat = build_return_matrix_for_date(
                trade_date=td,
                components_df=components_df,
                lookback_returns=lookback_returns,
                ticker_col=ticker_col,
                price_dir=price_dir,
            ) # Returns Day-to-Day returns matrix for the day with lookback_returns days(M*N) where N = tickers in the Universe ; Values ex: 0.05, 0.03, etc.
            # The paper uses a N*M matrix, our is transposed.
            Y, Sigma, means, stds = preprocess_for_pca(mat) # Preprocesses the day-to-day returns matrix.
            # Y = The normalized the M*N Matrix (Ours is transposed)
            # Sigma = The correlation matrix (N*N)
            # means = Tickers Means for all returns in the lookback_returns period
            # stds = Tickers stds for all returns in the lookback_returns period

            eigenvalues, eigenvectors, evr = pca_from_sigma(Sigma) # Applies PCA on correlation matrix
            ep_returns, ep_weights = compute_eigenportfolio_returns(
                mat=mat,
                stds=stds,
                eigenvectors=eigenvectors,
            ) # Applies Equation 9 to the PCA results (Standardizing)

            if is_select_dynamic_factors:
                k_max = ep_returns.shape[1]
                cum_evr = np.cumsum(evr[:k_max])
                n_factors = int(np.searchsorted(cum_evr, target_evr) + 1)
            else:
                n_factors = static_factors

            # Section 3
            residuals, betas = compute_factor_neutral_residuals(
                mat=mat,
                ep_returns=ep_returns,
                n_factors=n_factors,
            ) 
            # TODO: Suprisingly, one of my backtest days required only 2 components to explain at least 50%. AI says back in 2003 markets had a weaker correlation structure, and many sectors behaved independently.
            # TODO: A cool study would be to understand market regime through how many factors are needed to explain the variance and how it chaneged overtime (Study the Factor's component's dominant sectors)
            ou_params = estimate_ou_from_residuals(
                residuals=residuals,
                window=60,
                min_obs=60,
                annualization=252,
            )

            # Section 4: signals + bang-bang rule
            new_positions = apply_bang_bang_rule(
                s_score=ou_params["ou_z"],
                phi=ou_params["phi"],
                sbo=1.25, # open long threshold
                sso=1.25, # open short threshold
                sbc=0.75, # close-short
                ssc=0.50, # close-long
            )

            key = td
            results_by_day[key] = {
                "mat": mat,
                "Y": Y,
                "Sigma": Sigma,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "explained_var_ratio": evr,
                "ep_returns": ep_returns,
                "ep_weights": ep_weights,
                "means": means,
                "stds": stds,
                "residuals": residuals,
                "betas": betas,
                "ou_params": ou_params,
                "trades_next_day": new_positions.copy(),
            }

            print(
                f"{td.date()}: "
                f"Sigma shape {Sigma.shape}, "
                f"active names {len(active_positions)}, "
                f"ret={port_ret:.4%}, equity={equity:,.0f}"
            )

        except Exception as e:
            print(f"Skipped {td.date()}: {e}")

    # --- 3. Build portfolio equity curve and stats ---
    portfolio_df = pd.DataFrame(portfolio_records).set_index("date")

    return results_by_day, portfolio_df

def main(start_date, end_date, strategy_conf, market="usa", is_save_backtest=False, equity=1_000_000, leverage=1.0,):
    if market == "usa":
        df = get_sp500_last_n_months_df()
        fetch_all_constituent_data(df)
    
    results_by_day, portfolio_df = backtest_pure_mean_reversion(
        start_date=start_date,
        end_date=end_date,
        components_df=df,
        lookback_returns=252,
        ticker_col="yfinance_ticker",
        price_dir="./data/stocks_csv/",
        equity=equity,
        target_gross_leverage=leverage,

        is_select_dynamic_factors=strategy_conf['is_select_dynamic_factors'],
        target_evr=strategy_conf['target_evr'],
        static_factors=strategy_conf['static_factors'],
    )

    print(portfolio_df.head())

    if is_save_backtest:
        save_backtest_results(market, start_date, end_date, results_by_day, portfolio_df, strategy_conf)

if __name__ =="__main__":
    start_date = "2025-01-01"
    end_date = "2025-08-30"
    market = "usa"
    is_save_backtest=True
    equity=1_000

    strategy_conf = {
        'is_select_dynamic_factors': True,
        'target_evr': 0.55,
        'static_factors': 15,
    }

    main(
        start_date=start_date,
        end_date=end_date,
        strategy_conf=strategy_conf,

        market=market,
        is_save_backtest=is_save_backtest,
        equity=equity,
    )
