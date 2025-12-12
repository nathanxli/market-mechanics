from utils import *
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Loading helpers
# ============================================================

def load_results_by_day(backtest_dir: str | Path) -> dict:
    """
    Load results_by_day.pkl from a backtest directory.
    """
    backtest_dir = Path(backtest_dir)
    pkl_path = backtest_dir / "results_by_day.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"results_by_day.pkl not found in {backtest_dir}")

    with open(pkl_path, "rb") as f:
        results_by_day = pickle.load(f)

    return results_by_day


def load_equity_curve(backtest_dir: str | Path) -> pd.DataFrame:
    """
    Load equity_curve.csv from a backtest directory.
    """
    backtest_dir = Path(backtest_dir)
    csv_path = backtest_dir / "equity_curve.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"equity_curve.csv not found in {backtest_dir}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


# ============================================================
# 1. Function to show all PCA plots for a given day
# ============================================================

def show_pca_plots_for_day(
    backtest_dir: str | Path,
    day: str | pd.Timestamp,
    factor: str = "f2",
    top_n: int = 15,
) -> None:
    """
    Load results_by_day from backtest_dir and show:
      - Scree plot
      - Explained variance ratio
      - PCA loadings for a given factor
      - Eigenportfolio returns for that factor
      - Eigenportfolio weights for that factor
    """
    results_by_day = load_results_by_day(backtest_dir)
    day_ts = pd.Timestamp(day).normalize()

    if day_ts not in results_by_day:
        valid_days = list(results_by_day.keys())
        raise KeyError(
            f"Day {day_ts.date()} not found in results_by_day. "
            f"Available keys include something like: {valid_days[:5]}"
        )

    res = results_by_day[day_ts]

    eigenvalues = res["eigenvalues"]
    evr = res["explained_var_ratio"]
    eigenvectors = res["eigenvectors"]
    ep_returns = res["ep_returns"]
    ep_weights = res["ep_weights"]

    # Now make the plots
    plot_scree(eigenvalues)
    plot_explained_variance_ratio(evr)
    plot_pca_loadings(eigenvectors, factor=factor, top_n=top_n)
    plot_eigenportfolio_returns(ep_returns, factor=factor)
    plot_eigenportfolio_weights(ep_weights, factor=factor, top_n=top_n)


# ============================================================
# 2. Function to plot equity curve (optionally with VOO)
# ============================================================

def plot_equity_curve(
    backtest_dir: str | Path,
    include_market: bool = False,
    market: str | None = None,
) -> None:
    portfolio_df = load_equity_curve(backtest_dir)

    plt.figure(figsize=(10, 4))
    portfolio_df["equity"].plot(label="Strategy")

    if include_market:
        if market == "india":
            data_path =  "./data/stocks_csv/SETFNIF50.NS-data.csv"
        elif market == "usa":
            data_path = "./data/stocks_csv/VOO-data.csv"
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"VOO data file not found at {data_path}")

        voo = pd.read_csv(data_path, parse_dates=["Date"])
        voo["Date"] = pd.to_datetime(voo["Date"]).dt.normalize()
        voo = voo.set_index("Date").sort_index()

        if "AdjClose" not in voo.columns:
            raise ValueError("VOO CSV must contain an 'AdjClose' column")

        # Align dates to intersection
        common_idx = portfolio_df.index.intersection(voo.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between portfolio and VOO.")

        equity_start = portfolio_df.loc[common_idx[0], "equity"]
        voo_prices = voo.loc[common_idx, "AdjClose"]

        # Scale VOO so that it starts at the same equity as the strategy
        voo_equity = voo_prices / voo_prices.iloc[0] * equity_start

        voo_equity.plot(label=f"Index (scaled)")

    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_performance_stats(portfolio_df: pd.DataFrame, trading_days=252):
    r = portfolio_df["daily_return"].dropna()

    total_return = portfolio_df["equity"].iloc[-1] / portfolio_df["equity"].iloc[0] - 1.0
    ann_return = (1 + total_return) ** (trading_days / len(r)) - 1 if len(r) > 0 else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = portfolio_df["equity"]
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": len(r),
    }

def compute_alpha_beta(portfolio_df: pd.DataFrame, market: str, trading_days=252):
    if market == "india":
        data_path =  "./data/stocks_csv/SETFNIF50.NS-data.csv"
    elif market == "usa":
        data_path = "./data/stocks_csv/VOO-data.csv"

    # Strategy returns
    r_s = portfolio_df["daily_return"].dropna()

    # Load VOO returns
    voo = pd.read_csv(data_path, parse_dates=["Date"])
    voo["Date"] = pd.to_datetime(voo["Date"]).dt.normalize()
    voo = voo.set_index("Date").sort_index()
    voo["ret"] = voo["AdjClose"].pct_change().dropna()

    r_b = voo["ret"]

    # Align time series
    idx = r_s.index.intersection(r_b.index)
    if len(idx) == 0:
        return {"alpha_annual": np.nan, "alpha_daily": np.nan, "beta": np.nan}

    r_s = r_s.loc[idx].values
    r_b = r_b.loc[idx].values

    # CAPM regression
    X = np.vstack([np.ones(len(r_b)), r_b]).T
    params = np.linalg.inv(X.T @ X) @ (X.T @ r_s)

    alpha_daily, beta = params
    alpha_annual = (1 + alpha_daily)**trading_days - 1

    return {
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "beta": beta,
        "n_obs": len(idx),
    }

def get_OU_plot(results_by_day, day, ticker, show_options:bool=True):
    result_on_day = results_by_day[day]

    residuals = result_on_day['residuals']
    ou_params = result_on_day['ou_params']

    if show_options:
        print()
        print(f"AvailableTickers on day ({day}) = {list(ou_params['phi'].keys())}")
        print()

    plot_ou_fit(
        residuals=residuals,
        ou_params=ou_params,
        ticker=ticker,
        window=60,
    )

# ============================================================
# Example usage (you can delete or adapt this)
# ============================================================
if __name__ == "__main__":
    start_date = "2025-01-01"
    end_date = "2025-08-30"
    market = "usa"

    BACKTEST_DIR = f"backtest_results/{market}_{start_date}_{end_date}"

    results_by_day = load_results_by_day(BACKTEST_DIR)
    portfolio_df = load_equity_curve(BACKTEST_DIR)

    print(f"Loaded {len(portfolio_df)} trading days of equity curve.")
    print(f"Loaded {len(results_by_day)} days of PCA + OU diagnostics.")

    perf_stats = compute_performance_stats(portfolio_df)
    print("\n--- Performance Stats ---")
    for k, v in perf_stats.items():
        print(f"{k:20s}: {v}")
    
    alpha_stats = compute_alpha_beta(portfolio_df, market=market)
    print("\n--- Alpha / Beta vs Index ---")
    for k, v in alpha_stats.items():
        print(f"{k:20s}: {v}")

    # Example: show PCA plots for a specific day
    day = pd.Timestamp("2025-02-05")
    # show_pca_plots_for_day(BACKTEST_DIR, day, factor="f1", top_n=15)

    # Example: plot equity curve with VOO overlay
    plot_equity_curve(BACKTEST_DIR, include_market=True, market=market)

    # day = pd.Timestamp("2025-01-06").normalize()
    # get_OU_plot(results_by_day=results_by_day, day=day, ticker="ADANIPORTS.NS")
