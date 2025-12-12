import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import json

import calendar
from datetime import datetime
import pandas as pd
import numpy as np
import camelot
from pathlib import Path

# Data Extraction
def _get_last_day_of_month(year: int, month: int):
    last_day = calendar.monthrange(year, month)[1]
    day = 30 if last_day >= 30 else last_day
    return datetime(year, month, day).date()

def _read_pdf_tables(pdf_path: Path) -> pd.DataFrame:
    try:
        for flavor in ("lattice", "stream"):
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
            if len(tables):
                return pd.concat([t.df for t in tables if not t.df.empty], ignore_index=True)
    except Exception:
        pass

    return pd.DataFrame()

# Backtesting
def save_backtest_results(market, start_date, end_date, results_by_day, portfolio_df, strategy_conf, base_out="backtest_results"):
    base_out_dir = Path(base_out)
    base_out_dir.mkdir(exist_ok=True)
    
    out_dir = f"{base_out}/{market}_{start_date}_{end_date}"
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Equity curve
    portfolio_df.to_csv(out_dir / "equity_curve.csv")

    # FULL daily results, including PCA
    with open(out_dir / "results_by_day.pkl", "wb") as f:
        pickle.dump(results_by_day, f)

    print(f"Saved equity_curve.csv, performance.json, results_by_day.pkl to {out_dir}")

# Plotting
def plot_scree(eigenvalues):
    plt.figure(figsize=(8,4))
    eigenvalues.plot(marker="o")
    plt.title("Scree Plot (Eigenvalues)")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
    return eigenvalues.to_dict()

def plot_explained_variance_ratio(evr):
    plt.figure(figsize=(8,4))
    evr.plot(kind="bar")
    plt.title("Explained Variance Ratio per Factor")
    plt.xlabel("Factor")
    plt.ylabel("Variance Explained")
    plt.grid(True)
    plt.show()
    return evr.to_dict()

def plot_pca_loadings(eigenvectors, factor="f1", top_n=20):
    # Sort stocks by absolute loading magnitude
    loadings = eigenvectors[factor].sort_values(key=lambda x: x.abs(), ascending=False)
    top_loadings = loadings.head(top_n)

    plt.figure(figsize=(6,6))
    sns.barplot(x=top_loadings.values, y=top_loadings.index)
    plt.title(f"PCA Loadings for {factor} (Top {top_n})")
    plt.xlabel("Loading Weight")
    plt.ylabel("Stock")
    plt.tight_layout()
    plt.show()
    return top_loadings.to_dict()
    

def plot_eigenportfolio_weights(ep_weights, factor="f1", top_n=20):
    w = ep_weights[factor].sort_values(key=lambda x: x.abs(), ascending=False)
    w = w.head(top_n)

    plt.figure(figsize=(6,6))
    sns.barplot(x=w.values, y=w.index)
    plt.title(f"Eigenportfolio Weights for {factor} (Top {top_n})")
    plt.xlabel("Weight")
    plt.ylabel("Stock")
    plt.tight_layout()
    plt.show()
    return w.to_dict()

def plot_eigenportfolio_returns(ep_returns, factor="f1"):
    plt.figure(figsize=(10,4))
    ep_returns[factor].plot()
    plt.title(f"Eigenportfolio Returns: {factor}")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
    return ep_returns[factor].to_dict()

def plot_ou_fit(residuals: pd.DataFrame, ou_params: dict, ticker: str, window: int = 60):
    # ---- Extract values from ou_params ----
    b = ou_params["phi"].get(ticker, np.nan)
    var_eta = ou_params["eps_var"].get(ticker, np.nan)

    if np.isnan(b):
        raise ValueError(f"No OU parameters available for ticker {ticker}")

    # ---- Recreate X_k = cumulative residuals ----
    eps = residuals[ticker].dropna().tail(window).values
    if len(eps) < 2:
        raise ValueError(f"Not enough data to compute X for {ticker}")

    X = np.cumsum(eps)  # X_1, X_2, ..., X_T

    # AR(1) regression re-fit to recover intercept 'a'
    # (same as inside OU estimator; needed because 'a' is not saved)
    X_n = X[:-1]
    X_np1 = X[1:]
    Z = np.column_stack([np.ones_like(X_n), X_n])
    theta, *_ = np.linalg.lstsq(Z, X_np1, rcond=None)
    a, b_refit = theta  # b_refit should equal b

    # ---- Compute fitted values ----
    X_fitted = a + b * X_n  # predicted X_{n+1}

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    
    # Plot actual pairs (X_n, X_{n+1})
    plt.scatter(X_n, X_np1, color='blue', alpha=0.6, label="Actual")
    
    # Plot fitted AR(1) line
    xs = np.linspace(min(X_n), max(X_n), 200)
    ys = a + b * xs
    plt.plot(xs, ys, color='red', linewidth=2, label=f"Fitted AR(1): Xₙ₊₁ = {a:.4f} + {b:.4f} Xₙ")

    plt.title(f"OU Fit for {ticker}: X vs AR(1) Prediction")
    plt.xlabel("Xₙ")
    plt.ylabel("Xₙ₊₁")
    plt.grid(True)
    plt.legend()
    plt.show()
