import pandas as pd
import requests
from typing import Iterable

import os, re
import yfinance as yf
from pathlib import Path

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# -------------------------------------------
# 1. Fetch Wikipedia tables once (fast)
# -------------------------------------------
def fetch_wiki_tables():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    return tables[0], tables[1]  # current constituents, change log


# -------------------------------------------
# 2. Convert S&P tickers → yfinance tickers
# -------------------------------------------
def to_yf_ticker(sym: str) -> str:
    """
    Convert S&P 500 tickers to Yahoo Finance compatible tickers.
    Ex: BRK.B → BRK-B
        BF.B → BF-B
    """
    return sym.replace(".", "-")


# -------------------------------------------
# 3. Reconstruct S&P500 as of a given date
# -------------------------------------------
def get_sp500_constituents_as_of(target_date, current_df, changes_df):
    changes_df = changes_df.copy()
    changes_df.columns = [
        "effective_date", "added_ticker", "added_name",
        "removed_ticker", "removed_name", "reason"
    ]
    changes_df["effective_date"] = pd.to_datetime(changes_df["effective_date"])

    # Identify the column for ticker symbol
    symbol_col = [c for c in current_df.columns if "ymbol" in str(c)][0]
    security_col = [c for c in current_df.columns if "Security" in str(c)][0]

    members = {
        (row[symbol_col], row[security_col])
        for _, row in current_df.iterrows()
    }

    target_date = pd.to_datetime(target_date)

    for _, row in changes_df.sort_values("effective_date", ascending=False).iterrows():
        d = row["effective_date"]
        add_t = row["added_ticker"]
        rem_t = row["removed_ticker"]

        if d <= target_date:
            break

        # Reverse additions/removals
        if pd.notna(add_t):
            members = {(s, n) for (s, n) in members if s != add_t}
        if pd.notna(rem_t):
            members.add((rem_t, None))  # Name unknown from change log; will fix later

    # Re-join with current_df to recover names if missing
    final_df = (
        pd.DataFrame(list(members), columns=["Symbol", "Security Name"])
        .merge(
            current_df[[symbol_col, security_col]]
            .rename(columns={symbol_col: "Symbol", security_col: "Security Name"}),
            on="Symbol",
            how="left",
            suffixes=("", "_current")
        )
    )

    # Fill missing names
    final_df["Security Name"] = final_df["Security Name"].fillna(final_df["Security Name_current"])
    final_df = final_df.drop(columns=["Security Name_current"])

    return final_df


# -------------------------------------------
# 4. Generate monthly history DataFrame
# TODO: Get data when csv file doesn't have data for a different n.
# -------------------------------------------
def get_sp500_last_n_months_df(n=60):
    out_dir = Path("./data/sp500/sp500_csv/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sp500_components.csv"

    if out_path.exists():
        return pd.read_csv(out_path)

    current_df, changes_df = fetch_wiki_tables()

    # last N month-end dates
    month_ends = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="M")

    all_rows = []

    for dt in month_ends:
        print(f"Building constituents for {dt.date()}...")
        subset_df = get_sp500_constituents_as_of(dt, current_df, changes_df)
        subset_df["date"] = dt.date()

        # yfinance ticker mapping
        subset_df["yfinance_ticker"] = subset_df["Symbol"].apply(to_yf_ticker)

        all_rows.append(subset_df)

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df[["Symbol", "Security Name", "date", "yfinance_ticker"]]

    # Ensure datetime format
    final_df["date"] = pd.to_datetime(final_df["date"])

    final_df.to_csv(out_path, index=False)

    return final_df

def get_stock_data(ticker = "SETFNIF50.NS", return_df = False):
    out_folder="./data/stocks_csv/"
    out_path = f"{out_folder}{ticker}-data.csv"

    if os.path.exists(out_path):
        if not return_df:
            return True
        
        df = pd.read_csv(out_path, parse_dates=['AdjClose'], date_format="%Y-%m-%d")
    else:
        df = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False
        )

        rename_map = {"Adj Close": "AdjClose"}
        df.rename(columns=rename_map, inplace=True)

        df = df.reset_index()
        df.columns = df.columns.droplevel(1)
         
        directory_path = Path(out_folder)
        directory_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")

    return df if return_df else True

def fetch_all_constituent_data(df: pd.DataFrame) -> pd.DataFrame:
    if "yfinance_ticker" not in df.columns:
        raise ValueError("Expected a 'yfinance_ticker' column in the dataframe.")

    unique_tickers: Iterable[str] = df["yfinance_ticker"].dropna().unique()
    for t in unique_tickers:
        try:
            get_stock_data(ticker=f"{t}", return_df=False)
            print(f"✅ fetched {t}")
        except Exception as e:
            # Keep going even if one fails
            print(f"⚠️ failed {t}: {e}")

    return df

# -------------------------------------------
# 5. RUN
# -------------------------------------------
if __name__ == "__main__":
    df = get_sp500_last_n_months_df(180)
    fetch_all_constituent_data(df)
