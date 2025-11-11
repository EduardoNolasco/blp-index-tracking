# data_io.py
import numpy as np
import pandas as pd

def load_returns_from_csv(index_csv: str, assets_csv: str, date_col="Date") -> tuple[pd.Series, pd.DataFrame]:
    """Load index and asset returns (aligned on dates). CSVs must contain a shared date column."""
    idx = pd.read_csv(index_csv, parse_dates=[date_col]).set_index(date_col).sort_index()
    ast = pd.read_csv(assets_csv, parse_dates=[date_col]).set_index(date_col).sort_index()

    # inner-join on dates
    df = ast.join(idx, how="inner", lsuffix="", rsuffix="_INDEX")
    y = df.iloc[:, -1].rename("index_ret")              # last column = index
    X = df.iloc[:, :-1]                                 # all but last = assets
    # drop any all-NaN cols and rows with NaNs
    X = X.dropna(axis=1, how="all")
    df2 = pd.concat([y, X], axis=1).dropna()
    y = df2.iloc[:, 0]
    X = df2.iloc[:, 1:]
    return y, X

def standardise_returns(y: pd.Series, X: pd.DataFrame, demean=True, scale=False):
    """Optionally demean (recommended for tracking error). Scaling is off by default."""
    y2 = y.copy()
    X2 = X.copy()
    if demean:
        y2 = y2 - y2.mean()
        X2 = X2 - X2.mean(axis=0)
    if scale:
        std = X2.std(axis=0).replace(0.0, 1.0)
        X2 = X2 / std
    return y2, X2