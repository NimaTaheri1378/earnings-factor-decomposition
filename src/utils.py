"""
Utility functions: winsorization, caching, safe type conversion.
"""
import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize as _scipy_winz
from src.config import INPUT_DIR, WINSORIZE_PCT


def to_float64(x):
    """Safely cast pandas Series/DataFrame (including nullable dtypes) to float64."""
    if hasattr(x, 'to_numpy'):
        return x.to_numpy(dtype='float64', na_value=np.nan)
    return np.asarray(x, dtype=np.float64)


def wrds_to_pandas(df):
    """
    Convert WRDS nullable dtypes to standard pandas/numpy dtypes.

    Newer WRDS returns StringDtype ('string') and Int64/Float64 (nullable).
    These cause issues with numpy operations. This converts:
        string  → object (str)
        Int64   → float64 (because NaN forces float)
        Float64 → float64
    """
    df = df.copy()
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in ("string", "String"):
            df[col] = df[col].astype(object)
        elif dtype in ("Int8", "Int16", "Int32", "Int64"):
            df[col] = df[col].astype("float64")
        elif dtype in ("Float32", "Float64"):
            df[col] = df[col].astype("float64")
        elif dtype == "boolean":
            df[col] = df[col].astype("float64")
    return df


def winsorize(arr, limits=None):
    """Winsorize array at symmetric percentiles, handling nullable dtypes."""
    if limits is None:
        limits = [WINSORIZE_PCT, WINSORIZE_PCT]
    a = np.asarray(arr, dtype=np.float64)
    return np.asarray(_scipy_winz(a, limits=limits), dtype=np.float64)


def winsorize_series(df, columns, limits=None):
    """Winsorize multiple columns in-place, skipping columns with <10 non-null."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > 10:
                df.loc[valid, col] = winsorize(df.loc[valid, col].values, limits)
    return df


# ── Parquet cache (stored in INPUT_DIR on Google Drive) ──────────────────────

def cache_path(name, fiscal_year=None):
    """Return path to a cached parquet file in INPUT_DIR."""
    suffix = f"_{fiscal_year}" if fiscal_year else ""
    return os.path.join(INPUT_DIR, f"{name}{suffix}.parquet")


def cache_exists(name, fiscal_year=None):
    """Check if cached file exists."""
    return os.path.exists(cache_path(name, fiscal_year))


def load_cache(name, fiscal_year=None):
    """Load cached parquet, or None if missing."""
    p = cache_path(name, fiscal_year)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def save_cache(df, name, fiscal_year=None):
    """Save DataFrame to parquet cache."""
    df.to_parquet(cache_path(name, fiscal_year), index=False)


def clear_cache():
    """Remove all cached files."""
    for f in os.listdir(INPUT_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(INPUT_DIR, f))


# ── Logging ──────────────────────────────────────────────────────────────────

class Logger:
    """Simple logger that prints and stores lines for a diagnostics report."""

    def __init__(self):
        self.lines = []

    def log(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def section(self, title):
        self.log("")
        self.log("─" * 70)
        self.log(title)
        self.log("─" * 70)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines))

logger = Logger()
