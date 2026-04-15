"""
Statistical inference: bootstrap confidence intervals, cross-sectional regressions.

All methodological choices (SE type, covariates, windows) come from
src/methodology.py — do NOT hardcode them here.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from src.config import N_BOOTSTRAP, BOOTSTRAP_CI, WINSORIZE_PCT
from src.methodology import (
    SE_TYPE_CROSS_SECTION,
    COVARIATES_MODEL1, COVARIATES_MODEL2, COVARIATES_MODEL3,
    INDUSTRY_FE_COLUMN, INDUSTRY_FE_MIN_OBS,
    CAR_WINDOW, WINSORIZE_CONTROLS,
)
from src.api import compute_api
from src.utils import to_float64, winsorize


def bootstrap_spread(ep, ar_col, news_col, n_boot=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    """
    Firm-level block bootstrap for the good-news/bad-news spread at month 0.

    Resamples firms with replacement, preserving each firm's complete time series,
    following Petersen (2009).

    Returns
    -------
    dict with keys: mean, ci_lo, ci_hi, se, p_value, spreads (array)
    """
    firms = ep["permno"].unique()
    spreads = []

    for _ in range(n_boot):
        boot_firms = np.random.choice(firms, len(firms), replace=True)
        boot_data = ep[ep["permno"].isin(boot_firms)]
        api = compute_api(boot_data, ar_col, news_col)
        if len(api) == 0:
            continue

        m0 = api[api["event_month"] == 0]
        gv = m0[m0["news"] == "Good"]["api"].values
        bv = m0[m0["news"] == "Bad"]["api"].values
        if len(gv) > 0 and len(bv) > 0:
            spreads.append((gv[0] - bv[0]) * 100)

    if len(spreads) < 50:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "se": np.nan, "p_value": np.nan, "spreads": np.array([])}

    a = np.array(spreads)
    alpha = (1 - ci) / 2
    return {
        "mean": np.mean(a),
        "ci_lo": np.percentile(a, alpha * 100),
        "ci_hi": np.percentile(a, (1 - alpha) * 100),
        "se": np.std(a),
        "p_value": 2 * min(np.mean(a > 0), np.mean(a < 0)),
        "spreads": a,
    }


def cross_sectional_regression(ep, linked, controls, momentum,
                                ar_cols=None):
    """
    Cross-sectional regressions: CAR[-1,+1] on SUE with progressive controls.

    Models:
        (1) Univariate: SUE only
        (2) + Size, Book-to-Market, Momentum
        (3) Full + Analyst coverage, Earnings vol, Leverage, Industry FE

    Parameters
    ----------
    ep : DataFrame
        Announcement-date event panel.
    linked : DataFrame
        Firm characteristics from I/B/E/S.
    controls : DataFrame
        Compustat controls (ln_size, bm, leverage, sic2).
    momentum : DataFrame
        Prior 12-month returns.
    ar_cols : list, optional
        AR columns to use as dependent variables.

    Returns
    -------
    DataFrame
        Regression results table.
    """
    if ar_cols is None:
        ar_cols = ["ar_simple_ew", "ar_mm_vw", "ar_ffc"]

    # Compute CAR[-1, +1]
    car_window = ep[(ep["event_month"] >= -1) & (ep["event_month"] <= 1)]
    car_by_firm = {}
    for ac in ar_cols:
        c = (
            car_window.dropna(subset=[ac])
            .groupby("permno")[ac]
            .sum()
            .reset_index()
            .rename(columns={ac: f"car_{ac}"})
        )
        car_by_firm[ac] = c

    # Build cross-sectional dataset
    xs = car_by_firm[ar_cols[0]]
    for ac in ar_cols[1:]:
        xs = xs.merge(car_by_firm[ac], on="permno", how="outer")

    xs = xs.merge(
        linked[["permno", "sue", "ln_numest", "earn_vol"]].drop_duplicates("permno"),
        on="permno", how="left",
    )
    xs = xs.merge(controls, on="permno", how="left")
    xs = xs.merge(momentum, on="permno", how="left")

    # Winsorize continuous controls (from methodology.py)
    for v in WINSORIZE_CONTROLS:
        if v in xs.columns:
            valid = xs[v].notna()
            if valid.sum() > 10:
                xs.loc[valid, v] = winsorize(xs.loc[valid, v].values)

    # Run regressions using covariate specs from methodology.py
    ar_labels = {
        "car_ar_simple_ew": "Simple(EW)",
        "car_ar_mm_vw": "MM(VW)",
        "car_ar_ffc": "FFC",
    }

    rows = []
    for dep, label in ar_labels.items():
        if dep not in xs.columns:
            continue

        # Model 1: Univariate (from COVARIATES_MODEL1)
        avail = [c for c in COVARIATES_MODEL1
                 if c in xs.columns and xs[c].notna().sum() > 50]
        if "sue" in avail:
            m1 = _run_ols(dep, avail, xs)
            if m1 is not None:
                rows.append(_extract_sue_coef(m1, label, "(1) Univariate"))

        # Model 2: + Core controls (from COVARIATES_MODEL2)
        avail = [c for c in COVARIATES_MODEL2
                 if c in xs.columns and xs[c].notna().sum() > 50]
        if "sue" in avail and len(avail) >= 2:
            m2 = _run_ols(dep, avail, xs)
            if m2 is not None:
                rows.append(_extract_sue_coef(m2, label, "(2) +Size,BM,Mom"))

        # Model 3: Full + Industry FE (from COVARIATES_MODEL3)
        avail = [c for c in COVARIATES_MODEL3
                 if c in xs.columns and xs[c].notna().sum() > 50]
        if "sue" in avail and len(avail) >= 3:
            m3 = _run_ols(dep, avail, xs, add_fe=True)
            if m3 is not None:
                rows.append(_extract_sue_coef(m3, label, "(3) Full+IndFE"))

    return pd.DataFrame(rows)


def _run_ols(y_col, x_cols, data, add_fe=False, fe_col=INDUSTRY_FE_COLUMN):
    """OLS with robust SEs (type from methodology.py) and optional industry FE."""
    d = data.dropna(subset=[y_col] + x_cols).copy()
    if len(d) < 30:
        return None

    if add_fe and fe_col in d.columns and d[fe_col].notna().sum() > 0:
        d[fe_col] = d[fe_col].astype(str)
        fe = pd.get_dummies(d[fe_col], prefix="sic", drop_first=True, dtype=float)
        fe = fe.loc[:, fe.sum() >= INDUSTRY_FE_MIN_OBS]
        X_data = pd.concat(
            [d[x_cols].astype(float).reset_index(drop=True),
             fe.reset_index(drop=True)],
            axis=1,
        )
        X = sm.add_constant(X_data.astype(float))
    else:
        X = sm.add_constant(d[x_cols].astype(float))

    y = to_float64(d[y_col])
    X = to_float64(X)
    return sm.OLS(y, X).fit(cov_type=SE_TYPE_CROSS_SECTION)


def _extract_sue_coef(model, dep_label, spec_label):
    """Extract the SUE coefficient from a regression model."""
    # SUE is always the first regressor after the constant (index 1)
    return {
        "dep": dep_label,
        "model": spec_label,
        "beta_sue": model.params[1],
        "t_stat": model.tvalues[1],
        "p_value": model.pvalues[1],
        "r2_adj": model.rsquared_adj,
        "n": int(model.nobs),
    }


def spearman_monotonicity(quintile_api, at_month=0):
    """Test quintile monotonicity via Spearman rank correlation."""
    m0 = quintile_api[quintile_api["event_month"] == at_month][
        ["quintile", "api"]
    ].sort_values("quintile")
    rho, pv = stats.spearmanr(
        m0["quintile"].astype(float).values,
        m0["api"].astype(float).values,
    )
    return rho, pv
