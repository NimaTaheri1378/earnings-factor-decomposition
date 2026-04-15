"""
Abnormal return models: simple market-adjusted, market model, Fama-French-Carhart.

Handles estimation window parameter estimation and vectorized AR computation.
"""
import numpy as np
import pandas as pd
from src.config import (
    MIN_EST_MONTHS, EVENT_WINDOW_START, EVENT_WINDOW_END,
    ESTIMATION_START, ESTIMATION_END,
)
from src.utils import to_float64, load_cache, save_cache, logger


def estimate_firm_params(group):
    """
    Estimate market model and FFC parameters from estimation window data.

    Uses numpy.linalg.lstsq for speed (~3x faster than statsmodels per firm).
    We don't need robust SEs here — these params are only used to compute
    predicted returns, not for hypothesis testing.
    """
    y = to_float64(group["ret"])
    if len(y) < MIN_EST_MONTHS:
        return None

    res = {}

    # Market Model (VW): ret = a + b * vwretd
    try:
        X = np.column_stack([np.ones(len(y)), to_float64(group["vwretd"])])
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res["mm_vw_a"], res["mm_vw_b"] = float(params[0]), float(params[1])
    except Exception:
        return None

    # Market Model (EW): ret = a + b * ewretd
    try:
        X = np.column_stack([np.ones(len(y)), to_float64(group["ewretd"])])
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res["mm_ew_a"], res["mm_ew_b"] = float(params[0]), float(params[1])
    except Exception:
        pass

    # FFC four-factor: (ret - rf) = a + b*MKT + s*SMB + h*HML + p*UMD
    ffc_cols = ["mktrf", "smb", "hml", "umd", "rf"]
    ffc_data = group.dropna(subset=ffc_cols)
    if len(ffc_data) >= MIN_EST_MONTHS:
        try:
            y_ex = to_float64(ffc_data["ret"]) - to_float64(ffc_data["rf"])
            X = np.column_stack([
                np.ones(len(y_ex)),
                to_float64(ffc_data["mktrf"]),
                to_float64(ffc_data["smb"]),
                to_float64(ffc_data["hml"]),
                to_float64(ffc_data["umd"]),
            ])
            params, _, _, _ = np.linalg.lstsq(X, y_ex, rcond=None)
            res["ffc_a"] = float(params[0])
            res["ffc_b"] = float(params[1])
            res["ffc_s"] = float(params[2])
            res["ffc_h"] = float(params[3])
            res["ffc_u"] = float(params[4])
        except Exception:
            pass

    return res


def build_event_panel(linked, msf, mkt, event_col,
                      w_start=EVENT_WINDOW_START, w_end=EVENT_WINDOW_END,
                      cache_name=None, fiscal_year=None):
    """
    Construct an event-study panel with all five abnormal return measures.

    Parameters
    ----------
    linked : DataFrame
        Firm-level data with permno, event dates, and news classifications.
    msf : DataFrame
        Monthly stock returns merged with market returns and factors.
    mkt : DataFrame
        Market-level return data with month index.
    event_col : str
        Column in `linked` containing the event date ('fiscal_ye' or 'event_date').
    w_start, w_end : int
        Event window bounds in months relative to event.
    cache_name : str, optional
        Name for caching the result.
    fiscal_year : int, optional
        For cache key.

    Returns
    -------
    DataFrame
        Panel with columns: permno, event_month, ret, all AR measures,
        news classifications, and sue values.
    """
    if cache_name:
        cached = load_cache(cache_name, fiscal_year)
        if cached is not None:
            logger.log(f"  [cache] {cache_name}: {len(cached):,} obs")
            return cached

    # Map event dates → market month index
    evt = linked[["permno", event_col, "news_sue", "news_tsue", "sue"]].copy()
    evt.columns = ["permno", "evt_date", "news_sue", "news_tsue", "sue_value"]
    evt["evt_date"] = pd.to_datetime(evt["evt_date"])

    mkt_dates = mkt[["date", "mth"]].sort_values("date")
    evt = evt.sort_values("evt_date")
    evt = pd.merge_asof(
        evt, mkt_dates, left_on="evt_date", right_on="date", direction="backward"
    )
    evt = evt.rename(columns={"mth": "evt_mth"}).drop(columns="date", errors="ignore")
    evt = evt.dropna(subset=["evt_mth"])
    evt["evt_mth"] = evt["evt_mth"].astype(int)

    # Merge stock data with event info
    panel = msf.merge(evt, on="permno", how="inner")
    panel["event_month"] = panel["mth"] - panel["evt_mth"]

    # Estimate parameters from estimation window
    est_data = panel[
        (panel["event_month"] >= ESTIMATION_START)
        & (panel["event_month"] <= ESTIMATION_END)
    ].dropna(subset=["ret", "vwretd", "ewretd"])

    logger.log("    Estimating model parameters...")
    param_list = []
    for permno, grp in est_data.groupby("permno"):
        res = estimate_firm_params(grp)
        if res is not None and "mm_vw_a" in res:
            res["permno"] = int(permno)
            param_list.append(res)

    if not param_list:
        raise ValueError("No firms survived model estimation — check data window.")

    params = pd.DataFrame(param_list)
    logger.log(f"    Parameters estimated for {len(params)} firms")

    # Filter to event window and merge parameters
    ep = panel[(panel["event_month"] >= w_start) & (panel["event_month"] <= w_end)]
    ep = ep.merge(params, on="permno", how="inner")

    # Compute all abnormal returns (vectorized)
    ep = compute_abnormal_returns(ep)

    logger.log(f"    Panel: {len(ep):,} obs, {ep['permno'].nunique()} firms")

    if cache_name:
        save_cache(ep, cache_name, fiscal_year)

    return ep


def compute_abnormal_returns(ep):
    """
    Compute all five AR specifications from panel data (vectorized).

    Modifies ep in-place and returns it.
    """
    ret = to_float64(ep["ret"])
    ew = to_float64(ep["ewretd"])
    vw = to_float64(ep["vwretd"])

    # Models 1-2: Simple market-adjusted (Ball & Brown 1968)
    ep["ar_simple_ew"] = ret - ew
    ep["ar_simple_vw"] = ret - vw

    # Model 3: Market model (EW benchmark)
    ep["ar_mm_ew"] = np.nan
    if "mm_ew_a" in ep.columns:
        mask = ep["mm_ew_a"].notna().values
        if mask.any():
            ep.loc[mask, "ar_mm_ew"] = ret[mask] - (
                to_float64(ep.loc[mask, "mm_ew_a"])
                + to_float64(ep.loc[mask, "mm_ew_b"]) * ew[mask]
            )

    # Model 4: Market model (VW benchmark)
    ep["ar_mm_vw"] = ret - (
        to_float64(ep["mm_vw_a"]) + to_float64(ep["mm_vw_b"]) * vw
    )

    # Model 5: Fama-French-Carhart four-factor
    ep["ar_ffc"] = np.nan
    if "ffc_a" in ep.columns:
        mask = ep["ffc_a"].notna().values
        if mask.any():
            rf = to_float64(ep.loc[mask, "rf"])
            predicted = (
                to_float64(ep.loc[mask, "ffc_a"])
                + to_float64(ep.loc[mask, "ffc_b"]) * to_float64(ep.loc[mask, "mktrf"])
                + to_float64(ep.loc[mask, "ffc_s"]) * to_float64(ep.loc[mask, "smb"])
                + to_float64(ep.loc[mask, "ffc_h"]) * to_float64(ep.loc[mask, "hml"])
                + to_float64(ep.loc[mask, "ffc_u"]) * to_float64(ep.loc[mask, "umd"])
            )
            ep.loc[mask, "ar_ffc"] = (ret[mask] - rf) - predicted

    return ep
