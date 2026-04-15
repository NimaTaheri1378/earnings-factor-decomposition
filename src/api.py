"""
Abnormal Performance Index (API) — Ball & Brown (1968) multiplicative formulation.

API_τ = Π_{s=-12}^{τ} (1 + AAR_s), normalized to 1.0 at month -12.
"""
import numpy as np
import pandas as pd
from src.config import AR_MODELS


def compute_api(ep, ar_col, news_col):
    """
    Compute the multiplicative API for good-news and bad-news portfolios.

    Parameters
    ----------
    ep : DataFrame
        Event panel with event_month, ar_col, and news_col columns.
    ar_col : str
        Abnormal return column to use.
    news_col : str
        News classification column ('news_sue' or 'news_tsue').

    Returns
    -------
    DataFrame
        Columns: event_month, aar, std, n, se, t_stat, api, news
    """
    parts = []
    for news in ["Good", "Bad"]:
        g = ep[ep[news_col] == news].dropna(subset=[ar_col])
        if len(g) == 0:
            continue

        agg = g.groupby("event_month")[ar_col].agg(["mean", "std", "count"])
        agg.columns = ["aar", "std", "n"]
        agg = agg.sort_index()
        agg["se"] = agg["std"] / np.sqrt(agg["n"])
        agg["t_stat"] = agg["aar"] / agg["se"]

        raw = (1 + agg["aar"]).cumprod()
        agg["api"] = raw / raw.iloc[0]
        agg["news"] = news
        agg = agg.reset_index()
        parts.append(agg)

    return pd.concat(parts) if parts else pd.DataFrame()


def compute_spread(api_df, at_month=0):
    """
    Compute the good-news minus bad-news spread at a specific event month.

    Returns (spread_pp, good_api, bad_api) or (np.nan, np.nan, np.nan).
    """
    m = api_df[api_df["event_month"] == at_month]
    gv = m[m["news"] == "Good"]["api"].values
    bv = m[m["news"] == "Bad"]["api"].values
    if len(gv) > 0 and len(bv) > 0:
        return (gv[0] - bv[0]) * 100, gv[0], bv[0]
    return np.nan, np.nan, np.nan


def compute_all_20_specs(ep_fye, ep_ann):
    """
    Compute API for all 20 specifications (5 models × 2 surprises × 2 events).

    Returns (all_api_df, summary_table).
    """
    ar_cols = list(AR_MODELS.keys())
    configs = []
    for ev_label, ep in [("FYE", ep_fye), ("Ann", ep_ann)]:
        for ar_col, model_label in AR_MODELS.items():
            for news_col, surp_label in [
                ("news_sue", "SUE"),
                ("news_tsue", "TSUE"),
            ]:
                configs.append((ev_label, ep, ar_col, news_col, model_label, surp_label))

    all_api = []
    summary = []

    for ev, ep, ac, nc, mdl, surp in configs:
        api = compute_api(ep, ac, nc)
        if len(api) == 0:
            continue
        api["event"] = ev
        api["model"] = mdl
        api["surprise"] = surp
        all_api.append(api)

        sp, g, b = compute_spread(api, at_month=0)
        summary.append({
            "event": ev,
            "model": mdl,
            "surprise": surp,
            "good_api": g,
            "bad_api": b,
            "spread_pp": sp,
        })

    all_api_df = pd.concat(all_api, ignore_index=True) if all_api else pd.DataFrame()
    summary_df = pd.DataFrame(summary)

    return all_api_df, summary_df


def compute_quintile_api(ep, sue_col="sue_value", ar_col="ar_simple_ew",
                          n_quintiles=5):
    """
    Sort firms into quintiles by SUE and compute API for each.
    """
    valid = ep.dropna(subset=[sue_col, ar_col]).copy()
    fsue = valid.groupby("permno")[sue_col].first().reset_index()

    try:
        fsue["quintile"] = pd.qcut(fsue[sue_col], n_quintiles,
                                    labels=range(1, n_quintiles + 1))
    except ValueError:
        fsue["quintile"] = pd.qcut(
            fsue[sue_col].rank(method="first"), n_quintiles,
            labels=range(1, n_quintiles + 1),
        )
    fsue["quintile"] = fsue["quintile"].astype(int)

    valid = valid.merge(fsue[["permno", "quintile"]], on="permno", how="left")

    parts = []
    for q in range(1, n_quintiles + 1):
        g = valid[valid["quintile"] == q]
        aar = g.groupby("event_month")[ar_col].mean().sort_index()
        raw = (1 + aar).cumprod()
        api = raw / raw.iloc[0]
        df = pd.DataFrame({
            "event_month": aar.index,
            "aar": aar.values,
            "api": api.values,
            "quintile": q,
            "n_firms": g["permno"].nunique(),
        })
        parts.append(df)

    return pd.concat(parts, ignore_index=True)
