"""
Transaction cost estimation.

Primary source: taqmsec.wrds_iid_YYYY (WRDS Intraday Indicators, year-sharded)
  — pre-computed daily quoted and effective spreads per stock
  — keyed by (date, sym_root)
  — columns (lowercase): quotedspread_percent_tw, effectivespread_percent_ave

Fallback: crsp.dsf askhi/bidlo for quoted spread approximation
Additional: Corwin-Schultz (2012) estimator, Amihud (2002) illiquidity
"""
import numpy as np
import pandas as pd
from src.utils import to_float64, load_cache, save_cache, logger


def pull_spreads(permnos, start_date, end_date):
    """
    Pull bid-ask spreads. Tries taqmsec.wrds_iid first, then CRSP daily.

    Returns DataFrame with: permno, date, spread_bps
    """
    from src.data import _query

    pstr = ",".join(map(str, permnos))

    # ── 1. taqmsec.wrds_iid (pre-computed by WRDS, year-sharded) ────────
    iid_year = end_date[:4]
    iid_table = f"taqmsec.wrds_iid_{iid_year}"
    logger.log(f"  [TC] Querying {iid_table} (pre-computed spreads)...")
    try:
        df = _query(f"""
            SELECT a.permno,
                   b.date,
                   b.quotedspread_percent_tw    AS qs_pct,
                   b.effectivespread_percent_ave AS es_pct,
                   b.effectivespread_percent_dw  AS es_dw_pct
            FROM crsp.msenames a
            INNER JOIN {iid_table} b
              ON UPPER(a.tsymbol) = UPPER(b.sym_root)
              AND b.date BETWEEN '{start_date}' AND '{end_date}'
            WHERE a.permno IN ({pstr})
              AND b.date >= a.namedt
              AND (b.date <= a.nameendt OR a.nameendt IS NULL)
              AND b.quotedspread_percent_tw IS NOT NULL
        """)

        if df is not None and len(df) > 100:
            df["date"] = pd.to_datetime(df["date"])
            # iid_ms stores proportions despite "Percent" in name
            # e.g. 0.00045 = 4.5 bps.  Multiply by 10000.
            df["spread_bps"] = (
                df["es_pct"].fillna(df["qs_pct"]).astype(float) * 10000
            )
            df = df[df["spread_bps"] > 0]
            med = df["spread_bps"].median()
            logger.log(f"  [TC] {len(df):,} obs from wrds_iid "
                       f"(median effective spread: {med:.1f} bps)")
            return df[["permno", "date", "spread_bps"]]

    except Exception as e:
        logger.log(f"  [TC] wrds_iid failed: {e}")

    # ── 2. CRSP daily askhi/bidlo (always available) ─────────────────────
    logger.log("  [TC] Falling back to CRSP daily bid-ask...")
    try:
        df = _query(f"""
            SELECT permno, date,
                   CASE WHEN askhi > 0 AND bidlo > 0 AND askhi > bidlo
                        THEN (askhi - bidlo)
                             / ((askhi + bidlo) / 2.0) * 10000
                        ELSE NULL END AS spread_bps
            FROM crsp.dsf
            WHERE permno IN ({pstr})
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND askhi IS NOT NULL AND bidlo IS NOT NULL
              AND askhi > 0 AND bidlo > 0
              AND askhi / bidlo < 1.5
        """)
        if df is not None:
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["spread_bps"])
            logger.log(f"  [TC] {len(df):,} obs from CRSP daily")
            return df
    except Exception as e:
        logger.log(f"  ⚠ CRSP daily failed: {e}")

    return pd.DataFrame(columns=["permno", "date", "spread_bps"])


def corwin_schultz_spread(high, low):
    """
    Corwin-Schultz (2012) bid-ask spread estimator from daily high-low.

    Returns spread as a proportion (multiply by 10000 for bps).
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    if len(h) < 2:
        return np.nan

    gamma = np.log(h / l) ** 2
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    beta = np.log(h2 / l2) ** 2

    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2
    alpha = (np.sqrt(2 * np.mean(beta)) - np.sqrt(np.mean(beta))) / denom \
            - np.sqrt(np.mean(gamma) / denom)

    if alpha < 0:
        return 0.0
    return max(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0.0)


def estimate_spreads_corwin_schultz(permnos, fiscal_year):
    """Estimate spreads from CRSP daily high-low (Corwin-Schultz 2012)."""
    from src.data import _query

    pstr = ",".join(map(str, permnos))
    daily = _query(f"""
        SELECT permno, date, askhi, bidlo, ABS(prc) AS prc, vol
        FROM crsp.dsf
        WHERE permno IN ({pstr})
          AND date BETWEEN '{fiscal_year + 1}-01-01' AND '{fiscal_year + 1}-06-30'
          AND askhi IS NOT NULL AND bidlo IS NOT NULL
          AND askhi > 0 AND bidlo > 0
        ORDER BY permno, date
    """)
    if daily is None or len(daily) == 0:
        return pd.DataFrame(columns=["permno", "spread_bps"])

    daily["date"] = pd.to_datetime(daily["date"])
    results = []
    for permno, g in daily.groupby("permno"):
        g = g.sort_values("date")
        if len(g) < 5:
            continue
        spread = corwin_schultz_spread(
            g["askhi"].astype(float).values,
            g["bidlo"].astype(float).values,
        )
        results.append({
            "permno": permno,
            "spread_bps": spread * 10000,
            "avg_price": g["prc"].astype(float).abs().mean(),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        logger.log(f"  [CS] Spreads for {len(df)} firms "
                   f"(median: {df['spread_bps'].median():.0f} bps)")
    return df


def amihud_illiquidity(permnos, fiscal_year):
    """Amihud (2002) illiquidity: daily |return| / dollar volume."""
    from src.data import _query

    pstr = ",".join(map(str, permnos))
    daily = _query(f"""
        SELECT permno, ret, ABS(prc) * vol AS dvol
        FROM crsp.dsf
        WHERE permno IN ({pstr})
          AND date BETWEEN '{fiscal_year}-07-01' AND '{fiscal_year + 1}-06-30'
          AND ret IS NOT NULL AND vol > 0 AND prc IS NOT NULL
    """)
    if daily is None or len(daily) == 0:
        return pd.DataFrame(columns=["permno", "amihud"])

    daily["illiq"] = daily["ret"].astype(float).abs() / daily["dvol"].astype(float).clip(1)
    return daily.groupby("permno")["illiq"].mean().reset_index().rename(
        columns={"illiq": "amihud"}
    )


def compute_net_returns(strategy_returns, spreads_by_firm, turnover_rate=1.0):
    """Adjust strategy returns for transaction costs."""
    median_spread = spreads_by_firm["spread_bps"].median()
    rt_cost_bps = 2 * median_spread * turnover_rate  # round-trip, both legs

    df = strategy_returns.copy()
    df["tc_bps"] = rt_cost_bps
    df["tc_pct"] = rt_cost_bps / 10000
    if "ls_ret" in df.columns:
        df["ls_ret_net"] = df["ls_ret"] - 2 * df["tc_pct"]
    return df


def transaction_cost_analysis(ep, controls, fiscal_year):
    """
    Full TC pipeline: estimate spreads, split by size, compute net returns.
    """
    permnos = ep["permno"].unique()
    start = f"{fiscal_year + 1}-01-01"
    end = f"{fiscal_year + 1}-04-30"

    # Get spreads
    spreads = pull_spreads(permnos, start, end)
    if len(spreads) == 0:
        logger.log("  ⚠ No spread data")
        return None

    firm_spreads = spreads.groupby("permno")["spread_bps"].median().reset_index()

    # Split by size
    size_data = controls[["permno", "ln_size"]].dropna()
    median_size = size_data["ln_size"].astype(float).median()
    firm_spreads = firm_spreads.merge(size_data, on="permno", how="left")
    firm_spreads["size_group"] = np.where(
        firm_spreads["ln_size"].astype(float) >= median_size, "Large", "Small"
    )

    summary = firm_spreads.groupby("size_group")["spread_bps"].agg(
        ["median", "mean", "count"]
    )
    logger.log("\n  Spread by size:")
    logger.log(summary.to_string())

    overall = firm_spreads["spread_bps"].median()
    logger.log(f"\n  Overall median: {overall:.1f} bps")
    logger.log(f"  Round-trip (both legs): {2 * overall:.1f} bps")

    return {
        "firm_spreads": firm_spreads,
        "summary": summary,
        "median_spread_bps": overall,
    }
