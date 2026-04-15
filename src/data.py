"""
Data acquisition from WRDS: I/B/E/S, CRSP, Compustat, Fama-French factors.

All SQL queries are parameterized by fiscal year for multi-year support.
Results are cached as parquet for reproducibility and speed.
"""
import wrds
import pandas as pd
import numpy as np
from src.config import (
    MIN_ANALYSTS, MIN_PRICE, TSUE_HISTORY_START,
    WINSORIZE_PCT, PRIMARY_FY,
)
from src.utils import (
    to_float64, winsorize, cache_exists, load_cache, save_cache, logger,
    wrds_to_pandas,
)


_db = None


def get_connection():
    """Get or create a WRDS connection (singleton)."""
    global _db
    if _db is not None:
        return _db
    _db = wrds.Connection()
    logger.log("✓ WRDS connection established")
    return _db


def _query(sql):
    """Run WRDS SQL and convert nullable dtypes to standard numpy-backed ones."""
    db = get_connection()
    df = db.raw_sql(sql)
    return wrds_to_pandas(df)


def pull_ibes(fiscal_year=PRIMARY_FY, force=False):
    """
    Pull I/B/E/S annual EPS consensus data for December FYE firms.

    Returns DataFrame with columns:
        ticker, cusip, cname, fpedats, statpers, medest, meanest,
        numest, stdev, actual, anndats_act, sue, news_sue
    """
    cached = load_cache("ibes", fiscal_year)
    if cached is not None and not force:
        logger.log(f"  [cache] IBES FY{fiscal_year}: {len(cached)} firms")
        return cached

    ibes = _query(f"""
        WITH ranked AS (
            SELECT ticker, cusip, cname, fpedats, statpers,
                   medest, meanest, numest, stdev, actual, anndats_act,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, fpedats
                       ORDER BY statpers DESC
                   ) rn
            FROM ibes.statsum_epsus
            WHERE fpi = '1'
              AND EXTRACT(YEAR FROM fpedats) = {fiscal_year}
              AND EXTRACT(MONTH FROM fpedats) = 12
              AND actual IS NOT NULL
              AND numest >= {MIN_ANALYSTS}
              AND usfirm = 1
        )
        SELECT * FROM ranked WHERE rn = 1 ORDER BY ticker
    """)
    ibes["fpedats"] = pd.to_datetime(ibes["fpedats"])
    ibes["anndats_act"] = pd.to_datetime(ibes["anndats_act"])

    # Analyst-based SUE: (Actual - Median) / StdDev  [Foster et al. 1984]
    ibes["sue"] = (ibes["actual"] - ibes["medest"]) / ibes["stdev"]
    ibes["sue"] = ibes["sue"].replace([np.inf, -np.inf], np.nan)
    ibes = ibes.dropna(subset=["sue"])
    ibes["sue"] = winsorize(ibes["sue"].values)
    ibes["news_sue"] = np.where(
        ibes["sue"] > 0, "Good", np.where(ibes["sue"] < 0, "Bad", "Neutral")
    )

    save_cache(ibes, "ibes", fiscal_year)
    logger.log(f"  [pulled] IBES FY{fiscal_year}: {len(ibes)} firms")
    return ibes


def pull_tsue(fiscal_year=PRIMARY_FY, force=False):
    """
    Compute time-series unexpected earnings (Ball & Brown 1968 method).

    Regresses firm-level ΔEPSᵢ on market-wide ΔEPS over prior years,
    then defines TSUE as the FY residual.
    """
    cached = load_cache("tsue", fiscal_year)
    if cached is not None and not force:
        logger.log(f"  [cache] TSUE FY{fiscal_year}: {len(cached)} firms")
        return cached

    eh = _query(f"""
        WITH ranked AS (
            SELECT ticker, fpedats, actual,
                   EXTRACT(YEAR FROM fpedats) fyear,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, fpedats
                       ORDER BY statpers DESC
                   ) rn
            FROM ibes.statsum_epsus
            WHERE fpi = '1'
              AND EXTRACT(MONTH FROM fpedats) = 12
              AND actual IS NOT NULL AND usfirm = 1
              AND EXTRACT(YEAR FROM fpedats)
                  BETWEEN {TSUE_HISTORY_START} AND {fiscal_year}
        )
        SELECT ticker, fpedats, actual, fyear FROM ranked WHERE rn = 1
        ORDER BY ticker, fpedats
    """)
    eh["fyear"] = eh["fyear"].astype(int)
    eh = eh.sort_values(["ticker", "fpedats"])
    eh["eps_lag"] = eh.groupby("ticker")["actual"].shift(1)
    eh["delta_eps"] = pd.to_numeric(eh["actual"] - eh["eps_lag"], errors="coerce")

    mkt_delta = eh.dropna(subset=["delta_eps"]).groupby("fyear")["delta_eps"].mean()
    mkt_delta.name = "delta_mkt"
    eh = eh.join(mkt_delta, on="fyear")

    def _tsue_one(g):
        est = g[g["fyear"] < fiscal_year].dropna(subset=["delta_eps", "delta_mkt"])
        obs = g[g["fyear"] == fiscal_year].dropna(subset=["delta_eps", "delta_mkt"])
        if len(est) < 3 or len(obs) == 0:
            return None
        try:
            # Fast OLS via numpy (avoids statsmodels overhead per firm)
            y = est["delta_eps"].values.astype(float)
            x = est["delta_mkt"].values.astype(float)
            X = np.column_stack([np.ones(len(x)), x])
            params = np.linalg.lstsq(X, y, rcond=None)[0]  # [intercept, slope]
            row = obs.iloc[0]
            return float(row["delta_eps"] - (params[0] + params[1] * row["delta_mkt"]))
        except Exception:
            return None

    results = eh.groupby("ticker").apply(_tsue_one).dropna()
    tsue = results.reset_index()
    tsue.columns = ["ticker", "tsue"]
    tsue["news_tsue"] = np.where(
        tsue["tsue"] > 0, "Good", np.where(tsue["tsue"] < 0, "Bad", "Neutral")
    )

    save_cache(tsue, "tsue", fiscal_year)
    logger.log(f"  [computed] TSUE FY{fiscal_year}: {len(tsue)} firms")
    return tsue


def link_ibes_crsp(ibes, tsue, fiscal_year=PRIMARY_FY, force=False):
    """
    Link I/B/E/S tickers to CRSP PERMNOs via wrdsapps.ibcrsphist.
    Applies $5 minimum price filter.
    """
    cached = load_cache("linked", fiscal_year)
    if cached is not None and not force:
        logger.log(f"  [cache] Linked FY{fiscal_year}: {len(cached)} firms")
        return cached

    link = _query(
        "SELECT ticker, permno, sdate, edate, score FROM wrdsapps.ibcrsphist"
    )
    link["sdate"] = pd.to_datetime(link["sdate"])
    link["edate"] = pd.to_datetime(link["edate"])

    ibes = ibes.copy()
    ibes["event_date"] = ibes["anndats_act"]
    ibes["fiscal_ye"] = ibes["fpedats"]

    linked = ibes.merge(link, on="ticker", how="inner")
    mx_e = link["edate"].max()
    linked = linked[
        (linked["event_date"] >= linked["sdate"])
        & ((linked["event_date"] <= linked["edate"]) | (linked["edate"] == mx_e))
    ]
    linked = linked.sort_values("score").drop_duplicates("ticker", keep="first")
    linked = linked.merge(
        tsue[["ticker", "news_tsue", "tsue"]], on="ticker", how="left"
    )
    linked = linked[linked["permno"].notna()]
    linked["permno"] = linked["permno"].astype(int)

    # Price filter
    pstr = ",".join(map(str, linked["permno"].unique()))
    prices = _query(f"""
        SELECT permno, ABS(prc) prc FROM crsp.msf
        WHERE permno IN ({pstr})
          AND date = (SELECT MAX(date) FROM crsp.msf
                      WHERE date <= '{fiscal_year}-12-31')
          AND prc IS NOT NULL
    """).drop_duplicates("permno", keep="last")
    linked = linked.merge(prices, on="permno", how="left")
    linked = linked[linked["prc"] >= MIN_PRICE]

    save_cache(linked, "linked", fiscal_year)
    logger.log(f"  [linked] FY{fiscal_year}: {len(linked)} firms")
    return linked


def pull_returns_and_factors(permnos, fiscal_year=PRIMARY_FY, force=False):
    """
    Pull CRSP monthly returns, market indices, and FF+Carhart factors.

    Returns (msf, mkt) DataFrames.
    """
    cached_msf = load_cache("msf", fiscal_year)
    cached_mkt = load_cache("mkt", fiscal_year)
    if cached_msf is not None and cached_mkt is not None and not force:
        logger.log(f"  [cache] Returns FY{fiscal_year}")
        return cached_msf, cached_mkt

    # Date range: 2 years before FYE through 1 year after
    start = f"{fiscal_year - 3}-01-01"
    end = f"{fiscal_year + 1}-12-31"

    pstr = ",".join(map(str, permnos))
    msf = _query(f"""
        SELECT permno, date, ret FROM crsp.msf
        WHERE permno IN ({pstr})
          AND date BETWEEN '{start}' AND '{end}'
          AND ret IS NOT NULL
    """)
    msf["date"] = pd.to_datetime(msf["date"])
    msf["ret"] = winsorize(msf["ret"].values)

    msi = _query(f"""
        SELECT date, ewretd, vwretd FROM crsp.msi
        WHERE date BETWEEN '{start}' AND '{end}'
    """)
    msi["date"] = pd.to_datetime(msi["date"])

    ff = _query(f"""
        SELECT date, mktrf, smb, hml, umd, rf FROM ff.factors_monthly
        WHERE date BETWEEN '{start}' AND '{end}'
    """)
    ff["date"] = pd.to_datetime(ff["date"])

    msi["ym"] = msi["date"].dt.to_period("M")
    ff["ym"] = ff["date"].dt.to_period("M")
    mkt = msi.merge(ff.drop(columns="date"), on="ym", how="left")
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt["mth"] = np.arange(len(mkt))

    msf["ym"] = msf["date"].dt.to_period("M")
    msf = msf.merge(mkt.drop(columns="date"), on="ym", how="left")

    save_cache(msf, "msf", fiscal_year)
    save_cache(mkt, "mkt", fiscal_year)
    logger.log(f"  [pulled] {len(msf):,} stock-months, {len(mkt)} market-months")
    return msf, mkt


def pull_compustat_controls(permnos, fiscal_year=PRIMARY_FY, force=False):
    """
    Pull Compustat fundamentals for cross-sectional controls.

    Returns DataFrame with: permno, ln_size, bm, leverage, sic2, mktcap
    """
    cached = load_cache("controls", fiscal_year)
    if cached is not None and not force:
        logger.log(f"  [cache] Controls FY{fiscal_year}: {len(cached)} firms")
        return cached

    pstr = ",".join(map(str, permnos))
    ctrl = _query(f"""
        SELECT DISTINCT a.gvkey, a.fyear,
               a.at, a.ceq, a.lt, a.sale, a.ni, a.ib,
               a.prcc_f, a.csho, a.sich,
               b.lpermno AS permno
        FROM comp.funda a
        INNER JOIN crsp.ccmxpf_linktable b
          ON a.gvkey = b.gvkey
          AND b.linktype IN ('LU','LC')
          AND b.linkprim IN ('P','C')
          AND a.datadate >= b.linkdt
          AND (a.datadate <= b.linkenddt OR b.linkenddt IS NULL)
        WHERE a.fyear = {fiscal_year}
          AND a.indfmt = 'INDL' AND a.datafmt = 'STD'
          AND a.popsrc = 'D' AND a.consol = 'C'
          AND b.lpermno IN ({pstr})
    """)
    ctrl["permno"] = ctrl["permno"].astype(int)
    ctrl = ctrl.drop_duplicates("permno", keep="first")

    ctrl["mktcap"] = ctrl["prcc_f"] * ctrl["csho"]
    ctrl["ln_size"] = np.log(ctrl["mktcap"].clip(lower=1e-6))
    ctrl["bm"] = ctrl["ceq"] / ctrl["mktcap"].clip(1e-6)
    ctrl["leverage"] = ctrl["lt"] / ctrl["at"].clip(1e-6)
    ctrl["sic2"] = (ctrl["sich"] // 100).astype("Int64")

    controls = ctrl[["permno", "ln_size", "bm", "leverage", "sic2", "mktcap"]].copy()
    save_cache(controls, "controls", fiscal_year)
    logger.log(f"  [pulled] Controls FY{fiscal_year}: {len(controls)} firms")
    return controls


def compute_momentum(msf):
    """
    Prior 12-month return, skipping most recent month (Jegadeesh & Titman 1993).

    mom_i = Π_{t=-12}^{t=-2} (1 + ret_i,t) - 1

    Vectorized via log-returns and rolling sum to avoid slow groupby.apply.
    """
    df = msf[["permno", "date", "ret"]].copy()
    df = df.sort_values(["permno", "date"])

    # Log return for additive rolling
    df["log1r"] = np.log1p(df["ret"].astype(float))

    # Rolling 12-month sum of log returns, then shift by 1 to skip most recent
    df["cum12"] = df.groupby("permno")["log1r"].transform(
        lambda x: x.rolling(12, min_periods=10).sum()
    )
    # Shift forward by 1: this gives sum of months [-12, -1]
    # We want [-12, -2], so subtract the t-1 log return
    df["mom_log"] = df["cum12"] - df["log1r"]

    # Take the last observation per firm
    last = df.groupby("permno").tail(1)[["permno", "mom_log"]].copy()
    last["momentum"] = np.expm1(last["mom_log"])  # convert back from log
    last = last[["permno", "momentum"]].reset_index(drop=True)

    # Drop firms with too few observations (momentum will be NaN)
    return last


def pull_full_sample(fiscal_year=PRIMARY_FY, force=False):
    """
    Orchestrates the full data pipeline for a given fiscal year.

    Returns dict with keys:
        ibes, tsue, linked, msf, mkt, controls, momentum
    """
    logger.section(f"DATA PIPELINE — FY{fiscal_year}")

    ibes = pull_ibes(fiscal_year, force)
    tsue = pull_tsue(fiscal_year, force)
    linked = link_ibes_crsp(ibes, tsue, fiscal_year, force)
    msf, mkt = pull_returns_and_factors(linked["permno"].unique(), fiscal_year, force)
    controls = pull_compustat_controls(linked["permno"].unique(), fiscal_year, force)
    momentum = compute_momentum(msf)

    # Attach additional controls to linked
    linked["ln_numest"] = np.log(linked["numest"].clip(lower=1))
    linked["earn_vol"] = linked["stdev"]

    return {
        "ibes": ibes,
        "tsue": tsue,
        "linked": linked,
        "msf": msf,
        "mkt": mkt,
        "controls": controls,
        "momentum": momentum,
    }
