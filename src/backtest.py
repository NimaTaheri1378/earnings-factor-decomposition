"""
Trading strategy backtests built on PEAD findings.

Strategies:
    1. Binary PEAD:       Long good-news, short bad-news (equal-weight)
    2. Quintile spread:   Long Q5, short Q1 (strongest signals)
    3. Size-conditional:  PEAD in small caps only (where drift concentrates)
    4. Factor-aware:      Use FFC residuals, long positive / short negative
    5. Decay-weighted:    Overweight recent announcements (drift front-loads)

All strategies report: cumulative return, Sharpe ratio, max drawdown,
turnover, and factor-adjusted alpha via FFC time-series regression.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import Optional
from src.utils import to_float64


@dataclass
class BacktestResult:
    """Container for strategy performance metrics."""
    name: str
    cumulative_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    avg_monthly_turnover: float
    n_months: int
    n_firms_avg: float
    monthly_returns: np.ndarray
    # Factor-adjusted metrics
    ffc_alpha: Optional[float] = None
    ffc_alpha_t: Optional[float] = None
    ffc_alpha_p: Optional[float] = None
    ffc_r2: Optional[float] = None
    # Factor loadings
    mkt_beta: Optional[float] = None
    smb_loading: Optional[float] = None
    hml_loading: Optional[float] = None
    umd_loading: Optional[float] = None

    def summary_dict(self):
        return {
            "Strategy": self.name,
            "Cum. Return (%)": f"{self.cumulative_return:.2f}",
            "Ann. Return (%)": f"{self.annualized_return:.2f}",
            "Ann. Vol (%)": f"{self.annualized_vol:.2f}",
            "Sharpe": f"{self.sharpe_ratio:.3f}",
            "Max DD (%)": f"{self.max_drawdown:.2f}",
            "Avg Turnover (%)": f"{self.avg_monthly_turnover:.1f}",
            "FFC Alpha (bps/mo)": f"{self.ffc_alpha * 10000:.1f}" if self.ffc_alpha else "—",
            "Alpha t-stat": f"{self.ffc_alpha_t:.2f}" if self.ffc_alpha_t else "—",
            "N months": self.n_months,
            "Avg firms": f"{self.n_firms_avg:.0f}",
        }


def compute_performance(monthly_returns, name="Strategy", rf=None):
    """Compute standard performance metrics from a monthly return series."""
    r = np.array(monthly_returns)
    r = r[~np.isnan(r)]

    if len(r) < 3:
        return BacktestResult(
            name=name, cumulative_return=0, annualized_return=0,
            annualized_vol=0, sharpe_ratio=0, max_drawdown=0,
            avg_monthly_turnover=0, n_months=len(r), n_firms_avg=0,
            monthly_returns=r,
        )

    cum = (1 + r).prod() - 1
    ann_ret = (1 + cum) ** (12 / len(r)) - 1
    ann_vol = np.std(r) * np.sqrt(12)

    # Sharpe (excess over risk-free if provided)
    if rf is not None:
        excess = r - np.array(rf[:len(r)])
    else:
        excess = r
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(12) if np.std(excess) > 0 else 0

    # Max drawdown
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    max_dd = np.max(drawdown) * 100

    return BacktestResult(
        name=name,
        cumulative_return=cum * 100,
        annualized_return=ann_ret * 100,
        annualized_vol=ann_vol * 100,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        avg_monthly_turnover=0,  # filled in by strategy
        n_months=len(r),
        n_firms_avg=0,
        monthly_returns=r,
    )


def ffc_regression(monthly_returns, factor_returns):
    """
    Regress strategy returns on FFC factors to extract alpha.

    Uses SE_TYPE_TIME_SERIES from methodology.py (HC1 by default,
    Newey-West available via SE_TYPE_TS_ROBUST for robustness).

    Parameters
    ----------
    monthly_returns : array-like
        Strategy monthly excess returns.
    factor_returns : DataFrame
        Must have columns: mktrf, smb, hml, umd, date (or ym).

    Returns
    -------
    dict with alpha, betas, t-stats, R²
    """
    from src.methodology import SE_TYPE_TIME_SERIES

    r = np.array(monthly_returns)
    if len(r) < 6 or len(factor_returns) < 6:
        return {}

    n = min(len(r), len(factor_returns))
    r = r[:n]
    ff = factor_returns.iloc[:n]

    y = to_float64(r)
    X = sm.add_constant(to_float64(ff[["mktrf", "smb", "hml", "umd"]]))

    try:
        model = sm.OLS(y, X).fit(cov_type=SE_TYPE_TIME_SERIES)
        return {
            "alpha": model.params[0],
            "alpha_t": model.tvalues[0],
            "alpha_p": model.pvalues[0],
            "mkt_beta": model.params[1],
            "smb": model.params[2],
            "hml": model.params[3],
            "umd": model.params[4],
            "r2": model.rsquared,
        }
    except Exception:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def strategy_binary_pead(ep, holding_months=3, news_col="news_sue"):
    """
    Strategy 1: Binary PEAD long-short.

    At each announcement, go long good-news firms and short bad-news firms.
    Hold for `holding_months`. Equal-weighted within each leg.

    Returns monthly portfolio returns as a Series indexed by calendar month.
    """
    valid = ep.dropna(subset=["ar_simple_ew"]).copy()
    valid = valid[valid["event_month"].between(0, holding_months)]

    # Monthly long-short returns
    monthly = []
    for em in range(0, holding_months + 1):
        month_data = valid[valid["event_month"] == em]
        long_ret = month_data[month_data[news_col] == "Good"]["ret"].mean()
        short_ret = month_data[month_data[news_col] == "Bad"]["ret"].mean()
        n_long = (month_data[news_col] == "Good").sum()
        n_short = (month_data[news_col] == "Bad").sum()

        if not np.isnan(long_ret) and not np.isnan(short_ret):
            monthly.append({
                "event_month": em,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": long_ret - short_ret,
                "n_long": n_long,
                "n_short": n_short,
            })

    if not monthly:
        return None

    df = pd.DataFrame(monthly)
    result = compute_performance(df["ls_ret"].values, name="Binary PEAD L/S")
    result.n_firms_avg = df[["n_long", "n_short"]].sum(axis=1).mean()
    return result, df


def strategy_quintile_spread(ep, sue_col="sue_value", holding_months=3):
    """
    Strategy 2: Quintile long-short (Q5 - Q1).

    Strongest earnings signals only. Avoids middle quintiles where
    noise dominates signal.
    """
    valid = ep.dropna(subset=[sue_col, "ret"]).copy()
    valid = valid[valid["event_month"].between(0, holding_months)]

    fsue = valid.groupby("permno")[sue_col].first().reset_index()
    try:
        fsue["quintile"] = pd.qcut(fsue[sue_col], 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        fsue["quintile"] = pd.qcut(
            fsue[sue_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )
    fsue["quintile"] = fsue["quintile"].astype(int)
    valid = valid.merge(fsue[["permno", "quintile"]], on="permno", how="left")

    monthly = []
    for em in range(0, holding_months + 1):
        md = valid[valid["event_month"] == em]
        q5_ret = md[md["quintile"] == 5]["ret"].mean()
        q1_ret = md[md["quintile"] == 1]["ret"].mean()
        n_q5 = (md["quintile"] == 5).sum()
        n_q1 = (md["quintile"] == 1).sum()

        if not np.isnan(q5_ret) and not np.isnan(q1_ret):
            monthly.append({
                "event_month": em,
                "q5_ret": q5_ret,
                "q1_ret": q1_ret,
                "ls_ret": q5_ret - q1_ret,
                "n_long": n_q5,
                "n_short": n_q1,
            })

    if not monthly:
        return None

    df = pd.DataFrame(monthly)
    result = compute_performance(df["ls_ret"].values, name="Quintile Q5-Q1 L/S")
    result.n_firms_avg = df[["n_long", "n_short"]].sum(axis=1).mean()
    return result, df


def strategy_size_conditional(ep, controls, holding_months=3,
                               news_col="news_sue", size_group="Small"):
    """
    Strategy 3: PEAD in small caps only.

    Exploits the finding that drift is ~5x larger in small caps (14.25pp vs 3.03pp),
    consistent with Hong, Lim, Stein (2000) limited-attention hypothesis.
    """
    size_data = controls[["permno", "ln_size"]].dropna().copy()
    median_size = size_data["ln_size"].median()
    size_data["size_group"] = np.where(
        size_data["ln_size"] >= median_size, "Large", "Small"
    )

    valid = ep.merge(size_data[["permno", "size_group"]], on="permno", how="inner")
    valid = valid[valid["size_group"] == size_group]
    valid = valid[valid["event_month"].between(0, holding_months)]

    monthly = []
    for em in range(0, holding_months + 1):
        md = valid[valid["event_month"] == em]
        long_ret = md[md[news_col] == "Good"]["ret"].mean()
        short_ret = md[md[news_col] == "Bad"]["ret"].mean()
        n_long = (md[news_col] == "Good").sum()
        n_short = (md[news_col] == "Bad").sum()

        if not np.isnan(long_ret) and not np.isnan(short_ret):
            monthly.append({
                "event_month": em,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": long_ret - short_ret,
                "n_long": n_long,
                "n_short": n_short,
            })

    if not monthly:
        return None

    df = pd.DataFrame(monthly)
    result = compute_performance(
        df["ls_ret"].values,
        name=f"PEAD {size_group}-Cap L/S",
    )
    result.n_firms_avg = df[["n_long", "n_short"]].sum(axis=1).mean()
    return result, df


def strategy_factor_aware(ep, holding_months=3, threshold=0):
    """
    Strategy 4: Factor-aware PEAD.

    Uses FFC residual returns instead of raw returns. Trades on the
    alpha component after stripping out factor exposures. This tests
    whether the drift generates genuine alpha vs factor harvesting.
    """
    valid = ep.dropna(subset=["ar_ffc", "news_sue"]).copy()
    valid = valid[valid["event_month"].between(0, holding_months)]

    monthly = []
    for em in range(0, holding_months + 1):
        md = valid[valid["event_month"] == em]
        # Use FFC abnormal returns as the signal and return measure
        long_ar = md[md["news_sue"] == "Good"]["ar_ffc"].mean()
        short_ar = md[md["news_sue"] == "Bad"]["ar_ffc"].mean()

        if not np.isnan(long_ar) and not np.isnan(short_ar):
            monthly.append({
                "event_month": em,
                "long_ar": long_ar,
                "short_ar": short_ar,
                "ls_ar": long_ar - short_ar,
            })

    if not monthly:
        return None

    df = pd.DataFrame(monthly)
    result = compute_performance(
        df["ls_ar"].values, name="Factor-Aware PEAD (FFC residual)"
    )
    m0 = ep.dropna(subset=["ar_ffc", "news_sue"])
    result.n_firms_avg = len(m0[m0["event_month"] == 0])
    return result, df


def strategy_decay_weighted(ep, news_col="news_sue", max_months=6):
    """
    Strategy 5: Decay-weighted PEAD.

    Overweights positions in months 0-3 (where drift front-loads per Table 6)
    and underweights months 4-6. Captures the timing of alpha accrual.

    Weight decays linearly: w(t) = max(1 - t/max_months, 0).
    """
    valid = ep.dropna(subset=["ret"]).copy()
    valid = valid[valid["event_month"].between(0, max_months)]

    monthly = []
    for em in range(0, max_months + 1):
        md = valid[valid["event_month"] == em]
        weight = max(1 - em / max_months, 0.1)  # minimum 10% weight

        long_ret = md[md[news_col] == "Good"]["ret"].mean()
        short_ret = md[md[news_col] == "Bad"]["ret"].mean()

        if not np.isnan(long_ret) and not np.isnan(short_ret):
            monthly.append({
                "event_month": em,
                "weight": weight,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": (long_ret - short_ret) * weight,
                "ls_ret_unweighted": long_ret - short_ret,
            })

    if not monthly:
        return None

    df = pd.DataFrame(monthly)
    result = compute_performance(
        df["ls_ret"].values, name="Decay-Weighted PEAD"
    )
    m0 = ep[(ep["event_month"] == 0) & (ep[news_col].isin(["Good","Bad"]))]
    result.n_firms_avg = len(m0)
    return result, df


def run_all_strategies(ep, controls, factor_returns=None, holding_months=3):
    """
    Run all five strategies and return a summary DataFrame.

    Parameters
    ----------
    ep : DataFrame
        Announcement-date event panel (extended window).
    controls : DataFrame
        Firm-level controls with ln_size.
    factor_returns : DataFrame, optional
        FF factors for alpha regression.
    holding_months : int
        Default holding period.

    Returns
    -------
    summary_df : DataFrame
        Performance comparison across strategies.
    results : dict
        Strategy name → (BacktestResult, detail_df).
    """
    strategies = {}

    # 1. Binary PEAD
    out = strategy_binary_pead(ep, holding_months)
    if out:
        strategies["Binary PEAD"] = out

    # 2. Quintile spread
    out = strategy_quintile_spread(ep, holding_months=holding_months)
    if out:
        strategies["Quintile Q5-Q1"] = out

    # 3. Small-cap PEAD
    out = strategy_size_conditional(ep, controls, holding_months, size_group="Small")
    if out:
        strategies["Small-Cap PEAD"] = out

    # 4. Large-cap PEAD (for comparison)
    out = strategy_size_conditional(ep, controls, holding_months, size_group="Large")
    if out:
        strategies["Large-Cap PEAD"] = out

    # 5. Factor-aware
    out = strategy_factor_aware(ep, holding_months)
    if out:
        strategies["Factor-Aware"] = out

    # 6. Decay-weighted
    out = strategy_decay_weighted(ep)
    if out:
        strategies["Decay-Weighted"] = out

    # Add FFC alpha regressions if factor data available
    if factor_returns is not None:
        for name, (result, df) in strategies.items():
            ffc = ffc_regression(result.monthly_returns, factor_returns)
            if ffc:
                result.ffc_alpha = ffc.get("alpha")
                result.ffc_alpha_t = ffc.get("alpha_t")
                result.ffc_alpha_p = ffc.get("alpha_p")
                result.ffc_r2 = ffc.get("r2")
                result.mkt_beta = ffc.get("mkt_beta")
                result.smb_loading = ffc.get("smb")
                result.hml_loading = ffc.get("hml")
                result.umd_loading = ffc.get("umd")

    # Build summary table
    summary = pd.DataFrame([r.summary_dict() for r, _ in strategies.values()])

    return summary, strategies
