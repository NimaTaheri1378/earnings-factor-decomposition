"""
Publication-quality figures for the earnings factor decomposition paper.

All figures follow a consistent style and are saved as both PNG (300 dpi)
and PDF for LaTeX inclusion.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import COLORS, PLOT_PARAMS, FIGURE_DIR, AR_MODELS
import os


plt.rcParams.update(PLOT_PARAMS)


def _save(fig, name):
    """Save figure as PNG and PDF."""
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches="tight")


def _plot_api(ax, api_df, title, legend=True):
    """Plot API trajectories for good-news and bad-news portfolios."""
    for nw, c in [("Good", COLORS["good"]), ("Bad", COLORS["bad"])]:
        d = api_df[api_df["news"] == nw].sort_values("event_month")
        if len(d):
            ax.plot(d["event_month"], d["api"], label=nw, color=c,
                    lw=2, marker="o", markersize=3)
    ax.axhline(1, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("API")
    ax.set_title(title, fontsize=10, fontweight="bold")
    if legend:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


def _get_spread(all_api_df, ev, mdl, surp, at_month=0):
    """Extract spread from the all-specs API dataframe."""
    s = all_api_df[
        (all_api_df["event"] == ev)
        & (all_api_df["model"] == mdl)
        & (all_api_df["surprise"] == surp)
        & (all_api_df["event_month"] == at_month)
    ]
    g = s[s["news"] == "Good"]["api"].values
    b = s[s["news"] == "Bad"]["api"].values
    return (g[0] - b[0]) * 100 if len(g) > 0 and len(b) > 0 else 0


def fig1_ball_brown(all_api_df):
    """Figure 1: Simple EW — Ball & Brown (1968) exact replication (2×2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (ev, su) in zip(
        axes.flat,
        [("FYE", "SUE"), ("FYE", "TSUE"), ("Ann", "SUE"), ("Ann", "TSUE")],
    ):
        api = all_api_df[
            (all_api_df["event"] == ev)
            & (all_api_df["model"] == "Simple (EW)")
            & (all_api_df["surprise"] == su)
        ]
        sp = _get_spread(all_api_df, ev, "Simple (EW)", su)
        _plot_api(ax, api, f"{ev}+{su} ({sp:.1f}pp)")
    fig.suptitle(
        "Simple EW — Ball & Brown (1968) Exact",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig1_simple_ew")
    return fig


def fig2_attenuation(all_api_df):
    """Figure 2: Model comparison showing attenuation (Ann + SUE)."""
    models = ["Simple (EW)", "Simple (VW)", "MM (EW)", "MM (VW)", "FFC"]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, mdl in zip(axes, models):
        api = all_api_df[
            (all_api_df["model"] == mdl)
            & (all_api_df["surprise"] == "SUE")
            & (all_api_df["event"] == "Ann")
        ]
        sp = _get_spread(all_api_df, "Ann", mdl, "SUE")
        _plot_api(ax, api, f"{mdl}\n{sp:.1f}pp")
    fig.suptitle(
        "Attenuation: Spread shrinks with risk adjustment (Ann+SUE)",
        fontsize=13, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(fig, "fig2_attenuation")
    return fig


def fig3_quintiles(qapi):
    """Figure 3: SUE quintile portfolio API trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Q1 (Bad)", "Q2", "Q3", "Q4", "Q5 (Good)"]
    for q, c, lb in zip(range(1, 6), COLORS["quintiles"], labels):
        d = qapi[qapi["quintile"] == q].sort_values("event_month")
        ax.plot(d["event_month"], d["api"], label=lb, color=c,
                lw=2, marker="o", markersize=3)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("API")
    ax.set_title("SUE Quintile Portfolios", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "fig3_quintiles")
    return fig


def fig4_pead(pead_api):
    """Figure 4: Post-earnings announcement drift over [-12, +12]."""
    specs = ["Simple(EW)+SUE", "MM(VW)+SUE", "FFC+SUE"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, label in zip(axes, specs):
        sub = pead_api[pead_api["spec"] == label]
        _plot_api(ax, sub, label)
        ax.axvspan(0, 12, alpha=0.05, color="blue")
        ax.set_xlim(-12.5, 12.5)

        # Annotate drift
        g0 = sub[(sub["event_month"] == 0) & (sub["news"] == "Good")]["api"].values
        b0 = sub[(sub["event_month"] == 0) & (sub["news"] == "Bad")]["api"].values
        g12 = sub[(sub["event_month"] == 12) & (sub["news"] == "Good")]["api"].values
        b12 = sub[(sub["event_month"] == 12) & (sub["news"] == "Bad")]["api"].values
        if len(g0) and len(b0) and len(g12) and len(b12):
            sp0 = (g0[0] - b0[0]) * 100
            sp12 = (g12[0] - b12[0]) * 100
            ax.annotate(
                f"Spread(0)={sp0:.1f}pp\nSpread(+12)={sp12:.1f}pp\n"
                f"Drift={sp12 - sp0:+.1f}pp",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
            )
    fig.suptitle(
        "Post-Earnings Announcement Drift (PEAD) — Ann + SUE, Window [-12, +12]",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig4_pead")
    return fig


def fig5_pead_by_size(pead_size):
    """Figure 5: PEAD by firm size — small vs large cap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        ("Small", "Good"): (COLORS["good"], "-", 2.5),
        ("Small", "Bad"): (COLORS["bad"], "-", 2.5),
        ("Large", "Good"): (COLORS["good"], "--", 1.5),
        ("Large", "Bad"): (COLORS["bad"], "--", 1.5),
    }
    for (sz, nw), (c, ls, lw) in styles.items():
        d = pead_size[
            (pead_size["size"] == sz) & (pead_size["news"] == nw)
        ].sort_values("event_month")
        if len(d):
            ax.plot(d["event_month"], d["api"], label=f"{sz}-{nw}",
                    color=c, ls=ls, lw=lw, marker="o", markersize=2)
    ax.axhline(1, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvspan(0, 12, alpha=0.05, color="blue")
    ax.set_xlabel("Month Relative to Announcement")
    ax.set_ylabel("API")
    ax.set_title(
        "PEAD by Firm Size — Simple EW + SUE, Window [-12, +12]",
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-12.5, 12.5)
    plt.tight_layout()
    _save(fig, "fig5_pead_by_size")
    return fig


def fig6_strategy_comparison(strategy_results):
    """Figure 6: Strategy performance comparison bar chart."""
    names = []
    sharpes = []
    alphas = []

    for name, (result, _) in strategy_results.items():
        names.append(name)
        sharpes.append(result.sharpe_ratio)
        alphas.append(
            result.ffc_alpha * 10000 if result.ffc_alpha else 0
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(names))
    bars1 = ax1.bar(x, sharpes, color=[COLORS["good"] if s > 0 else COLORS["bad"]
                                        for s in sharpes], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Annualized Sharpe Ratio")
    ax1.set_title("Sharpe Ratio by Strategy", fontweight="bold")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.grid(True, alpha=0.2, axis="y")

    bars2 = ax2.bar(x, alphas, color=[COLORS["good"] if a > 0 else COLORS["bad"]
                                       for a in alphas], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("FFC Alpha (bps/month)")
    ax2.set_title("Four-Factor Alpha by Strategy", fontweight="bold")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    _save(fig, "fig6_strategy_comparison")
    return fig


def fig7_net_of_cost(gross_returns, net_returns, spread_bps):
    """Figure 7: Gross vs net-of-transaction-cost cumulative returns."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gross_cum = (1 + gross_returns).cumprod()
    net_cum = (1 + net_returns).cumprod()
    months = np.arange(len(gross_cum))

    ax.plot(months, gross_cum, label="Gross", color=COLORS["good"], lw=2)
    ax.plot(months, net_cum, label="Net of TC", color=COLORS["bad"],
            lw=2, ls="--")
    ax.fill_between(months, net_cum, gross_cum, alpha=0.1, color="gray")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(
        f"Gross vs Net Returns (median spread: {spread_bps:.0f} bps)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "fig7_net_of_cost")
    return fig
