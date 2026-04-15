"""
═══════════════════════════════════════════════════════════════════════════════
METHODOLOGY — Single source of truth for all methodological decisions
═══════════════════════════════════════════════════════════════════════════════

Every regression, standard error, covariate set, and statistical test in this
project is defined HERE. No module should hardcode a method choice. This
ensures consistency across replication (inference.py), backtests (backtest.py),
and robustness checks.

Change a method once here → it propagates everywhere.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1. STANDARD ERRORS
# ═════════════════════════════════════════════════════════════════════════════
# All cross-sectional OLS:  HC1 (White 1980)
# All time-series OLS:      HC1 for consistency
#   - Note: Newey-West (HAC) would be more conservative for time-series
#     regressions (backtest alpha). We use HC1 for uniformity; robustness
#     with HAC(3) is available via se_type_ts_robust below.
SE_TYPE_CROSS_SECTION = "HC1"       # White (1980) — all cross-sectional regs
SE_TYPE_TIME_SERIES = "HC1"         # used in FFC alpha regressions
SE_TYPE_TS_ROBUST = "HAC"           # Newey-West alternative (robustness only)
HAC_MAXLAGS = 3                     # for Newey-West if used

# ═════════════════════════════════════════════════════════════════════════════
# 2. COVARIATES — Cross-Sectional Regressions (Table 5)
# ═════════════════════════════════════════════════════════════════════════════
# Three nested specifications, each building on the previous:

# Model 1: Univariate
COVARIATES_MODEL1 = ["sue"]

# Model 2: + Asset pricing controls
COVARIATES_MODEL2 = [
    "sue",
    "ln_size",      # Banz (1981), Fama & French (1992)
    "bm",           # Fama & French (1992, 1993)
    "momentum",     # Jegadeesh & Titman (1993) — prior 12mo return, skip t-1
]

# Model 3: Full specification
COVARIATES_MODEL3 = [
    "sue",
    "ln_size",      # Banz (1981), Fama & French (1992)
    "bm",           # Fama & French (1992, 1993)
    "momentum",     # Jegadeesh & Titman (1993)
    "ln_numest",    # Hong, Lim, Stein (2000) — log analyst coverage
    "earn_vol",     # Foster, Olsen, Shevlin (1984) — analyst forecast dispersion
    "leverage",     # Standard control — total liabilities / total assets
]

# Industry fixed effects
INDUSTRY_FE_COLUMN = "sic2"         # 2-digit SIC code
INDUSTRY_FE_MIN_OBS = 5             # drop industries with < 5 firms

# ═════════════════════════════════════════════════════════════════════════════
# 3. VARIABLE CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
# How each covariate is built (for documentation / test validation):
VARIABLE_DEFINITIONS = {
    "sue":        "(actual - medest) / stdev, winsorized 1%/99%",
    "ln_size":    "log(prcc_f × csho), from Compustat",
    "bm":         "ceq / (prcc_f × csho), from Compustat",
    "momentum":   "cumulative return months [-12, -2], from CRSP",
    "ln_numest":  "log(numest), from I/B/E/S",
    "earn_vol":   "stdev of analyst forecasts, from I/B/E/S",
    "leverage":   "lt / at, from Compustat",
    "sic2":       "floor(sich / 100), from Compustat",
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. WINSORIZATION
# ═════════════════════════════════════════════════════════════════════════════
WINSORIZE_LIMITS = [0.01, 0.01]     # 1st and 99th percentile, symmetric

# Variables that get winsorized:
WINSORIZE_RETURNS = True            # all stock returns (ret) in CRSP
WINSORIZE_SUE = True                # SUE before classification
WINSORIZE_CONTROLS = [              # continuous controls before regression
    "ln_size", "bm", "leverage", "momentum", "earn_vol",
]

# ═════════════════════════════════════════════════════════════════════════════
# 5. ESTIMATION WINDOWS
# ═════════════════════════════════════════════════════════════════════════════
# Market model & FFC estimation: months [-24, -4] relative to event
# Following Brown & Warner (1985) with 21-month estimation, 3-month gap
ESTIMATION_WINDOW = (-24, -4)
MIN_ESTIMATION_OBS = 12

# Event windows
EVENT_WINDOW_STANDARD = (-12, 6)    # standard Ball & Brown
EVENT_WINDOW_PEAD = (-12, 12)       # extended for drift analysis
CAR_WINDOW = (-1, 1)                # for cross-sectional regressions

# ═════════════════════════════════════════════════════════════════════════════
# 6. BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════
BOOTSTRAP_METHOD = "firm_block"     # resample firms, keep full time-series
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LEVEL = 0.95           # 95% confidence interval
BOOTSTRAP_SEED = None               # set for reproducibility (e.g., 42)

# ═════════════════════════════════════════════════════════════════════════════
# 7. ABNORMAL RETURN MODELS
# ═════════════════════════════════════════════════════════════════════════════
# Model 1: AR = ret - ewretd               Simple EW (Ball & Brown 1968)
# Model 2: AR = ret - vwretd               Simple VW
# Model 3: AR = ret - (α̂ + β̂·ewretd)      Market Model EW (Brown & Warner 1985)
# Model 4: AR = ret - (α̂ + β̂·vwretd)      Market Model VW
# Model 5: AR = (ret-rf) - (α̂ + β̂·MKT + ŝ·SMB + ĥ·HML + p̂·UMD)
#                                           Fama-French-Carhart (1993/1997)

# ═════════════════════════════════════════════════════════════════════════════
# 8. WRDS TABLE REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
WRDS_TABLES = {
    "ibes_summary":     "ibes.statsum_epsus",
    "ibes_crsp_link":   "wrdsapps.ibcrsphist",
    "crsp_monthly":     "crsp.msf",
    "crsp_daily":       "crsp.dsf",
    "crsp_market_idx":  "crsp.msi",
    "crsp_names":       "crsp.msenames",
    "ff_factors":       "ff.factors_monthly",
    "compustat_annual":  "comp.funda",
    "ccm_link":         "crsp.ccmxpf_linktable",
    # TAQ — table names vary by WRDS subscription tier
    # The test script auto-detects which TAQ library is available
    "taq_nbbo":         None,   # auto-detected
    "taq_trades":       None,   # auto-detected
}
