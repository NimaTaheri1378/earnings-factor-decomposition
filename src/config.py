"""
Global configuration — all tunable parameters and paths in one place.
"""
import os

# ── Google Drive base (Colab) ────────────────────────────────────────────────
BASE_DIR = os.environ.get(
    "EFD_BASE_DIR",
    "/content/drive/MyDrive/Suresh1.github",
)
INPUT_DIR  = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR  = os.path.join(OUTPUT_DIR, "tables")

for _d in [BASE_DIR, INPUT_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Fiscal years ─────────────────────────────────────────────────────────────
FISCAL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
PRIMARY_FY   = 2023

# ── Sample filters ───────────────────────────────────────────────────────────
TSUE_HISTORY_START = 2010
MIN_ANALYSTS       = 3
MIN_PRICE          = 5.0
MIN_EST_MONTHS     = 12
WINSORIZE_PCT      = 0.01

# ── Event study windows ─────────────────────────────────────────────────────
EVENT_WINDOW_START = -12
EVENT_WINDOW_END   =   6
PEAD_WINDOW_END    =  12
ESTIMATION_START   = -24
ESTIMATION_END     =  -4

# ── Inference ────────────────────────────────────────────────────────────────
N_BOOTSTRAP  = 1000
BOOTSTRAP_CI = 0.95

# ── Backtest ─────────────────────────────────────────────────────────────────
HOLDING_PERIODS = [1, 3, 6, 12]
QUINTILE_BINS   = 5

# ── Model labels ─────────────────────────────────────────────────────────────
AR_MODELS = {
    "ar_simple_ew": "Simple (EW)",
    "ar_simple_vw": "Simple (VW)",
    "ar_mm_ew":     "MM (EW)",
    "ar_mm_vw":     "MM (VW)",
    "ar_ffc":       "FFC",
}

# ── Plot style ───────────────────────────────────────────────────────────────
PLOT_PARAMS = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
}
COLORS = {
    "good": "#2ca02c", "bad": "#d62728", "neutral": "#7f7f7f",
    "quintiles": ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"],
}
