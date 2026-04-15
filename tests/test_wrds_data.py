"""
═══════════════════════════════════════════════════════════════════════════════
WRDS DATA VALIDATION & PIPELINE TEST
═══════════════════════════════════════════════════════════════════════════════

Run this FIRST before any notebook. Validates every WRDS table, column name,
data type, value range, and cross-table linkage.

Usage (Colab):
    %run tests/test_wrds_data.py

Usage (local):
    python tests/test_wrds_data.py

Runtime: ~3-5 minutes
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Also try common Colab locations
for _candidate in [
    "/content/drive/MyDrive/Suresh1.github",
    "/content/drive/MyDrive/Suresh1.github/repo",
    "/content/drive/MyDrive/Research.Suresh1/repo",
    "/content/drive/MyDrive/Research.Suresh1",
    "/content/drive/MyDrive/earnings-factor-decomposition",
]:
    if os.path.isdir(os.path.join(_candidate, "src")):
        sys.path.insert(0, _candidate)
        break
import wrds
import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.db = None
        self._lines = []

    def _out(self, msg=""):
        print(msg)
        self._lines.append(msg)

    def section(self, title):
        self._out(f"\n{'═' * 75}")
        self._out(f"  {title}")
        self._out(f"{'═' * 75}")

    def ok(self, desc, cond, detail=""):
        if cond:
            self._out(f"  ✓ {desc}")
            self.passed += 1
        else:
            self._out(f"  ✗ FAIL: {desc}")
            if detail:
                self._out(f"    → {detail}")
            self.failed += 1
            self.errors.append(desc)

    def warn(self, desc, detail=""):
        self._out(f"  ⚠ {desc}")
        if detail:
            self._out(f"    → {detail}")
        self.warnings += 1

    def info(self, msg):
        self._out(f"    {msg}")

    def q(self, sql, label="query"):
        """Run SQL; return DataFrame or None."""
        try:
            return self.db.raw_sql(sql)
        except Exception as e:
            self._out(f"  ✗ {label}: {e}")
            self.failed += 1
            self.errors.append(f"{label}: {e}")
            return None

    def summary(self):
        self.section("SUMMARY")
        tot = self.passed + self.failed
        self._out(f"  Passed:   {self.passed}/{tot}")
        self._out(f"  Failed:   {self.failed}/{tot}")
        self._out(f"  Warnings: {self.warnings}")
        if self.errors:
            self._out(f"\n  FAILURES:")
            for e in self.errors:
                self._out(f"    • {e}")
        if self.failed == 0:
            self._out(f"\n  ✓ ALL TESTS PASSED — safe to run notebooks")
        else:
            self._out(f"\n  ✗ {self.failed} test(s) failed — fix before running")
        # Save
        try:
            from src.config import OUTPUT_DIR
            rpt = os.path.join(OUTPUT_DIR, "test_report.txt")
        except ImportError:
            rpt = os.path.join(os.path.dirname(__file__), "..", "outputs", "test_report.txt")
        os.makedirs(os.path.dirname(rpt), exist_ok=True)
        with open(rpt, "w") as f:
            f.write("\n".join(self._lines))
        self._out(f"\n  Report → {rpt}")


T = TestRunner()
FY = 2023

# ═════════════════════════════════════════════════════════════════════════════
# WRDS TABLE SCHEMA — exact column names, expected pandas dtypes after query,
# and value constraints. This is the single source of truth.
#
# WRDS/PostgreSQL → pandas dtype mapping:
#   varchar/text   → object (str)
#   numeric/float8 → float64
#   int4/int8      → float64 (because NULL → NaN forces float)
#   date           → object (str, must call pd.to_datetime)
#   boolean        → object or bool
# ═════════════════════════════════════════════════════════════════════════════

SCHEMA = {
    "ibes.statsum_epsus": {
        "columns": {
            "ticker":      {"dtype": "object",  "nullable": False, "desc": "I/B/E/S ticker symbol"},
            "cusip":       {"dtype": "object",  "nullable": True,  "desc": "CUSIP (8-char)"},
            "cname":       {"dtype": "object",  "nullable": True,  "desc": "Company name"},
            "fpedats":     {"dtype": "object",  "nullable": False, "desc": "Fiscal period end date",    "convert": "datetime"},
            "statpers":    {"dtype": "object",  "nullable": False, "desc": "Statistical period date",   "convert": "datetime"},
            "medest":      {"dtype": "float64", "nullable": True,  "desc": "Median EPS forecast"},
            "meanest":     {"dtype": "float64", "nullable": True,  "desc": "Mean EPS forecast"},
            "numest":      {"dtype": "float64", "nullable": False, "desc": "Number of analysts",        "min": 1},
            "stdev":       {"dtype": "float64", "nullable": True,  "desc": "Forecast std dev",          "min": 0},
            "actual":      {"dtype": "float64", "nullable": False, "desc": "Actual reported EPS"},
            "anndats_act": {"dtype": "object",  "nullable": True,  "desc": "Actual announcement date",  "convert": "datetime"},
            "fpi":         {"dtype": "object",  "nullable": False, "desc": "Forecast period indicator",  "values": ["0", "1", "2", "6", "7", "8", "9"]},
            "usfirm":      {"dtype": "float64", "nullable": True,  "desc": "US firm indicator",          "values_approx": [0, 1]},
        },
        "test_query": f"""
            SELECT ticker, cusip, cname, fpedats, statpers,
                   medest, meanest, numest, stdev, actual, anndats_act, fpi, usfirm
            FROM ibes.statsum_epsus
            WHERE fpi = '1'
              AND EXTRACT(YEAR FROM fpedats) = {FY}
              AND EXTRACT(MONTH FROM fpedats) = 12
              AND actual IS NOT NULL AND numest >= 3 AND usfirm = 1
            LIMIT 100
        """,
    },
    "wrdsapps.ibcrsphist": {
        "columns": {
            "ticker": {"dtype": "object",  "nullable": False, "desc": "I/B/E/S ticker"},
            "permno": {"dtype": "float64", "nullable": False, "desc": "CRSP PERMNO",    "min": 10000},
            "sdate":  {"dtype": "object",  "nullable": True,  "desc": "Link start date", "convert": "datetime"},
            "edate":  {"dtype": "object",  "nullable": True,  "desc": "Link end date",   "convert": "datetime"},
            "score":  {"dtype": "float64", "nullable": True,  "desc": "Link quality score"},
        },
        "test_query": "SELECT ticker, permno, sdate, edate, score FROM wrdsapps.ibcrsphist LIMIT 100",
    },
    "crsp.msf": {
        "columns": {
            "permno": {"dtype": "float64", "nullable": False, "desc": "CRSP permanent ID",  "min": 10000},
            "date":   {"dtype": "object",  "nullable": False, "desc": "Month-end date",     "convert": "datetime"},
            "ret":    {"dtype": "float64", "nullable": True,  "desc": "Monthly return",     "min": -0.99, "max": 20.0},
            "prc":    {"dtype": "float64", "nullable": True,  "desc": "Closing price (neg=bid)"},
            "vol":    {"dtype": "float64", "nullable": True,  "desc": "Trading volume",     "min": 0},
            "shrout": {"dtype": "float64", "nullable": True,  "desc": "Shares outstanding", "min": 0},
        },
        "test_query": f"""
            SELECT permno, date, ret, prc, vol, shrout FROM crsp.msf
            WHERE date BETWEEN '{FY-1}-01-01' AND '{FY+1}-12-31' AND ret IS NOT NULL
            LIMIT 100
        """,
    },
    "crsp.msi": {
        "columns": {
            "date":   {"dtype": "object",  "nullable": False, "desc": "Month-end date", "convert": "datetime"},
            "ewretd": {"dtype": "float64", "nullable": False, "desc": "Equal-weighted market return"},
            "vwretd": {"dtype": "float64", "nullable": False, "desc": "Value-weighted market return"},
        },
        "test_query": f"""
            SELECT date, ewretd, vwretd FROM crsp.msi
            WHERE date BETWEEN '{FY-3}-01-01' AND '{FY+1}-12-31'
        """,
    },
    "crsp.dsf": {
        "columns": {
            "permno": {"dtype": "float64", "nullable": False, "desc": "CRSP permanent ID"},
            "date":   {"dtype": "object",  "nullable": False, "desc": "Trading date",    "convert": "datetime"},
            "ret":    {"dtype": "float64", "nullable": True,  "desc": "Daily return"},
            "prc":    {"dtype": "float64", "nullable": True,  "desc": "Closing price"},
            "vol":    {"dtype": "float64", "nullable": True,  "desc": "Volume"},
            "askhi":  {"dtype": "float64", "nullable": True,  "desc": "Daily high ask",  "min": 0},
            "bidlo":  {"dtype": "float64", "nullable": True,  "desc": "Daily low bid",   "min": 0},
        },
        "test_query": f"""
            SELECT permno, date, ret, prc, vol, askhi, bidlo FROM crsp.dsf
            WHERE date BETWEEN '{FY+1}-01-01' AND '{FY+1}-03-31'
              AND askhi IS NOT NULL AND bidlo IS NOT NULL
            LIMIT 100
        """,
    },
    "ff.factors_monthly": {
        "columns": {
            "date":  {"dtype": "object",  "nullable": False, "desc": "Month-end date",   "convert": "datetime"},
            "mktrf": {"dtype": "float64", "nullable": False, "desc": "Market excess return"},
            "smb":   {"dtype": "float64", "nullable": False, "desc": "Small minus Big"},
            "hml":   {"dtype": "float64", "nullable": False, "desc": "High minus Low"},
            "umd":   {"dtype": "float64", "nullable": True,  "desc": "Momentum (Up minus Down)"},
            "rf":    {"dtype": "float64", "nullable": False, "desc": "Risk-free rate",    "min": -0.01},
        },
        "test_query": f"""
            SELECT date, mktrf, smb, hml, umd, rf FROM ff.factors_monthly
            WHERE date BETWEEN '{FY-3}-01-01' AND '{FY+1}-12-31'
        """,
    },
    "comp.funda": {
        "columns": {
            "gvkey":   {"dtype": "object",  "nullable": False, "desc": "Compustat firm key"},
            "fyear":   {"dtype": "float64", "nullable": False, "desc": "Fiscal year"},
            "datadate": {"dtype": "object", "nullable": False, "desc": "Data date",    "convert": "datetime"},
            "at":      {"dtype": "float64", "nullable": True,  "desc": "Total assets",  "min": 0},
            "ceq":     {"dtype": "float64", "nullable": True,  "desc": "Common equity"},
            "lt":      {"dtype": "float64", "nullable": True,  "desc": "Total liabilities", "min": 0},
            "prcc_f":  {"dtype": "float64", "nullable": True,  "desc": "FY-end stock price", "min": 0},
            "csho":    {"dtype": "float64", "nullable": True,  "desc": "Shares outstanding", "min": 0},
            "sich":    {"dtype": "float64", "nullable": True,  "desc": "SIC industry code"},
            "indfmt":  {"dtype": "object",  "nullable": False, "desc": "Industry format"},
            "datafmt": {"dtype": "object",  "nullable": False, "desc": "Data format"},
            "popsrc":  {"dtype": "object",  "nullable": False, "desc": "Population source"},
            "consol":  {"dtype": "object",  "nullable": False, "desc": "Consolidation level"},
        },
        "test_query": f"""
            SELECT gvkey, fyear, datadate, at, ceq, lt, prcc_f, csho, sich,
                   indfmt, datafmt, popsrc, consol
            FROM comp.funda
            WHERE fyear = {FY} AND indfmt = 'INDL' AND datafmt = 'STD'
              AND popsrc = 'D' AND consol = 'C'
            LIMIT 100
        """,
    },
    "crsp.ccmxpf_linktable": {
        "columns": {
            "gvkey":     {"dtype": "object",  "nullable": False, "desc": "Compustat key"},
            "lpermno":   {"dtype": "float64", "nullable": False, "desc": "CRSP PERMNO"},
            "linktype":  {"dtype": "object",  "nullable": False, "desc": "Link type (LU/LC/LN/...)"},
            "linkprim":  {"dtype": "object",  "nullable": True,  "desc": "Primary link (P/C/J/N)"},
            "linkdt":    {"dtype": "object",  "nullable": True,  "desc": "Link start date", "convert": "datetime"},
            "linkenddt": {"dtype": "object",  "nullable": True,  "desc": "Link end date",   "convert": "datetime"},
        },
        "test_query": "SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt FROM crsp.ccmxpf_linktable LIMIT 100",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONNECTION
# ═════════════════════════════════════════════════════════════════════════════

T.section("1. WRDS CONNECTION")
T._out(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    T.db = wrds.Connection()
    T.ok("WRDS connection", True)
except Exception as e:
    T.ok("WRDS connection", False, str(e))
    T.summary()
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# 2. SCHEMA VALIDATION — every table, column, dtype
# ═════════════════════════════════════════════════════════════════════════════

T.section("2. SCHEMA VALIDATION (all tables)")

for table_name, spec in SCHEMA.items():
    T._out(f"\n  ── {table_name} ──")

    df = T.q(spec["test_query"], label=table_name)
    if df is None:
        T.ok(f"{table_name} accessible", False, "Query failed — check WRDS subscription")
        continue

    T.ok(f"{table_name}: {len(df)} rows returned", len(df) > 0)

    actual_cols = set(df.columns.str.lower())
    expected_cols = spec["columns"]

    for col_name, col_spec in expected_cols.items():
        col_lower = col_name.lower()

        # Column exists?
        if col_lower not in actual_cols:
            T.ok(f"  {col_name} exists", False,
                 f"Missing. Available: {sorted(actual_cols)}")
            continue

        col_data = df[col_lower] if col_lower in df.columns else df[col_name]
        actual_dtype = str(col_data.dtype)
        expected_dtype = col_spec["dtype"]

        # Dtype check — flexible for string/object and Int64/float64
        dtype_ok = (
            actual_dtype == expected_dtype
            or (expected_dtype == "float64" and actual_dtype in ("float64", "int64", "Float64", "Int64"))
            or (expected_dtype == "object" and actual_dtype in ("object", "string", "str"))
        )
        T.ok(f"  {col_name:<14} dtype={actual_dtype:<10} (expect {expected_dtype})", dtype_ok,
             f"Got {actual_dtype}, expected {expected_dtype}")

        # Null check
        nulls = col_data.isna().sum()
        if not col_spec.get("nullable", True) and nulls > 0:
            T.warn(f"  {col_name}: {nulls} nulls (expected non-nullable)")

        # Value range
        if "min" in col_spec and pd.api.types.is_numeric_dtype(col_data):
            below = (col_data.dropna() < col_spec["min"]).sum()
            if below > 0:
                T.warn(f"  {col_name}: {below} values below min={col_spec['min']}")

        if "max" in col_spec and pd.api.types.is_numeric_dtype(col_data):
            above = (col_data.dropna() > col_spec["max"]).sum()
            if above > 0:
                T.warn(f"  {col_name}: {above} values above max={col_spec['max']}")

    # Summary stats
    T.info(f"Shape: {df.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. I/B/E/S SAMPLE SIZE & SUE
# ═════════════════════════════════════════════════════════════════════════════

T.section("3. I/B/E/S SAMPLE + SUE COMPUTATION")

ibes_full = T.q(f"""
    WITH ranked AS (
        SELECT ticker, cusip, fpedats, actual, medest, stdev, numest, anndats_act,
               ROW_NUMBER() OVER (PARTITION BY ticker, fpedats ORDER BY statpers DESC) rn
        FROM ibes.statsum_epsus
        WHERE fpi = '1'
          AND EXTRACT(YEAR FROM fpedats) = {FY}
          AND EXTRACT(MONTH FROM fpedats) = 12
          AND actual IS NOT NULL AND numest >= 3 AND usfirm = 1
    )
    SELECT * FROM ranked WHERE rn = 1
""", label="IBES full sample")

if ibes_full is not None:
    n = len(ibes_full)
    T.ok(f"IBES FY{FY}: {n} firms (expect ~3000)", n > 1000, f"Only {n}")

    # SUE
    ibes_full["sue"] = (ibes_full["actual"] - ibes_full["medest"]) / ibes_full["stdev"]
    sue_ok = ibes_full["sue"].notna().sum()
    T.ok(f"SUE computable: {sue_ok}/{n} ({sue_ok/n*100:.0f}%)", sue_ok > n * 0.7)

    sue = ibes_full["sue"].dropna().astype(float)
    sue = sue[np.isfinite(sue)]  # drop inf from zero-stdev firms
    n_good = (sue > 0).sum()
    n_bad  = (sue < 0).sum()
    T.ok(f"Good={n_good}, Bad={n_bad} (both > 100)", n_good > 100 and n_bad > 100)
    T.info(f"SUE: mean={sue.mean():.3f} median={sue.median():.3f} "
           f"std={sue.std():.3f} [{sue.min():.1f}, {sue.max():.1f}]")

    # Announcement dates
    ibes_full["anndats_act"] = pd.to_datetime(ibes_full["anndats_act"])
    ann = ibes_full["anndats_act"].dropna()
    pct_q1 = ((ann.dt.year == FY + 1) & (ann.dt.month <= 4)).mean()
    T.ok(f"Announcements: {pct_q1:.0%} in Q1 of {FY+1} (expect >50%)", pct_q1 > 0.5)
    T.info(f"Ann dates: {ann.min().date()} to {ann.max().date()}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. TSUE HISTORICAL COVERAGE
# ═════════════════════════════════════════════════════════════════════════════

T.section("4. TSUE HISTORICAL DATA")

tsue_ct = T.q(f"""
    SELECT EXTRACT(YEAR FROM fpedats)::int AS fyear, COUNT(DISTINCT ticker) AS n
    FROM ibes.statsum_epsus
    WHERE fpi = '1' AND EXTRACT(MONTH FROM fpedats) = 12
      AND actual IS NOT NULL AND usfirm = 1
      AND EXTRACT(YEAR FROM fpedats) BETWEEN 2010 AND {FY}
    GROUP BY 1 ORDER BY 1
""", label="TSUE year counts")

if tsue_ct is not None:
    T.ok(f"Historical years: {len(tsue_ct)} (expect ≥10)", len(tsue_ct) >= 10)
    for _, row in tsue_ct.iterrows():
        T.info(f"FY{int(row['fyear'])}: {int(row['n'])} firms")


# ═════════════════════════════════════════════════════════════════════════════
# 5. IBES→CRSP LINKING
# ═════════════════════════════════════════════════════════════════════════════

T.section("5. IBES→CRSP LINK")

link_ct = T.q("SELECT COUNT(*) AS n FROM wrdsapps.ibcrsphist", label="link count")
if link_ct is not None:
    T.ok(f"Link table: {int(link_ct['n'].iloc[0]):,} rows", link_ct["n"].iloc[0] > 10000)

# Test match rate on our IBES sample
if ibes_full is not None:
    sample_tickers = ibes_full["ticker"].unique()[:50]
    tick_str = "','".join(sample_tickers)
    matches = T.q(f"""
        SELECT DISTINCT ticker FROM wrdsapps.ibcrsphist
        WHERE ticker IN ('{tick_str}')
    """, label="link match test")
    if matches is not None:
        rate = len(matches) / len(sample_tickers) * 100
        T.ok(f"Link match: {rate:.0f}% of 50 sample tickers", rate > 70)


# ═════════════════════════════════════════════════════════════════════════════
# 6. FF-MSI MERGE TEST
# ═════════════════════════════════════════════════════════════════════════════

T.section("6. FF ↔ MSI MERGE")

msi = T.q(f"SELECT date, ewretd, vwretd FROM crsp.msi WHERE date BETWEEN '{FY-3}-01-01' AND '{FY+1}-12-31'")
ff  = T.q(f"SELECT date, mktrf, smb, hml, umd, rf FROM ff.factors_monthly WHERE date BETWEEN '{FY-3}-01-01' AND '{FY+1}-12-31'")

if msi is not None and ff is not None:
    msi["ym"] = pd.to_datetime(msi["date"]).dt.to_period("M")
    ff["ym"]  = pd.to_datetime(ff["date"]).dt.to_period("M")
    merged = msi.merge(ff.drop(columns="date"), on="ym", how="inner")
    T.ok(f"MSI-FF merged: {len(merged)} months (expect ~60)", len(merged) >= 48)

    for col in ["mktrf", "smb", "hml", "umd", "rf", "ewretd", "vwretd"]:
        nulls = merged[col].isna().sum()
        T.ok(f"  {col}: {nulls} nulls after merge", nulls == 0)

    T.ok("UMD available for Carhart model", merged["umd"].notna().all(),
         f"{merged['umd'].isna().sum()} null UMD values")


# ═════════════════════════════════════════════════════════════════════════════
# 7. COMPUSTAT CONTROLS CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

T.section("7. COMPUSTAT CONTROL VARIABLES")

comp = T.q(f"""
    SELECT a.at, a.ceq, a.lt, a.prcc_f, a.csho, a.sich,
           b.lpermno AS permno
    FROM comp.funda a
    INNER JOIN crsp.ccmxpf_linktable b
      ON a.gvkey = b.gvkey
      AND b.linktype IN ('LU','LC') AND b.linkprim IN ('P','C')
      AND a.datadate >= b.linkdt
      AND (a.datadate <= b.linkenddt OR b.linkenddt IS NULL)
    WHERE a.fyear = {FY} AND a.indfmt = 'INDL' AND a.datafmt = 'STD'
      AND a.popsrc = 'D' AND a.consol = 'C'
    LIMIT 500
""", label="Compustat + CCM")

if comp is not None:
    # Test every control variable computation
    comp["mktcap"]   = comp["prcc_f"] * comp["csho"]
    comp["ln_size"]  = np.log(comp["mktcap"].clip(lower=1e-6))
    comp["bm"]       = comp["ceq"] / comp["mktcap"].clip(1e-6)
    comp["leverage"] = comp["lt"] / comp["at"].clip(1e-6)
    comp["sic2"]     = (comp["sich"] // 100).astype("Int64")

    for var, desc, expect_med in [
        ("ln_size",  "log(Market Cap)",        (5, 12)),
        ("bm",       "Book-to-Market",         (0, 3)),
        ("leverage", "Liabilities/Assets",     (0, 2)),
    ]:
        valid = comp[var].replace([np.inf, -np.inf], np.nan).dropna()
        med = valid.median()
        T.ok(f"{var} ({desc}): median={med:.2f} (expect [{expect_med[0]},{expect_med[1]}])",
             expect_med[0] <= med <= expect_med[1])

    T.ok(f"SIC2: {comp['sic2'].nunique()} industries (expect 30+)",
         comp["sic2"].nunique() > 20)


# ═════════════════════════════════════════════════════════════════════════════
# 8. CRSP DAILY BID-ASK (for transaction costs)
# ═════════════════════════════════════════════════════════════════════════════

T.section("8. CRSP DAILY BID-ASK (Transaction Costs)")

# Pick 20 PERMNOs from CRSP monthly to test daily data
dsf_permnos = T.q(f"""
    SELECT DISTINCT permno FROM crsp.msf
    WHERE date BETWEEN '{FY+1}-01-01' AND '{FY+1}-03-31'
      AND ret IS NOT NULL
    LIMIT 20
""", label="sample permnos for daily")

if dsf_permnos is not None and len(dsf_permnos) > 0:
    pstr = ",".join(dsf_permnos["permno"].astype(int).astype(str))
    dsf = T.q(f"""
        SELECT permno, date, askhi, bidlo, ABS(prc) AS prc
        FROM crsp.dsf
        WHERE permno IN ({pstr})
          AND date BETWEEN '{FY+1}-01-01' AND '{FY+1}-03-31'
          AND askhi IS NOT NULL AND bidlo IS NOT NULL
          AND askhi > 0 AND bidlo > 0
    """, label="CRSP daily bid-ask")

    if dsf is not None and len(dsf) > 0:
        T.ok(f"Daily obs: {len(dsf):,}", len(dsf) > 100)

        askhi = dsf["askhi"].astype(float)
        bidlo = dsf["bidlo"].astype(float)
        T.ok("askhi > bidlo (basic sanity)", (askhi > bidlo).mean() > 0.90)

        spread = (askhi - bidlo) / ((askhi + bidlo) / 2) * 10000
        T.info(f"Spread (bps): median={spread.median():.0f}, mean={spread.mean():.0f}")

        dsf["date"] = pd.to_datetime(dsf["date"])
        days_per_firm = dsf.groupby("permno").size()
        n_with_5 = (days_per_firm >= 5).sum()
        T.ok(f"Firms with ≥5 days: {n_with_5}/{len(days_per_firm)}", n_with_5 > 5)
    else:
        T.warn("No CRSP daily data returned")


# ═════════════════════════════════════════════════════════════════════════════
# 9. TAQ AUTO-DETECT
# ═════════════════════════════════════════════════════════════════════════════

T.section("9. WRDS INTRADAY INDICATORS (taqmsec.wrds_iid_2024)")

iid = T.q(f"""
    SELECT date, sym_root,
           quotedspread_percent_tw,
           effectivespread_percent_ave,
           effectivespread_percent_dw
    FROM taqmsec.wrds_iid_2024
    WHERE date BETWEEN '{FY+1}-01-01' AND '{FY+1}-03-31'
    LIMIT 100
""", label="taqmsec.wrds_iid_2024")

if iid is not None and len(iid) > 0:
    T.ok(f"taqmsec.wrds_iid_2024: {len(iid)} rows", True)
    for col in ["quotedspread_percent_tw", "effectivespread_percent_ave"]:
        if col in iid.columns:
            vals = iid[col].astype(float).dropna()
            T.ok(f"  {col}: {len(vals)} non-null", len(vals) > 0)
            if len(vals) > 0:
                T.info(f"    median={vals.median()*100:.0f} bps, "
                       f"mean={vals.mean()*100:.0f} bps")
    T.info("(This is the primary TC source — pre-computed by WRDS)")
else:
    T.warn("taqmsec.wrds_iid_2024 not accessible",
           "Will fall back to CRSP daily bid-ask for TC estimation")


# ═════════════════════════════════════════════════════════════════════════════
# 10. SMOKE TEST — 5-firm end-to-end
# ═════════════════════════════════════════════════════════════════════════════

T.section("10. SMOKE TEST (5 firms)")

try:
    if ibes_full is not None and msi is not None and ff is not None:
        # Pick 5 firms, link, pull returns, estimate market model
        ticks = ibes_full["ticker"].head(10).tolist()
        tick_str = "','".join(ticks)
        lnk = T.q(f"SELECT DISTINCT ticker, permno FROM wrdsapps.ibcrsphist WHERE ticker IN ('{tick_str}')")

        if lnk is not None and len(lnk) > 0:
            permnos = lnk["permno"].dropna().astype(int).unique()[:5]
            pstr = ",".join(map(str, permnos))
            T.info(f"PERMNOs: {permnos.tolist()}")

            rets = T.q(f"""
                SELECT permno, date, ret FROM crsp.msf
                WHERE permno IN ({pstr})
                  AND date BETWEEN '{FY-2}-01-01' AND '{FY+1}-12-31'
                  AND ret IS NOT NULL
            """, label="smoke test returns")

            if rets is not None and len(rets) > 0:
                rets["date"] = pd.to_datetime(rets["date"])
                rets["ym"] = rets["date"].dt.to_period("M")
                msi_dt = msi.copy()
                msi_dt["date"] = pd.to_datetime(msi_dt["date"])
                msi_dt["ym"] = msi_dt["date"].dt.to_period("M")
                ff_dt = ff.copy()
                ff_dt["date"] = pd.to_datetime(ff_dt["date"])
                ff_dt["ym"] = ff_dt["date"].dt.to_period("M")

                mkt = msi_dt.merge(ff_dt.drop(columns="date"), on="ym", how="left")
                panel = rets.merge(mkt.drop(columns="date"), on="ym", how="inner")

                # Market model
                import statsmodels.api as sm
                p = permnos[0]
                firm = panel[panel["permno"] == p].dropna(subset=["ret", "vwretd"])
                T.ok(f"PERMNO {p}: {len(firm)} months", len(firm) >= 12)

                if len(firm) >= 12:
                    y = firm["ret"].astype(float).values
                    X = sm.add_constant(firm["vwretd"].astype(float).values)
                    m = sm.OLS(y, X).fit(cov_type="HC1")
                    T.ok(f"Market model OLS (HC1): α={m.params[0]:.4f}, β={m.params[1]:.4f}", True)

                    # FFC
                    ffc_data = firm.dropna(subset=["mktrf", "smb", "hml", "umd", "rf"])
                    if len(ffc_data) >= 12:
                        y_ex = ffc_data["ret"].astype(float).values - ffc_data["rf"].astype(float).values
                        X_ffc = sm.add_constant(ffc_data[["mktrf", "smb", "hml", "umd"]].astype(float).values)
                        m_ffc = sm.OLS(y_ex, X_ffc).fit(cov_type="HC1")
                        T.ok(f"FFC model (HC1): α={m_ffc.params[0]:.4f}, R²={m_ffc.rsquared:.3f}", True)
                    else:
                        T.warn(f"FFC: only {len(ffc_data)} obs with all factors")
except Exception as e:
    T.ok("Smoke test", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# 11. MULTI-YEAR AVAILABILITY
# ═════════════════════════════════════════════════════════════════════════════

T.section("11. MULTI-YEAR AVAILABILITY")

for fy in [2019, 2020, 2021, 2022, 2023]:
    ct = T.q(f"""
        SELECT COUNT(DISTINCT ticker) AS n
        FROM ibes.statsum_epsus
        WHERE fpi = '1' AND EXTRACT(YEAR FROM fpedats) = {fy}
          AND EXTRACT(MONTH FROM fpedats) = 12
          AND actual IS NOT NULL AND numest >= 3 AND usfirm = 1
    """, label=f"FY{fy}")
    if ct is not None:
        n = int(ct["n"].iloc[0])
        T.ok(f"FY{fy}: {n} firms", n > 1000, f"Only {n}")


# ═════════════════════════════════════════════════════════════════════════════
# 12. METHODOLOGY MODULE
# ═════════════════════════════════════════════════════════════════════════════

T.section("12. METHODOLOGY CONSISTENCY")

try:
    from src.methodology import (
        SE_TYPE_CROSS_SECTION, SE_TYPE_TIME_SERIES,
        COVARIATES_MODEL1, COVARIATES_MODEL2, COVARIATES_MODEL3,
        WINSORIZE_LIMITS, ESTIMATION_WINDOW, CAR_WINDOW,
        BOOTSTRAP_ITERATIONS, INDUSTRY_FE_COLUMN, INDUSTRY_FE_MIN_OBS,
    )

    T.ok("methodology.py imports", True)
    T.ok("SE consistent (HC1)", SE_TYPE_CROSS_SECTION == SE_TYPE_TIME_SERIES == "HC1")
    T.ok("Model 2 ⊇ Model 1", all(c in COVARIATES_MODEL2 for c in COVARIATES_MODEL1))
    T.ok("Model 3 ⊇ Model 2", all(c in COVARIATES_MODEL3 for c in COVARIATES_MODEL2))
    T.ok("CAR window is [-1, +1]", CAR_WINDOW == (-1, 1))
    T.ok("Estimation window is [-24, -4]", ESTIMATION_WINDOW == (-24, -4))
    T.info(f"M1: {COVARIATES_MODEL1}")
    T.info(f"M2: {COVARIATES_MODEL2}")
    T.info(f"M3: {COVARIATES_MODEL3}")
    T.info(f"SE: {SE_TYPE_CROSS_SECTION} | FE: {INDUSTRY_FE_COLUMN} (min {INDUSTRY_FE_MIN_OBS})")
except ImportError as e:
    T.ok("methodology.py imports", False, str(e))

# All src modules
import importlib
for mod in ["config", "utils", "data", "abnormal_returns", "api",
            "inference", "backtest", "transaction_costs", "plotting", "methodology"]:
    full = f"src.{mod}"
    try:
        # Clear any cached failed import from previous test runs
        if full in sys.modules:
            del sys.modules[full]
        importlib.import_module(full)
        T.ok(f"src.{mod} imports", True)
    except Exception as e:
        T.ok(f"src.{mod} imports", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
T.summary()
