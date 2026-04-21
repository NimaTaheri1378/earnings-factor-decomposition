# Earnings "Alpha" or Factor Compensation?

**99% of measured earnings information content is compensation for systematic risk factors.**

This project replicates and extends [Ball & Brown (1968)](https://doi.org/10.2307/2490232), systematically decomposing earnings announcement abnormal returns across five asset pricing models, then testing whether the residual generates implementable alpha.

## Key Results

### Attenuation: 77–97% across 8 fiscal years

| FY | B&B Spread (pp) | FFC Spread (pp) | Attenuation |
|---|---|---|---|
| 2016 | 10.7 | −0.3 | 103%* |
| 2017 | 9.4 | 0.5 | 94% |
| 2018 | 9.8 | 6.5 | 34%† |
| 2019 | 14.7 | 1.2 | 92% |
| 2020 | 18.6 | 3.0 | 84% |
| 2021 | 20.5 | 0.6 | 97% |
| 2022 | 23.5 | 2.6 | 89% |
| 2023 | 13.0 | 3.0 | 77% |

*\*FY2016: >100% attenuation means FFC spread is slightly negative (factor adjustment flips sign).*
*†FY2018: Low attenuation coincides with TCJA tax reform creating genuine idiosyncratic earnings surprises that factor models cannot absorb.*

*Spread between good-news and bad-news portfolios at month 0. B&B = Simple EW market-adjusted (Ball & Brown exact). FFC = Fama–French–Carhart four-factor. Results for FY2016–2018 generated on first run.*

### Pooled calendar-time alpha (FY2016–2023)

| Strategy | Months | Sharpe | Alpha (bps/mo) | t-stat | p-value |
|---|---|---|---|---|---|
| Binary L/S (Good − Bad) | 60 | 1.75 | 69.0 | 3.97 | <0.001 |
| Decile D10 − D1 | 60 | 1.62 | 103.9 | 3.73 | <0.001 |

*Calendar-time portfolio returns regressed on MKT, SMB, HML, UMD. Alpha is the intercept. Factor loadings are near zero (market neutral, no factor harvesting).*

### Post-earnings drift concentrates in small caps

| Size | Spread(0) | Spread(+12) | Drift |
|---|---|---|---|
| Small cap | 7.2pp | 21.2pp | +14.0pp |
| Large cap | 8.3pp | 10.8pp | +2.5pp |

*Consistent with [Hong, Lim & Stein (2000)](https://doi.org/10.1111/0022-1082.00206) limited-attention hypothesis.*

### Single-year strategies (FY2023, 12-month hold)

| Strategy | Cum. Return | Ann. Return | Sharpe | FFC Alpha (bps/mo) | t-stat | Firms |
|---|---|---|---|---|---|---|
| Binary PEAD L/S | 11.96% | 10.99% | 1.85 | 181.4 | 1.38 | 1,978 |
| Quintile Q5−Q1 | 14.00% | 12.86% | 1.34 | 236.4 | 1.06 | 828 |
| **Decile D10−D1** | **19.87%** | — | **1.53** | **316.0** | 1.09 | 414 |
| Small-Cap PEAD | 18.53% | 16.99% | 2.42 | 255.1 | 1.59 | 951 |
| Large-Cap PEAD | 3.73% | 3.44% | 0.79 | 113.8 | 1.25 | 1,008 |
| Factor-Aware (FFC residual) | 5.21% | 4.80% | 1.04 | 153.6 | 1.79 | 2,290 |
| Decay-Weighted | 7.05% | 12.39% | 2.26 | 344.5 | 2.88 | 2,185 |

*Single-year alphas have low power (13 observations, 5 parameters). See pooled results below for proper inference.*

### Pooled calendar-time strategies (FY2016–2023)

| Strategy | Months | Avg Monthly | Ann. Sharpe | FFC Alpha (bps/mo) | t-stat | p-value |
|---|---|---|---|---|---|---|
| **Binary L/S** | 60 | 0.73% | **1.75** | **69.0** | **3.97** | **<0.001** |
| **Decile D10−D1** | 60 | 1.11% | **1.62** | **103.9** | **3.73** | **<0.001** |

Factor loadings (pooled binary): MKT β = 0.015, SMB = −0.200, HML = 0.012, UMD = 0.010 — near-zero market exposure, slight large-cap tilt, no value or momentum harvesting.

*The significant pooled alpha (t > 3.7) contrasts with the 77–97% attenuation of the raw spread, suggesting a small but real residual of ~69–104 bps/month survives comprehensive factor adjustment. This residual concentrates in small caps and is consistent with limited-attention theories rather than risk compensation.*

### Transaction costs (FY2023)

| Size Group | Median Eff. Spread (bps) | Mean (bps) |
|---|---|---|
| Large cap | 4.5 | 6.5 |
| Small cap | 12.1 | 25.4 |

| Metric | Gross | Net of TC |
|---|---|---|
| Cumulative Return | 11.96% | 7.57% |
| Sharpe | 1.85 | 1.20 |
| Breakeven spread | 2,216 bps | — |
| Actual median spread | 7.7 bps | — |

*Strategy survives transaction costs with wide margin. Spreads from WRDS Intraday Indicators (`taqmsec.wrds_iid`), measuring Lee-Ready effective spreads.*

### Transaction costs

Median effective spread from TAQ intraday indicators (`taqmsec.wrds_iid`):
- Large caps: 4.5 bps
- Small caps: 12.1 bps
- Strategy survives after costs (breakeven ≫ actual spread)

## Methodology

### 20 event-study specifications

5 abnormal return models × 2 surprise measures × 2 event dates:

| Model | Reference |
|---|---|
| Simple EW market-adjusted | Ball & Brown (1968) |
| Simple VW market-adjusted | — |
| Market model (EW & VW) | Brown & Warner (1985) |
| Fama–French–Carhart 4-factor | Fama & French (1993), Carhart (1997) |

| Surprise | Definition |
|---|---|
| SUE (analyst-based) | (Actual − Median Forecast) / σ |
| TSUE (time-series) | Residual from ΔEPSᵢ regressed on market ΔEPS |

### Statistical methods

All choices are centralized in `src/methodology.py`:

- **Standard errors**: HC1 (White 1980) everywhere — cross-sectional and time-series
- **Bootstrap**: 1,000 firm-level block resamples, preserving complete time series
- **Covariates** (3 nested models):
  - M1: SUE only
  - M2: + log(size), book-to-market, momentum
  - M3: + log(analyst coverage), earnings volatility, leverage, SIC2 industry FE
- **Winsorization**: 1st/99th percentile on returns, SUE, and all continuous controls
- **Estimation window**: months [−24, −4] relative to event
- **CAR window**: months [−1, +1] for cross-sectional regressions

### Trading strategies

| Strategy | Logic | Hold |
|---|---|---|
| Binary PEAD L/S | Long Good, short Bad | 12 mo |
| Quintile Q5−Q1 | Long top 20%, short bottom 20% | 12 mo |
| Decile D10−D1 | Long top 10%, short bottom 10% | 12 mo |
| Small-Cap PEAD | Binary in below-median size only | 12 mo |
| Large-Cap PEAD | Binary in above-median size only | 12 mo |
| Factor-Aware | Trade on FFC residuals | 12 mo |
| Decay-Weighted | Overweight months 0–3 | 7 mo |

Multi-year pooled alpha estimated in calendar time across FY2016–2023 with FFC regression.

## Differences from the paper

The code closely replicates the methodology described in the research paper, with these minor differences:

| Item | Paper | Code | Impact |
|---|---|---|---|
| Sample size | 2,358 firms | 2,342 firms | Trivial — date-matching edge cases |
| TSUE history | 2015–2022 | 2010–2022 | Slightly different residuals; more estimation data |
| B&B spread | 12.98pp | 13.01pp | Within bootstrap CI; random winsorization boundary |
| Bootstrap CIs | Fixed | Vary by run | No fixed seed (set `BOOTSTRAP_SEED=42` in methodology.py for exact replication) |
| R² reporting | Unadjusted (some tables) | Adjusted R² throughout | Consistent; slightly lower values |
| Fiscal years | FY2023 only | FY2016–2023 | Extension, not a discrepancy |

## Data sources

All data from [WRDS](https://wrds-www.wharton.upenn.edu/):

| Dataset | WRDS table | What it provides |
|---|---|---|
| I/B/E/S Summary | `ibes.statsum_epsus` | Analyst forecasts, actual EPS, announcement dates |
| I/B/E/S–CRSP Link | `wrdsapps.ibcrsphist` | Ticker → PERMNO mapping |
| CRSP Monthly | `crsp.msf` | Stock returns |
| CRSP Market Index | `crsp.msi` | EW and VW market returns |
| CRSP Daily | `crsp.dsf` | Daily bid-ask for Corwin-Schultz spreads |
| Fama-French Factors | `ff.factors_monthly` | MKT, SMB, HML, UMD, RF |
| Compustat | `comp.funda` | Size, book equity, leverage, SIC codes |
| CCM Link | `crsp.ccmxpf_linktable` | GVKEY → PERMNO mapping |
| TAQ Intraday | `taqmsec.wrds_iid_YYYY` | Pre-computed effective spreads |

**Raw data is not included** (WRDS license). You need a WRDS account to reproduce.

## Repo structure

```
├── run_all.py              # Single script — runs everything
├── src/
│   ├── config.py           # All parameters and paths
│   ├── methodology.py      # SE types, covariates, windows (single source of truth)
│   ├── data.py             # WRDS data pipeline with parquet caching
│   ├── abnormal_returns.py # 5 AR models, vectorized estimation
│   ├── api.py              # Abnormal Performance Index (Ball & Brown)
│   ├── inference.py        # Bootstrap CIs, cross-sectional regressions
│   ├── backtest.py         # 7 trading strategies + FFC alpha
│   ├── transaction_costs.py# TAQ spreads, Corwin-Schultz, net returns
│   ├── plotting.py         # Publication-quality figures
│   └── utils.py            # Winsorization, dtype conversion, caching
├── tests/
│   └── test_wrds_data.py   # Validates all WRDS tables before running
├── paper/
│   └── paper.pdf           # Research paper
├── outputs/
│   ├── tables/             # CSV results
│   └── figures/            # PNG + PDF figures
└── requirements.txt
```

## Quick start
**Requires a [WRDS account](https://wrds-www.wharton.upenn.edu/).**
```bash
# In Google Colab:
!pip install wrds statsmodels scipy -q
!git clone https://github.com/NimaTaheri1378/earnings-factor-decomposition.git
%cd earnings-factor-decomposition
import os, getpass
os.environ['WRDS_USERNAME'] = 'your_wrds_username'
os.environ['WRDS_PASSWORD'] = getpass.getpass('WRDS Password: ')
%run run_all.py
```

Or locally:
```bash
git clone https://github.com/NimaTaheri1378/earnings-factor-decomposition.git
cd earnings-factor-decomposition
pip install -r requirements.txt
export EFD_BASE_DIR=./output
python run_all.py
```

Prompts for WRDS credentials once. First run pulls ~15 minutes of data (cached for subsequent runs). Full pipeline takes ~20 minutes.

## References

- Ball, R. & Brown, P. (1968). An empirical evaluation of accounting income numbers. *JAR*, 6(2), 159–178.
- Bernard, V.L. & Thomas, J.K. (1989). Post-earnings-announcement drift. *JAR*, 27, 1–36.
- Brown, S.J. & Warner, J.B. (1985). Using daily stock returns. *JFE*, 14(1), 3–31.
- Carhart, M. (1997). On persistence in mutual fund performance. *JF*, 52(1), 57–82.
- Fama, E.F. & French, K.R. (1993). Common risk factors. *JFE*, 33(1), 3–56.
- Foster, G., Olsen, C. & Shevlin, T. (1984). Earnings releases and anomalies. *TAR*, 59(4), 574–603.
- Hong, H., Lim, T. & Stein, J.C. (2000). Bad news travels slowly. *JF*, 55(1), 265–295.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners. *JF*, 48(1), 65–91.

## Citation

If you use this code or build on this work, please cite:

> Taheri Hosseinkhani, Nima. "How Much of Earnings Information is Factor Compensation?" (2026). Available at SSRN: [https://papers.ssrn.com/abstract=6440878](https://papers.ssrn.com/abstract=6440878)

```bibtex
@article{taheri2026earnings,
  title={How Much of Earnings Information is Factor Compensation?},
  author={Taheri Hosseinkhani, Nima},
  year={2026},
  journal={SSRN Working Paper},
  note={Available at \url{https://papers.ssrn.com/abstract=6440878}}
}
```


## License

MIT
