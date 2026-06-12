# Residual Information in Earnings Announcements

**Earnings surprise returns, expected-return compensation, and post-announcement drift in U.S. equities, 2000–2024.**

<p align="center">
  <img src="docs/figures/visual_abstract.svg" alt="Visual abstract: residual information in earnings announcements" width="1000"/>
</p>

## Overview

This project studies how equity prices respond to earnings news. The central question is whether the return spread associated with earnings surprises reflects residual information processing or compensation for expected-return components captured by standard asset-pricing models.

The evidence points to a sharp distinction. Earnings announcements generate large residual abnormal returns, but longer-horizon post-earnings-announcement drift is largely absorbed once factor exposure and pre-event alpha/intercept components are separated from residual returns.

## Main exhibits

<p align="center">
  <img src="docs/figures/announcement_vs_pead.svg" alt="Announcement versus PEAD return spreads" width="1000"/>
</p>

<p align="center">
  <img src="docs/figures/decomposition.svg" alt="FF5 plus momentum decomposition" width="1000"/>
</p>

<p align="center">
  <img src="docs/figures/sample_funnel.svg" alt="Sample construction funnel" width="1000"/>
</p>

## Empirical design

For each earnings announcement, the project computes a price-scaled analyst surprise using the latest eligible pre-announcement consensus forecast and the stock price two trading days before the event. Firms are sorted into annual surprise quintiles. Event-window returns are then measured under raw returns, market-adjusted returns, and factor-model abnormal returns. Finally, raw Q5−Q1 spreads are decomposed into factor-exposure, alpha/intercept, and residual abnormal-return components.

## Data and sample

| Stage | Events | PERMNOs | I/B/E/S tickers |
|---|---:|---:|---:|
| I/B/E/S actual EPS announcements | 501,536 | — | 16,444 |
| Matched to pre-announcement consensus | 411,325 | — | 13,408 |
| Linked to CRSP PERMNO | 396,038 | 12,676 | 12,477 |
| Mapped to CRSP and price-scaled SUE | 268,622 | 9,331 | 9,209 |
| **Final clean main sample** | **238,529** | **8,260** | **8,155** |

The public repository does not include proprietary WRDS source data. The research design links I/B/E/S earnings announcements and analyst forecasts to CRSP returns, Compustat accounting variables, and Fama–French factor data.

## Main results

### Announcement-window returns remain residual

| Window | Benchmark | Q5−Q1 spread (pp) | p-value |
|---|---|---:|---:|
| Announcement [−1,+1] | Raw CAR | 6.33 | <0.001 |
| Announcement [−1,+1] | Market-adjusted CAR | 6.28 | <0.001 |
| Announcement [−1,+1] | FF5+Momentum AR | 6.25 | <0.001 |

### Post-announcement drift is absorbed by expected-return components

| Window | Benchmark | Q5−Q1 spread (pp) | p-value |
|---|---|---:|---:|
| PEAD [+2,+60] | Raw CAR | 1.92 | <0.001 |
| PEAD [+2,+60] | Market-adjusted CAR | 1.90 | <0.001 |
| PEAD [+2,+60] | FF5+Momentum AR | −0.08 | 0.669 |
| PEAD [+2,+252] | Raw CAR | 6.60 | <0.001 |
| PEAD [+2,+252] | FF5+Momentum AR | −0.55 | 0.389 |

### FF5+Momentum decomposition

| Window | Raw spread | Factor exposure | Alpha/intercept | Residual AR | Residual share |
|---|---:|---:|---:|---:|---:|
| Announcement [−1,+1] | 6.42 | 0.10 | 0.07 | 6.25 | 97.4% |
| PEAD [+2,+60] | 1.99 | 0.63 | 1.44 | −0.08 | −3.9% |
| PEAD [+2,+252] | 6.96 | 1.27 | 6.24 | −0.55 | −7.9% |

## Repository contents

- `docs/figures/` — polished SVG figures for browser viewing.
- `docs/project_abstract.md` — compact research abstract.
- `src/` — implementation modules from the research workflow.
- `outputs/` — non-proprietary output structure and result artifacts where available.
- `paper/` — manuscript-related materials included in the public release.

## Interpretation

The findings refine the standard PEAD interpretation. The announcement-window response is not explained away by factor adjustment: it is a residual information reaction. The apparent drift after the announcement is different. In raw and market-adjusted returns it accumulates over time, but under event-level factor decompositions it is largely captured by expected-return and alpha/intercept components rather than residual abnormal performance.

## References

- Ball, R. and Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159–178.
- Bernard, V.L. and Thomas, J.K. (1989). Post-earnings-announcement drift. *Journal of Accounting Research*, 27, 1–36.
- Brown, S.J. and Warner, J.B. (1985). Using daily stock returns: The case of event studies. *Journal of Financial Economics*, 14(1), 3–31.
- Carhart, M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
- Fama, E.F. and French, K.R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Foster, G., Olsen, C. and Shevlin, T. (1984). Earnings releases, anomalies, and the behavior of security returns. *The Accounting Review*, 59(4), 574–603.
- Hong, H., Lim, T. and Stein, J.C. (2000). Bad news travels slowly: Size, analyst coverage, and the profitability of momentum strategies. *Journal of Finance*, 55(1), 265–295.
