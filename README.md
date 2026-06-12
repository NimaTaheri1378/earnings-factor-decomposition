# Residual Information in Earnings Announcements

A public, GitHub-safe companion to an empirical asset-pricing project on earnings surprises, announcement-window returns, and post-earnings-announcement drift.

<p align="center">
  <img src="docs/figures/visual_abstract.svg" alt="Visual abstract" width="900"/>
</p>

## Main result

Large announcement-window earnings-surprise spreads remain almost entirely intact after factor adjustment, whereas longer-horizon drift is largely absorbed by expected-return components.

## At a glance

- Final clean sample: **238,529** earnings-announcement events from **2000-2024**.
- Data backbone: I/B/E/S forecasts and actuals linked to CRSP, Compustat, and Fama-French factors.
- Core design: annual analyst-SUE sorts, event-window returns, and event-level expected-return decompositions.

## Key findings

| Result | Evidence |
|---|---:|
| Announcement raw Q5-Q1 spread | 6.33 pp |
| Announcement market-adjusted spread | 6.28 pp |
| Announcement FF5+Momentum residual spread | 6.25 pp |
| 60-day raw PEAD spread | 1.92 pp |
| 60-day FF5+Momentum residual spread | -0.08 pp |

## Sample construction

| Stage | Events | PERMNOs | I/B/E/S tickers |
|---|---:|---:|---:|
| I/B/E/S actual EPS announcements | 501,536 | - | 16,444 |
| Matched to pre-announcement consensus | 411,325 | - | 13,408 |
| Linked to CRSP PERMNO | 396,038 | 12,676 | 12,477 |
| Mapped to CRSP and price-scaled SUE | 268,622 | 9,331 | 9,209 |
| Final clean main sample | 238,529 | 8,260 | 8,155 |

## Research design

For each earnings announcement, the project computes a price-scaled analyst surprise using the latest eligible pre-announcement consensus forecast and the stock price two trading days before the event. Firms are sorted into annual quintiles. Event-window returns are evaluated under raw returns, market-adjusted returns, and factor-model abnormal returns. Raw spreads are then decomposed into factor-exposure, alpha/intercept, and residual abnormal-return components.

## Repository notes

This public version omits proprietary raw WRDS data. It is intended as a professional research showcase for event-study design, factor modeling, abnormal-return measurement, and empirical finance communication.

See also:

- `docs/github_safe_summary.md`
- `docs/project_abstract.md`
