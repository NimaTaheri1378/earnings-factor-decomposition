# GitHub-Safe Project Summary

## Project overview

This project studies how the stock market responds to earnings news. The central question is whether the return spread associated with earnings surprises reflects residual information processing or whether much of it is compensation for expected-return components already captured by standard asset-pricing models.

The empirical design links I/B/E/S earnings announcements and analyst forecasts to CRSP returns, Compustat accounting variables, and Fama–French factor data. The final clean sample contains 238,529 events between 2000 and 2024.

## Research question

The paper separates two ideas that are often bundled together in the PEAD literature:

1. **Immediate information reaction** at the earnings announcement.
2. **Delayed return drift** after the announcement.

The project asks whether these two phenomena behave differently once returns are decomposed at the event level.

## Main findings

### Announcement window

The Q5−Q1 spread in the announcement window is large under every benchmark. The raw spread is 6.33 percentage points, the market-adjusted spread is 6.28 percentage points, and the FF5+Momentum abnormal-return spread is 6.25 percentage points. This indicates that the market’s immediate response to earnings news is overwhelmingly residual rather than a simple artifact of factor exposure.

### Post-announcement drift

Over the 60-trading-day post-announcement window, the raw spread is 1.92 percentage points and the market-adjusted spread is 1.90 percentage points, but the FF5+Momentum residual spread is −0.08 percentage points and statistically insignificant. Over the 252-trading-day window, the raw spread is 6.60 percentage points, whereas the FF5+Momentum residual spread is −0.55 percentage points and insignificant.

### Decomposition result

The FF5+Momentum decomposition sharpens the interpretation. In the announcement window, the raw spread is 6.42 percentage points, of which 6.25 percentage points are residual abnormal return. By contrast, in the 60-day PEAD window, the 1.99 percentage-point raw spread is accounted for by 0.63 percentage points of factor exposure and 1.44 percentage points of alpha/intercept, leaving a residual component of −0.08 percentage points.

## Why this matters

The findings suggest that researchers and practitioners should not treat all earnings-surprise-related return continuation as residual information inefficiency. The evidence supports a more nuanced view: earnings announcements themselves generate strong residual information reactions, but longer-horizon drift is substantially weaker once expected-return components are isolated.

## Why this repository is public-safe

- It contains polished, browser-friendly SVG figures.
- It reports reader-facing summary statistics and findings without exposing proprietary raw data.
- It is suitable for showing research communication, event-study design, empirical finance workflow, and figure presentation.
- It avoids private review files and journal-only submission materials.
