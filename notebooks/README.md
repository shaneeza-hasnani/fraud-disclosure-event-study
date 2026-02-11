Jupyter notebook implementing the full event study workflow:

- Data collection (Yahoo Finance, Fama–French factors)
- Log return computation and date alignment
- Fama–French three-factor OLS estimation
- Abnormal and cumulative abnormal return calculation
- Parametric and non-parametric statistical tests
- Publication-quality visualizations (AAR/CAAR, spaghetti, waterfall, heatmap)
- Robustness checks (winsorization, outlier exclusion, alternative windows, CAPM comparison)

Core analysis logic is imported from `src/event_study.py`.
