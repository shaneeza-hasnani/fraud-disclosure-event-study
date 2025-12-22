# Market Reaction to Corporate Fraud Disclosures

## Executive Summary
This project examines how equity markets respond to corporate fraud disclosures. Using an event study framework grounded in the Fama–French Three-Factor Model, the analysis measures abnormal stock returns around confirmed fraud announcement dates. Results show that fraud disclosures trigger economically meaningful and statistically significant negative abnormal returns, with evidence of continued market adjustment in the days following disclosure.

---

## Business Problem
Corporate fraud represents a sudden and severe information shock to investors. When fraud is disclosed, markets must rapidly reassess a firm’s future cash flows, litigation risk, regulatory exposure, and reputational damage.

This analysis addresses the following questions:
- Do fraud disclosures generate abnormal stock price reactions?
- How quickly does the market incorporate fraud-related information?
- Are price effects confined to the announcement day, or do they persist afterward?

Understanding these dynamics is critical for investors, regulators, risk managers, and compliance professionals.

---

## Data Overview
- **Fraud Events**: 25 publicly listed firms with well-documented fraud disclosures between 2015 and 2022
  - Sources include SEC enforcement actions, DOJ settlements, whistleblower reports, and investigative journalism
- **Stock Prices**: Daily adjusted closing prices from Yahoo Finance
- **Market Benchmark**: S&P 500 Index (^GSPC)
- **Risk Factors**: Fama–French Three-Factor daily data (Market, Size, Value) and risk-free rate

The final dataset includes over 2,100 trading days of returns per firm, providing sufficient depth for reliable estimation windows.

---

## Methodology
- Identified a single, validated disclosure date for each fraud event
- Adjusted disclosure dates to the nearest trading day when necessary
- Computed daily log returns for all firms
- Estimated expected returns using the Fama–French Three-Factor Model
- Calculated:
  - **Abnormal Returns (AR)**: Actual return minus expected return
  - **Cumulative Abnormal Returns (CAR)** over the event window
- Applied cross-sectional t-tests to assess statistical significance of abnormal returns

**Windows**
- Estimation Window: −150 to −30 trading days before disclosure
- Event Window: −10 to +10 trading days around disclosure

This structure isolates fraud-related price effects while controlling for market-wide movements.

---

## Skills Demonstrated
**Technical**
- Event study design and implementation
- Financial time series analysis
- Factor-based asset pricing (Fama–French)
- Regression modeling and hypothesis testing
- Abnormal return and cumulative return calculations
- Data validation and trading-calendar alignment

**Analytical & Business**
- Translating financial theory into empirical analysis
- Interpreting market reactions to corporate misconduct
- Risk and compliance-oriented financial reasoning
- Communicating statistical findings to non-technical audiences

---

## Key Results
- Fraud disclosures are associated with sharp negative abnormal returns
- Day 0 shows a large negative Average Abnormal Return, though not statistically significant
- Statistically significant negative abnormal returns occur on:
  - **Day +1**
  - **Day +6**
- Cumulative Average Abnormal Returns (CAAR) decline steadily after disclosure, reaching approximately −20% by Day +10
- Results suggest that market adjustment to fraud information is not instantaneous and unfolds over multiple trading days

---

## Interpretation
Markets react quickly to fraud disclosures, but not all information is fully priced on the announcement day. Continued negative abnormal returns indicate ongoing reassessment of firm risk, legal exposure, and credibility. This delayed response highlights how complex and uncertain fraud-related information can be for investors.

---

## Limitations
- Analysis uses the Fama–French Three-Factor Model and excludes momentum or industry-specific factors
- Some disclosures may coincide with other firm-specific news
- Sample consists of high-profile fraud cases, limiting generalizability
- CAAR significance is marginal for longer windows and should be interpreted cautiously

---

## Business Implications
- Fraud imposes immediate and sustained shareholder value loss
- Markets penalize not just the event itself, but the uncertainty that follows
- Event studies can support enforcement impact assessments, litigation analysis, and risk monitoring
- Compliance failures have measurable financial consequences beyond reputational harm

---

## Next Steps
- Extend the event window to analyze long-term drift
- Incorporate additional risk factors such as momentum
- Examine cross-sectional differences by firm size or industry
- Test sensitivity to alternative model specifications

---

## Repository Contents
- `/notebooks`: Python notebook implementing the full event study workflow
- `/data`: Curated fraud event list used for analysis
