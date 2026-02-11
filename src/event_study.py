"""
Event study framework for analyzing abnormal returns around corporate fraud disclosures.

Methodology based on MacKinlay (1997) "Event Studies in Economics and Finance"
and the Fama-French Three-Factor Model (Fama & French, 1993).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_fraud_dataframe():
    """Return the curated DataFrame of 25 fraud disclosure events."""
    fraud_data = {
        "Company": [
            "Wells Fargo", "Volkswagen", "Nvidia", "Luckin Coffee", "Kraft Heinz",
            "General Electric", "Under Armour", "Bausch Health (Valeant)", "Equifax", "Tesla",
            "Goldman Sachs", "Mattel", "Workhorse", "Canoo", "Clover Health",
            "Rio Tinto", "JPMorgan", "Meta Platforms", "Nissan", "Vale",
            "Boeing", "Raytheon", "Airbus", "Deutsche Bank", "Ericsson",
        ],
        "Ticker": [
            "WFC", "VWAGY", "NVDA", "LKNCY", "KHC",
            "GE", "UAA", "BHC", "EFX", "TSLA",
            "GS", "MAT", "WKHS", "GOEV", "CLOV",
            "RIO", "JPM", "META", "NSANY", "VALE",
            "BA", "RTX", "EADSY", "DB", "ERIC",
        ],
        "Disclosure_Date": [
            "2016-09-08", "2015-09-18", "2022-05-06", "2020-04-02", "2019-02-21",
            "2019-08-15", "2019-11-03", "2015-10-21", "2017-09-07", "2018-08-07",
            "2020-10-22", "2019-10-29", "2021-11-08", "2021-05-17", "2021-02-04",
            "2017-10-17", "2020-09-29", "2019-07-24", "2018-11-19", "2019-01-25",
            "2022-09-22", "2020-10-27", "2020-01-31", "2016-09-15", "2019-12-06",
        ],
        "Source_Event": [
            "CFPB/OCC Fines (Fake Accounts Scandal)",
            "EPA Notice of Violation (Dieselgate Emissions)",
            "SEC Charges (Crypto Mining Revenue Disclosure)",
            "Company Admission (Fabricated Sales)",
            "SEC Subpoena (Accounting Policies)",
            "Harry Markopolos Report (Accounting Fraud Allegations)",
            "WSJ Report (Federal Accounting Probe)",
            "Citron Research Report (Philidor/Accounting)",
            "Data Breach Disclosure (Insider Trading Allegations)",
            "Elon Musk 'Funding Secured' Tweet (SEC Fraud Charge)",
            "DOJ Settlement (1MDB Scandal)",
            "Whistleblower Letter (Accounting Errors)",
            "SEC Investigation Disclosure (Short Report)",
            "SEC Investigation Disclosure (Subscription Model)",
            "Hindenburg Research Report (DOJ Investigation)",
            "SEC Charges (Mozambique Coal Valuation)",
            "DOJ/CFTC Penalty (Spoofing/Market Manipulation)",
            "SEC Settlement (Privacy Disclosures/Cambridge Analytica)",
            "Arrest of Carlos Ghosn (Financial Misconduct)",
            "Brumadinho Dam Collapse (Safety Certification Fraud)",
            "SEC Charges (737 MAX Safety Statements)",
            "DOJ Investigation (Pricing/Accounting)",
            "SFO/DOJ/PBF Bribery Settlement (Airbus Scandal)",
            "DOJ Demand (MBS Investigation)",
            "DOJ/SEC Bribery Settlement (FCPA)",
        ],
    }
    df = pd.DataFrame(fraud_data)
    df["Disclosure_Date"] = pd.to_datetime(df["Disclosure_Date"])
    return df


# ---------------------------------------------------------------------------
# Core event-study engine
# ---------------------------------------------------------------------------

def run_event_study(data, fraud_df, estimation_start=150, estimation_end=30,
                    event_window=10):
    """
    Run the Fama-French three-factor event study for every firm in *fraud_df*.

    Returns
    -------
    ar_df : DataFrame  – abnormal returns (days x firms)
    car_df : DataFrame – cumulative abnormal returns (days x firms)
    diagnostics : DataFrame – per-firm regression diagnostics
    """
    ar_data = {}
    diagnostics = []

    for _, row in fraud_df.iterrows():
        ticker = row["Ticker"]
        event_date = row["Disclosure_Date"]

        if ticker not in data.columns:
            continue

        event_loc = data.index.get_indexer([event_date], method="nearest")[0]

        est_start = event_loc - estimation_start
        est_end = event_loc - estimation_end
        evt_start = event_loc - event_window
        evt_end = event_loc + event_window + 1

        if est_start < 0 or evt_end > len(data):
            continue

        # --- Estimation window regression ---
        est = data.iloc[est_start:est_end]
        Y = est[ticker] - est["RF"]
        X = est[["Mkt-RF", "SMB", "HML"]]
        valid = ~Y.isna() & ~X.isna().any(axis=1)
        Y, X = Y[valid], X[valid]

        if len(Y) < 60:
            continue

        X_c = sm.add_constant(X)
        model = sm.OLS(Y, X_c).fit()

        diagnostics.append({
            "Ticker": ticker,
            "Company": row["Company"],
            "R_squared": model.rsquared,
            "Alpha": model.params["const"],
            "Alpha_pval": model.pvalues["const"],
            "Beta_Mkt": model.params["Mkt-RF"],
            "Beta_SMB": model.params["SMB"],
            "Beta_HML": model.params["HML"],
            "N_obs": int(model.nobs),
        })

        # --- Event window abnormal returns ---
        evt = data.iloc[evt_start:evt_end]
        Y_evt = evt[ticker] - evt["RF"]
        X_evt = sm.add_constant(evt[["Mkt-RF", "SMB", "HML"]], has_constant="add")
        expected = model.predict(X_evt)
        ar = Y_evt - expected

        days = list(range(-event_window, event_window + 1))
        if len(ar) == len(days):
            ar.index = days
            ar_data[ticker] = ar

    ar_df = pd.DataFrame(ar_data)
    ar_df.index.name = "Relative_Day"
    car_df = ar_df.cumsum()
    diag_df = pd.DataFrame(diagnostics)

    return ar_df, car_df, diag_df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compute_summary_stats(ar_df, car_df):
    """Cross-sectional t-tests for AAR and CAAR at each event day."""
    aar = ar_df.mean(axis=1)
    caar = car_df.mean(axis=1)
    t_ar, p_ar = stats.ttest_1samp(ar_df, 0, axis=1, nan_policy="omit")
    t_car, p_car = stats.ttest_1samp(car_df, 0, axis=1, nan_policy="omit")

    return pd.DataFrame({
        "AAR": aar, "AAR_t_stat": t_ar, "AAR_p_val": p_ar,
        "CAAR": caar, "CAAR_t_stat": t_car, "CAAR_p_val": p_car,
    }, index=ar_df.index)


def wilcoxon_test(ar_df):
    """Non-parametric Wilcoxon signed-rank test for each event day."""
    results = []
    for day in ar_df.index:
        vals = ar_df.loc[day].dropna()
        if len(vals) < 10:
            continue
        stat, p = stats.wilcoxon(vals, alternative="two-sided")
        n_neg = (vals < 0).sum()
        results.append({
            "Relative_Day": day,
            "Wilcoxon_stat": stat,
            "Wilcoxon_p_val": p,
            "N_negative": int(n_neg),
            "N_total": len(vals),
            "Pct_negative": n_neg / len(vals),
        })
    return pd.DataFrame(results).set_index("Relative_Day")


def sign_test(ar_df):
    """Binomial sign test: proportion of negative ARs vs 50 %."""
    results = []
    for day in ar_df.index:
        vals = ar_df.loc[day].dropna()
        n = len(vals)
        n_neg = (vals < 0).sum()
        p = stats.binom_test(n_neg, n, 0.5) if n > 0 else np.nan
        results.append({
            "Relative_Day": day,
            "N_negative": int(n_neg),
            "N_total": n,
            "Pct_negative": n_neg / n if n else np.nan,
            "Sign_test_p_val": p,
        })
    return pd.DataFrame(results).set_index("Relative_Day")


def bh_correction(p_values):
    """Benjamini-Hochberg FDR correction for a Series of p-values."""
    n = len(p_values)
    ranked = p_values.rank()
    adjusted = p_values * n / ranked
    # enforce monotonicity (backward pass)
    adjusted = adjusted.sort_index(ascending=False).cummin().sort_index()
    return adjusted.clip(upper=1.0)


def winsorize_ar(ar_df, limits=(0.05, 0.05)):
    """Winsorize each day's cross-section at the given percentile limits."""
    from scipy.stats import mstats
    winsorized = ar_df.copy()
    for day in winsorized.index:
        row = winsorized.loc[day].dropna()
        w = mstats.winsorize(row.values, limits=limits)
        winsorized.loc[day, row.index] = w
    return winsorized


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_aar_caar(summary, ar_df, car_df, save_path=None):
    """Publication-quality AAR and CAAR plots with confidence intervals."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    idx = summary.index
    ar_sem = ar_df.sem(axis=1)
    car_sem = car_df.sem(axis=1)

    # --- AAR ---
    ax1.bar(idx, summary["AAR"] * 100, color="steelblue", alpha=0.7, zorder=2)
    ax1.errorbar(idx, summary["AAR"] * 100, yerr=1.96 * ar_sem * 100,
                 fmt="none", color="black", capsize=3, zorder=3)
    sig_days = summary[summary["AAR_p_val"] < 0.05].index
    for d in sig_days:
        y = summary.loc[d, "AAR"] * 100
        offset = -1.5 if y < 0 else 1.0
        ax1.annotate("*", xy=(d, y), xytext=(d, y + offset),
                      fontsize=16, ha="center", color="red", fontweight="bold")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Average Abnormal Returns (AAR) Around Fraud Disclosure", fontsize=14)
    ax1.set_ylabel("AAR (%)", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    # --- CAAR ---
    ax2.plot(idx, summary["CAAR"] * 100, marker="o", color="darkgreen",
             linewidth=2, zorder=3)
    ax2.fill_between(idx,
                     (summary["CAAR"] - 1.96 * car_sem) * 100,
                     (summary["CAAR"] + 1.96 * car_sem) * 100,
                     color="green", alpha=0.15, zorder=2)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Cumulative Average Abnormal Returns (CAAR) Around Fraud Disclosure",
                  fontsize=14)
    ax2.set_xlabel("Event Day (0 = Disclosure Date)", fontsize=12)
    ax2.set_ylabel("CAAR (%)", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spaghetti(car_df, summary, save_path=None):
    """Individual-firm CAR trajectories behind the cross-sectional average."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    for col in car_df.columns:
        ax.plot(car_df.index, car_df[col] * 100, color="grey", alpha=0.25,
                linewidth=0.8)
    ax.plot(summary.index, summary["CAAR"] * 100, color="darkgreen",
            linewidth=2.5, label="CAAR (mean)", zorder=5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Individual CARs and Cross-Sectional Average (CAAR)", fontsize=14)
    ax.set_xlabel("Event Day (0 = Disclosure Date)", fontsize=12)
    ax.set_ylabel("CAR (%)", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_waterfall(ar_df, fraud_df, day=0, save_path=None):
    """Horizontal bar chart of Day-0 ARs sorted by magnitude."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    day_ar = ar_df.loc[day].dropna().sort_values()
    labels = []
    ticker_to_company = dict(zip(fraud_df["Ticker"], fraud_df["Company"]))
    for t in day_ar.index:
        labels.append(f"{ticker_to_company.get(t, t)} ({t})")

    colors = ["#c0392b" if v < 0 else "#27ae60" for v in day_ar.values]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(day_ar)), day_ar.values * 100, color=colors, edgecolor="white")
    ax.set_yticks(range(len(day_ar)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Abnormal Return (%)", fontsize=12)
    ax.set_title(f"Day {day} Abnormal Returns by Company", fontsize=14)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_heatmap(ar_df, save_path=None):
    """Heatmap of abnormal returns (firms x event days)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ar_df.T * 100, cmap="RdYlGn_r", center=0, linewidths=0.3,
                ax=ax, cbar_kws={"label": "AR (%)", "format": "%+.0f%%"},
                xticklabels=True, yticklabels=True)
    ax.set_title("Abnormal Returns Heatmap (Companies × Event Days)", fontsize=14)
    ax.set_xlabel("Event Day", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
