"""Run the event study and regenerate the README figures and result tables."""

from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_study import (  # noqa: E402
    build_fraud_dataframe,
    compute_summary_stats,
    plot_aar_caar,
    plot_heatmap,
    plot_spaghetti,
    plot_waterfall,
    run_event_study,
)


def main():
    output = ROOT / "output"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fraud_df = build_fraud_dataframe()
    start_date = fraud_df["Disclosure_Date"].min() - timedelta(days=300)
    end_date = fraud_df["Disclosure_Date"].max() + timedelta(days=300)
    tickers = fraud_df["Ticker"].tolist() + ["^GSPC"]

    print("Downloading adjusted prices...")
    prices = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True,
        progress=False, threads=False,
    )["Close"]

    print("Downloading Fama-French daily factors...")
    factors = web.DataReader(
        "F-F_Research_Data_Factors_daily", "famafrench",
        start=start_date, end=end_date,
    )[0]
    if not isinstance(factors.index, pd.DatetimeIndex):
        factors.index = factors.index.to_timestamp()

    log_returns = np.log(prices / prices.shift(1))
    data = log_returns.join(factors / 100.0, how="inner").iloc[1:]
    ar_df, car_df, diagnostics = run_event_study(data, fraud_df)
    if ar_df.empty:
        raise RuntimeError("No firms completed the event study; check the downloaded inputs.")
    summary = compute_summary_stats(ar_df, car_df)

    plot_aar_caar(summary, ar_df, car_df, figures / "aar_caar.png")
    plot_spaghetti(car_df, summary, figures / "car_spaghetti.png")
    plot_waterfall(ar_df, fraud_df, day=0, save_path=figures / "day0_waterfall.png")
    plot_heatmap(ar_df, figures / "ar_heatmap.png")

    ar_df.to_csv(output / "abnormal_returns.csv")
    car_df.to_csv(output / "cumulative_abnormal_returns.csv")
    summary.to_csv(output / "event_study_summary.csv")
    diagnostics.to_csv(output / "regression_diagnostics.csv", index=False)
    fraud_df.to_csv(output / "fraud_disclosure_dates.csv", index=False)

    print(f"Completed {ar_df.shape[1]} firms.")
    print(f"Day +10 CAAR: {summary['CAAR'].iloc[-1] * 100:+.2f}%")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
