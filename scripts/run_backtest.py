from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  
import pandas as pd  
from src.backtest import (  
    compute_vix_timeseries,
    fetch_gld_ohlc,
    realized_vol_close,
    realized_vol_parkinson,)

plt.style.use("seaborn-v0_8-whitegrid")

def load_data(data_dir: Path):
    """Load GLD options, rates, and calendar."""
    options = pd.read_csv(data_dir / "gld_options_daily.csv")
    options["tradeDate"] = pd.to_datetime(options["tradeDate"])
    options["expirationDate"] = pd.to_datetime(options["expirationDate"])
    rates = pd.read_csv(data_dir / "treasury yield curve rates.csv")
    rates["Date"] = pd.to_datetime(rates["Date"])
    calendar = pd.read_csv(data_dir / "cme holidays.csv")
    return options, rates, calendar

def plot_vix_vs_rv(vix_df: pd.DataFrame, rv_df: pd.DataFrame, output_path: Path):
    """Plot VIX time series overlaid with realized vol estimators."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(vix_df["trade_date"], vix_df["vix"], label="GLD Implied Vol (VIX-style)",
            color="royalblue", linewidth=1.5)
    if "rv_cc" in rv_df.columns:
        ax.plot(rv_df.index, rv_df["rv_cc"] * 100, label="Realized Vol (Close-Close, 30d)",
                color="crimson", linewidth=1.0, alpha=0.8)
    if "rv_pk" in rv_df.columns:
        ax.plot(rv_df.index, rv_df["rv_pk"] * 100, label="Realized Vol (Parkinson, 30d)",
                color="darkorange", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility (%)", fontsize=12)
    ax.set_title("GLD Implied Vol (VIX-style 30d) vs Realized Volatility", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close(fig)

def plot_vrp(vix_df: pd.DataFrame, rv_df: pd.DataFrame, output_path: Path):
    """Plot Variance Risk Premium over time."""
    merged = vix_df.set_index("trade_date")
    rv_aligned = rv_df["rv_cc"].reindex(merged.index)
    valid = merged["vix"].notna() & rv_aligned.notna()
    dates = merged.index[valid]
    vrp = (merged.loc[valid, "vix"] / 100) ** 2 - rv_aligned[valid] ** 2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, vrp, 0, where=vrp >= 0,
                    color="steelblue", alpha=0.5, label="VRP > 0 (IV > RV)")
    ax.fill_between(dates, vrp, 0, where=vrp < 0,
                    color="crimson", alpha=0.5, label="VRP < 0 (IV < RV)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VRP (variance units)", fontsize=12)
    ax.set_title("GLD Variance Risk Premium: IV² − RV²", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close(fig)

def main():
    data_dir = ROOT / "data"
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    print("Loading data...")
    options, rates, calendar = load_data(data_dir)
    print("Computing VIX time series (this may take several minutes)...")
    vix_df = compute_vix_timeseries(
        options_df=options,
        rates_df=rates,
        calendar_df=calendar,
        target_days=30,
        price_scale=1.0,
        min_dte=7,
    )
    if vix_df.empty:
        print("ERROR: No VIX values computed. Check data.")
        return 1
    csv_path = output_dir / "gld_vix_timeseries.csv"
    vix_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(vix_df)} dates)")
    print("Fetching GLD OHLC prices...")
    ohlc = fetch_gld_ohlc(start="2024-01-01", end="2026-03-01")
    if ohlc.empty:
        print("  WARNING: Could not fetch GLD prices. Skipping realized vol.")
        rv_df = pd.DataFrame()
    else:
        print(f"  Got {len(ohlc)} daily prices")
        rv_df = pd.DataFrame(index=ohlc.index)
        rv_df["rv_cc"] = realized_vol_close(ohlc["Close"], window=30)
        rv_df["rv_pk"] = realized_vol_parkinson(ohlc["High"], ohlc["Low"], window=30)
    print("Generating charts...")
    plot_vix_vs_rv(vix_df, rv_df, output_dir / "gld_vix_timeseries.png")
    if not rv_df.empty:
        plot_vrp(vix_df, rv_df, output_dir / "gld_vrp.png")
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Date range:     {vix_df['trade_date'].min().date()} → "
          f"{vix_df['trade_date'].max().date()}")
    print(f"  Dates computed: {len(vix_df)}")
    print(f"  VIX mean:       {vix_df['vix'].mean():.2f}%")
    print(f"  VIX min:        {vix_df['vix'].min():.2f}%")
    print(f"  VIX max:        {vix_df['vix'].max():.2f}%")
    print(f"  VIX std:        {vix_df['vix'].std():.2f}%")
    print("=" * 60)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())