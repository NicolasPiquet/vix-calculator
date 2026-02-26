#!/usr/bin/env python3
"""
Run VIX-style volatility index calculation.
Usage:
    python run.py              # Run Henry Hub (default)
    python run.py --gld        # Run GLD ETF options
Expects CSV files in data/ directory:
    - henry hub european options.csv  (for Henry Hub mode)
    - gld_options_daily.csv           (for GLD mode – generate with src/preprocess_gld.py)
    - treasury yield curve rates.csv
    - cme holidays.csv
"""
import sys
from pathlib import Path
import pandas as pd
from src.vix_index import vix_index, vix_index_equity
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required CSV files from data directory."""
    options_path = data_dir / "henry hub european options.csv"
    rates_path = data_dir / "treasury yield curve rates.csv"
    calendar_path = data_dir / "cme holidays.csv"
    for path in [options_path, rates_path, calendar_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    options = pd.read_csv(options_path)
    options["futures-expirationDate"] = pd.to_datetime(options["futures-expirationDate"])
    options["tradeDate"] = pd.to_datetime(options["tradeDate"])
    options["futures-updated"] = pd.to_datetime(options["futures-updated"])
    options["options-updated"] = pd.to_datetime(options["options-updated"])
    rates = pd.read_csv(rates_path)
    rates["Date"] = pd.to_datetime(rates["Date"])
    calendar = pd.read_csv(calendar_path)
    return options, rates, calendar

def print_report(result: dict) -> None:
    """Print a formatted report of VIX calculation results."""
    print("=" * 60)
    print("VIX-STYLE VOLATILITY INDEX CALCULATION")
    print("=" * 60)
    print()
    print(f"VIX Index:  {result['vix']:.2f}")
    print()
    print("-" * 60)
    print("NEAR-TERM CONTRACT")
    print("-" * 60)
    print(f"  Time to expiration (T1):  {result['T1']:.6f} years")
    print(f"  Forward price (F1):       {result['F1']:.4f}")
    print(f"  ATM strike (K0_1):        {result['K0_1']:.4f}")
    print(f"  Variance (σ²_1):          {result['sigma2_1']:.6f}")
    print(f"  Risk-free rate (r1):      {result['r1']*100:.4f}%")
    print(f"  Strikes used:             {result['n_strikes_1']}")
    print()
    print("-" * 60)
    print("NEXT-TERM CONTRACT")
    print("-" * 60)
    print(f"  Time to expiration (T2):  {result['T2']:.6f} years")
    print(f"  Forward price (F2):       {result['F2']:.4f}")
    print(f"  ATM strike (K0_2):        {result['K0_2']:.4f}")
    print(f"  Variance (σ²_2):          {result['sigma2_2']:.6f}")
    print(f"  Risk-free rate (r2):      {result['r2']*100:.4f}%")
    print(f"  Strikes used:             {result['n_strikes_2']}")
    print()
    print("=" * 60)

def load_gld_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load GLD options data and shared rate/calendar files."""
    options_path = data_dir / "gld_options_daily.csv"
    rates_path = data_dir / "treasury yield curve rates.csv"
    calendar_path = data_dir / "cme holidays.csv"
    for path in [options_path, rates_path, calendar_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    options = pd.read_csv(options_path)
    options["tradeDate"] = pd.to_datetime(options["tradeDate"])
    options["expirationDate"] = pd.to_datetime(options["expirationDate"])
    rates = pd.read_csv(rates_path)
    rates["Date"] = pd.to_datetime(rates["Date"])
    calendar = pd.read_csv(calendar_path)
    return options, rates, calendar

def main_henry_hub():
    """Run example VIX calculation for Henry Hub options."""
    data_dir = Path(__file__).parent / "data"
    params = {
        "options_id": 1352,
        "trade_date": "2020-11-12",
        "front_months": 2,
        "rear_months": 4,
        "expiration_hour": 16,
        "expiration_day_offset": 4,
        "target_days": 90,
        "price_scale": 100,}
    print(f"Loading data from: {data_dir}")
    try:
        options, rates, calendar = load_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure required CSV files are in the data/ directory:")
        print("  - henry hub european options.csv")
        print("  - treasury yield curve rates.csv")
        print("  - cme holidays.csv")
        return 1
    print(f"Calculating VIX for trade date: {params['trade_date']}")
    print(f"Options ID: {params['options_id']}")
    print(f"Term structure: {params['front_months']}M / {params['rear_months']}M")
    print()
    try:
        result = vix_index(
            options_df=options,
            rates_df=rates,
            calendar_df=calendar,
            **params,)
        print_report(result)
        return 0
    except ValueError as e:
        print(f"Calculation error: {e}")
        return 1

def main_gld(trade_date: str = "2025-01-02"):
    """Run VIX-style calculation for GLD ETF options."""
    data_dir = Path(__file__).parent / "data"
    print(f"Loading GLD data from: {data_dir}")
    try:
        options, rates, calendar = load_gld_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure required CSV files are in the data/ directory:")
        print("  - gld_options_daily.csv  (run: python -m src.preprocess_gld)")
        print("  - treasury yield curve rates.csv")
        print("  - cme holidays.csv")
        return 1
    print(f"Calculating GLD VIX for trade date: {trade_date}")
    print(f"Target maturity: 30 days")
    print()
    try:
        result = vix_index_equity(
            options_df=options,
            rates_df=rates,
            calendar_df=calendar,
            trade_date=trade_date,
            target_days=30,
            price_scale=1.0,)
        print_report_equity(result)
        return 0
    except ValueError as e:
        print(f"Calculation error: {e}")
        return 1

def print_report_equity(result: dict) -> None:
    """Print a formatted report for equity VIX calculation."""
    print("=" * 60)
    print("VIX-STYLE VOLATILITY INDEX — GLD")
    print("=" * 60)
    print()
    print(f"VIX Index:  {result['vix']:.2f}")
    print()
    print("-" * 60)
    print(f"NEAR-TERM  (exp: {result['near_exp'].date()})")
    print("-" * 60)
    print(f"  Time to expiration (T1):  {result['T1']:.6f} years")
    print(f"  Forward price (F1):       {result['F1']:.4f}")
    print(f"  ATM strike (K0_1):        {result['K0_1']:.4f}")
    print(f"  Variance (σ²_1):          {result['sigma2_1']:.6f}")
    print(f"  Risk-free rate (r1):      {result['r1']*100:.4f}%")
    print(f"  Strikes used:             {result['n_strikes_1']}")
    print()
    print("-" * 60)
    print(f"NEXT-TERM  (exp: {result['next_exp'].date()})")
    print("-" * 60)
    print(f"  Time to expiration (T2):  {result['T2']:.6f} years")
    print(f"  Forward price (F2):       {result['F2']:.4f}")
    print(f"  ATM strike (K0_2):        {result['K0_2']:.4f}")
    print(f"  Variance (σ²_2):          {result['sigma2_2']:.6f}")
    print(f"  Risk-free rate (r2):      {result['r2']*100:.4f}%")
    print(f"  Strikes used:             {result['n_strikes_2']}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    if "--gld" in sys.argv:
        # Optional: python run.py --gld 2025-03-15
        date_arg = None
        for arg in sys.argv[1:]:
            if arg != "--gld":
                date_arg = arg
                break
        if date_arg:
            exit(main_gld(trade_date=date_arg))
        else:
            exit(main_gld())
    else:
        exit(main_henry_hub())
