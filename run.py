#!/usr/bin/env python3
"""
Run VIX-style volatility index calculation.

Usage:
    python run.py

Expects CSV files in data/ directory:
    - henry hub european options.csv
    - treasury yield curve rates.csv
    - cme holidays.csv
"""

from pathlib import Path

import pandas as pd

from src.vix_index import vix_index


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required CSV files from data directory."""
    options_path = data_dir / "henry hub european options.csv"
    rates_path = data_dir / "treasury yield curve rates.csv"
    calendar_path = data_dir / "cme holidays.csv"

    # Check files exist
    for path in [options_path, rates_path, calendar_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load options
    options = pd.read_csv(options_path)
    options["futures-expirationDate"] = pd.to_datetime(options["futures-expirationDate"])
    options["tradeDate"] = pd.to_datetime(options["tradeDate"])
    options["futures-updated"] = pd.to_datetime(options["futures-updated"])
    options["options-updated"] = pd.to_datetime(options["options-updated"])

    # Load rates
    rates = pd.read_csv(rates_path)
    rates["Date"] = pd.to_datetime(rates["Date"])

    # Load calendar
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


def main():
    """Run example VIX calculation for Henry Hub options."""
    # Configuration
    data_dir = Path(__file__).parent / "data"

    # Parameters for Henry Hub Natural Gas options
    # See: https://www.cmegroup.com/trading/energy/natural-gas/natural-gas_contractSpecs_options.html
    params = {
        "options_id": 1352,
        "trade_date": "2020-11-12",
        "front_months": 2,  # 2 months ahead = Jan (from Nov)
        "rear_months": 4,    # 4 months ahead = Mar (from Nov)
        "expiration_hour": 16,
        "expiration_day_offset": 4,
        "target_days": 90,  # 3-month horizon (90 days)
        "price_scale": 100,  # Strikes in cents, prices in dollars
    }

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
            **params,
        )
        print_report(result)
        return 0

    except ValueError as e:
        print(f"Calculation error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
