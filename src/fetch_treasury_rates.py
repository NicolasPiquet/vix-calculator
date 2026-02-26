"""
Fetch US Treasury CMT yield curve rates from the Fiscal Data API.
API docs: https://fiscaldata.treasury.gov/datasets/average-interest-rates-treasury-securities/
          https://api.fiscaldata.treasury.gov/services/api/fiscal_service/
Outputs data in the same long format as the existing treasury yield curve rates.csv:
    Date, maturity, value
"""

from __future__ import annotations
import io
from pathlib import Path
from urllib.request import urlopen
import pandas as pd


_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
_CMT_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"

def fetch_cmt_rates(year: int) -> pd.DataFrame:
    """Fetch CMT rates for a given year from Treasury.gov.
    Returns DataFrame with columns: Date, maturity, value
    """
    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
    print(f"Fetching CMT rates for {year} ...")
    try:
        with urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  Failed to fetch from primary URL: {e}")
        # Fallback: try alternative URL format
        url_alt = (
            f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/{year}/all"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
        with urlopen(url_alt) as resp:
            raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    date_col = df.columns[0]  # Usually "Date"
    maturity_cols = [c for c in df.columns if c != date_col]
    records = []
    for _, row in df.iterrows():
        for mat in maturity_cols:
            val = row[mat]
            if pd.notna(val):
                records.append({
                    "Date": row[date_col],
                    "maturity": mat,
                    "value": float(val),})
    result = pd.DataFrame(records)
    print(f"  Got {len(result)} rate observations for {year}")
    return result

def fetch_and_merge(
    years: list[int],
    existing_path: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch rates for given years and merge with existing file."""
    if output_path is None:
        output_path = existing_path
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        print(f"Existing rates: {len(existing)} rows")
    else:
        existing = pd.DataFrame(columns=["Date", "maturity", "value"])
    new_dfs = []
    for year in years:
        try:
            df = fetch_cmt_rates(year)
            new_dfs.append(df)
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
    if not new_dfs:
        print("No new data fetched.")
        return existing
    new_data = pd.concat(new_dfs, ignore_index=True)
    mat_map = {
        "1 Month": "1 Mo", "2 Month": "2 Mo", "3 Month": "3 Mo",
        "4 Month": "4 Mo", "6 Month": "6 Mo",
        "1 Year": "1 Yr", "2 Year": "2 Yr", "3 Year": "3 Yr",
        "5 Year": "5 Yr", "7 Year": "7 Yr", "10 Year": "10 Yr",
        "20 Year": "20 Yr", "30 Year": "30 Yr",}
    new_data["maturity"] = new_data["maturity"].replace(mat_map)
    valid_mats = set(existing["maturity"].unique()) if not existing.empty else None
    if valid_mats:
        new_data = new_data[new_data["maturity"].isin(valid_mats)]
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "maturity"], keep="last")
    combined = combined.sort_values(["Date", "maturity"]).reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"Written {len(combined)} rows to {output_path}")
    return combined

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    rates_path = data_dir / "treasury yield curve rates.csv"
    fetch_and_merge(years=[2024, 2025, 2026], existing_path=rates_path)
