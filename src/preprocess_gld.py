"""
Preprocess GLD minute-level options data from Polygon.io into daily settlement proxies.
Reads: data/GLD_minute_aggs_*.csv.gz  (minute OHLCV bars per option contract)
Writes: data/gld_options_daily.csv     (daily close per option, one row per trade-date/expiry/strike/type)
OCC ticker format: O:GLD240221C00176000
  - GLD      = underlying
  - 240221   = expiration YYMMDD
  - C        = call (P = put)
  - 00176000 = strike * 1000  (i.e. $176.000)
"""
from __future__ import annotations
import gzip
import logging
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
logger = logging.getLogger(__name__)
_OCC_RE = re.compile(r"^O:GLD(\d{6})([CP])(\d{8})$")

def parse_occ_ticker(ticker: str) -> tuple[str, str, float] | None:
    """Parse an OCC ticker into (expiration_YYMMDD, option_type, strike).
    Returns None if the ticker doesn't match GLD format.
    """
    m = _OCC_RE.match(ticker)
    if m is None:
        return None
    exp_str = m.group(1)       
    opt_type = m.group(2)      
    strike = int(m.group(3)) / 1000.0  
    return exp_str, opt_type, strike
def expiration_str_to_date(exp_str: str) -> str:
    """Convert YYMMDD to YYYY-MM-DD."""
    dt = datetime.strptime(exp_str, "%y%m%d")
    return dt.strftime("%Y-%m-%d")
def preprocess_gld(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read gzipped minute data, aggregate to daily, write CSV.
    For each (trade_date, expiration, strike, option_type), we keep the
    **last minute bar's close price** as a settlement proxy.
    """
    logger.info("Reading %s ...", input_path)
    records: list[dict] = []
    skipped = 0
    with gzip.open(input_path, "rt") as f:
        f.readline()  
        for i, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split(",")
            ticker = parts[0]
            parsed = parse_occ_ticker(ticker)
            if parsed is None:
                skipped += 1
                continue
            exp_str, opt_type, strike = parsed
            close_price = float(parts[3])
            window_start_ns = int(parts[6])
            dt = datetime.fromtimestamp(window_start_ns / 1e9)
            trade_date = dt.strftime("%Y-%m-%d")
            records.append({
                "tradeDate": trade_date,
                "expirationDate": expiration_str_to_date(exp_str),
                "strikePrice": strike,
                "optionType": "call" if opt_type == "C" else "put",
                "close": close_price,
                "window_start_ns": window_start_ns,})
            if i % 2_000_000 == 0:
                logger.info("  processed %s rows (%s GLD records) ...", f"{i:,}", f"{len(records):,}")
    logger.info("  total rows: %s, GLD records: %s, skipped: %s", f"{i:,}", f"{len(records):,}", f"{skipped:,}")
    df = pd.DataFrame(records)
    logger.info("Aggregating to daily settlement proxies ...")
    idx = df.groupby(["tradeDate", "expirationDate", "strikePrice", "optionType"])[
        "window_start_ns"
    ].idxmax()
    daily = df.loc[idx].copy()
    daily = daily.rename(columns={"close": "settlement"})
    daily = daily.drop(columns=["window_start_ns"])
    daily = daily.sort_values(["tradeDate", "expirationDate", "strikePrice", "optionType"])
    daily = daily.reset_index(drop=True)
    logger.info("Daily options rows: %s", f"{len(daily):,}")
    logger.info("Trade dates: %d", daily["tradeDate"].nunique())
    logger.info("Expirations: %d", daily["expirationDate"].nunique())
    daily.to_csv(output_path, index=False)
    logger.info("Written to %s", output_path)
    return daily

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    gz_files = list(data_dir.glob("GLD_minute_aggs_*.csv.gz"))
    if not gz_files:
        logger.error("No GLD_minute_aggs_*.csv.gz file found in data/")
        exit(1)
    input_path = gz_files[0]
    output_path = data_dir / "gld_options_daily.csv"
    preprocess_gld(input_path, output_path)
