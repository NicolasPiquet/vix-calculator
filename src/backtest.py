"""
Backtesting framework for VIX-style volatility indices.
Provides:
- VIX time series computation across all trade dates
- Realized volatility estimators (close-to-close, Parkinson, Yang-Zhang)
- Variance Risk Premium (VRP) analysis
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)
from .vix_index import vix_index_equity

# ---------------------------------------------------------------------------
# VIX time series
# ---------------------------------------------------------------------------

def compute_vix_timeseries(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    target_days: int = 30,
    price_scale: float = 1.0,
    min_dte: int = 7,
    trade_date_col: str = "tradeDate",
) -> pd.DataFrame:
    """Compute VIX for every available trade date.
    Returns DataFrame with columns: trade_date, vix, T1, T2, near_exp, next_exp, ...
    Dates where the calculation fails are skipped (logged).
    """
    trade_dates = sorted(options_df[trade_date_col].unique())
    results = []
    errors = []
    for i, td in enumerate(trade_dates):
        td_str = str(td)[:10]
        try:
            result = vix_index_equity(
                options_df=options_df,
                rates_df=rates_df,
                calendar_df=calendar_df,
                trade_date=td_str,
                target_days=target_days,
                price_scale=price_scale,
                min_dte=min_dte,
            )
            result["trade_date"] = td_str
            results.append(result)
        except (ValueError, KeyError) as e:
            errors.append({"trade_date": td_str, "error": str(e)})
        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] computed ...", i + 1, len(trade_dates))
    logger.info(
        "Computed VIX for %d/%d dates (%d errors)",
        len(results), len(trade_dates), len(errors),)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Realized volatility estimators
# ---------------------------------------------------------------------------

def realized_vol_close(prices: pd.Series, window: int = 30) -> pd.Series:
    """Close-to-close realized volatility (annualized).
    σ_cc = std(log returns) × √252
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def realized_vol_parkinson(
    high: pd.Series, low: pd.Series, window: int = 30,
) -> pd.Series:
    """Parkinson (1980) realized volatility estimator using high-low range.
    σ_P = √( (1/4ln2) × E[(ln(H/L))²] ) × √252
    """
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

def realized_vol_yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Yang-Zhang (2000) realized volatility estimator.
    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    components for a minimum-variance unbiased estimator.
    """
    n = window
    log_oc = np.log(open_ / close.shift(1))  
    log_co = np.log(close / open_)            
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)
    sigma2_o = log_oc.rolling(n).var()
    sigma2_c = log_co.rolling(n).var()
    rs = log_ho * log_hc + log_lo * log_lc
    sigma2_rs = rs.rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    sigma2_yz = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    return np.sqrt(sigma2_yz.clip(lower=0) * 252)

# ---------------------------------------------------------------------------
# Variance Risk Premium
# ---------------------------------------------------------------------------

def variance_risk_premium(
    vix_series: pd.Series,
    realized_vol_series: pd.Series,
) -> pd.Series:
    """Compute Variance Risk Premium: VRP = IV² - RV²
    Both inputs should be in decimal (e.g. 0.15 for 15%).
    Returns VRP in variance units.
    """
    return (vix_series / 100) ** 2 - realized_vol_series ** 2

def vrp_with_forward_rv(
    vix_df: pd.DataFrame,
    underlying_prices: pd.Series,
    target_days: int = 30,
    vix_col: str = "vix",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Compute VRP using forward-looking realized vol.
    VRP_t = (VIX_t / 100)² - RV²_{t → t+target_days}
    This is the "ex-post" VRP that measures whether implied vol was
    a good predictor of subsequent realized vol.
    """
    log_ret = np.log(underlying_prices / underlying_prices.shift(1))
    records = []
    for _, row in vix_df.iterrows():
        td = row[date_col]
        iv = row[vix_col] / 100  
        future_rets = log_ret.loc[td:].iloc[1:target_days + 1]
        if len(future_rets) < target_days * 0.8:  
            continue
        rv_fwd = future_rets.std() * np.sqrt(252)
        vrp = iv**2 - rv_fwd**2
        records.append({
            "trade_date": td,
            "implied_vol": iv,
            "realized_vol_fwd": rv_fwd,
            "vrp": vrp,
            "vrp_vol": iv - rv_fwd,})
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Fetch GLD underlying prices (for realized vol)
# ---------------------------------------------------------------------------

def fetch_gld_prices(start: str = "2024-02-01", end: str = "2026-03-01") -> pd.Series:
    """Fetch GLD daily close prices via yfinance.
    Falls back to a simple message if yfinance is not installed.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("GLD")
        hist = ticker.history(start=start, end=end)
        return hist["Close"]
    except ImportError:
        logger.warning("yfinance not installed. Install with: pip install yfinance")
        return pd.Series(dtype=float)

def fetch_gld_ohlc(start: str = "2024-02-01", end: str = "2026-03-01") -> pd.DataFrame:
    """Fetch GLD daily OHLC via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GLD")
        hist = ticker.history(start=start, end=end)
        return hist[["Open", "High", "Low", "Close"]]
    except ImportError:
        logger.warning("yfinance not installed.")
        return pd.DataFrame()
