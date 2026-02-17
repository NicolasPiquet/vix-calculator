"""
VIX-style volatility index calculator.

Implements the CBOE VIX methodology adapted for commodity options.
Reference: CBOE VIX White Paper (http://www.cboe.com/micro/vix/vixwhite.pdf)

Key differences from official VIX:
- Uses priorSettle as proxy for bid-ask midpoint (no live quotes)
- Configurable expiration parameters for different underlyings
- Supports custom time horizons (not limited to 30-day)
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import dateutil.relativedelta
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MINUTES_PER_YEAR = 365 * 24 * 60
MINUTES_30_DAYS = 30 * 24 * 60


# -----------------------------------------------------------------------------
# Rate interpolation
# -----------------------------------------------------------------------------

def fill_cmt_missing_dates(cmt_rate: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates (weekends, holidays) in CMT rate data using forward-fill.

    Parameters
    ----------
    cmt_rate : pd.DataFrame
        Columns: Date, maturity, value

    Returns
    -------
    pd.DataFrame
        Same structure with all calendar dates filled.
    """
    cmt = cmt_rate.copy()
    complete_dates = pd.date_range(cmt["Date"].min(), cmt["Date"].max())

    # Pivot, reindex, forward-fill
    pivoted = cmt.pivot(index="Date", columns="maturity", values="value")
    pivoted = pivoted.reindex(complete_dates)
    pivoted = pivoted.ffill()

    # Back to long format
    pivoted.index.name = "Date"
    result = pivoted.reset_index().melt(id_vars="Date", var_name="maturity", value_name="value")
    return result


def get_rate_for_maturity(
    cmt_rate: pd.DataFrame,
    trade_date: pd.Timestamp,
    maturity_months: int,
) -> float:
    """
    Get the CMT rate for a given maturity and date.

    If the exact maturity is not available, performs linear interpolation
    between the two nearest available maturities.

    Parameters
    ----------
    cmt_rate : pd.DataFrame
        CMT rate data with Date, maturity, value columns.
    trade_date : pd.Timestamp
        The date to look up.
    maturity_months : int
        Maturity in months (e.g., 1, 2, 3).

    Returns
    -------
    float
        Interest rate as decimal (e.g., 0.02 for 2%).

    Raises
    ------
    ValueError
        If rate cannot be found or interpolated for the given date/maturity.
    """
    # Try exact match first
    maturity_label = f"{maturity_months} Mo"
    mask = (cmt_rate["maturity"] == maturity_label) & (cmt_rate["Date"] == trade_date)
    matches = cmt_rate.loc[mask, "value"]

    if not matches.empty:
        return matches.iloc[0] / 100.0

    # Exact match not found - try linear interpolation
    date_rates = cmt_rate[cmt_rate["Date"] == trade_date].copy()
    if date_rates.empty:
        raise ValueError(f"No CMT rates found for date {trade_date}")

    # Extract numeric months from maturity labels (e.g., "3 Mo" -> 3)
    date_rates["months"] = date_rates["maturity"].str.extract(r"(\d+)").astype(float)
    date_rates = date_rates.dropna(subset=["months"]).sort_values("months")

    available_months = date_rates["months"].values
    available_rates = date_rates["value"].values

    # Find bracketing maturities
    lower_mask = available_months <= maturity_months
    upper_mask = available_months >= maturity_months

    if not lower_mask.any() or not upper_mask.any():
        raise ValueError(
            f"Cannot interpolate rate for {maturity_months} Mo on {trade_date}. "
            f"Available maturities: {sorted(available_months)}"
        )

    lower_idx = np.where(lower_mask)[0][-1]
    upper_idx = np.where(upper_mask)[0][0]

    if lower_idx == upper_idx:
        # Exact match found through rounding
        return available_rates[lower_idx] / 100.0

    # Linear interpolation
    m1, r1 = available_months[lower_idx], available_rates[lower_idx]
    m2, r2 = available_months[upper_idx], available_rates[upper_idx]
    interpolated = r1 + (r2 - r1) * (maturity_months - m1) / (m2 - m1)

    return interpolated / 100.0


# -----------------------------------------------------------------------------
# Settlement date calculation
# -----------------------------------------------------------------------------

def get_settlement_day(
    current_day: dt.datetime,
    months_ahead: int,
    expiration_day_offset: int,
    expiration_hour: int,
    holidays: list[str],
) -> dt.datetime:
    """
    Calculate the settlement day for options expiring N months ahead.

    The settlement day is the Nth last business day of the expiration month,
    excluding weekends and provided holidays.

    Parameters
    ----------
    current_day : dt.datetime
        Current datetime.
    months_ahead : int
        Number of months until expiration.
    expiration_day_offset : int
        Which business day from month end (e.g., 4 = 4th last business day).
    expiration_hour : int
        Hour of expiration (0-23).
    holidays : list[str]
        List of holiday dates as 'YYYY-MM-DD' strings.

    Returns
    -------
    dt.datetime
        Settlement datetime.
    """
    # Start from end of expiration month
    month_end = current_day + dateutil.relativedelta.relativedelta(
        day=31,
        months=months_ahead - 1,
        hour=expiration_hour,
        minute=0,
        second=0,
        microsecond=0,
    )

    settlement = month_end
    business_day_count = 0

    # Count backwards from month end
    while True:
        is_weekday = settlement.weekday() < 5
        is_holiday = str(settlement)[:10] in holidays

        if is_weekday:
            business_day_count += 1
            if not is_holiday and business_day_count >= expiration_day_offset:
                break

        settlement -= dt.timedelta(days=1)

    return settlement


def time_to_expiration_years(
    current_day: dt.datetime,
    months_ahead: int,
    expiration_day_offset: int,
    expiration_hour: int,
    holidays: list[str],
) -> float:
    """
    Calculate time to expiration in years.

    Parameters
    ----------
    current_day : dt.datetime
        Current datetime.
    months_ahead : int
        Months until expiration.
    expiration_day_offset : int
        Business day offset from month end.
    expiration_hour : int
        Hour of expiration.
    holidays : list[str]
        Holiday dates.

    Returns
    -------
    float
        Time to expiration in years (T).
    """
    settlement = get_settlement_day(
        current_day, months_ahead, expiration_day_offset, expiration_hour, holidays
    )
    minutes = (settlement - current_day).total_seconds() / 60
    return minutes / MINUTES_PER_YEAR


# -----------------------------------------------------------------------------
# Forward and K0 calculation
# -----------------------------------------------------------------------------

def compute_forward_and_k0(
    options: pd.DataFrame,
    r: float,
    T: float,
    strike_col: str = "options-strikePrice",
    type_col: str = "options-optiontype",
    price_col: str = "options-priorSettle",
) -> tuple[float, float]:
    """
    Compute forward price F and at-the-money strike K0.

    CBOE methodology:
    1. For each strike, compute |Call - Put|
    2. Find strike K* that minimizes this difference
    3. F = K* + exp(rT) * (Call(K*) - Put(K*))
    4. K0 = max{K : K <= F}

    Parameters
    ----------
    options : pd.DataFrame
        Options data with strike, type, and price columns.
    r : float
        Risk-free rate (decimal).
    T : float
        Time to expiration (years).
    strike_col, type_col, price_col : str
        Column names.

    Returns
    -------
    tuple[float, float]
        (F, K0) - forward price and ATM strike.

    Raises
    ------
    ValueError
        If insufficient data to compute forward/K0.
    """
    opts = options.copy()
    opts = opts.sort_values(strike_col)

    # Pivot to get call/put prices by strike (use pivot_table to handle duplicates)
    pivoted = opts.pivot_table(index=strike_col, columns=type_col, values=price_col, aggfunc="mean")

    if "call" not in pivoted.columns or "put" not in pivoted.columns:
        raise ValueError("Options data must contain both calls and puts")

    # Find strike with minimum |Call - Put|
    diff = (pivoted["call"] - pivoted["put"]).abs()
    k_star = diff.idxmin()

    call_price = pivoted.loc[k_star, "call"]
    put_price = pivoted.loc[k_star, "put"]

    # Forward level
    F = k_star + np.exp(r * T) * (call_price - put_price)

    # K0 = largest strike <= F
    valid_strikes = pivoted.index[pivoted.index <= F]
    if len(valid_strikes) == 0:
        raise ValueError(f"No strike found <= forward price F={F:.4f}")

    K0 = valid_strikes[-1]

    return float(F), float(K0)


# -----------------------------------------------------------------------------
# Q(K) construction
# -----------------------------------------------------------------------------

def build_qk(
    options: pd.DataFrame,
    K0: float,
    strike_col: str = "options-strikePrice",
    type_col: str = "options-optiontype",
    price_col: str = "options-priorSettle",
    consecutive_zeros: int = 2,
) -> pd.Series:
    """
    Build the Q(K) series according to CBOE methodology.

    - K < K0: use OTM puts
    - K > K0: use OTM calls
    - K = K0: average of put and call (counted once)

    Exclusion rule: stop including strikes after N consecutive zero prices.

    Parameters
    ----------
    options : pd.DataFrame
        Options chain data.
    K0 : float
        At-the-money strike.
    strike_col, type_col, price_col : str
        Column names.
    consecutive_zeros : int
        Stop after this many consecutive zero-price strikes.

    Returns
    -------
    pd.Series
        Q(K) indexed by strike price.
    """
    opts = options.copy()
    # Use pivot_table to handle potential duplicates gracefully
    pivoted = opts.pivot_table(index=strike_col, columns=type_col, values=price_col, aggfunc="mean").sort_index()

    strikes = pivoted.index.values
    qk_values = {}

    # --- Process puts (K < K0), going outward from K0 ---
    put_strikes = strikes[strikes < K0][::-1]  # Reverse to go from K0 outward
    zero_count = 0

    for K in put_strikes:
        price = pivoted.loc[K, "put"] if "put" in pivoted.columns else np.nan
        if pd.isna(price):
            continue
        if price == 0:
            zero_count += 1
            if zero_count >= consecutive_zeros:
                break
        else:
            zero_count = 0
            qk_values[K] = price

    # --- Process calls (K > K0), going outward from K0 ---
    call_strikes = strikes[strikes > K0]
    zero_count = 0

    for K in call_strikes:
        price = pivoted.loc[K, "call"] if "call" in pivoted.columns else np.nan
        if pd.isna(price):
            continue
        if price == 0:
            zero_count += 1
            if zero_count >= consecutive_zeros:
                break
        else:
            zero_count = 0
            qk_values[K] = price

    # --- K0: average of put and call ---
    if K0 in strikes:
        put_k0 = pivoted.loc[K0, "put"] if "put" in pivoted.columns else np.nan
        call_k0 = pivoted.loc[K0, "call"] if "call" in pivoted.columns else np.nan

        if pd.notna(put_k0) and pd.notna(call_k0):
            qk_values[K0] = (put_k0 + call_k0) / 2
        elif pd.notna(put_k0):
            qk_values[K0] = put_k0
        elif pd.notna(call_k0):
            qk_values[K0] = call_k0

    qk = pd.Series(qk_values).sort_index()
    return qk


def compute_delta_k(strikes: np.ndarray) -> np.ndarray:
    """
    Compute strike intervals ΔK for VIX formula.

    - Interior strikes: ΔK_i = (K_{i+1} - K_{i-1}) / 2
    - First strike: ΔK_1 = K_2 - K_1
    - Last strike: ΔK_n = K_n - K_{n-1}

    Parameters
    ----------
    strikes : np.ndarray
        Sorted array of strike prices.

    Returns
    -------
    np.ndarray
        Array of ΔK values.
    """
    n = len(strikes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])

    delta_k = np.zeros(n)

    # First element
    delta_k[0] = strikes[1] - strikes[0]

    # Interior elements
    for i in range(1, n - 1):
        delta_k[i] = (strikes[i + 1] - strikes[i - 1]) / 2

    # Last element
    delta_k[n - 1] = strikes[n - 1] - strikes[n - 2]

    return delta_k


# -----------------------------------------------------------------------------
# Sigma squared calculation
# -----------------------------------------------------------------------------

def compute_sigma2(
    F: float,
    K0: float,
    qk: pd.Series,
    r: float,
    T: float,
    price_scale: float = 1.0,
) -> float:
    """
    Compute sigma^2 (variance contribution) using CBOE formula.

    σ² = (2/T) * Σ (ΔK_i / K_i²) * e^(rT) * Q(K_i) - (1/T) * (F/K0 - 1)²

    Parameters
    ----------
    F : float
        Forward price.
    K0 : float
        At-the-money strike.
    qk : pd.Series
        Q(K) values indexed by strike.
    r : float
        Risk-free rate.
    T : float
        Time to expiration (years).
    price_scale : float
        Multiplier to convert option prices to same units as strikes.
        E.g., if strikes are in cents and prices in dollars, use 100.

    Returns
    -------
    float
        σ² value.

    Raises
    ------
    ValueError
        If insufficient strikes for calculation, or if computed σ² <= 0
        (indicates unit mismatch - check price_scale).
    """
    if len(qk) < 2:
        raise ValueError(f"Need at least 2 strikes for sigma2, got {len(qk)}")

    strikes = qk.index.values
    prices = qk.values * price_scale
    delta_k = compute_delta_k(strikes)

    # Vectorized sum: Σ (ΔK / K²) * e^(rT) * Q(K)
    contribution = np.sum((delta_k / strikes**2) * np.exp(r * T) * prices)

    # CBOE formula
    sigma2 = (2 / T) * contribution - (1 / T) * (F / K0 - 1) ** 2

    if sigma2 <= 0:
        raise ValueError(
            f"Computed σ² = {sigma2:.6f} <= 0. "
            "This typically indicates a unit mismatch between strikes and option prices. "
            "Check the 'price_scale' parameter."
        )

    return float(sigma2)


# -----------------------------------------------------------------------------
# VIX interpolation
# -----------------------------------------------------------------------------

def interpolate_to_target(
    T1: float,
    T2: float,
    sigma2_1: float,
    sigma2_2: float,
    target_minutes: int = MINUTES_30_DAYS,
) -> float:
    """
    Interpolate between two term sigma² to target maturity.

    Uses the CBOE weighting formula for 30-day VIX.

    Parameters
    ----------
    T1 : float
        Time to expiration for near-term (years).
    T2 : float
        Time to expiration for next-term (years).
    sigma2_1 : float
        σ² for near-term.
    sigma2_2 : float
        σ² for next-term.
    target_minutes : int
        Target maturity in minutes (default 30 days).

    Returns
    -------
    float
        VIX value (volatility in percentage points).

    Raises
    ------
    ValueError
        If target maturity is not bracketed by the two term expirations.
    """
    N1 = T1 * MINUTES_PER_YEAR
    N2 = T2 * MINUTES_PER_YEAR
    N_target = target_minutes

    # Validate bracketing: N1 < N_target < N2
    if not (N1 < N_target < N2):
        raise ValueError(
            f"Target maturity not bracketed by expirations. "
            f"N1={N1:.0f} min, N_target={N_target:.0f} min, N2={N2:.0f} min. "
            f"Pick different front_months/rear_months to bracket the target."
        )

    # CBOE interpolation weights
    w1 = T1 * sigma2_1 * (N2 - N_target) / (N2 - N1)
    w2 = T2 * sigma2_2 * (N_target - N1) / (N2 - N1)

    # VIX = 100 * sqrt(weighted_avg * scale)
    vix = 100 * np.sqrt((w1 + w2) * MINUTES_PER_YEAR / N_target)

    return float(vix)


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def vix_index(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    options_id: int,
    trade_date: pd.Timestamp | str,
    front_months: int = 1,
    rear_months: int = 2,
    expiration_hour: int = 16,
    expiration_day_offset: int = 4,
    target_days: int = 30,
    price_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate VIX-style volatility index.

    Parameters
    ----------
    options_df : pd.DataFrame
        Options chain with columns:
        - options-id, tradeDate, futures-expirationDate
        - options-strikePrice, options-optiontype, options-priorSettle
        - futures-updated
    rates_df : pd.DataFrame
        CMT rates with columns: Date, maturity, value
    calendar_df : pd.DataFrame
        Holiday calendar with column: DATE
    options_id : int
        Options product identifier.
    trade_date : pd.Timestamp or str
        Calculation date.
    front_months : int
        Near-term expiration (months ahead from trade_date).
        E.g., front_months=1 in November targets December contracts.
    rear_months : int
        Next-term expiration (months ahead from trade_date).
        E.g., rear_months=2 in November targets January contracts.
    expiration_hour : int
        Hour of settlement.
    expiration_day_offset : int
        Business day from month end.
    target_days : int
        Target maturity for interpolation (days).
    price_scale : float
        Multiplier to convert option prices to same units as strikes.
        E.g., if strikes are in cents and prices in dollars, use 100.

    Returns
    -------
    dict
        Results including:
        - vix: final VIX value
        - T1, T2: times to expiration
        - F1, F2: forward prices
        - K0_1, K0_2: ATM strikes
        - sigma2_1, sigma2_2: variance terms
        - r1, r2: interest rates
        - n_strikes_1, n_strikes_2: number of strikes used

    Raises
    ------
    ValueError
        If data is insufficient or invalid.
    """
    # Normalize trade_date
    if isinstance(trade_date, str):
        trade_date = pd.Timestamp(trade_date)

    # Prepare holidays
    holidays = calendar_df["DATE"].astype(str).tolist()

    # Fill missing CMT dates
    rates = fill_cmt_missing_dates(rates_df)

    # Get interest rates
    r1 = get_rate_for_maturity(rates, trade_date, front_months)
    r2 = get_rate_for_maturity(rates, trade_date, rear_months)

    # Filter options for this product and date
    mask = (options_df["options-id"] == options_id) & (options_df["tradeDate"] == trade_date)
    current_options = options_df.loc[mask].copy()

    if current_options.empty:
        raise ValueError(f"No options found for id={options_id} on {trade_date}")

    # Determine expiration months (front_months=1 means next month's contract)
    front_exp = pd.Timestamp(f"{trade_date.year}-{trade_date.month}-1") + pd.DateOffset(months=front_months)
    rear_exp = pd.Timestamp(f"{trade_date.year}-{trade_date.month}-1") + pd.DateOffset(months=rear_months)

    # Adjust to first of month for matching
    front_exp = pd.Timestamp(f"{front_exp.year}-{front_exp.month}-1")
    rear_exp = pd.Timestamp(f"{rear_exp.year}-{rear_exp.month}-1")

    # Split by expiration
    opts_front = current_options[current_options["futures-expirationDate"] == front_exp].copy()
    opts_rear = current_options[current_options["futures-expirationDate"] == rear_exp].copy()

    if opts_front.empty:
        raise ValueError(f"No front-month options found for expiration {front_exp}")
    if opts_rear.empty:
        raise ValueError(f"No rear-month options found for expiration {rear_exp}")

    # Get current timestamps from futures data
    current_front = opts_front["futures-updated"].iloc[0]
    current_rear = opts_rear["futures-updated"].iloc[0]

    # Time to expiration
    T1 = time_to_expiration_years(current_front, front_months, expiration_day_offset, expiration_hour, holidays)
    T2 = time_to_expiration_years(current_rear, rear_months, expiration_day_offset, expiration_hour, holidays)

    if T1 <= 0 or T2 <= 0:
        raise ValueError(f"Invalid time to expiration: T1={T1:.4f}, T2={T2:.4f}")

    # Compute forward and K0 for each term
    F1, K0_1 = compute_forward_and_k0(opts_front, r1, T1)
    F2, K0_2 = compute_forward_and_k0(opts_rear, r2, T2)

    # Build Q(K) for each term
    qk1 = build_qk(opts_front, K0_1)
    qk2 = build_qk(opts_rear, K0_2)

    if len(qk1) < 2:
        raise ValueError(f"Insufficient strikes for front-term: {len(qk1)}")
    if len(qk2) < 2:
        raise ValueError(f"Insufficient strikes for rear-term: {len(qk2)}")

    # Compute sigma^2
    sigma2_1 = compute_sigma2(F1, K0_1, qk1, r1, T1, price_scale)
    sigma2_2 = compute_sigma2(F2, K0_2, qk2, r2, T2, price_scale)

    # Interpolate to target maturity
    target_minutes = target_days * 24 * 60
    vix = interpolate_to_target(T1, T2, sigma2_1, sigma2_2, target_minutes)

    return {
        "vix": vix,
        "T1": T1,
        "T2": T2,
        "F1": F1,
        "F2": F2,
        "K0_1": K0_1,
        "K0_2": K0_2,
        "sigma2_1": sigma2_1,
        "sigma2_2": sigma2_2,
        "r1": r1,
        "r2": r2,
        "n_strikes_1": len(qk1),
        "n_strikes_2": len(qk2),
    }
