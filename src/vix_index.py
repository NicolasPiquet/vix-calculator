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
    """Forward-fill missing dates in Treasury CMT rate data.
    Pivots rates to wide format, reindexes to a continuous daily calendar,
    forward-fills gaps (weekends, holidays), then melts back to long format.
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
    """Return the annualised risk-free rate for a given maturity.
    Looks up the CMT rate table for *trade_date* and *maturity_months*.
    If no exact maturity match exists, linearly interpolates between the
    two closest available tenors.  Returns the rate as a decimal (e.g. 0.045).
    """
    maturity_label = f"{maturity_months} Mo"
    mask = (cmt_rate["maturity"] == maturity_label) & (cmt_rate["Date"] == trade_date)
    matches = cmt_rate.loc[mask, "value"]
    if not matches.empty:
        return matches.iloc[0] / 100.0
    date_rates = cmt_rate[cmt_rate["Date"] == trade_date].copy()
    if date_rates.empty:
        raise ValueError(f"No CMT rates found for date {trade_date}")
    date_rates["months"] = date_rates["maturity"].str.extract(r"(\d+)").astype(float)
    date_rates = date_rates.dropna(subset=["months"]).sort_values("months")
    available_months = date_rates["months"].values
    available_rates = date_rates["value"].values
    lower_mask = available_months <= maturity_months
    upper_mask = available_months >= maturity_months
    if not lower_mask.any() or not upper_mask.any():
        raise ValueError(
            f"Cannot interpolate rate for {maturity_months} Mo on {trade_date}. "
            f"Available maturities: {sorted(available_months)}")
    lower_idx = np.where(lower_mask)[0][-1]
    upper_idx = np.where(upper_mask)[0][0]
    if lower_idx == upper_idx:
        return available_rates[lower_idx] / 100.0
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
    holidays: list[str],) -> dt.datetime:
    """Compute the settlement datetime for a CME futures contract.
    Counts *expiration_day_offset* business days backwards from the end
    of the expiration month, skipping holidays.
    """
    month_end = current_day + dateutil.relativedelta.relativedelta(
        day=31,
        months=months_ahead - 1,
        hour=expiration_hour,
        minute=0,
        second=0,
        microsecond=0,)
    settlement = month_end
    business_day_count = 0
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
    """Return time to expiration in years (minute-precise, CBOE convention)."""
    settlement = get_settlement_day(
        current_day, months_ahead, expiration_day_offset, expiration_hour, holidays)
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
    """Derive the forward price F and ATM strike K₀ from put-call parity.
    F = K* + e^{rT} (C(K*) - P(K*))  where K* minimises |C - P|.
    K₀ = max{K : K ≤ F}.
    """
    opts = options.copy()
    opts = opts.sort_values(strike_col)
    pivoted = opts.pivot_table(index=strike_col, columns=type_col, values=price_col, aggfunc="mean")
    if "call" not in pivoted.columns or "put" not in pivoted.columns:
        raise ValueError("Options data must contain both calls and puts")
    diff = (pivoted["call"] - pivoted["put"]).abs()
    k_star = diff.idxmin()
    call_price = pivoted.loc[k_star, "call"]
    put_price = pivoted.loc[k_star, "put"]
    F = k_star + np.exp(r * T) * (call_price - put_price)
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
    consecutive_zeros: int = 2,) -> pd.Series:
    """Construct the Q(K) series of OTM option prices per the CBOE methodology.
    Uses puts for K < K₀, calls for K > K₀, and the average at K₀.
    Stops including strikes after *consecutive_zeros* zero-price quotes
    when moving outward from K₀.
    """
    opts = options.copy()
    pivoted = opts.pivot_table(index=strike_col, columns=type_col, values=price_col, aggfunc="mean").sort_index()
    strikes = pivoted.index.values
    qk_values = {}
    put_strikes = strikes[strikes < K0][::-1]  
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
    """Compute CBOE strike intervals ΔK for each strike.
    Interior: ΔKᵢ = (Kᵢ₊₁ − Kᵢ₋₁) / 2.  Boundaries use the adjacent gap.
    """
    n = len(strikes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])
    delta_k = np.empty(n)
    delta_k[0] = strikes[1] - strikes[0]
    delta_k[-1] = strikes[-1] - strikes[-2]
    delta_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2
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
    price_scale: float = 1.0,) -> float:
    """Compute the CBOE variance contribution σ² for one expiration term.
    σ² = (2/T) Σ (ΔK/K²) e^{rT} Q(K) − (1/T)(F/K₀ − 1)²
    """
    if len(qk) < 2:
        raise ValueError(f"Need at least 2 strikes for sigma2, got {len(qk)}")
    strikes = qk.index.values
    prices = qk.values * price_scale
    delta_k = compute_delta_k(strikes)
    contribution = np.sum((delta_k / strikes**2) * np.exp(r * T) * prices)
    sigma2 = (2 / T) * contribution - (1 / T) * (F / K0 - 1) ** 2
    if sigma2 <= 0:
        raise ValueError(
            f"Computed σ² = {sigma2:.6f} <= 0. "
            "This typically indicates a unit mismatch between strikes and option prices. "
            "Check the 'price_scale' parameter.")
    return float(sigma2)

# -----------------------------------------------------------------------------
# VIX interpolation
# -----------------------------------------------------------------------------

def interpolate_to_target(
    T1: float,
    T2: float,
    sigma2_1: float,
    sigma2_2: float,
    target_minutes: int = MINUTES_30_DAYS,) -> float:
    """Interpolate near/next-term variances to the target maturity (CBOE formula).
    Returns the VIX value (in percentage points).
    """
    N1 = T1 * MINUTES_PER_YEAR
    N2 = T2 * MINUTES_PER_YEAR
    N_target = target_minutes
    if not (N1 < N_target < N2):
        raise ValueError(
            f"Target maturity not bracketed by expirations. "
            f"N1={N1:.0f} min, N_target={N_target:.0f} min, N2={N2:.0f} min. "
            f"Pick different front_months/rear_months to bracket the target.")
    w1 = T1 * sigma2_1 * (N2 - N_target) / (N2 - N1)
    w2 = T2 * sigma2_2 * (N_target - N1) / (N2 - N1)
    vix = 100 * np.sqrt((w1 + w2) * MINUTES_PER_YEAR / N_target)
    return float(vix)

# -----------------------------------------------------------------------------
# Common VIX computation core
# -----------------------------------------------------------------------------

def _compute_vix_from_options(
    opts_front: pd.DataFrame,
    opts_rear: pd.DataFrame,
    r1: float,
    r2: float,
    T1: float,
    T2: float,
    target_days: int,
    price_scale: float = 1.0,
) -> dict[str, Any]:
    """Shared VIX computation: forward, Q(K), sigma², interpolation.
    Both ``vix_index`` and ``vix_index_equity`` delegate here once they
    have resolved their front/rear option slices, rates and times.
    """
    F1, K0_1 = compute_forward_and_k0(opts_front, r1, T1)
    F2, K0_2 = compute_forward_and_k0(opts_rear, r2, T2)
    qk1 = build_qk(opts_front, K0_1)
    qk2 = build_qk(opts_rear, K0_2)
    if len(qk1) < 2:
        raise ValueError(f"Insufficient strikes for front-term: {len(qk1)}")
    if len(qk2) < 2:
        raise ValueError(f"Insufficient strikes for rear-term: {len(qk2)}")
    sigma2_1 = compute_sigma2(F1, K0_1, qk1, r1, T1, price_scale)
    sigma2_2 = compute_sigma2(F2, K0_2, qk2, r2, T2, price_scale)
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
        "n_strikes_2": len(qk2),}

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
    """Compute VIX-style volatility index for CME futures options (e.g. Henry Hub).
    Selects front/rear month contracts by *front_months*/*rear_months*,
    resolves settlement dates via CME calendar, and delegates to the
    shared ``_compute_vix_from_options`` core.
    """
    if isinstance(trade_date, str):
        trade_date = pd.Timestamp(trade_date)
    holidays = calendar_df["DATE"].astype(str).tolist()
    rates = fill_cmt_missing_dates(rates_df)
    r1 = get_rate_for_maturity(rates, trade_date, front_months)
    r2 = get_rate_for_maturity(rates, trade_date, rear_months)
    mask = (options_df["options-id"] == options_id) & (options_df["tradeDate"] == trade_date)
    current_options = options_df.loc[mask].copy()
    if current_options.empty:
        raise ValueError(f"No options found for id={options_id} on {trade_date}")
    front_exp = pd.Timestamp(f"{trade_date.year}-{trade_date.month}-1") + pd.DateOffset(months=front_months)
    rear_exp = pd.Timestamp(f"{trade_date.year}-{trade_date.month}-1") + pd.DateOffset(months=rear_months)
    front_exp = pd.Timestamp(f"{front_exp.year}-{front_exp.month}-1")
    rear_exp = pd.Timestamp(f"{rear_exp.year}-{rear_exp.month}-1")
    opts_front = current_options[current_options["futures-expirationDate"] == front_exp].copy()
    opts_rear = current_options[current_options["futures-expirationDate"] == rear_exp].copy()
    if opts_front.empty:
        raise ValueError(f"No front-month options found for expiration {front_exp}")
    if opts_rear.empty:
        raise ValueError(f"No rear-month options found for expiration {rear_exp}")
    current_front = opts_front["futures-updated"].iloc[0]
    current_rear = opts_rear["futures-updated"].iloc[0]
    T1 = time_to_expiration_years(current_front, front_months, expiration_day_offset, expiration_hour, holidays)
    T2 = time_to_expiration_years(current_rear, rear_months, expiration_day_offset, expiration_hour, holidays)
    if T1 <= 0 or T2 <= 0:
        raise ValueError(f"Invalid time to expiration: T1={T1:.4f}, T2={T2:.4f}")
    return _compute_vix_from_options(
        opts_front, opts_rear, r1, r2, T1, T2, target_days, price_scale,)

# -----------------------------------------------------------------------------
# Equity options orchestrator
# -----------------------------------------------------------------------------

def _select_bracketing_expirations(
    available_expirations: list[pd.Timestamp],
    trade_date: pd.Timestamp,
    target_days: int,
    min_dte: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Select the two expirations that bracket the target maturity.
    Only considers expirations with at least *min_dte* days to expiry
    (to avoid near-expiry micro-structure noise).
    Returns (near_exp, next_exp) where:
        near_exp < trade_date + target_days <= next_exp
    """
    target_date = trade_date + pd.Timedelta(days=target_days)
    candidates = sorted(
        e for e in available_expirations
        if (e - trade_date).days >= min_dte)
    if len(candidates) < 2:
        raise ValueError(
            f"Need at least 2 valid expirations after {trade_date}, "
            f"got {len(candidates)}. Available: {candidates}")
    near_candidates = [e for e in candidates if e <= target_date]
    next_candidates = [e for e in candidates if e > target_date]
    if not near_candidates or not next_candidates:
        raise ValueError(
            f"Cannot bracket {target_days}-day target ({target_date.date()}) "
            f"with available expirations. Near: {near_candidates}, Next: {next_candidates}")
    near_exp = near_candidates[-1]   
    next_exp = next_candidates[0]     
    return near_exp, next_exp

def vix_index_equity(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    trade_date: pd.Timestamp | str,
    target_days: int = 30,
    price_scale: float = 1.0,
    expiration_hour: int = 16,
    min_dte: int = 7,
    strike_col: str = "strikePrice",
    type_col: str = "optionType",
    price_col: str = "settlement",
    expiration_col: str = "expirationDate",
    trade_date_col: str = "tradeDate",
) -> dict[str, Any]:
    """Compute VIX-style volatility index for equity/ETF options.
    Unlike ``vix_index`` (designed for CME futures options), this function
    works with options that carry explicit expiration dates and selects the
    two expirations that bracket the target maturity automatically.
    Parameters
    ----------
    options_df : DataFrame
        Daily options data with columns for trade date, expiration, strike,
        option type ("call"/"put"), and settlement/close price.
    rates_df : DataFrame
        Treasury CMT rates (Date, maturity, value).
    calendar_df : DataFrame
        Holiday calendar with DATE column.
    trade_date : str or Timestamp
        The calculation date.
    target_days : int
        Target maturity in calendar days (default 30 for standard VIX).
    price_scale : float
        Multiplier to align strike and option price units (usually 1.0).
    expiration_hour : int
        Hour of day at which options expire (for minute-level T calc).
    min_dte : int
        Minimum days-to-expiry for an expiration to be considered.
    strike_col, type_col, price_col, expiration_col, trade_date_col : str
        Column name overrides.
    """
    if isinstance(trade_date, str):
        trade_date = pd.Timestamp(trade_date)
    rates = fill_cmt_missing_dates(rates_df)
    mask = pd.to_datetime(options_df[trade_date_col]) == trade_date
    current_options = options_df.loc[mask].copy()
    if current_options.empty:
        raise ValueError(f"No options found on {trade_date}")
    current_options["_exp"] = pd.to_datetime(current_options[expiration_col])
    available_expirations = current_options["_exp"].unique().tolist()
    near_exp, next_exp = _select_bracketing_expirations(
        available_expirations, trade_date, target_days, min_dte)
    opts_front = current_options[current_options["_exp"] == near_exp].copy()
    opts_rear = current_options[current_options["_exp"] == next_exp].copy()
    near_exp_dt = near_exp + pd.Timedelta(hours=expiration_hour)
    next_exp_dt = next_exp + pd.Timedelta(hours=expiration_hour)
    trade_dt = trade_date + pd.Timedelta(hours=16)  
    T1 = (near_exp_dt - trade_dt).total_seconds() / 60 / MINUTES_PER_YEAR
    T2 = (next_exp_dt - trade_dt).total_seconds() / 60 / MINUTES_PER_YEAR
    if T1 <= 0 or T2 <= 0:
        raise ValueError(f"Invalid time to expiration: T1={T1:.4f}, T2={T2:.4f}")
    near_months = max(1, round((near_exp - trade_date).days / 30))
    next_months = max(1, round((next_exp - trade_date).days / 30))
    near_months = min(near_months, 6)
    next_months = min(next_months, 6)
    r1 = get_rate_for_maturity(rates, trade_date, near_months)
    r2 = get_rate_for_maturity(rates, trade_date, next_months)
    col_map = {
        strike_col: "options-strikePrice",
        type_col: "options-optiontype",
        price_col: "options-priorSettle",}
    opts_front = opts_front.rename(columns=col_map)
    opts_rear = opts_rear.rename(columns=col_map)
    result = _compute_vix_from_options(
        opts_front, opts_rear, r1, r2, T1, T2, target_days, price_scale,)
    result["near_exp"] = near_exp
    result["next_exp"] = next_exp
    return result