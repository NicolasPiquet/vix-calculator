"""
Volatility skew analysis.
Provides:
- Strike-to-delta conversion and delta-based vol interpolation
- Risk Reversal (25Δ) and Butterfly (25Δ) metrics
- Skew term structure extraction
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .implied_vol import bs_delta

# ---------------------------------------------------------------------------
# Delta-based vol extraction
# ---------------------------------------------------------------------------

def strike_to_delta(
    K: float, S: float, T: float, r: float, sigma: float, option_type: str = "call",
) -> float:
    """Convert a strike to its BS delta."""
    return bs_delta(S, K, T, r, sigma, option_type)

def vol_at_delta(
    target_delta: float,
    strikes: np.ndarray,
    ivs: np.ndarray,
    forward: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Interpolate the implied vol at a given delta level.
    Computes BS delta for each (strike, iv) pair, then interpolates.
    """
    deltas = np.array([
        bs_delta(forward, K, T, r, iv, option_type) for K, iv in zip(strikes, ivs)])
    order = np.argsort(deltas)
    deltas_sorted = deltas[order]
    ivs_sorted = ivs[order]
    mask = np.isfinite(deltas_sorted) & np.isfinite(ivs_sorted)
    deltas_sorted = deltas_sorted[mask]
    ivs_sorted = ivs_sorted[mask]
    if len(deltas_sorted) < 2:
        return np.nan
    if target_delta < deltas_sorted[0] or target_delta > deltas_sorted[-1]:
        return np.nan
    f = interp1d(deltas_sorted, ivs_sorted, kind="linear", fill_value="extrapolate")
    return float(f(target_delta))

# ---------------------------------------------------------------------------
# Skew metrics
# ---------------------------------------------------------------------------

def compute_skew_metrics(
    iv_slice: pd.DataFrame,
    forward: float,
    T: float,
    r: float,
    strike_col: str = "strike",
    iv_col: str = "iv",
    type_col: str = "option_type",
) -> dict:
    """Compute RR25, BF25, and ATM vol for a single expiration slice.
    Returns dict with: atm_vol, call_25d_vol, put_25d_vol, rr25, bf25
    """
    calls = iv_slice[iv_slice[type_col] == "call"]
    puts = iv_slice[iv_slice[type_col] == "put"]
    atm_vol = vol_at_delta(
        0.50, calls[strike_col].values, calls[iv_col].values, forward, T, r, "call")
    if np.isnan(atm_vol):
        closest_idx = (iv_slice[strike_col] - forward).abs().idxmin()
        atm_vol = iv_slice.loc[closest_idx, iv_col]
    call_25d = vol_at_delta(
        0.25, calls[strike_col].values, calls[iv_col].values, forward, T, r, "call")
    put_25d = vol_at_delta(
        -0.25, puts[strike_col].values, puts[iv_col].values, forward, T, r, "put")
    rr25 = np.nan
    bf25 = np.nan
    if np.isfinite(call_25d) and np.isfinite(put_25d) and np.isfinite(atm_vol):
        rr25 = call_25d - put_25d                       # Risk Reversal
        bf25 = (call_25d + put_25d) / 2 - atm_vol       # Butterfly
    return {
        "atm_vol": atm_vol,
        "call_25d_vol": call_25d,
        "put_25d_vol": put_25d,
        "rr25": rr25,
        "bf25": bf25,}

def skew_term_structure(
    surface_df: pd.DataFrame,
    forwards: dict[str, float],
    times: dict[str, float],
    rates: dict[str, float],
) -> pd.DataFrame:
    """Compute skew metrics across the term structure.
    Parameters
    ----------
    surface_df : IV surface from build_full_surface()
    forwards, times, rates : dicts keyed by expiration string
    Returns DataFrame with: expiration, T, atm_vol, rr25, bf25, call_25d_vol, put_25d_vol
    """
    results = []
    for exp, group in surface_df.groupby("expiration"):
        exp_str = str(exp)
        if exp_str not in forwards:
            continue
        F = forwards[exp_str]
        T = times[exp_str]
        r = rates[exp_str]
        metrics = compute_skew_metrics(group, F, T, r)
        metrics["expiration"] = exp_str
        metrics["T"] = T
        results.append(metrics)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df
