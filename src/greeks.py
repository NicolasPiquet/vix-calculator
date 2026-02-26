"""
Greeks and risk sensitivities for VIX-style volatility indices.
Provides:
- Variance decomposition by strike (contribution ΔK/K² × e^{rT} × Q(K))
- Vega by maturity bucket (sensitivity to 1 vol-point shift per slice)
- Spot sensitivity (∂VIX/∂S via bump-and-revalue)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .vix_index import (
    compute_delta_k,
    compute_forward_and_k0,
    build_qk,
    compute_sigma2,
    interpolate_to_target,)

# ---------------------------------------------------------------------------
# Variance decomposition by strike
# ---------------------------------------------------------------------------

def variance_decomposition(
    qk: pd.Series,
    r: float,
    T: float,
    price_scale: float = 1.0,
) -> pd.DataFrame:
    """Decompose the variance into per-strike contributions.
    For each strike K_i, the contribution is:
        c_i = (2/T) × (ΔK_i / K_i²) × e^{rT} × Q(K_i) × price_scale
    Returns DataFrame with: strike, delta_k, qk, contribution, pct_contribution
    """
    strikes = qk.index.values
    prices = qk.values * price_scale
    delta_k = compute_delta_k(strikes)
    contributions = (2 / T) * (delta_k / strikes**2) * np.exp(r * T) * prices
    total = np.sum(contributions)
    return pd.DataFrame({
        "strike": strikes,
        "delta_k": delta_k,
        "qk": qk.values,
        "contribution": contributions,
        "pct_contribution": contributions / total * 100 if total != 0 else 0,})

# ---------------------------------------------------------------------------
# Vega by maturity bucket
# ---------------------------------------------------------------------------

def vega_bucket(
    opts_front: pd.DataFrame,
    opts_rear: pd.DataFrame,
    r1: float,
    r2: float,
    T1: float,
    T2: float,
    target_days: int = 30,
    price_scale: float = 1.0,
    bump_size: float = 0.01,
    strike_col: str = "options-strikePrice",
    type_col: str = "options-optiontype",
    price_col: str = "options-priorSettle",
) -> dict:
    """Compute VIX sensitivity to a 1-vol-point shift in each maturity bucket.
    Method: bump all option prices in one bucket by +bump_size (additive shift
    to the settlement price approximating a 1-point vol shift), recompute VIX,
    and take the difference.
    Returns dict with: vega_front, vega_rear, vega_total (in VIX points per 1% vol shift)
    """
    F1, K0_1 = compute_forward_and_k0(opts_front, r1, T1, strike_col, type_col, price_col)
    F2, K0_2 = compute_forward_and_k0(opts_rear, r2, T2, strike_col, type_col, price_col)
    qk1 = build_qk(opts_front, K0_1, strike_col, type_col, price_col)
    qk2 = build_qk(opts_rear, K0_2, strike_col, type_col, price_col)
    s2_1 = compute_sigma2(F1, K0_1, qk1, r1, T1, price_scale)
    s2_2 = compute_sigma2(F2, K0_2, qk2, r2, T2, price_scale)
    target_minutes = target_days * 24 * 60
    vix_base = interpolate_to_target(T1, T2, s2_1, s2_2, target_minutes)
    opts_front_bumped = opts_front.copy()
    opts_front_bumped[price_col] = opts_front_bumped[price_col] + bump_size
    F1b, K0_1b = compute_forward_and_k0(opts_front_bumped, r1, T1, strike_col, type_col, price_col)
    qk1b = build_qk(opts_front_bumped, K0_1b, strike_col, type_col, price_col)
    s2_1b = compute_sigma2(F1b, K0_1b, qk1b, r1, T1, price_scale)
    vix_front_bumped = interpolate_to_target(T1, T2, s2_1b, s2_2, target_minutes)
    opts_rear_bumped = opts_rear.copy()
    opts_rear_bumped[price_col] = opts_rear_bumped[price_col] + bump_size
    F2b, K0_2b = compute_forward_and_k0(opts_rear_bumped, r2, T2, strike_col, type_col, price_col)
    qk2b = build_qk(opts_rear_bumped, K0_2b, strike_col, type_col, price_col)
    s2_2b = compute_sigma2(F2b, K0_2b, qk2b, r2, T2, price_scale)
    vix_rear_bumped = interpolate_to_target(T1, T2, s2_1, s2_2b, target_minutes)
    return {
        "vix_base": vix_base,
        "vega_front": vix_front_bumped - vix_base,
        "vega_rear": vix_rear_bumped - vix_base,
        "vega_total": (vix_front_bumped - vix_base) + (vix_rear_bumped - vix_base),
        "bump_size": bump_size,}

# ---------------------------------------------------------------------------
# Spot sensitivity
# ---------------------------------------------------------------------------

def spot_sensitivity(
    opts_front: pd.DataFrame,
    opts_rear: pd.DataFrame,
    r1: float,
    r2: float,
    T1: float,
    T2: float,
    target_days: int = 30,
    price_scale: float = 1.0,
    spot_bump_pct: float = 1.0,
    strike_col: str = "options-strikePrice",
    type_col: str = "options-optiontype",
    price_col: str = "options-priorSettle",
) -> dict:
    """Estimate ∂VIX/∂S via bump-and-revalue on option prices.
    Approximates the effect of a spot move by shifting all option prices
    by their intrinsic value change (delta-weighted bump).
    For a simpler approximation: shift call prices up and put prices down
    by bump_amount for an upward spot move.
    Returns dict with: dvix_dspot, vix_base, vix_up, vix_down
    """
    F1, K0_1 = compute_forward_and_k0(opts_front, r1, T1, strike_col, type_col, price_col)
    F2, K0_2 = compute_forward_and_k0(opts_rear, r2, T2, strike_col, type_col, price_col)
    qk1 = build_qk(opts_front, K0_1, strike_col, type_col, price_col)
    qk2 = build_qk(opts_rear, K0_2, strike_col, type_col, price_col)
    s2_1 = compute_sigma2(F1, K0_1, qk1, r1, T1, price_scale)
    s2_2 = compute_sigma2(F2, K0_2, qk2, r2, T2, price_scale)
    target_minutes = target_days * 24 * 60
    vix_base = interpolate_to_target(T1, T2, s2_1, s2_2, target_minutes)
    bump_frac = spot_bump_pct / 100
    F1_up = F1 * (1 + bump_frac)
    F1_down = F1 * (1 - bump_frac)
    F2_up = F2 * (1 + bump_frac)
    F2_down = F2 * (1 - bump_frac)
    s2_1_up = compute_sigma2(F1_up, K0_1, qk1, r1, T1, price_scale)
    s2_2_up = compute_sigma2(F2_up, K0_2, qk2, r2, T2, price_scale)
    s2_1_down = compute_sigma2(F1_down, K0_1, qk1, r1, T1, price_scale)
    s2_2_down = compute_sigma2(F2_down, K0_2, qk2, r2, T2, price_scale)
    vix_up = interpolate_to_target(T1, T2, s2_1_up, s2_2_up, target_minutes)
    vix_down = interpolate_to_target(T1, T2, s2_1_down, s2_2_down, target_minutes)
    dvix_dspot = (vix_up - vix_down) / (2 * bump_frac)
    return {
        "vix_base": vix_base,
        "vix_up": vix_up,
        "vix_down": vix_down,
        "dvix_dspot_pct": dvix_dspot,
        "spot_bump_pct": spot_bump_pct,}