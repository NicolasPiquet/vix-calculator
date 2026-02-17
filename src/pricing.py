"""
Variance swap and volatility derivatives pricing.

Provides pricing tools for variance-based derivatives using the VIX methodology.
"""

from __future__ import annotations

import numpy as np


def variance_swap_fair_strike(sigma2: float) -> float:
    """
    Compute the fair strike of a variance swap.

    The fair variance strike K_var is simply the risk-neutral expected
    realized variance, which equals σ² from the VIX calculation.

    Parameters
    ----------
    sigma2 : float
        Annualized variance (σ²) from VIX calculation.

    Returns
    -------
    float
        Fair variance strike K_var (annualized).
    """
    return sigma2


def variance_swap_pnl(
    k_var_entry: float,
    realized_variance: float,
    notional_vega: float,
) -> float:
    """
    Compute P&L of a variance swap at maturity.

    P&L = Notional_vega × (RealizedVariance - K_var) / (2 × √K_var)

    For a vega notional position, the P&L scales with the difference
    between realized and strike variance.

    Parameters
    ----------
    k_var_entry : float
        Variance strike at entry (annualized).
    realized_variance : float
        Realized variance over the period (annualized).
    notional_vega : float
        Vega notional (exposure per volatility point).

    Returns
    -------
    float
        P&L in currency units.
    """
    k_vol = np.sqrt(k_var_entry)
    return notional_vega * (realized_variance - k_var_entry) / (2 * k_vol)


def variance_swap_mark_to_market(
    k_var_entry: float,
    k_var_current: float,
    notional_vega: float,
    time_elapsed_frac: float,
    realized_variance_so_far: float,
) -> float:
    """
    Mark-to-market a variance swap before maturity.

    Uses the weighted combination of realized variance (so far) and
    implied variance (for remaining period).

    Parameters
    ----------
    k_var_entry : float
        Variance strike at entry.
    k_var_current : float
        Current fair variance strike (from VIX).
    notional_vega : float
        Vega notional.
    time_elapsed_frac : float
        Fraction of time elapsed (0 to 1).
    realized_variance_so_far : float
        Annualized realized variance observed so far.

    Returns
    -------
    float
        Mark-to-market P&L.
    """
    # Expected total variance = weighted avg of realized and implied
    expected_var = (
        time_elapsed_frac * realized_variance_so_far
        + (1 - time_elapsed_frac) * k_var_current
    )

    k_vol = np.sqrt(k_var_entry)
    return notional_vega * (expected_var - k_var_entry) / (2 * k_vol)


def vol_swap_convexity_adjustment(sigma2: float, vol_of_vol: float = 0.0) -> float:
    """
    Approximate fair strike of a volatility swap.

    Vol swap strike ≈ √(K_var) - convexity_adjustment

    The convexity adjustment accounts for the fact that E[√X] < √E[X]
    (Jensen's inequality).

    Parameters
    ----------
    sigma2 : float
        Fair variance (σ²).
    vol_of_vol : float
        Volatility of volatility (used for convexity adjustment).
        If 0, returns simple √σ².

    Returns
    -------
    float
        Approximate fair volatility swap strike.
    """
    k_vol = np.sqrt(sigma2)

    if vol_of_vol <= 0:
        return k_vol

    # Approximate convexity adjustment: σ³ × volofvol² / (8 × σ²)
    # Simplified: ~ volofvol² / (8 × σ)
    adjustment = (vol_of_vol ** 2) / (8 * k_vol)
    return k_vol - adjustment


def vix_forward(
    sigma2_near: float,
    sigma2_far: float,
    T_near: float,
    T_far: float,
    T_forward_start: float,
    T_forward_end: float,
) -> float:
    """
    Compute VIX forward (forward-starting variance swap strike).

    Uses the variance term structure to extract forward variance.

    Parameters
    ----------
    sigma2_near : float
        Variance for near-term expiration.
    sigma2_far : float
        Variance for far-term expiration.
    T_near : float
        Time to near expiration (years).
    T_far : float
        Time to far expiration (years).
    T_forward_start : float
        Forward start time (years).
    T_forward_end : float
        Forward end time (years).

    Returns
    -------
    float
        Forward VIX (annualized volatility, in %).
    """
    # Total variance to each maturity
    total_var_near = sigma2_near * T_near
    total_var_far = sigma2_far * T_far

    # Interpolate to get total variance at forward start and end
    # Linear interpolation in total variance space
    if T_far <= T_near:
        raise ValueError("T_far must be greater than T_near")

    slope = (total_var_far - total_var_near) / (T_far - T_near)

    total_var_start = total_var_near + slope * (T_forward_start - T_near)
    total_var_end = total_var_near + slope * (T_forward_end - T_near)

    # Forward variance
    forward_var = (total_var_end - total_var_start) / (T_forward_end - T_forward_start)

    if forward_var <= 0:
        raise ValueError(f"Negative forward variance: {forward_var:.6f}")

    return 100 * np.sqrt(forward_var)


def delta_vix_to_spot(
    vix: float,
    spot_change_pct: float = 1.0,
    beta: float = -0.5,
) -> float:
    """
    Estimate VIX delta (sensitivity to spot move).

    Empirically, VIX tends to move inversely to spot with a leverage effect.
    This provides a rough approximation.

    Parameters
    ----------
    vix : float
        Current VIX level.
    spot_change_pct : float
        Spot move in percentage points (e.g., 1.0 = 1% move).
    beta : float
        Empirical beta (typically -0.5 to -1.0 for equity indices).
        Negative because VIX rises when spot falls.

    Returns
    -------
    float
        Expected VIX change (in volatility points).
    """
    # Approximate: ΔVIX ≈ β × VIX × (ΔSpot / Spot)
    return beta * vix * spot_change_pct / 100


def vega_variance_swap(
    notional_vega: float,
    k_var: float,
) -> float:
    """
    Compute vega exposure of a variance swap.

    Vega = ∂V/∂σ for a variance swap.

    Parameters
    ----------
    notional_vega : float
        Vega notional of the swap.
    k_var : float
        Variance strike.

    Returns
    -------
    float
        Vega exposure (P&L per 1 vol point move in implied vol).
    """
    k_vol = np.sqrt(k_var)
    # For small moves: dV/dσ ≈ Notional_vega × σ / k_vol
    # At inception (σ = k_vol): vega ≈ Notional_vega
    return notional_vega
