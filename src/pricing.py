"""
Variance swap and volatility derivatives pricing.
Provides pricing tools for variance-based derivatives using the VIX methodology.
"""
from __future__ import annotations
import numpy as np
import pandas as pd  # noqa: F401 — used in type annotations

def variance_swap_fair_strike(sigma2: float) -> float:
    return sigma2

def variance_swap_pnl(
    k_var_entry: float,
    realized_variance: float,
    notional_vega: float,) -> float:
    k_vol = np.sqrt(k_var_entry)
    return notional_vega * (realized_variance - k_var_entry) / (2 * k_vol)

def variance_swap_mark_to_market(
    k_var_entry: float,
    k_var_current: float,
    notional_vega: float,
    time_elapsed_frac: float,
    realized_variance_so_far: float,) -> float:
    expected_var = (
        time_elapsed_frac * realized_variance_so_far
        + (1 - time_elapsed_frac) * k_var_current)
    k_vol = np.sqrt(k_var_entry)
    return notional_vega * (expected_var - k_var_entry) / (2 * k_vol)

def vol_swap_convexity_adjustment(sigma2: float, vol_of_vol: float = 0.0) -> float:
    k_vol = np.sqrt(sigma2)
    if vol_of_vol <= 0:
        return k_vol
    adjustment = (vol_of_vol ** 2) / (8 * k_vol)
    return k_vol - adjustment

def vix_forward(
    sigma2_near: float,
    sigma2_far: float,
    T_near: float,
    T_far: float,
    T_forward_start: float,
    T_forward_end: float,) -> float:
    total_var_near = sigma2_near * T_near
    total_var_far = sigma2_far * T_far
    if T_far <= T_near:
        raise ValueError("T_far must be greater than T_near")
    slope = (total_var_far - total_var_near) / (T_far - T_near)
    total_var_start = total_var_near + slope * (T_forward_start - T_near)
    total_var_end = total_var_near + slope * (T_forward_end - T_near)
    forward_var = (total_var_end - total_var_start) / (T_forward_end - T_forward_start)
    if forward_var <= 0:
        raise ValueError(f"Negative forward variance: {forward_var:.6f}")
    return 100 * np.sqrt(forward_var)

def delta_vix_to_spot(
    vix: float,
    spot_change_pct: float = 1.0,
    beta: float = -0.5,
) -> float:
    return beta * vix * spot_change_pct / 100

def vega_variance_swap(
    notional_vega: float,
    k_var: float,
) -> float:
    return notional_vega

# ---------------------------------------------------------------------------
# Notional conventions
# ---------------------------------------------------------------------------

def vega_to_variance_notional(notional_vega: float, k_var: float) -> float:
    """Convert vega notional to variance notional.
    Variance notional = Vega notional / (2 × √K_var)
    """
    return notional_vega / (2 * np.sqrt(k_var))

def variance_to_vega_notional(notional_var: float, k_var: float) -> float:
    """Convert variance notional to vega notional.
    Vega notional = Variance notional × 2 × √K_var
    """
    return notional_var * 2 * np.sqrt(k_var)

def variance_swap_pnl_from_notional(
    k_var: float,
    realized_variance: float,
    notional_var: float,
) -> float:
    """P&L of a variance swap using variance notional.
    P&L = N_var × (σ²_realized - K_var)
    """
    return notional_var * (realized_variance - k_var)

# ---------------------------------------------------------------------------
# Corridor variance swap
# ---------------------------------------------------------------------------

def corridor_variance(
    qk: "pd.Series",
    F: float,
    K0: float,
    r: float,
    T: float,
    K_low: float,
    K_high: float,
    price_scale: float = 1.0,
) -> float:
    """Compute corridor variance (variance conditional on spot in [K_low, K_high]).
    Uses the same CBOE formula but restricts the strike sum to the corridor:
        σ²_corridor = (2/T) Σ (ΔK/K²) e^{rT} Q(K) × 𝟙{K_low ≤ K ≤ K_high}
                      - (1/T)(F/K0 - 1)²
    Parameters
    ----------
    qk : Series indexed by strike with Q(K) values
    F, K0, r, T : as in compute_sigma2
    K_low, K_high : corridor bounds
    price_scale : strike/price unit alignment
    """
    from .vix_index import compute_delta_k
    strikes = qk.index.values
    prices = qk.values * price_scale
    delta_k = compute_delta_k(strikes)
    mask = (strikes >= K_low) & (strikes <= K_high)
    contribution = np.sum(
        (delta_k[mask] / strikes[mask]**2) * np.exp(r * T) * prices[mask])
    sigma2 = (2 / T) * contribution - (1 / T) * (F / K0 - 1)**2
    return float(sigma2)

def corridor_variance_swap_pnl(
    k_var_corridor: float,
    realized_variance_in_corridor: float,
    notional_var: float,
) -> float:
    """P&L of a corridor variance swap."""
    return notional_var * (realized_variance_in_corridor - k_var_corridor)

# ---------------------------------------------------------------------------
# Forward-starting variance swap
# ---------------------------------------------------------------------------

def forward_variance(
    sigma2_near: float,
    sigma2_far: float,
    T_near: float,
    T_far: float,
) -> float:
    """Extract forward variance between two maturities.
    σ²_fwd = (σ²_far × T_far - σ²_near × T_near) / (T_far - T_near)
    Returns the forward variance (annualized).
    """
    if T_far <= T_near:
        raise ValueError("T_far must be greater than T_near")
    fwd = (sigma2_far * T_far - sigma2_near * T_near) / (T_far - T_near)
    if fwd < 0:
        raise ValueError(f"Negative forward variance: {fwd:.6f}. Calendar arbitrage.")
    return fwd

def forward_vol(sigma2_near: float, sigma2_far: float, T_near: float, T_far: float) -> float:
    """Forward implied volatility (annualized, in %)."""
    return 100 * np.sqrt(forward_variance(sigma2_near, sigma2_far, T_near, T_far))
