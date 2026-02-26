"""
Implied volatility surface construction and SVI calibration
Provides:
- Black-Scholes pricing and Newton-Raphson implied vol inversion
- Implied vol surface extraction from an options chain
- SVI (Stochastic Volatility Inspired) parametric smile calibration
Reference:
    Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility derivatives."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
) -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        # Intrinsic value
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (∂Price/∂σ)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
) -> float:
    """Black-Scholes delta (∂Price/∂S)."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0

# ---------------------------------------------------------------------------
# Implied volatility inversion (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
    initial_guess: float = 0.20,
) -> float:
    """Extract implied volatility via Newton-Raphson.
    Returns NaN if the algorithm fails to converge or the price is
    below intrinsic value.
    """
    if T <= 0 or price <= 0:
        return np.nan
    # Check intrinsic bounds
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0.0)
    if price < intrinsic - 1e-10:
        return np.nan
    sigma = initial_guess
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12:
            # Vega too small — switch to bisection fallback
            return _implied_vol_bisection(price, S, K, T, r, option_type, tol)
        sigma_new = sigma - (p - price) / v
        if sigma_new <= 0:
            sigma_new = sigma / 2
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return np.nan

def _implied_vol_bisection(
    price: float, S: float, K: float, T: float, r: float,
    option_type: str, tol: float, lo: float = 1e-4, hi: float = 5.0,
) -> float:
    """Bisection fallback for implied vol when Newton fails."""
    for _ in range(200):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, r, mid, option_type)
        if abs(p - price) < tol:
            return mid
        if p > price:
            hi = mid
        else:
            lo = mid
    return np.nan

# ---------------------------------------------------------------------------
# IV surface construction
# ---------------------------------------------------------------------------

def build_iv_surface(
    options_df: pd.DataFrame,
    forward: float,
    T: float,
    r: float,
    strike_col: str = "strikePrice",
    type_col: str = "optionType",
    price_col: str = "settlement",
) -> pd.DataFrame:
    """Build implied vol by strike for a single expiration slice.
    Uses OTM options (puts for K < F, calls for K >= F) following VIX convention.
    Returns DataFrame with columns: strike, moneyness, iv, option_type
    """
    rows = []
    for _, row in options_df.iterrows():
        K = row[strike_col]
        opt_type = row[type_col]
        price = row[price_col]
        if K < forward and opt_type != "put":
            continue
        if K >= forward and opt_type != "call":
            continue
        if price <= 0:
            continue
        iv = implied_vol(price, forward, K, T, r, opt_type)
        if np.isnan(iv) or iv <= 0 or iv > 3.0:
            continue
        rows.append({
            "strike": K,
            "moneyness": np.log(K / forward),  # log-moneyness k
            "iv": iv,
            "option_type": opt_type,})
    return pd.DataFrame(rows)

def build_full_surface(
    options_df: pd.DataFrame,
    trade_date: str,
    forwards: dict[str, float],
    times: dict[str, float],
    rates: dict[str, float],
    trade_date_col: str = "tradeDate",
    expiration_col: str = "expirationDate",
    **kwargs,
) -> pd.DataFrame:
    """Build IV surface across multiple expirations for a given trade date.
    Parameters
    ----------
    forwards : dict mapping expiration str -> forward price F
    times : dict mapping expiration str -> time to expiry T
    rates : dict mapping expiration str -> risk-free rate r
    Returns DataFrame with: expiration, T, strike, moneyness, iv, option_type
    """
    mask = options_df[trade_date_col] == trade_date
    all_slices = []
    for exp_str, F in forwards.items():
        T = times[exp_str]
        r = rates[exp_str]
        exp_mask = mask & (options_df[expiration_col] == exp_str)
        slice_df = options_df.loc[exp_mask]
        if slice_df.empty or T <= 0:
            continue
        iv_slice = build_iv_surface(slice_df, F, T, r, **kwargs)
        if iv_slice.empty:
            continue
        iv_slice["expiration"] = exp_str
        iv_slice["T"] = T
        all_slices.append(iv_slice)
    if not all_slices:
        return pd.DataFrame()
    return pd.concat(all_slices, ignore_index=True)

# ---------------------------------------------------------------------------
# SVI calibration
# ---------------------------------------------------------------------------

def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """SVI raw parameterization: w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))
    where w = σ_imp² × T  (total implied variance).
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def calibrate_svi(
    moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Calibrate SVI parameters to a smile slice.
    Parameters
    ----------
    moneyness : array of log(K/F) values
    total_variance : array of σ²×T values
    weights : optional weights (e.g. inverse vega)
    Returns dict with keys: a, b, rho, m, sigma, rmse
    """
    if weights is None:
        weights = np.ones_like(moneyness)
    def objective(params):
        a, b, rho, m, sig = params
        model = svi_raw(moneyness, a, b, rho, m, sig)
        residuals = (model - total_variance) * weights
        return np.sum(residuals**2)
    atm_var = np.interp(0.0, moneyness, total_variance)
    x0 = [atm_var, 0.1, -0.2, 0.0, 0.1]
    bounds = [
        (1e-6, None),   
        (1e-6, None),   
        (-0.99, 0.99),  
        (-1.0, 1.0),    
        (1e-4, 2.0),]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sig = result.x
    fitted = svi_raw(moneyness, a, b, rho, m, sig)
    rmse = np.sqrt(np.mean((fitted - total_variance)**2))
    return {
        "a": a, "b": b, "rho": rho, "m": m, "sigma": sig,
        "rmse": rmse, "success": result.success,}

def calibrate_svi_surface(
    surface_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calibrate SVI to each expiration slice of an IV surface.
    Returns DataFrame with one row per expiration and SVI parameters.
    """
    results = []
    for exp, group in surface_df.groupby("expiration"):
        T = group["T"].iloc[0]
        k = group["moneyness"].values
        w = (group["iv"].values ** 2) * T  # total variance
        if len(k) < 5:
            continue
        params = calibrate_svi(k, w)
        params["expiration"] = exp
        params["T"] = T
        params["n_points"] = len(k)
        results.append(params)
    return pd.DataFrame(results)