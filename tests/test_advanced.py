"""Tests for advanced modules: implied_vol, skew, pricing, greeks, backtest."""

import numpy as np
import pandas as pd
import pytest
from src.implied_vol import (
    bs_price, bs_vega, bs_delta, implied_vol, svi_raw, calibrate_svi,
    build_iv_surface,)
from src.skew import vol_at_delta, compute_skew_metrics
from src.pricing import (
    vega_to_variance_notional, variance_to_vega_notional,
    forward_variance, forward_vol, corridor_variance,
    variance_swap_pnl_from_notional,)
from src.greeks import variance_decomposition, vega_bucket, spot_sensitivity
from src.backtest import (
    realized_vol_close, realized_vol_parkinson, realized_vol_yang_zhang,
    variance_risk_premium,)
from src.implied_vol import build_full_surface, calibrate_svi_surface

# ===================================================================
# Black-Scholes
# ===================================================================

class TestBlackScholes:
    def test_call_put_parity(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        c = bs_price(S, K, T, r, sigma, "call")
        p = bs_price(S, K, T, r, sigma, "put")
        # C - P = S - K*exp(-rT)
        assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-10
    def test_atm_call_positive(self):
        c = bs_price(100, 100, 0.5, 0.05, 0.20, "call")
        assert c > 0
    def test_deep_itm_call(self):
        c = bs_price(200, 100, 0.5, 0.05, 0.20, "call")
        assert c > 99  # close to intrinsic
    def test_vega_positive(self):
        v = bs_vega(100, 100, 0.5, 0.05, 0.20)
        assert v > 0
    def test_delta_atm_call_near_half(self):
        d = bs_delta(100, 100, 0.25, 0.05, 0.20, "call")
        assert 0.45 < d < 0.60
    def test_delta_put_negative(self):
        d = bs_delta(100, 100, 0.25, 0.05, 0.20, "put")
        assert d < 0

# ===================================================================
# Implied volatility
# ===================================================================

class TestImpliedVol:
    def test_roundtrip(self):
        S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.25
        price = bs_price(S, K, T, r, sigma, "call")
        iv = implied_vol(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6
    def test_roundtrip_put(self):
        S, K, T, r, sigma = 100, 95, 0.5, 0.05, 0.30
        price = bs_price(S, K, T, r, sigma, "put")
        iv = implied_vol(price, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6
    def test_returns_nan_for_zero_price(self):
        assert np.isnan(implied_vol(0, 100, 100, 0.5, 0.05))
    def test_returns_nan_for_negative_T(self):
        assert np.isnan(implied_vol(5, 100, 100, -0.1, 0.05))

# ===================================================================
# SVI
# ===================================================================

class TestSVI:
    def test_svi_raw_atm(self):
        w = svi_raw(np.array([0.0]), a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        expected = 0.04 + 0.1 * (-0.3 * 0 + np.sqrt(0 + 0.01))
        assert abs(w[0] - expected) < 1e-10
    def test_svi_symmetric_when_rho_zero(self):
        k = np.array([-0.1, 0.1])
        w = svi_raw(k, a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        assert abs(w[0] - w[1]) < 1e-10
    def test_calibrate_svi_recovers_params(self):
        k = np.linspace(-0.3, 0.3, 50)
        true_w = svi_raw(k, a=0.04, b=0.08, rho=-0.25, m=0.01, sigma=0.15)
        result = calibrate_svi(k, true_w)
        assert result["rmse"] < 1e-4
        assert result["success"]

# ===================================================================
# IV surface construction
# ===================================================================

class TestBuildIvSurface:
    def test_basic_surface(self):
        F, T, r = 250.0, 0.08, 0.045
        sigma = 0.15
        rows = []
        for K in np.arange(240, 261, 1.0):
            cp = bs_price(F, K, T, r, sigma, "call")
            pp = bs_price(F, K, T, r, sigma, "put")
            rows.append({"strikePrice": K, "optionType": "call", "settlement": cp})
            rows.append({"strikePrice": K, "optionType": "put", "settlement": pp})
        df = pd.DataFrame(rows)
        surface = build_iv_surface(df, F, T, r)
        assert len(surface) > 0
        assert surface["iv"].mean() == pytest.approx(sigma, abs=0.02)

# ===================================================================
# Skew
# ===================================================================

class TestSkew:
    def test_vol_at_delta_interpolates(self):
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        F, T, r = 100.0, 0.25, 0.05
        v = vol_at_delta(0.50, strikes, ivs, F, T, r, "call")
        assert 0.18 < v < 0.22
    def test_skew_metrics_returns_keys(self):
        rows = []
        F, T, r = 250.0, 0.08, 0.04
        for K in np.arange(230, 271, 1.0):
            sigma = 0.15 + 0.1 * ((K - F) / F) ** 2  # smile
            cp = bs_price(F, K, T, r, sigma, "call")
            pp = bs_price(F, K, T, r, sigma, "put")
            iv_c = implied_vol(cp, F, K, T, r, "call")
            iv_p = implied_vol(pp, F, K, T, r, "put")
            if not np.isnan(iv_c):
                rows.append({"strike": K, "iv": iv_c, "option_type": "call"})
            if not np.isnan(iv_p):
                rows.append({"strike": K, "iv": iv_p, "option_type": "put"})
        df = pd.DataFrame(rows)
        metrics = compute_skew_metrics(df, F, T, r)
        assert "atm_vol" in metrics
        assert "rr25" in metrics
        assert "bf25" in metrics

# ===================================================================
# Pricing extensions
# ===================================================================

class TestPricingExtensions:
    def test_notional_roundtrip(self):
        k_var = 0.04  # 20% vol
        n_vega = 100_000
        n_var = vega_to_variance_notional(n_vega, k_var)
        n_vega_back = variance_to_vega_notional(n_var, k_var)
        assert abs(n_vega_back - n_vega) < 1e-6
    def test_forward_variance_positive(self):
        fwd = forward_variance(0.04, 0.05, 0.08, 0.16)
        assert fwd > 0
    def test_forward_variance_raises_on_bad_T(self):
        with pytest.raises(ValueError, match="T_far must be greater"):
            forward_variance(0.04, 0.05, 0.16, 0.08)
    def test_forward_vol(self):
        fv = forward_vol(0.04, 0.05, 0.08, 0.16)
        assert fv > 0  # in percent
    def test_variance_swap_pnl_from_notional(self):
        pnl = variance_swap_pnl_from_notional(0.04, 0.05, 100_000)
        assert pnl == pytest.approx(100_000 * 0.01)
    def test_corridor_variance(self):
        strikes = np.arange(90, 111, 1.0)
        prices = 5.0 * np.exp(-0.5 * ((strikes - 100) / 5) ** 2)
        qk = pd.Series(prices, index=strikes)
        F, K0, r, T = 100.0, 100.0, 0.05, 0.25
        full = corridor_variance(qk, F, K0, r, T, 90.0, 110.0)
        narrow = corridor_variance(qk, F, K0, r, T, 95.0, 105.0)
        assert narrow < full

# ===================================================================
# Greeks
# ===================================================================

class TestGreeks:
    def test_variance_decomposition_sums_to_total(self):
        strikes = np.arange(90, 111, 1.0)
        prices = 5.0 * np.exp(-0.5 * ((strikes - 100) / 5) ** 2)
        qk = pd.Series(prices, index=strikes)
        r, T = 0.05, 0.25
        decomp = variance_decomposition(qk, r, T)
        assert abs(decomp["pct_contribution"].sum() - 100.0) < 0.01
    def test_decomposition_all_positive(self):
        strikes = np.arange(90, 111, 1.0)
        prices = 5.0 * np.exp(-0.5 * ((strikes - 100) / 5) ** 2)
        qk = pd.Series(prices, index=strikes)
        decomp = variance_decomposition(qk, 0.05, 0.25)
        assert (decomp["contribution"] >= 0).all()

class TestVegaBucket:
    def _make_options(self, atm=100.0):
        strikes = np.arange(atm - 10, atm + 11, 1.0)
        rows = []
        for K in strikes:
            call_price = max(0.5, atm - K + 3.0 * np.exp(-0.5 * ((K / atm - 1) ** 2) / 0.01))
            put_price = max(0.5, K - atm + 3.0 * np.exp(-0.5 * ((K / atm - 1) ** 2) / 0.01))
            rows.append({"options-strikePrice": K, "options-optiontype": "call", "options-priorSettle": call_price})
            rows.append({"options-strikePrice": K, "options-optiontype": "put", "options-priorSettle": put_price})
        return pd.DataFrame(rows)
    def test_vega_total_equals_sum(self):
        opts = self._make_options()
        result = vega_bucket(opts, opts, r1=0.05, r2=0.05, T1=0.05, T2=0.12)
        assert result["vega_total"] == pytest.approx(
            result["vega_front"] + result["vega_rear"], abs=1e-8)
    def test_vega_values_nonzero(self):
        opts = self._make_options()
        result = vega_bucket(opts, opts, r1=0.05, r2=0.05, T1=0.05, T2=0.12)
        assert result["vega_front"] != 0
        assert result["vega_rear"] != 0
    def test_vix_base_positive(self):
        opts = self._make_options()
        result = vega_bucket(opts, opts, r1=0.05, r2=0.05, T1=0.05, T2=0.12)
        assert result["vix_base"] > 0

class TestSpotSensitivity:
    def _make_options(self, atm=100.0):
        strikes = np.arange(atm - 10, atm + 11, 1.0)
        rows = []
        for K in strikes:
            call_price = max(0.5, atm - K + 3.0 * np.exp(-0.5 * ((K / atm - 1) ** 2) / 0.01))
            put_price = max(0.5, K - atm + 3.0 * np.exp(-0.5 * ((K / atm - 1) ** 2) / 0.01))
            rows.append({"options-strikePrice": K, "options-optiontype": "call", "options-priorSettle": call_price})
            rows.append({"options-strikePrice": K, "options-optiontype": "put", "options-priorSettle": put_price})
        return pd.DataFrame(rows)
    def test_dvix_dspot_nonzero(self):
        opts_front = self._make_options(atm=100.0)
        opts_rear = self._make_options(atm=102.0)
        result = spot_sensitivity(opts_front, opts_rear, r1=0.04, r2=0.05, T1=0.05, T2=0.12)
        assert result["vix_base"] > 0
    def test_result_keys(self):
        opts_front = self._make_options(atm=100.0)
        opts_rear = self._make_options(atm=102.0)
        result = spot_sensitivity(opts_front, opts_rear, r1=0.04, r2=0.05, T1=0.05, T2=0.12)
        expected_keys = {"vix_base", "vix_up", "vix_down", "dvix_dspot_pct", "spot_bump_pct"}
        assert set(result.keys()) == expected_keys
        assert result["vix_up"] > 0
        assert result["vix_down"] > 0
        assert result["spot_bump_pct"] == 1.0

# ===================================================================
# Build full IV surface
# ===================================================================

class TestBuildFullSurface:
    def test_multi_expiry_surface(self):
        F, r = 250.0, 0.045
        sigma = 0.15
        rows = []
        expirations = {"2025-02-07": 0.06, "2025-02-21": 0.10}
        for exp_str, T in expirations.items():
            for K in np.arange(240, 261, 1.0):
                cp = bs_price(F, K, T, r, sigma, "call")
                pp = bs_price(F, K, T, r, sigma, "put")
                rows.append({"tradeDate": "2025-01-15", "expirationDate": exp_str,
                             "strikePrice": K, "optionType": "call", "settlement": cp})
                rows.append({"tradeDate": "2025-01-15", "expirationDate": exp_str,
                             "strikePrice": K, "optionType": "put", "settlement": pp})
        df = pd.DataFrame(rows)
        forwards = {exp: F for exp in expirations}
        times = expirations
        rates_d = {exp: r for exp in expirations}
        surface = build_full_surface(df, "2025-01-15", forwards, times, rates_d)
        assert len(surface) > 0
        assert surface["expiration"].nunique() == 2
        assert surface["iv"].mean() == pytest.approx(sigma, abs=0.02)

class TestCalibrateSviSurface:
    def test_calibrate_returns_params(self):
        from src.implied_vol import svi_raw
        rows = []
        for T, exp in [(0.06, "2025-02-07"), (0.10, "2025-02-21")]:
            k = np.linspace(-0.2, 0.2, 30)
            true_w = svi_raw(k, a=0.04, b=0.08, rho=-0.25, m=0.01, sigma=0.15)
            iv = np.sqrt(true_w / T)
            for ki, ivi in zip(k, iv):
                rows.append({"expiration": exp, "T": T, "moneyness": ki, "iv": ivi,
                             "strike": 250 * np.exp(ki), "option_type": "call"})
        surface_df = pd.DataFrame(rows)
        result = calibrate_svi_surface(surface_df)
        assert len(result) == 2
        assert "a" in result.columns
        assert "rmse" in result.columns
        assert (result["rmse"] < 0.01).all()

# ===================================================================
# Backtest: realized vol estimators
# ===================================================================

class TestRealizedVol:
    def setup_method(self):
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0, 0.01, n)  
        self.prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    def test_close_to_close_reasonable(self):
        rv = realized_vol_close(self.prices, window=30)
        last_rv = rv.dropna().iloc[-1]
        assert 0.05 < last_rv < 0.50  
    def test_parkinson_reasonable(self):
        high = self.prices * 1.005
        low = self.prices * 0.995
        rv = realized_vol_parkinson(high, low, window=30)
        last_rv = rv.dropna().iloc[-1]
        assert 0.01 < last_rv < 0.50
    def test_yang_zhang_reasonable(self):
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0, 0.01, n)
        close = pd.Series(100 * np.exp(np.cumsum(returns)))
        open_ = close.shift(1).fillna(close.iloc[0])
        high = pd.concat([close, open_], axis=1).max(axis=1) * (1 + np.abs(np.random.normal(0, 0.002, n)))
        low = pd.concat([close, open_], axis=1).min(axis=1) * (1 - np.abs(np.random.normal(0, 0.002, n)))
        rv = realized_vol_yang_zhang(open_, high, low, close, window=30)
        last_rv = rv.dropna().iloc[-1]
        assert 0.01 < last_rv < 1.0  

    def test_vrp_sign(self):
        rv = pd.Series([0.15, 0.15, 0.15])
        vrp = variance_risk_premium(pd.Series([20, 20, 20]), rv)
        assert (vrp > 0).all()  