"""
Unit tests for VIX-style volatility index calculator.

Tests use synthetic data to verify CBOE methodology compliance.
"""

import numpy as np
import pandas as pd
import pytest

from src.vix_index import (
    build_qk,
    compute_delta_k,
    compute_forward_and_k0,
    compute_sigma2,
    fill_cmt_missing_dates,
    interpolate_to_target,
)


# -----------------------------------------------------------------------------
# Fixtures: synthetic data
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_options():
    """Create a synthetic options chain for testing."""
    strikes = [90, 95, 100, 105, 110, 115, 120]
    data = []

    for K in strikes:
        # Simple synthetic prices: OTM options have lower prices
        if K < 100:
            put_price = (100 - K) * 0.3 + 0.5
            call_price = 0.2
        elif K > 100:
            call_price = (K - 100) * 0.25 + 0.4
            put_price = 0.15
        else:
            call_price = 2.5
            put_price = 2.5

        data.append({
            "options-strikePrice": K,
            "options-optiontype": "call",
            "options-priorSettle": call_price,
        })
        data.append({
            "options-strikePrice": K,
            "options-optiontype": "put",
            "options-priorSettle": put_price,
        })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_options_with_zeros():
    """Options chain with zero prices to test exclusion logic."""
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    data = []

    for K in strikes:
        if K < 100:
            # Put prices, with zeros for far OTM
            put_price = 0 if K <= 85 else (100 - K) * 0.3
            call_price = 0.1
        elif K > 100:
            # Call prices, with zeros for far OTM
            call_price = 0 if K >= 115 else (K - 100) * 0.25
            put_price = 0.05
        else:
            call_price = 2.0
            put_price = 2.0

        data.append({
            "options-strikePrice": K,
            "options-optiontype": "call",
            "options-priorSettle": call_price,
        })
        data.append({
            "options-strikePrice": K,
            "options-optiontype": "put",
            "options-priorSettle": put_price,
        })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_cmt_rates():
    """Create synthetic CMT rate data with gaps."""
    dates = pd.date_range("2020-11-09", "2020-11-13", freq="D")
    # Skip weekend (Nov 14-15 would be Sat-Sun if we extended)
    maturities = ["1 Mo", "2 Mo", "3 Mo"]

    data = []
    for date in dates:
        # Skip Nov 11 (Veterans Day holiday)
        if date.day == 11:
            continue
        for mat in maturities:
            data.append({
                "Date": date,
                "maturity": mat,
                "value": 0.1 + (int(mat[0]) * 0.02),  # Simple rate curve
            })

    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Tests: compute_delta_k
# -----------------------------------------------------------------------------

class TestComputeDeltaK:
    """Tests for strike interval calculation."""

    def test_interior_strikes(self):
        """Interior ΔK = (K_{i+1} - K_{i-1}) / 2."""
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        delta_k = compute_delta_k(strikes)

        # Interior elements
        assert delta_k[1] == pytest.approx((100 - 90) / 2)  # 5.0
        assert delta_k[2] == pytest.approx((105 - 95) / 2)  # 5.0
        assert delta_k[3] == pytest.approx((110 - 100) / 2)  # 5.0

    def test_boundary_strikes(self):
        """First and last ΔK use adjacent strikes."""
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        delta_k = compute_delta_k(strikes)

        # First: K_2 - K_1
        assert delta_k[0] == pytest.approx(95 - 90)  # 5.0
        # Last: K_n - K_{n-1}
        assert delta_k[4] == pytest.approx(110 - 105)  # 5.0

    def test_non_uniform_strikes(self):
        """ΔK with non-uniform strike spacing."""
        strikes = np.array([90.0, 92.5, 100.0, 110.0])
        delta_k = compute_delta_k(strikes)

        assert delta_k[0] == pytest.approx(2.5)  # 92.5 - 90
        assert delta_k[1] == pytest.approx((100 - 90) / 2)  # 5.0
        assert delta_k[2] == pytest.approx((110 - 92.5) / 2)  # 8.75
        assert delta_k[3] == pytest.approx(110 - 100)  # 10.0

    def test_single_strike(self):
        """Single strike returns zero delta."""
        strikes = np.array([100.0])
        delta_k = compute_delta_k(strikes)
        assert len(delta_k) == 1
        assert delta_k[0] == 0.0

    def test_empty_strikes(self):
        """Empty array returns empty array."""
        strikes = np.array([])
        delta_k = compute_delta_k(strikes)
        assert len(delta_k) == 0


# -----------------------------------------------------------------------------
# Tests: compute_forward_and_k0
# -----------------------------------------------------------------------------

class TestComputeForwardAndK0:
    """Tests for forward price and ATM strike calculation."""

    def test_forward_greater_than_k0(self, synthetic_options):
        """K0 must be <= F (CBOE definition)."""
        F, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        assert K0 <= F

    def test_k0_is_valid_strike(self, synthetic_options):
        """K0 must be an actual strike price."""
        F, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        strikes = synthetic_options["options-strikePrice"].unique()
        assert K0 in strikes

    def test_forward_calculation(self, synthetic_options):
        """F = K* + exp(rT) * (C - P) at the min |C-P| strike."""
        r, T = 0.01, 0.1
        F, K0 = compute_forward_and_k0(synthetic_options, r=r, T=T)

        # At K=100, C=P=2.5, so F should be close to 100
        # F = 100 + exp(0.001) * (2.5 - 2.5) = 100
        assert F == pytest.approx(100.0, rel=0.01)

    def test_raises_on_missing_calls_or_puts(self):
        """Should raise if missing call or put data."""
        calls_only = pd.DataFrame({
            "options-strikePrice": [100, 105],
            "options-optiontype": ["call", "call"],
            "options-priorSettle": [2.0, 1.5],
        })
        with pytest.raises(ValueError, match="must contain both"):
            compute_forward_and_k0(calls_only, r=0.01, T=0.1)


# -----------------------------------------------------------------------------
# Tests: build_qk
# -----------------------------------------------------------------------------

class TestBuildQK:
    """Tests for Q(K) construction."""

    def test_k0_included_once(self, synthetic_options):
        """K0 should appear exactly once in Q(K)."""
        _, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        qk = build_qk(synthetic_options, K0)

        # K0 should be in the index
        assert K0 in qk.index

        # Count occurrences of K0
        k0_count = (qk.index == K0).sum()
        assert k0_count == 1

    def test_k0_is_average(self, synthetic_options):
        """Q(K0) = (Call + Put) / 2."""
        _, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        qk = build_qk(synthetic_options, K0)

        # Get original call and put at K0
        pivoted = synthetic_options.pivot(
            index="options-strikePrice",
            columns="options-optiontype",
            values="options-priorSettle"
        )
        expected = (pivoted.loc[K0, "call"] + pivoted.loc[K0, "put"]) / 2
        assert qk[K0] == pytest.approx(expected)

    def test_puts_below_k0_calls_above(self, synthetic_options):
        """K < K0 uses puts, K > K0 uses calls."""
        _, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        qk = build_qk(synthetic_options, K0)

        pivoted = synthetic_options.pivot(
            index="options-strikePrice",
            columns="options-optiontype",
            values="options-priorSettle"
        )

        for K in qk.index:
            if K < K0:
                # Should be put price
                assert qk[K] == pytest.approx(pivoted.loc[K, "put"])
            elif K > K0:
                # Should be call price
                assert qk[K] == pytest.approx(pivoted.loc[K, "call"])

    def test_exclusion_after_consecutive_zeros(self, synthetic_options_with_zeros):
        """Strikes are excluded after N consecutive zeros."""
        K0 = 100.0
        qk = build_qk(synthetic_options_with_zeros, K0, consecutive_zeros=2)

        # K=80, 85 have zero put prices -> excluded
        assert 80 not in qk.index
        assert 85 not in qk.index

        # K=115, 120 have zero call prices -> excluded
        assert 115 not in qk.index
        assert 120 not in qk.index


# -----------------------------------------------------------------------------
# Tests: compute_sigma2
# -----------------------------------------------------------------------------

class TestComputeSigma2:
    """Tests for variance calculation."""

    def test_sigma2_positive(self, synthetic_options):
        """σ² should be positive for reasonable inputs."""
        F, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        qk = build_qk(synthetic_options, K0)
        sigma2 = compute_sigma2(F, K0, qk, r=0.01, T=0.1)

        assert sigma2 > 0

    def test_sigma2_reasonable_range(self, synthetic_options):
        """σ² should be in a reasonable range (0 < σ² < 10)."""
        F, K0 = compute_forward_and_k0(synthetic_options, r=0.01, T=0.1)
        qk = build_qk(synthetic_options, K0)
        sigma2 = compute_sigma2(F, K0, qk, r=0.01, T=0.1)

        # Implied vol between 0% and ~316% (sqrt(10))
        assert 0 < sigma2 < 10

    def test_raises_on_insufficient_strikes(self):
        """Should raise if fewer than 2 strikes."""
        qk = pd.Series({100: 2.5})  # Only one strike
        with pytest.raises(ValueError, match="at least 2 strikes"):
            compute_sigma2(F=100, K0=100, qk=qk, r=0.01, T=0.1)


# -----------------------------------------------------------------------------
# Tests: interpolate_to_target
# -----------------------------------------------------------------------------

class TestInterpolateToTarget:
    """Tests for VIX interpolation."""

    def test_vix_positive(self):
        """VIX should always be positive."""
        vix = interpolate_to_target(
            T1=0.05, T2=0.12,
            sigma2_1=0.04, sigma2_2=0.05,
            target_minutes=30 * 24 * 60
        )
        assert vix > 0

    def test_vix_scales_with_volatility(self):
        """Higher σ² should give higher VIX."""
        vix_low = interpolate_to_target(T1=0.05, T2=0.12, sigma2_1=0.01, sigma2_2=0.02)
        vix_high = interpolate_to_target(T1=0.05, T2=0.12, sigma2_1=0.04, sigma2_2=0.05)

        assert vix_high > vix_low


# -----------------------------------------------------------------------------
# Tests: fill_cmt_missing_dates
# -----------------------------------------------------------------------------

class TestFillCmtMissingDates:
    """Tests for rate data gap filling."""

    def test_fills_missing_dates(self, synthetic_cmt_rates):
        """Missing dates should be filled."""
        filled = fill_cmt_missing_dates(synthetic_cmt_rates)

        # Nov 11 was missing, should now be present
        dates = filled["Date"].unique()
        assert pd.Timestamp("2020-11-11") in dates

    def test_forward_fills_values(self, synthetic_cmt_rates):
        """Missing values should be forward-filled."""
        filled = fill_cmt_missing_dates(synthetic_cmt_rates)

        # Get Nov 11 value (forward-filled from Nov 10)
        nov11_1mo = filled[(filled["Date"] == "2020-11-11") & (filled["maturity"] == "1 Mo")]["value"]
        nov10_1mo = filled[(filled["Date"] == "2020-11-10") & (filled["maturity"] == "1 Mo")]["value"]

        assert nov11_1mo.iloc[0] == nov10_1mo.iloc[0]


# -----------------------------------------------------------------------------
# Integration test
# -----------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests with synthetic data."""

    def test_full_calculation_synthetic(self, synthetic_options):
        """Full VIX calculation on synthetic data."""
        # Using two copies with slight time offset
        F1, K0_1 = compute_forward_and_k0(synthetic_options, r=0.001, T=0.08)
        F2, K0_2 = compute_forward_and_k0(synthetic_options, r=0.002, T=0.16)

        qk1 = build_qk(synthetic_options, K0_1)
        qk2 = build_qk(synthetic_options, K0_2)

        sigma2_1 = compute_sigma2(F1, K0_1, qk1, r=0.001, T=0.08)
        sigma2_2 = compute_sigma2(F2, K0_2, qk2, r=0.002, T=0.16)

        vix = interpolate_to_target(T1=0.08, T2=0.16, sigma2_1=sigma2_1, sigma2_2=sigma2_2)

        # Sanity checks
        assert vix > 0
        assert vix < 500  # Not astronomically high
        assert K0_1 <= F1
        assert K0_2 <= F2
