"""Tests for GLD preprocessing and equity VIX calculation."""
import numpy as np
import pandas as pd
import pytest
from src.preprocess_gld import parse_occ_ticker, expiration_str_to_date
from src.vix_index import vix_index_equity, _select_bracketing_expirations

# ---------------------------------------------------------------------------
# OCC ticker parsing
# ---------------------------------------------------------------------------

class TestParseOccTicker:
    def test_call(self):
        result = parse_occ_ticker("O:GLD240221C00176000")
        assert result is not None
        exp, opt_type, strike = result
        assert exp == "240221"
        assert opt_type == "C"
        assert strike == 176.0
    def test_put(self):
        result = parse_occ_ticker("O:GLD250117P00250000")
        assert result is not None
        exp, opt_type, strike = result
        assert exp == "250117"
        assert opt_type == "P"
        assert strike == 250.0
    def test_fractional_strike(self):
        result = parse_occ_ticker("O:GLD240315C00245500")
        assert result is not None
        _, _, strike = result
        assert strike == 245.5
    def test_rejects_rgld(self):
        assert parse_occ_ticker("O:RGLD280121C00290000") is None
    def test_rejects_gldd(self):
        assert parse_occ_ticker("O:GLDD250117C00010000") is None
    def test_rejects_non_option(self):
        assert parse_occ_ticker("GLD") is None
        assert parse_occ_ticker("O:GLD") is None

class TestExpirationStrToDate:
    def test_basic(self):
        assert expiration_str_to_date("240221") == "2024-02-21"
    def test_year_boundary(self):
        assert expiration_str_to_date("251231") == "2025-12-31"

# ---------------------------------------------------------------------------
# Bracketing expiration selection
# ---------------------------------------------------------------------------

class TestSelectBracketingExpirations:
    def setup_method(self):
        self.trade_date = pd.Timestamp("2025-01-15")
        self.expirations = [
            pd.Timestamp("2025-01-24"),
            pd.Timestamp("2025-01-31"),
            pd.Timestamp("2025-02-07"),
            pd.Timestamp("2025-02-14"),
            pd.Timestamp("2025-02-21"),
            pd.Timestamp("2025-03-21"),]
    def test_brackets_30_day_target(self):
        near, nxt = _select_bracketing_expirations(
            self.expirations, self.trade_date, target_days=30)
        assert near <= self.trade_date + pd.Timedelta(days=30)
        assert nxt > self.trade_date + pd.Timedelta(days=30)
    def test_near_before_next(self):
        near, nxt = _select_bracketing_expirations(
            self.expirations, self.trade_date, target_days=30)
        assert near < nxt
    def test_min_dte_filter(self):
        near, nxt = _select_bracketing_expirations(
            self.expirations, self.trade_date, target_days=30, min_dte=20)
        assert (near - self.trade_date).days >= 20
    def test_raises_if_not_enough_expirations(self):
        with pytest.raises(ValueError, match="Need at least 2"):
            _select_bracketing_expirations(
                [pd.Timestamp("2025-01-16")],
                self.trade_date,
                target_days=30,)
    def test_raises_if_cannot_bracket(self):
        short_exp = [pd.Timestamp("2025-01-25"), pd.Timestamp("2025-01-31")]
        with pytest.raises(ValueError, match="Cannot bracket"):
            _select_bracketing_expirations(
                short_exp, self.trade_date, target_days=30)

# ---------------------------------------------------------------------------
# vix_index_equity with synthetic data
# ---------------------------------------------------------------------------

class TestVixIndexEquitySynthetic:
    def _make_synthetic_options(self, trade_date, near_exp, next_exp, atm=250.0):
        rows = []
        strikes = np.arange(atm - 20, atm + 21, 1.0)
        for exp in [near_exp, next_exp]:
            for K in strikes:
                # Rough Black-Scholes-ish prices
                moneyness = K / atm
                call_price = max(0.01, atm - K + 5.0 * np.exp(-0.5 * (moneyness - 1) ** 2 / 0.01))
                put_price = max(0.01, K - atm + 5.0 * np.exp(-0.5 * (moneyness - 1) ** 2 / 0.01))
                rows.append({
                    "tradeDate": trade_date,
                    "expirationDate": exp,
                    "strikePrice": K,
                    "optionType": "call",
                    "settlement": call_price,})
                rows.append({
                    "tradeDate": trade_date,
                    "expirationDate": exp,
                    "strikePrice": K,
                    "optionType": "put",
                    "settlement": put_price,})
        return pd.DataFrame(rows)
    def _make_rates(self, trade_date):
        td = pd.Timestamp(trade_date)
        return pd.DataFrame([
            {"Date": td, "maturity": "1 Mo", "value": 4.5},
            {"Date": td, "maturity": "2 Mo", "value": 4.5},
            {"Date": td, "maturity": "3 Mo", "value": 4.5},
            {"Date": td, "maturity": "6 Mo", "value": 4.5},])
    def _make_calendar(self):
        return pd.DataFrame({"DATE": ["2025-01-01"]})
    def test_vix_positive(self):
        td = "2025-01-15"
        options = self._make_synthetic_options(td, "2025-02-07", "2025-02-21")
        rates = self._make_rates(td)
        calendar = self._make_calendar()
        result = vix_index_equity(options, rates, calendar, trade_date=td, target_days=30)
        assert result["vix"] > 0
    def test_result_keys(self):
        td = "2025-01-15"
        options = self._make_synthetic_options(td, "2025-02-07", "2025-02-21")
        rates = self._make_rates(td)
        calendar = self._make_calendar()
        result = vix_index_equity(options, rates, calendar, trade_date=td, target_days=30)
        expected_keys = {
            "vix", "T1", "T2", "F1", "F2", "K0_1", "K0_2",
            "sigma2_1", "sigma2_2", "r1", "r2",
            "n_strikes_1", "n_strikes_2", "near_exp", "next_exp",}
        assert set(result.keys()) == expected_keys
    def test_raises_on_empty_date(self):
        td = "2025-01-15"
        options = self._make_synthetic_options(td, "2025-02-07", "2025-02-21")
        rates = self._make_rates(td)
        calendar = self._make_calendar()
        with pytest.raises(ValueError, match="No options found"):
            vix_index_equity(options, rates, calendar, trade_date="2099-01-01", target_days=30)