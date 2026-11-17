"""Unit tests: one function at a time."""
import json
import uuid

import pytest

import config
from optimizer import apply_cash_buffer, compute_target_weights
from portfolio_state import (Position, build_trade_list, compute_current_weights,
                             compute_drift, needs_rebalance)
from rebalance_engine import _dumps
from tests.conftest import narrate


def _targets(prices, **kw):
    return compute_target_weights(prices, config.MIN_WEIGHT, config.MAX_WEIGHT,
                                  config.RISK_FREE_RATE, **kw)


def test_weights_respect_floor_cap_and_sum_to_one(risky_prices):
    w = _targets(risky_prices)
    narrate("Optimizer weights: " + ", ".join(f"{k} {v:.1%}" for k, v in w.items()))
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    for t, v in w.items():
        assert config.MIN_WEIGHT - 1e-6 <= v <= config.MAX_WEIGHT + 1e-6, t


def test_cash_buffer_sums_to_one_and_scales(risky_prices):
    out = apply_cash_buffer(_targets(risky_prices), config.CASH_BUFFER)
    narrate(f"After 7% buffer: cash {out['CASH']:.1%}; smallest position {min(v for k, v in out.items() if k != 'CASH'):.2%}")
    assert out["CASH"] == config.CASH_BUFFER
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-3)
    # floor is scaled by (1 - buffer): 5% * 0.93 = 4.65%
    assert min(v for k, v in out.items() if k != "CASH") >= config.MIN_WEIGHT * (1 - config.CASH_BUFFER) - 1e-6


def test_mu_override_missing_ticker_raises(risky_prices):
    import pandas as pd
    bad = pd.Series({"NVDA": 0.1, "AAPL": 0.1})
    with pytest.raises(ValueError, match="missing values"):
        _targets(risky_prices, mu_override=bad)


def test_drift_gating():
    tgt = {"NVDA": 0.30, "AAPL": 0.30, "GOOGL": 0.30, "CASH": 0.10}
    small = {"NVDA": 0.32, "AAPL": 0.29, "GOOGL": 0.29, "CASH": 0.10}
    big = {"NVDA": 0.36, "AAPL": 0.27, "GOOGL": 0.27, "CASH": 0.10}
    assert not needs_rebalance(compute_drift(small, tgt), config.DRIFT_THRESHOLD)
    assert needs_rebalance(compute_drift(big, tgt), config.DRIFT_THRESHOLD)


def test_drift_ignores_cash():
    assert not needs_rebalance({"NVDA": 0.0, "CASH": 0.5}, 0.04)


def test_current_weights_include_cash():
    w = compute_current_weights([Position("NVDA", 1, 900.0)], 100.0)
    assert w["NVDA"] == pytest.approx(0.9) and w["CASH"] == pytest.approx(0.1)


def test_zero_value_portfolio_rejected():
    with pytest.raises(ValueError):
        compute_current_weights([], 0.0)


def test_trade_list_direction_and_dust_filter():
    pos = [Position("NVDA", 10, 1000.0), Position("AAPL", 5, 1000.0), Position("GOOGL", 4, 1000.0)]
    target = {"NVDA": 0.40, "AAPL": 1 / 3, "GOOGL": 0.2666667, "CASH": 0.0}   # AAPL delta ~ $0 -> dust
    trades = build_trade_list(pos, 0.0, target, {"NVDA": 100.0, "AAPL": 200.0, "GOOGL": 250.0})
    assert {t.ticker: t.side for t in trades} == {"NVDA": "buy", "GOOGL": "sell"}


def test_trade_list_refuses_missing_price():
    with pytest.raises(ValueError, match="Missing/invalid"):
        build_trade_list([], 1000.0, {"NVDA": 1.0}, {})


def test_audit_json_handles_uuid():                        # regression: UUID crashed the audit log
    out = json.loads(_dumps({"order_id": uuid.uuid4(), "n": 1}))
    assert isinstance(out["order_id"], str)
