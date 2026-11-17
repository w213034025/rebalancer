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
    """What: Runs the optimizer on fake prices and checks every weight is between 5% and 40% and they total 100%.
    Why: These limits are the diversification promise. The optimizer is not allowed to zero out a stock or go all-in on one.
    """
    w = _targets(risky_prices)
    narrate("Optimizer weights: " + ", ".join(f"{k} {v:.1%}" for k, v in w.items()))
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    for t, v in w.items():
        assert config.MIN_WEIGHT - 1e-6 <= v <= config.MAX_WEIGHT + 1e-6, t


def test_cash_buffer_sums_to_one_and_scales(risky_prices):
    """What: Applies the 7% cash buffer and checks cash is exactly 7%, everything still totals 100%, and the smallest position scaled to 4.65%.
    Why: Shows how the buffer and the floor interact: 5% of the 93% that is invested is 4.65% of the whole portfolio.
    """
    out = apply_cash_buffer(_targets(risky_prices), config.CASH_BUFFER)
    narrate(f"After 7% buffer: cash {out['CASH']:.1%}; smallest position {min(v for k, v in out.items() if k != 'CASH'):.2%}")
    assert out["CASH"] == config.CASH_BUFFER
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-3)
    # floor is scaled by (1 - buffer): 5% * 0.93 = 4.65%
    assert min(v for k, v in out.items() if k != "CASH") >= config.MIN_WEIGHT * (1 - config.CASH_BUFFER) - 1e-6


def test_mu_override_missing_ticker_raises(risky_prices):
    """What: Passes the optimizer expected returns that are missing most tickers and expects a clear error.
    Why: Missing inputs must fail loudly. A silent fallback could produce plausible-looking but wrong weights.
    """
    import pandas as pd
    bad = pd.Series({"NVDA": 0.1, "AAPL": 0.1})
    with pytest.raises(ValueError, match="missing values"):
        _targets(risky_prices, mu_override=bad)


def test_drift_gating():
    """What: Feeds the drift rule one case under 4% and one over, and checks only the second triggers a rebalance.
    Why: This is the core trading decision. It keeps the engine from trading on noise while still reacting to real drift.
    """
    tgt = {"NVDA": 0.30, "AAPL": 0.30, "GOOGL": 0.30, "CASH": 0.10}
    small = {"NVDA": 0.32, "AAPL": 0.29, "GOOGL": 0.29, "CASH": 0.10}
    big = {"NVDA": 0.36, "AAPL": 0.27, "GOOGL": 0.27, "CASH": 0.10}
    assert not needs_rebalance(compute_drift(small, tgt), config.DRIFT_THRESHOLD)
    assert needs_rebalance(compute_drift(big, tgt), config.DRIFT_THRESHOLD)


def test_drift_ignores_cash():
    """What: Gives the drift rule a portfolio that is far off only in cash and checks it does not trigger.
    Why: Cash moves as a side effect of price changes. Only stock positions drifting should cause trades.
    """
    assert not needs_rebalance({"NVDA": 0.0, "CASH": 0.5}, 0.04)


def test_current_weights_include_cash():
    """What: Computes weights for a $900 stock position and $100 cash and checks they come out 90% / 10%.
    Why: Everything downstream depends on correct current weights, including the cash line.
    """
    w = compute_current_weights([Position("NVDA", 1, 900.0)], 100.0)
    assert w["NVDA"] == pytest.approx(0.9) and w["CASH"] == pytest.approx(0.1)


def test_zero_value_portfolio_rejected():
    """What: Gives the weight calculation an empty, zero-value portfolio and expects an error.
    Why: Dividing by zero would produce nonsense weights. Refusing is safer than guessing.
    """
    with pytest.raises(ValueError):
        compute_current_weights([], 0.0)


def test_trade_list_direction_and_dust_filter():
    """What: Builds trades for three holdings: one underweight, one overweight, one essentially on target.
    Why: Checks buys and sells go the right way, and that trivial trades under $50 are skipped so we don't churn on dust.
    """
    pos = [Position("NVDA", 10, 1000.0), Position("AAPL", 5, 1000.0), Position("GOOGL", 4, 1000.0)]
    target = {"NVDA": 0.40, "AAPL": 1 / 3, "GOOGL": 0.2666667, "CASH": 0.0}   # AAPL delta ~ $0 -> dust
    trades = build_trade_list(pos, 0.0, target, {"NVDA": 100.0, "AAPL": 200.0, "GOOGL": 250.0})
    assert {t.ticker: t.side for t in trades} == {"NVDA": "buy", "GOOGL": "sell"}


def test_trade_list_refuses_missing_price():
    """What: Asks for a trade list with no price for the stock and expects the engine to refuse.
    Why: The system fails closed: it will not size an order from a missing price, rather than trade blind.
    """
    with pytest.raises(ValueError, match="Missing/invalid"):
        build_trade_list([], 1000.0, {"NVDA": 1.0}, {})


def test_audit_json_handles_uuid():                        # regression: UUID crashed the audit log
    """What: Writes a record containing a UUID through the audit logger and checks it saves as text.
    Why: Direct regression test for the audit-log bug that originally crashed on Alpaca order ids.
    """
    out = json.loads(_dumps({"order_id": uuid.uuid4(), "n": 1}))
    assert isinstance(out["order_id"], str)
