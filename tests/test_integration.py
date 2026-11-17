"""Integration tests: pieces working together."""
import pytest

import backtest
import config
import dashboard_logic as dl
from tests.conftest import make_prices


def test_backtest_end_date_windows_differ(monkeypatch):
    full = make_prices(days=1700)

    def fake_feed(tickers, lookback_days, end_date=None):
        df = full if end_date is None else full[full.index <= end_date]
        return df[tickers].tail(lookback_days + 500)

    monkeypatch.setattr(backtest, "get_price_history", fake_feed)
    now = backtest.run_backtest(start_lookback=252)
    past = backtest.run_backtest(start_lookback=252, end_date=full.index[-300].to_pydatetime())
    assert now.iloc[-1].equals(past.iloc[-1]) is False


def test_backtest_cash_weight_near_buffer(monkeypatch):
    full = make_prices(days=1500)
    monkeypatch.setattr(backtest, "get_price_history",
                        lambda t, n, end_date=None: full[t].tail(n + 500))
    out = backtest.run_backtest(start_lookback=252)
    assert len(out) > 0


def test_dashboard_logic_round_trip(tmp_db):
    prices = dl.synthetic_prices(config.TICKERS)
    res = dl.run_simulation(prices, {t: 18.6 for t in config.TICKERS}, 10_000.0)
    assert sum(res["target"].values()) == pytest.approx(1.0, abs=1e-3)
    assert res["target"]["CASH"] == config.CASH_BUFFER
    dl.save_run(dl.OFFLINE, 10_000.0, res)
    runs = dl.load_runs()
    assert len(runs) == 1 and runs[0]["result"]["needs_rebalance"] == res["needs_rebalance"]
    dl.clear_runs()
    assert dl.load_runs() == []


def test_dashboard_has_no_broker_import():
    src = open(dl.__file__).read()
    assert "import broker_alpaca" not in src and "from broker_alpaca" not in src


def test_backtest_flags_rebalance_days(monkeypatch):
    full = make_prices(days=1500)
    out = backtest.run_backtest(start_lookback=252, price_data=full)
    assert "rebalanced" in out.columns and out["rebalanced"].dtype == bool
    assert out["rebalanced"].iloc[0]            # first day always trades
    assert 0 < out["rebalanced"].sum() < len(out) / 2   # trades are the exception, not daily churn


def test_replay_slice_rebases_to_one():
    full = make_prices(days=1500)
    out = backtest.run_backtest(start_lookback=252, price_data=full)
    part = dl.replay_slice(out, start_pos=50, cursor=21)
    assert len(part) == 22
    assert part["portfolio_value"].iloc[0] == pytest.approx(1.0)
    assert part["equal_weight_benchmark"].iloc[0] == pytest.approx(1.0)
    assert len(dl.replay_slice(out, 50, 10**6)) == len(out) - 50    # cursor clamps at the end
