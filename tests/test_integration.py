"""Integration tests: pieces working together."""
import pytest

import backtest
import config
import dashboard_logic as dl
from tests.conftest import make_prices


def test_backtest_end_date_windows_differ(monkeypatch):
    """What: Runs the backtest twice on fake prices: once as of today, once as of ~300 days ago, and compares the results.
    Why: Our six-window robustness check only means something if each window really uses different data. This guards against the earlier bug where windows silently matched.
    """
    full = make_prices(days=1700)

    def fake_feed(tickers, lookback_days, end_date=None):
        df = full if end_date is None else full[full.index <= end_date]
        return df[tickers].tail(lookback_days + 500)

    monkeypatch.setattr(backtest, "get_price_history", fake_feed)
    now = backtest.run_backtest(start_lookback=252)
    past = backtest.run_backtest(start_lookback=252, end_date=full.index[-300].to_pydatetime())
    assert now.iloc[-1].equals(past.iloc[-1]) is False


def test_backtest_cash_weight_near_buffer(monkeypatch):
    """What: Runs a backtest and checks the average cash weight stays near the 7% target.
    Why: The buffer must survive the whole simulation, not just the first day. An earlier bug let drift quietly erode it.
    """
    full = make_prices(days=1500)
    monkeypatch.setattr(backtest, "get_price_history",
                        lambda t, n, end_date=None: full[t].tail(n + 500))
    out = backtest.run_backtest(start_lookback=252)
    assert len(out) > 0
    assert abs(out["CASH"].mean() - config.CASH_BUFFER) < 0.02


def test_dashboard_logic_round_trip(tmp_db):
    """What: Simulates holdings in the dashboard adapter, saves the run to SQLite, reloads it, then clears it.
    Why: Proves the dashboard's save/load/clear path works end to end and that targets still sum to 100% with a 7% cash line.
    """
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
    """What: Reads the dashboard adapter's source and confirms it never imports the broker module.
    Why: The dashboard is advertised as simulation-only. This turns that promise into a check that fails if someone adds a broker import.
    """
    src = open(dl.__file__).read()
    assert "import broker_alpaca" not in src and "from broker_alpaca" not in src


def test_backtest_flags_rebalance_days(monkeypatch):
    """What: Checks the backtest marks which days it actually traded, that day one trades, and that trades are the exception.
    Why: The Backtest Replay page draws a diamond on each trade day. This keeps those markers correct and confirms the drift rule prevents daily churn.
    """
    full = make_prices(days=1500)
    out = backtest.run_backtest(start_lookback=252, price_data=full)
    assert "rebalanced" in out.columns and out["rebalanced"].dtype == bool
    assert out["rebalanced"].iloc[0]            # first day always trades
    assert 0 < out["rebalanced"].sum() < len(out) / 2   # trades are the exception, not daily churn


def test_replay_slice_rebases_to_one():
    """What: Slices a backtest for the replay page and checks strategy and benchmark both start at 1.0 and the slice stops at the data's end.
    Why: Replay compares two lines from a shared starting point. Without rebasing, a later start date would make the comparison misleading.
    """
    full = make_prices(days=1500)
    out = backtest.run_backtest(start_lookback=252, price_data=full)
    part = dl.replay_slice(out, start_pos=50, cursor=21)
    assert len(part) == 22
    assert part["portfolio_value"].iloc[0] == pytest.approx(1.0)
    assert part["equal_weight_benchmark"].iloc[0] == pytest.approx(1.0)
    assert len(dl.replay_slice(out, 50, 10**6)) == len(out) - 50    # cursor clamps at the end
