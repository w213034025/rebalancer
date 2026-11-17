"""System tests: whole flow, stubbed network and broker."""
import json
import sqlite3
import sys

import pytest

import config
import rebalance_engine as eng
from broker_alpaca import AlpacaBroker
from tests.conftest import FakeClient, make_prices, narrate


def _stub_world(monkeypatch, client):
    prices = make_prices()
    monkeypatch.setattr(eng, "get_price_history", lambda t, n, end_date=None: prices[t].tail(n))
    last = prices.iloc[-1].to_dict()
    monkeypatch.setattr(eng, "get_latest_prices", lambda t: {k: last[k] for k in t})

    def fake_broker():
        b = AlpacaBroker.__new__(AlpacaBroker)      # skip key check / real client
        b.client = client
        return b
    monkeypatch.setattr(eng, "AlpacaBroker", fake_broker)


def _rows(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT trades, executed FROM rebalance_events").fetchall()


def test_full_cycle_dry_run_submits_nothing(monkeypatch, tmp_db):
    client = FakeClient(cash=10_000.0)               # 100% cash -> big drift -> trades
    _stub_world(monkeypatch, client)
    out = eng.run_rebalance_cycle(dry_run=True)
    tw = {k: f"{v:.1%}" for k, v in out["target_weights"].items()}
    narrate(f"Account is 100% cash. Optimizer targets: {tw}")
    narrate(f"Max drift {max(abs(v) for k, v in out['drift'].items() if k != 'CASH'):.1%} "
            f"vs {config.DRIFT_THRESHOLD:.0%} threshold -> rebalance triggered, {len(out['trades'])} orders built")
    assert out["executed"] is False
    assert len(out["trades"]) == len(config.TICKERS)
    assert client.orders == []                       # nothing sent to the broker
    narrate(f"DRY RUN: orders sent to broker = {len(client.orders)}")
    rows = _rows(tmp_db)
    narrate(f"Audit log rows written = {len(rows)}")
    assert len(rows) == 1 and rows[0][1] == 0


def test_full_cycle_live_path_logs_uuid_order_ids(monkeypatch, tmp_db):
    client = FakeClient(cash=10_000.0)
    _stub_world(monkeypatch, client)
    out = eng.run_rebalance_cycle(dry_run=False)      # stub client, so still no real order
    narrate(f"Live path against a FAKE broker: {len(client.orders)} orders accepted (no network involved)")
    assert out["executed"] is True and len(client.orders) == len(config.TICKERS)
    trades = json.loads(_rows(tmp_db)[0][0])
    narrate("Broker returned UUID order ids; audit log stored them as text (earlier bug, now fixed)")
    assert all(isinstance(t["order_id"], str) for t in trades)   # UUID bug regression


def test_no_trade_when_already_on_target(monkeypatch, tmp_db):
    client = FakeClient(cash=10_000.0)
    _stub_world(monkeypatch, client)
    first = eng.run_rebalance_cycle(dry_run=True)
    positions = [type("P", (), {"symbol": t, "qty": "1",
                                "market_value": str(first["target"] if False else w * 10_000.0)})()
                 for t, w in first["target_weights"].items() if t != "CASH"]
    client2 = FakeClient(positions=positions, cash=first["target_weights"]["CASH"] * 10_000.0)
    _stub_world(monkeypatch, client2)
    second = eng.run_rebalance_cycle(dry_run=True)
    narrate("Holdings already at target -> drift under threshold -> no trades, no orders")
    assert second["trades"] == [] and client2.orders == []


def test_dashboard_smoke(tmp_db):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard.py")
    at = AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    narrate("Dashboard loaded headlessly and Calculate clicked without errors")
    for page in ["Backtest & Model", "Run History", "About & Methodology"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_replay_page_steps_forward(tmp_db):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard.py")
    at = AppTest.from_file(path, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Backtest Replay").run()
    assert not at.exception
    assert at.session_state["cursor"] == 0
    [b for b in at.button if b.label.startswith("Next")][0].click().run()
    assert not at.exception and at.session_state["cursor"] == 5      # one week = 5 trading days
    [r for r in at.radio if r.label == "Step"][0].set_value("Month").run()
    [b for b in at.button if b.label.startswith("Next")][0].click().run()
    assert at.session_state["cursor"] == 26                          # +21
    [b for b in at.button if b.label.startswith("Reset")][0].click().run()
    assert at.session_state["cursor"] == 0
    narrate("Replay page: Next week = +5 trading days, Next month = +21, Reset returns to the start")
