"""Shared offline fixtures: seeded fake prices, temp DB. No network, no broker."""
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

DEMO = os.environ.get("TEST_DEMO") == "1"
if DEMO:                       # keep engine INFO logs off the projector
    import logging
    logging.disable(logging.WARNING)


def narrate(msg: str) -> None:
    """Plain-English line shown only in demo mode (./run.sh test --demo)."""
    if DEMO:
        print(f"\n      ▸ {msg}")


def make_prices(days=1400, seed=11, tickers=None):
    tickers = tickers or config.TICKERS + ["SPY"]
    rng = np.random.default_rng(seed)
    n = len(tickers)
    vol = np.linspace(0.40, 0.18, n) / np.sqrt(252)
    mu = np.linspace(0.40, 0.25, n) / 252
    corr = np.full((n, n), 0.5) + 0.5 * np.eye(n)
    rets = rng.multivariate_normal(mu, corr * np.outer(vol, vol), size=days)
    idx = pd.bdate_range(end="2026-11-13", periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def risky_prices(prices):
    return prices[config.TICKERS].tail(config.LOOKBACK_DAYS)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "test.sqlite3")
    monkeypatch.setattr(config, "LOG_DB_PATH", path)
    return path


class FakeOrder:
    def __init__(self):
        self.id = uuid.uuid4()          # Alpaca returns uuid.UUID, not str


class FakeClient:
    """Stands in for alpaca TradingClient. Records orders; never touches a network."""
    def __init__(self, positions=None, cash=10_000.0):
        self.orders, self._positions, self._cash = [], positions or [], cash

    def get_all_positions(self):
        return self._positions

    def get_account(self):
        return type("A", (), {"cash": str(self._cash)})()

    def submit_order(self, req):
        self.orders.append(req)
        return FakeOrder()


def pytest_runtest_setup(item):
    """Demo mode only: print each test's plain-English What/Why before it runs."""
    if DEMO and item.function.__doc__:
        import inspect
        print()   # start on a fresh line after pytest's test-name prefix
        for line in inspect.cleandoc(item.function.__doc__).splitlines():
            print(f"      {line}")
