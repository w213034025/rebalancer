"""
Interface adapter between the Streamlit dashboard and the Phase 3 engine.

SIMULATION ONLY: nothing here imports broker_alpaca or submits orders.
Reuses config, data_feed, optimizer and portfolio_state unchanged.
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from optimizer import compute_target_weights, apply_cash_buffer
from portfolio_state import (Position, build_trade_list, compute_drift,
                             needs_rebalance)

OFFLINE, YAHOO = "Offline demo (synthetic)", "Yahoo Finance (historical)"

# Cached output of robustness_check.py (run 2026-10-25, 6 "as of" windows).
# Static snapshot so the page works offline; re-run the script to refresh.
ROBUSTNESS = pd.DataFrame({
    "window": ["Today", "4 mo ago", "8 mo ago", "12 mo ago", "16 mo ago", "20 mo ago"],
    "Historical-mean (production)": [1.321, 1.380, 1.480, 2.339, 1.883, 2.372],
    "Ridge ML (experimental)":      [1.780, 1.299, 1.420, 2.356, 1.607, 1.862],
    "Equal-weight":                 [1.573, 1.394, 1.611, 2.194, 1.824, 2.289],
})
GATE = pd.DataFrame({
    "Model": ["Naive historical mean", "Ridge", "LightGBM"],
    "Info. coefficient": [-0.066, 0.023, 0.054],
    "Hit rate": [0.529, 0.526, 0.527],
    "Clipped at rail": ["-", "0%", "59%"],
    "Gate (beat naive on BOTH)": ["baseline", "FAIL (hit rate)", "FAIL (hit rate)"],
})


# ---------- market data ----------
def synthetic_prices(tickers: List[str], days: int = config.LOOKBACK_DAYS, seed: int = 7) -> pd.DataFrame:
    """Seeded correlated geometric random walk. NOT real market history."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    vol = np.linspace(0.45, 0.28, n) / np.sqrt(252)
    drift = (np.linspace(0.22, 0.12, n) / 252) - 0.5 * vol**2  # arithmetic mu minus Ito term
    corr = np.full((n, n), 0.55) + 0.45 * np.eye(n)
    cov = corr * np.outer(vol, vol)
    rets = rng.multivariate_normal(drift, cov, size=days)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


def load_prices(source: str) -> Tuple[pd.DataFrame, str]:
    """Returns (prices, label). Yahoo failures raise — no silent fallback to fake data."""
    if source == OFFLINE:
        return synthetic_prices(config.TICKERS), "Synthetic demonstration data"
    from data_feed import get_price_history
    return get_price_history(config.TICKERS, config.LOOKBACK_DAYS), "Yahoo Finance daily closes"


# ---------- simulation ----------
def run_simulation(prices: pd.DataFrame, holdings_pct: Dict[str, float], total_value: float) -> dict:
    """Optimizer -> cash buffer -> drift -> simulated trades for user-entered holdings."""
    raw = compute_target_weights(prices, config.MIN_WEIGHT, config.MAX_WEIGHT,
                                 config.RISK_FREE_RATE, config.OPTIMIZATION_OBJECTIVE)
    target = apply_cash_buffer(raw, config.CASH_BUFFER)

    last = prices.iloc[-1].to_dict()
    cur = {t: holdings_pct.get(t, 0.0) / 100 for t in config.TICKERS}
    cur["CASH"] = max(0.0, 1 - sum(cur.values()))
    drift = compute_drift(cur, target)
    gated = needs_rebalance(drift, config.DRIFT_THRESHOLD)

    positions = [Position(t, cur[t] * total_value / last[t], cur[t] * total_value)
                 for t in config.TICKERS]
    trades = build_trade_list(positions, cur["CASH"] * total_value, target, last) if gated else []

    # portfolio stats of the optimal (pre-cash) weights
    r = np.log(prices).diff().dropna()
    w = np.array([raw[t] for t in config.TICKERS])
    mu = float(((1 + r.mean()) ** 252 - 1) @ w)
    vol = float(np.sqrt(w @ (r.cov().values * 252) @ w))
    return {
        "target": target, "current": cur, "drift": drift, "needs_rebalance": gated,
        "trades": [t.__dict__ for t in trades], "last_prices": last,
        "exp_return": mu, "volatility": vol,
        "sharpe": (mu - config.RISK_FREE_RATE) / vol if vol else float("nan"),
    }


# ---------- saved runs (SQLite, same DB file as the engine's audit log) ----------
def _conn():
    os.makedirs(os.path.dirname(config.LOG_DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(config.LOG_DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS dashboard_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT,
        portfolio_value REAL, payload TEXT)""")
    return c


def save_run(source: str, value: float, result: dict) -> None:
    with _conn() as c:
        c.execute("INSERT INTO dashboard_runs(ts,source,portfolio_value,payload) VALUES(?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), source, value, json.dumps(result, default=float)))


def clear_runs() -> None:
    with _conn() as c:
        c.execute("DELETE FROM dashboard_runs")


def load_runs() -> List[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id,ts,source,portfolio_value,payload FROM dashboard_runs ORDER BY id DESC").fetchall()
    return [{"id": i, "ts": ts, "source": s, "value": v, "result": json.loads(p)} for i, ts, s, v, p in rows]


def load_audit_events(limit: int = 50) -> pd.DataFrame:
    """Engine's own rebalance_events table (written by main.py / rebalance_engine)."""
    try:
        with sqlite3.connect(config.LOG_DB_PATH) as c:
            raw = pd.read_sql_query(
                "SELECT ts,drift,trades,executed FROM rebalance_events ORDER BY rowid DESC LIMIT ?",
                c, params=(limit,))
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return raw
    def _max_drift(j):
        try:
            d = json.loads(j)
            return max(abs(v) for k, v in d.items() if k != "CASH") * 100
        except Exception:
            return float("nan")
    return pd.DataFrame({
        "Time": raw["ts"].map(fmt_ts),
        "Max drift (pp)": raw["drift"].map(_max_drift).round(2),
        "Trades": raw["trades"].map(lambda j: len(json.loads(j)) if j else 0),
        "Mode": raw["executed"].map(lambda e: "executed" if e else "dry-run"),
    })


def fmt_ts(iso: str) -> str:
    try:
        return pd.Timestamp(iso).tz_convert(None).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso


# ---------- backtest replay ----------
STEP_DAYS = {"Week": 5, "Month": 21}


def replay_result(source: str) -> pd.DataFrame:
    """Full backtest (production strategy) used by the replay page. Offline = synthetic prices."""
    import backtest
    if source == OFFLINE:
        px = synthetic_prices(config.TICKERS, days=config.LOOKBACK_DAYS * 2)
        return backtest.run_backtest(price_data=px)
    return backtest.run_backtest()


def replay_slice(result: pd.DataFrame, start_pos: int, cursor: int) -> pd.DataFrame:
    """Rows from start_pos through start_pos+cursor, value and benchmark rebased to 1.0 at the start."""
    end = min(start_pos + cursor, len(result) - 1)
    part = result.iloc[start_pos:end + 1].copy()
    for col in ("portfolio_value", "equal_weight_benchmark"):
        part[col] = part[col] / part[col].iloc[0]
    return part
