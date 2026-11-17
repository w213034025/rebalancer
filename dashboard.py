"""Top-5 Mega-Cap Rebalancing Engine — Streamlit dashboard. SIMULATION ONLY."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import dashboard_logic as dl

# One palette, used on every page: grey = what you hold now, blue = the production
# strategy / target, orange = experimental ML, green = equal-weight benchmark.
C_CURRENT, C_TARGET, C_ML, C_EW = "#9AA5B1", "#1F6FEB", "#E8883A", "#3FA66B"

st.set_page_config(page_title="Top-5 Rebalancing Engine", page_icon="📈", layout="wide")
st.title("📈 Top-5 Mega-Cap Rebalancing Engine")
st.caption("Phase 4 • Educational portfolio analytics • SIMULATION ONLY — no brokerage orders are submitted")

with st.sidebar:
    st.header("Dashboard settings")
    page = st.radio("Navigate", ["Overview & Rebalance", "Backtest Replay", "Backtest & Model", "Run History", "About & Methodology"])
    source = st.radio("Market data", [dl.OFFLINE, dl.YAHOO])
    value = st.number_input("Portfolio value (USD)", min_value=100.0, value=10000.0, step=500.0)
    st.info("This application never connects to a broker or submits orders.")


@st.cache_data(show_spinner="Loading market data…", ttl=3600)
def _prices(src):
    return dl.load_prices(src)


def overview():
    st.header("1. Market data")
    try:
        prices, label = _prices(source)
    except Exception as e:
        st.error(f"Could not load {source}: {e}\n\nSwitch to **Offline demo** to continue.")
        return
    st.success(f"Data loaded: {label} • {len(prices)} trading days • through {prices.index[-1]:%Y-%m-%d}")
    tk = st.selectbox("Stock price chart", config.TICKERS)
    fig = go.Figure(go.Scatter(x=prices.index, y=prices[tk], mode="lines"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Price (USD)")
    st.plotly_chart(fig, width="stretch")
    if source == dl.OFFLINE:  # caption only when it is actually true
        st.caption("Offline demo uses generated prices. It is not actual market history and must not be reported as real backtest performance.")

    st.header("2. Enter your current allocation")
    st.caption("Percent of your portfolio in each stock. The remaining percentage is cash.")
    cols = st.columns(len(config.TICKERS))
    hold = {t: c.number_input(f"{t} %", 0.0, 100.0, 18.6, 0.5, key=f"h_{t}") for t, c in zip(config.TICKERS, cols)}
    cash = 100 - sum(hold.values())
    if cash < 0:
        st.error(f"Holdings total {sum(hold.values()):.1f}% — over 100%. Reduce them.")
        return
    st.metric("Remaining cash allocation", f"{cash:.1f}%")
    st.caption("These are simulated holdings entered by you, not broker-reported positions.")

    if st.button("Calculate portfolio and rebalance", type="primary"):
        try:
            res = dl.run_simulation(prices, hold, value)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            return
        dl.save_run(source, value, res)
        st.session_state["res"] = res
    res = st.session_state.get("res")
    if res:
        show(res)


def weight_rows(res):
    """Compact target-vs-now rows: bar = target, tick = current, dashed lines = floor/cap."""
    keep = 1 - config.CASH_BUFFER
    floor, cap = config.MIN_WEIGHT * keep * 100, config.MAX_WEIGHT * keep * 100
    scale = 50.0  # bar track spans 0-50%
    thr = config.DRIFT_THRESHOLD * 100
    rows = ['<div style="display:grid;grid-template-columns:60px 1fr 64px 64px 80px;gap:6px 10px;align-items:center;font-size:0.95rem">'
            '<div></div><div style="opacity:.7">Target allocation (after cash buffer)</div>'
            '<div style="opacity:.7">Target</div><div style="opacity:.7">Now</div><div style="opacity:.7">Drift</div>']
    for n in config.TICKERS + ["CASH"]:
        t, c = res["target"][n] * 100, res["current"][n] * 100
        dr = c - t
        breach = n != "CASH" and abs(dr) > thr
        col = "#B3261E" if breach else "inherit"
        fx = "" if n == "CASH" else (
            f'<div style="position:absolute;left:{floor/scale*100:.2f}%;top:-3px;bottom:-3px;border-left:1px dashed #888"></div>'
            f'<div style="position:absolute;left:{cap/scale*100:.2f}%;top:-3px;bottom:-3px;border-left:1px dashed #888"></div>')
        rows.append(
            f'<div style="font-weight:600">{n}</div>'
            f'<div style="position:relative;height:14px;background:rgba(128,128,128,.15);border-radius:4px">'
            f'<div style="width:{min(t/scale,1)*100:.2f}%;height:100%;background:{C_TARGET if n!="CASH" else C_CURRENT};border-radius:4px"></div>'
            f'<div style="position:absolute;left:{min(c/scale,1)*100:.2f}%;top:-3px;bottom:-3px;border-left:3px solid #222"></div>'
            f'{fx}</div>'
            f'<div>{t:.1f}%</div><div>{c:.1f}%</div>'
            f'<div style="color:{col};font-weight:{600 if breach else 400}">{dr:+.1f} pts</div>')
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption(f"Bar = target, black tick = your current holding, dashed lines = {config.MIN_WEIGHT:.0%} floor / "
               f"{config.MAX_WEIGHT:.0%} cap (scaled for the {config.CASH_BUFFER:.0%} cash buffer). "
               f"Drift in red exceeds the {thr:.0f}-point threshold.")


def show(res):
    st.header("3. Recommendation")
    a, b, c, d = st.columns(4)
    a.metric("Expected return", f"{res['exp_return']:.1%}")
    b.metric("Volatility", f"{res['volatility']:.1%}")
    c.metric("Sharpe", f"{res['sharpe']:.2f}")
    d.metric("Target cash", f"{res['target']['CASH']:.1%}")
    st.caption("Historical-sample estimates — in-sample, not a forecast.")

    weight_rows(res)
    thr = config.DRIFT_THRESHOLD * 100
    n_breach = sum(abs(res["drift"][t]) * 100 > thr for t in config.TICKERS)
    if res["needs_rebalance"]:
        st.warning(f"Rebalance triggered: {n_breach} of {len(config.TICKERS)} positions exceed the {thr:.0f}% drift threshold.")
    else:
        st.success(f"All drifts within {thr:.0f}% → no trade needed this cycle.")

    if res["trades"]:
        st.subheader("Simulated order list")
        t = pd.DataFrame(res["trades"])
        t = pd.DataFrame({
            "Action": t["side"].str.capitalize(),
            "Ticker": t["ticker"],
            "Shares": t["qty"].round(2),
            "Est. price": t["est_price"],
            "Amount": t["est_notional"],
            "Why": t["side"].map({"buy": "Underweight vs target", "sell": "Overweight vs target"}),
        })
        st.dataframe(t, width="stretch", hide_index=True, column_config={
            "Shares": st.column_config.NumberColumn(format="%.2f"),
            "Est. price": st.column_config.NumberColumn(format="$%.2f"),
            "Amount": st.column_config.NumberColumn(format="$%.0f"),
        })
        st.caption("Estimate only: fractional quantities, assumed slippage. Not an executable trading plan.")


@st.cache_data(show_spinner="Running backtest…", ttl=3600)
def _replay(src):
    return dl.replay_result(src)


def replay():
    import time
    st.header("Backtest replay")
    st.caption("Step through history and watch the production strategy decide. It re-runs the same rules on "
               "past data (in-sample) and says nothing about the future.")
    if source == dl.OFFLINE:
        st.caption("Offline demo uses generated prices. It is not actual market history and must not be reported "
                   "as real backtest performance.")
    try:
        res = _replay(source)
    except Exception as e:
        st.error(f"Could not run backtest for {source}: {e}\n\nSwitch to **Offline demo** to continue.")
        return
    dates = res.index
    c1, c2 = st.columns([2, 1])
    start = c1.date_input("Start date", value=dates[0].date(), min_value=dates[0].date(),
                          max_value=dates[-20].date())
    step = c2.radio("Step", list(dl.STEP_DAYS), horizontal=True)
    n = dl.STEP_DAYS[step]
    start_pos = int(dates.searchsorted(pd.Timestamp(start)))

    key = (source, start_pos)
    if st.session_state.get("replay_key") != key:
        st.session_state["replay_key"], st.session_state["cursor"] = key, 0
    max_cursor = len(res) - 1 - start_pos

    b1, b2, b3, b4 = st.columns(4)
    if b1.button(f"Next {step.lower()} ▶", type="primary"):
        st.session_state["cursor"] = min(st.session_state["cursor"] + n, max_cursor)
    play = b2.button("Auto-play ⏵")
    b3.button("Stop ⏹")          # clicking triggers a rerun, which interrupts auto-play
    if b4.button("Reset ↺"):
        st.session_state["cursor"] = 0

    slot = st.empty()
    draw = lambda: _draw_replay(slot, dl.replay_slice(res, start_pos, st.session_state["cursor"]))
    draw()
    if play:
        while st.session_state["cursor"] < max_cursor:
            time.sleep(0.5)
            st.session_state["cursor"] = min(st.session_state["cursor"] + n, max_cursor)
            draw()
        st.success("Reached the end of the data.")


def _draw_replay(slot, part):
    last = part.iloc[-1]
    ev = part[part["rebalanced"]]
    with slot.container():
        a, b, c, d = st.columns(4)
        a.metric("Date", f"{part.index[-1]:%Y-%m-%d}")
        b.metric("Strategy", f"{last['portfolio_value']:.3f}",
                 f"{last['portfolio_value'] - last['equal_weight_benchmark']:+.3f} vs equal-weight")
        c.metric("Rebalances so far", len(ev))
        d.metric("Days since last trade", int((part.index[-1] - ev.index[-1]).days) if len(ev) else "—")
        fig = go.Figure()
        fig.add_scatter(x=part.index, y=part["portfolio_value"], name="Strategy", line=dict(color=C_TARGET))
        fig.add_scatter(x=part.index, y=part["equal_weight_benchmark"], name="Equal-weight",
                        line=dict(color=C_EW, dash="dot"))
        if len(ev):
            fig.add_scatter(x=ev.index, y=ev["portfolio_value"], mode="markers", name="Rebalance",
                            marker=dict(color=C_ML, size=11, symbol="diamond"))
        fig.update_layout(height=320, margin=dict(t=10, b=0), yaxis_title="Value of $1 (rebased)")
        st.plotly_chart(fig, width="stretch")
        wfig = go.Figure()
        palette = ["#1F6FEB", "#5B9BF3", "#8E7CC3", "#3FA66B", "#E8883A", C_CURRENT]   # 5 tickers + cash
        for name, col in zip(config.TICKERS + ["CASH"], palette):
            wfig.add_bar(y=["Weights"], x=[last[name] * 100], name=name, orientation="h", marker_color=col,
                         text=f"{name} {last[name]:.0%}" if last[name] > 0.07 else "", textposition="inside",
                         textfont_color="white")
        wfig.update_layout(barmode="stack", height=130, margin=dict(t=0, b=0), showlegend=False,
                           xaxis=dict(range=[0, 100], title="% of portfolio on this date"))
        st.plotly_chart(wfig, width="stretch")


def backtest():
    st.header("Strategy comparison (cached)")
    st.caption("Final value of $1 across six 'as of' windows — snapshot of robustness_check.py (run 2026-10-25). "
               "Historical-mean beat equal-weight in 3/6; ML beat historical in 2/6.")
    colors = [C_TARGET, C_ML, C_EW]
    fig = go.Figure()
    for col, color in zip(dl.ROBUSTNESS.columns[1:], colors):
        fig.add_bar(name=col, x=dl.ROBUSTNESS["window"], y=dl.ROBUSTNESS[col], marker_color=color)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#444", annotation_text="$1 start (break-even)")
    lo = float(dl.ROBUSTNESS.iloc[:, 1:].min().min())
    fig.update_yaxes(range=[min(0.9, lo - 0.1), None], title_text="Final value of $1")
    fig.update_layout(barmode="group", height=380, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Bars above the dashed line made money over that window; the axis is truncated just below it "
               "so differences between strategies are visible.")
    st.header("Model status")
    st.warning("ML forecaster NOT promoted: it beat the naive baseline on information coefficient but not on hit rate "
               "(pre-registered gate required both). USE_ML_FORECASTER = "
               f"{config.USE_ML_FORECASTER}.")
    st.dataframe(dl.GATE, width="stretch", hide_index=True)


def history():
    st.header("Saved dashboard runs")
    runs = dl.load_runs()[:20]  # most recent 20
    if runs and st.button("Clear saved runs"):
        dl.clear_runs()
        st.rerun()
    if not runs:
        st.info("No saved runs yet. Run a calculation on the Overview page.")
    for r in runs:
        with st.expander(f"Run #{r['id']} • {dl.fmt_ts(r['ts'])} • ${r['value']:,.2f} • {r['source']}"):
            res = r["result"]
            st.dataframe(pd.DataFrame({"Current %": {k: v * 100 for k, v in res["current"].items()},
                                       "Target %": {k: v * 100 for k, v in res["target"].items()}}).round(2))
            st.write(f"Rebalance triggered: **{res['needs_rebalance']}** • {len(res['trades'])} simulated trades")
    st.subheader("Engine audit log (CLI / paper-trading cycles)")
    ev = dl.load_audit_events()
    if ev.empty:
        st.caption("No engine events logged yet.")
    else:
        st.dataframe(ev, width="stretch", hide_index=True)


def about():
    st.header("System architecture and limitations")
    st.markdown("**Pipeline:** price data → historical-mean expected returns → Ledoit-Wolf covariance → "
                "maximum-Sharpe optimization (5–40% bounds) → 7% cash reserve → current-vs-target drift check → "
                "simulated trade list → SQLite audit log.")

    st.subheader("Modules (most to least important)")
    st.markdown("""
| # | Module | Purpose | Key functionality |
|---|---|---|---|
| 1 | `optimizer.py` | Decide what the portfolio *should* hold. | `compute_target_weights`: max-Sharpe optimization on historical-mean returns and a Ledoit-Wolf covariance, bounded 5–40% per name. `apply_cash_buffer`: scales weights down and adds the 7% cash line. |
| 2 | `rebalance_engine.py` | Run one complete rebalance cycle. | `run_rebalance_cycle`: fetch prices → targets → compare with holdings → trade only if drift exceeds the threshold → submit (dry-run by default) → write an audit row to SQLite. |
| 3 | `portfolio_state.py` | Compare what you hold with what you should hold. | Current weights, drift per position, the rebalance trigger, and `build_trade_list`: buy/sell quantities with assumed slippage, skipping dust trades under $50. |
| 4 | `config.py` | Keep every setting in one place. | Tickers, weight bounds, drift threshold, cash buffer, costs, risk-free rate, paper/live flag and the ML on/off switch. Keys come from environment variables. |
| 5 | `data_feed.py` | Supply clean market data. | `get_price_history` and `get_latest_prices` via Yahoo Finance; rejects missing tickers, excessive gaps and warns on stale data. Supports "as of" dates for historical tests. |
| 6 | `broker_alpaca.py` | Talk to the brokerage (CLI only). | Reads positions and cash; submits marketable limit orders to an Alpaca **paper** account, or only logs them in dry-run. The dashboard never imports it. |
| 7 | `backtest.py` | Test the strategy on history. | `run_backtest`: day-by-day simulation with drift-triggered trades, costs, and cash earning the risk-free rate, versus an equal-weight benchmark. `compare_strategies` adds the experimental ML variant. Powers Backtest Replay. |
| 8 | `dashboard.py`, `dashboard_logic.py` | Let a person explore the system safely. | This Streamlit app: enter holdings, see targets, drift and simulated orders, replay a backtest, review saved runs. The adapter supplies synthetic offline prices and saves runs to SQLite. |
| 9 | `main.py` | Give the engine a command line. | `check` (one dry-run cycle), `backtest` (optionally `--compare-ml`) and `schedule` (start the scheduler). |
| 10 | `scheduler.py` | Run the cycle automatically. | Calendar or drift-based schedule using APScheduler; currently dry-run only, and a failed cycle places no trades. |
| 11 | `forecaster.py`, `features.py` | Test whether machine learning can beat the baseline. | Ridge / LightGBM return forecaster with walk-forward evaluation, information coefficient and hit rate; `features.py` builds its inputs (volatility, momentum, drawdown, beta). Experimental, and it failed the pre-registered gate. |
| 12 | `robustness_check.py` | Check results aren't a lucky window. | Runs the strategy comparison across six historical end dates and reports how often each approach wins. |
| 13 | `tests/` | Prove the code does what we claim. | 21 offline tests (no network, keys or broker): unit, integration and a stubbed full-cycle system test, including a regression test for the audit-log bug. |
""")
    st.markdown("**ML status:** Ridge and LightGBM are experimental and intentionally NOT used for the main "
                "recommendation because the Phase 3 validation gate did not pass.")

    st.subheader("Security")
    st.markdown("""
- **Simulation-only dashboard:** it never imports `broker_alpaca.py`, submits orders or accesses brokerage accounts (a test checks this).
- **Safe broker defaults:** the command-line engine is dry-run by default and targets an Alpaca **paper** account (`ALPACA_PAPER`).
- **Secrets stay out of code:** API keys come from environment variables; `.env` is git-ignored; the dashboard needs no keys.
- **Audit trail:** every engine cycle is logged to SQLite with timestamp, targets, current weights, drift and trades.
- **Fails closed:** it refuses to trade on a missing or invalid price rather than trading blind.
- **Validated inputs:** holdings that total over 100% are rejected; weights are bounded to 5–40%.
- **Tested:** 21 offline tests, including a regression test for the audit-log serialization bug.
""")
    st.caption("This is a student project. These are good engineering practices, not a claim of enterprise "
               "certification or regulatory compliance.")

    st.subheader("Limitations")
    st.markdown("Offline data is synthetic. Simulated position quantities may be fractional and the order list is "
                "an estimate, not an executable trading plan.")


{"Overview & Rebalance": overview, "Backtest Replay": replay, "Backtest & Model": backtest,
 "Run History": history, "About & Methodology": about}[page]()
