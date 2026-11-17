# Phase 04 demo script (~5 min)

**Before class:** `./run.sh test` (should show 18 passed) → `./run.sh` (dashboard opens). Keep the sidebar on
**Offline demo** as the safe default; switch to Yahoo only if the network is confirmed.

| Min | Do | Say |
|---|---|---|
| 0:00 | Overview page, point at the sidebar banner | "Simulation only — this app never touches a broker." |
| 0:30 | Price chart, then the allocation inputs | "You enter your holdings; the rest is cash." |
| 1:00 | Click **Calculate portfolio and rebalance** | "Max-Sharpe optimizer, Ledoit-Wolf covariance, 5–40% bounds, 7% cash buffer." |
| 1:45 | Weight rows: bar = target, tick = now, dashed = floor/cap | "Red drift is over the 4-point threshold, so a rebalance triggers." |
| 2:30 | Order list | "Estimated trades with assumed slippage. Not an executable plan." |
| 3:00 | **Backtest Replay**: click Next month a few times (orange diamonds = actual trades) | "Trades are rare: it only acts when drift crosses 4%." |
| 3:30 | **Backtest & Model** page | "Production beat equal-weight in 3 of 6 windows. ML beat it in 2 of 6, and it failed our pre-registered gate, so it is not used." |
| 4:00 | **Run History** page | "Every run is saved to SQLite. The engine's paper-trading cycles log to the same file." |
| 4:30 | Terminal: `./run.sh test --demo` | "18 offline tests: unit, integration, and a stubbed full-cycle system test. Includes a regression test for the audit-log bug we found and fixed." |

**If something breaks:** Yahoo down → switch to Offline demo. Dashboard won't start → show `./run.sh test` output and the
screenshots. **Do not** present offline-demo numbers as real performance (the page says so).

**Likely questions**
- *Why isn't the ML model used?* Pre-registered gate required beating the naive baseline on both information coefficient and hit rate; it passed one.
- *Does it trade?* Dashboard: never. CLI: dry-run by default; Alpaca **paper** account only, behind explicit flags.
- *What's the biggest limitation?* Tiny universe, in-sample historical-mean returns, and a short test history.
