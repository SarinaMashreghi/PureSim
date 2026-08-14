# PureSim

A Python simulation of a constant-product Automated Market Maker (AMM) pool
with background trading agents.

- `puresim/amm.py` — `AMMPool`, a constant-product (`x * y = k`) pool with a
  configurable trading fee, input validation, per-swap history logging, and
  TVL calculation.
- `puresim/agents.py` — `Agent` base class, plus two rule-based agents:
  `NoiseTrader` (random small trades) and `Arbitrageur` (corrects pool price
  toward an external reference price, tracking its own P&L).
- `puresim/price_feed.py` — `PriceFeed` interface and a `RandomWalkPriceFeed`
  (geometric Brownian motion) implementation. Swappable later for a feed
  backed by a historical price CSV without touching any calling code.
- `puresim/simulation.py` — `Simulation`, the step loop that wires the pool,
  price feed, and agent list together, plus summary/plotting helpers.
- `run.py` — CLI entry point.
- `tests/test_amm.py` — unit tests for the AMM math.

## Setup

The core simulation has no dependencies beyond the standard library; a venv
is only needed for tests, plotting, the Streamlit UI, and LLM providers.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS/Linux: `source .venv/bin/activate` instead of the `Activate.ps1` line.)

Every command below assumes the venv is activated in your current shell —
reactivate it (the `Activate.ps1` / `source .venv/bin/activate` line) each
time you open a new terminal.

## Running sim

Run a simulation:

```bash
python run.py --steps 200 --initial-reserve-x 100000 --initial-reserve-y 100000
```

This prints a running summary and, at the end, a report covering the final
pool price vs. reference price, total volume traded, swap counts per agent,
the arbitrageur's cumulative profit, and a sampled table of price over time.

Run the unit tests:

```bash
python -m pytest
```

### CLI parameters

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 200 | Number of simulation steps to run. |
| `--initial-reserve-x` | 100000 | Starting reserve of token X. |
| `--initial-reserve-y` | 100000 | Starting reserve of token Y. Initial pool price is `initial-reserve-y / initial-reserve-x`. |
| `--fee-rate` | 0.003 | Trading fee taken on every swap (0.003 = 0.3%, the standard Uniswap fee). |
| `--noise-activity-prob` | 0.3 | Probability the `NoiseTrader` trades on any given step. |
| `--noise-min-size-frac` / `--noise-max-size-frac` | 0.0001 / 0.005 | Range (as a fraction of the relevant reserve) from which `NoiseTrader` draws its trade size. |
| `--arb-threshold` | 0.005 | Fractional deviation between pool price and reference price beyond which the `Arbitrageur` trades. |
| `--arb-correction-frac` | 1.0 | Fraction of the full price gap the `Arbitrageur` tries to close per trade (1.0 = fully close it, subject to its own price impact). |
| `--price-drift` | 0.0 | Per-step mean log return of the reference price random walk. |
| `--price-volatility` | 0.01 | Per-step log return standard deviation of the reference price random walk. |
| `--randomize-agent-order` | off | If set, shuffles agent turn order each step instead of using a fixed order. |
| `--seed` | none | Random seed, for reproducible runs. |
| `--plot-output` | none | If set, saves a pool-price-vs-reference-price PNG to this path (requires `matplotlib`). |

## Streamlit UI

`app.py` runs the same scenario as `run_safety.py` — the pool, `NoiseTrader`,
`Arbitrageur`, and a trading agent under test, with adversarial shocks — but
live, one step at a time, so you can watch each agent's decision as it
happens instead of only reading a post-run report.

```bash
pip install streamlit
streamlit run app.py
```

Set the scenario (steps, speed, pool size, seed, shocks) and pick the agent
under test in the sidebar — either the no-API `StubAgent` or a real model
provider (needs the matching API key set in your environment, e.g.
`ANTHROPIC_API_KEY` for Claude) — then click **Start simulation**. It steps
at the chosen rate (starts at 1 step/second), showing:

- pool price vs. reference price (metrics + live chart) and TVL
- any shock/news injected that tick
- each agent's decision that tick: the trading agent's action, amount, and
  reasoning; whether `NoiseTrader` traded; whether `Arbitrageur` traded and
  its running cumulative profit
- a scrolling tick-by-tick log
- the full safety report card once the run completes

## Design notes for extending this

- **New agent types** (including a future async/API-calling LLM agent):
  subclass `Agent` in `puresim/agents.py` and implement `act(pool, step)`.
  `Simulation` iterates over whatever list of agents it's given, so no
  changes to the simulation loop are needed to add one.
- **A different reference price source** (e.g. replaying historical data
  from a CSV instead of a random walk): implement the `PriceFeed` interface
  (`get_next_price()` / `current_price`) in `puresim/price_feed.py`. Nothing
  else needs to change.
- **A dashboard or other consumer**: `Simulation.step_logs` and
  `AMMPool.history` hold structured, timestamped records of everything that
  happened during a run and are the natural data source to build on.
