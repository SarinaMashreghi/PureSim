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

## Running sim

Install dependencies (only `pytest` for tests and, optionally, `matplotlib`
for plotting are required — the simulation itself has no dependencies beyond
the standard library):

```bash
pip install pytest matplotlib
```

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
