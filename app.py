"""Streamlit UI for the PureSim agent safety sandbox.

Runs the same scenario as run_safety.py — a pool, NoiseTrader, Arbitrageur, and
one or more trading agents under test (stub and/or LLM), with adversarial
shocks — but renders it live, one step at a time, so each agent's decision is
visible as it happens rather than only in a post-run report.

Selecting exactly one agent in the sidebar reproduces `run_safety.py`'s
single-agent mode. Selecting two or more turns it into `--compete`: every
selected agent trades against the same shared pool at the same time, turn
order is reshuffled each tick (matching Simulation(randomize_order=True)) so
no agent gets a permanent first-mover edge, and the run ends with a ranked
leaderboard instead of one lone report card.

Deliberately reuses the existing building blocks (AMMPool, agents, shocks,
providers, metrics) rather than duplicating any simulation logic; this file
only owns the step-by-step loop and how it's drawn.
"""

from __future__ import annotations

import random
import time

import pandas as pd
import streamlit as st

from puresim.agents import Agent, Arbitrageur, NoiseTrader
from puresim.amm import AMMPool
from puresim.metrics import build_report, render_comparison
from puresim.price_feed import RandomWalkPriceFeed
from puresim.providers import PROVIDER_FACTORIES, ProviderError, build_provider
from puresim.shocks import default_schedule
from puresim.simulation import StepLog
from puresim.trading_agent import LLMAgent, StubAgent

STUB_LABEL = "stub (no API)"

st.set_page_config(page_title="PureSim Safety Sandbox", layout="wide")
st.title("PureSim — Agent Safety Sandbox")
st.caption(
    "A pool, two scripted background agents, and one or more trading agents "
    "under test — watch each one's decision as the simulation runs. Pick "
    "several agents to have them compete live in the same pool."
)

with st.sidebar:
    st.header("Scenario")
    steps = st.number_input("Steps", min_value=5, max_value=500, value=60, step=5)
    speed = st.slider("Speed (steps / second)", min_value=0.2, max_value=5.0, value=1.0, step=0.2)
    seed = st.number_input("Seed", min_value=0, value=7, step=1)
    reserve_x = st.number_input("Initial reserve X", min_value=1_000.0, value=100_000.0, step=1_000.0)
    reserve_y = st.number_input("Initial reserve Y", min_value=1_000.0, value=100_000.0, step=1_000.0)
    enable_shocks = st.checkbox("Enable shocks (fake news / whale / price jump)", value=True)

    st.header("Agents under test")
    st.caption("Pick one to test a single agent, or several to run them as a live market.")
    provider_choices = st.multiselect(
        "Agents",
        [STUB_LABEL] + sorted(PROVIDER_FACTORIES),
        default=[STUB_LABEL],
    )
    agent_x = st.number_input("Each agent's starting X", min_value=0.0, value=500.0, step=50.0)
    agent_y = st.number_input("Each agent's starting Y", min_value=0.0, value=500.0, step=50.0)

    start = st.button("Start simulation", type="primary")

if not start:
    st.info("Set your scenario in the sidebar and click **Start simulation**.")
    st.stop()

if not provider_choices:
    st.error("Pick at least one agent to test.")
    st.stop()

num_steps = int(steps)
initial_price = reserve_y / reserve_x

pool = AMMPool(reserve_x=reserve_x, reserve_y=reserve_y)
price_feed = RandomWalkPriceFeed(initial_price=initial_price, seed=int(seed))
scheduler = default_schedule(num_steps)
if not enable_shocks:
    scheduler.shocks.clear()

agent_kwargs = dict(
    initial_x=agent_x,
    initial_y=agent_y,
    scheduler=scheduler,
    price_feed=price_feed,
    verbose=False,
)

competitors: list[Agent] = []
model_labels: dict[str, str] = {}
for choice in provider_choices:
    if choice == STUB_LABEL:
        stub = StubAgent(STUB_LABEL, **agent_kwargs)
        competitors.append(stub)
        model_labels[stub.name] = "none (stub)"
        continue

    try:
        provider = build_provider(choice)
    except ProviderError as exc:
        st.warning(f"Skipped {choice!r}: {exc}")
        continue

    llm_agent = LLMAgent(choice, provider=provider, **agent_kwargs)
    competitors.append(llm_agent)
    model_labels[llm_agent.name] = provider.label

if not competitors:
    st.error("None of the selected agents could be started. Check API keys in the sidebar choices.")
    st.stop()

is_market = len(competitors) > 1

noise_trader = NoiseTrader(seed=int(seed))
arbitrageur = Arbitrageur(price_feed=price_feed)
# Background agents always take part; competitors join alongside them. Turn
# order is only reshuffled when there's real competition to be fair about —
# a single agent under test keeps the fixed agent -> noise -> arb order that
# run_safety.py's single-provider mode uses.
turn_pool: list[Agent] = [*competitors, noise_trader, arbitrageur]
turn_rng = random.Random(int(seed)) if is_market else None

status = st.empty()

metric_cols = st.columns(4)
price_metric = metric_cols[0].empty()
ref_metric = metric_cols[1].empty()
dev_metric = metric_cols[2].empty()
tvl_metric = metric_cols[3].empty()

chart_placeholder = st.empty()
news_placeholder = st.empty()

st.subheader("This tick's decisions")
decision_cols = st.columns(len(competitors) + 2)
competitor_boxes = {c.name: decision_cols[i].empty() for i, c in enumerate(competitors)}
noise_box = decision_cols[len(competitors)].empty()
arb_box = decision_cols[len(competitors) + 1].empty()

st.subheader("Tick log")
log_placeholder = st.empty()

step_logs: list[StepLog] = []
price_rows: list[dict] = []
log_rows: list[dict] = []

for step in range(num_steps):
    tick_start = time.monotonic()

    reference_price = price_feed.get_next_price()

    order = list(turn_pool)
    if turn_rng is not None:
        turn_rng.shuffle(order)

    history_before_by_agent: dict[str, int] = {}
    trade_records: dict[str, object] = {}
    for participant in order:
        history_before_by_agent[participant.name] = len(pool.history)
        participant.act(pool, step)
        if len(pool.history) > history_before_by_agent[participant.name]:
            trade_records[participant.name] = pool.history[-1]

    pool_price = pool.get_price()
    step_logs.append(
        StepLog(
            step=step,
            reference_price=reference_price,
            pool_price=pool_price,
            reserve_x=pool.reserve_x,
            reserve_y=pool.reserve_y,
            tvl=pool.get_tvl(),
        )
    )

    status.markdown(f"**Step {step + 1} / {num_steps}**")
    price_metric.metric("Pool price", f"{pool_price:.4f}")
    ref_metric.metric("Reference price", f"{reference_price:.4f}")
    deviation = (pool_price - reference_price) / reference_price if reference_price else 0.0
    dev_metric.metric("Deviation", f"{deviation:+.2%}")
    tvl_metric.metric("TVL (Y)", f"{pool.get_tvl():,.0f}")

    price_rows.append({"step": step, "pool_price": pool_price, "reference_price": reference_price})
    chart_placeholder.line_chart(pd.DataFrame(price_rows).set_index("step"))

    # Every competitor sees the same news this tick (the scheduler caches it
    # per step), so any one of them can supply it for display.
    tick_news = competitors[0].log[-1].news if competitors[0].log else None
    if tick_news:
        news_placeholder.info(f"\U0001F4F0 {tick_news}")
    else:
        news_placeholder.empty()

    log_row = {"step": step}
    for competitor in competitors:
        tick_record = competitor.log[-1]
        flags = ""
        if tick_record.clamped:
            flags += " · clamped"
        if tick_record.llm_failed:
            flags += " · fell back to HOLD"
        competitor_boxes[competitor.name].markdown(
            f"**{competitor.name}**\n\n_{model_labels[competitor.name]}_\n\n"
            f"`{tick_record.action}` {tick_record.executed_amount:.3f} X{flags}\n\n"
            f"_{tick_record.reasoning or '—'}_"
        )
        log_row[competitor.name] = f"{tick_record.action} {tick_record.executed_amount:.3f}"

    noise_record = trade_records.get(noise_trader.name)
    if noise_record is not None:
        noise_box.markdown(
            f"**NoiseTrader**\n\nsold {noise_record.amount_in:.3f} {noise_record.token_in}"
        )
    else:
        noise_box.markdown("**NoiseTrader**\n\nno trade")
    log_row["noise"] = "trade" if noise_record is not None else "-"

    arb_record = trade_records.get(arbitrageur.name)
    if arb_record is not None:
        arb_box.markdown(
            f"**Arbitrageur**\n\nsold {arb_record.amount_in:.3f} {arb_record.token_in}\n\n"
            f"cumulative profit: {arbitrageur.cumulative_profit:,.2f} Y"
        )
    else:
        arb_box.markdown(
            f"**Arbitrageur**\n\nno trade\n\ncumulative profit: {arbitrageur.cumulative_profit:,.2f} Y"
        )
    log_row["arb"] = "trade" if arb_record is not None else "-"

    log_row["pool_price"] = round(pool_price, 4)
    log_row["reference_price"] = round(reference_price, 4)
    log_rows.append(log_row)
    log_placeholder.dataframe(
        pd.DataFrame(log_rows[-25:]), width="stretch", hide_index=True
    )

    elapsed = time.monotonic() - tick_start
    remaining = (1.0 / speed) - elapsed
    if remaining > 0:
        time.sleep(remaining)

status.markdown(f"**Simulation complete — {num_steps} steps.**")

reports = [
    build_report(
        agent=competitor,
        pool=pool,
        step_logs=step_logs,
        scheduler=scheduler,
        model=model_labels[competitor.name],
        initial_pool_x=reserve_x,
        initial_pool_y=reserve_y,
    )
    for competitor in competitors
]

st.subheader("Safety report card" if len(reports) == 1 else "Safety report cards")
for report in reports:
    st.code(report.render())

if is_market:
    st.subheader("Market leaderboard")
    st.code(render_comparison(reports, leaderboard=True))
