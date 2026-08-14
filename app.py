"""Streamlit UI for the PureSim agent safety sandbox.

Runs the same scenario as run_safety.py — a pool, NoiseTrader, Arbitrageur,
and a trading agent under test (stub or LLM), with adversarial shocks — but
renders it live, one step at a time, so each agent's decision is visible as
it happens rather than only in a post-run report.

Deliberately reuses the existing building blocks (AMMPool, agents, shocks,
providers, metrics) rather than duplicating any simulation logic; this file
only owns the step-by-step loop and how it's drawn.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from puresim.agents import Arbitrageur, NoiseTrader
from puresim.amm import AMMPool
from puresim.metrics import build_report
from puresim.price_feed import RandomWalkPriceFeed
from puresim.providers import PROVIDER_FACTORIES, ProviderError, build_provider
from puresim.shocks import default_schedule
from puresim.simulation import StepLog
from puresim.trading_agent import LLMAgent, StubAgent

STUB_LABEL = "stub (no API)"

st.set_page_config(page_title="PureSim Safety Sandbox", layout="wide")
st.title("PureSim — Agent Safety Sandbox")
st.caption(
    "A pool, two scripted background agents, and a trading agent under test — "
    "watch each one's decision as the simulation runs."
)

with st.sidebar:
    st.header("Scenario")
    steps = st.number_input("Steps", min_value=5, max_value=500, value=60, step=5)
    speed = st.slider("Speed (steps / second)", min_value=0.2, max_value=5.0, value=1.0, step=0.2)
    seed = st.number_input("Seed", min_value=0, value=7, step=1)
    reserve_x = st.number_input("Initial reserve X", min_value=1_000.0, value=100_000.0, step=1_000.0)
    reserve_y = st.number_input("Initial reserve Y", min_value=1_000.0, value=100_000.0, step=1_000.0)
    enable_shocks = st.checkbox("Enable shocks (fake news / whale / price jump)", value=True)

    st.header("Agent under test")
    provider_choice = st.selectbox("Provider", [STUB_LABEL] + sorted(PROVIDER_FACTORIES))
    agent_x = st.number_input("Agent starting X", min_value=0.0, value=500.0, step=50.0)
    agent_y = st.number_input("Agent starting Y", min_value=0.0, value=500.0, step=50.0)

    start = st.button("Start simulation", type="primary")

if not start:
    st.info("Set your scenario in the sidebar and click **Start simulation**.")
    st.stop()

try:
    provider = None if provider_choice == STUB_LABEL else build_provider(provider_choice)
except ProviderError as exc:
    st.error(f"Could not start provider {provider_choice!r}: {exc}")
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
if provider is None:
    agent = StubAgent("StubAgent", **agent_kwargs)
    model_label = "none (stub)"
else:
    agent = LLMAgent("LLMAgent", provider=provider, **agent_kwargs)
    model_label = provider.label

noise_trader = NoiseTrader(seed=int(seed))
arbitrageur = Arbitrageur(price_feed=price_feed)

status = st.empty()

metric_cols = st.columns(4)
price_metric = metric_cols[0].empty()
ref_metric = metric_cols[1].empty()
dev_metric = metric_cols[2].empty()
tvl_metric = metric_cols[3].empty()

chart_placeholder = st.empty()
news_placeholder = st.empty()

st.subheader("This tick's decisions")
decision_cols = st.columns(3)
agent_box = decision_cols[0].empty()
noise_box = decision_cols[1].empty()
arb_box = decision_cols[2].empty()

st.subheader("Tick log")
log_placeholder = st.empty()

step_logs: list[StepLog] = []
price_rows: list[dict] = []
log_rows: list[dict] = []

for step in range(num_steps):
    tick_start = time.monotonic()

    reference_price = price_feed.get_next_price()

    agent.act(pool, step)
    tick_record = agent.log[-1]

    history_before = len(pool.history)
    noise_trader.act(pool, step)
    noise_record = pool.history[-1] if len(pool.history) > history_before else None

    history_before = len(pool.history)
    arbitrageur.act(pool, step)
    arb_record = pool.history[-1] if len(pool.history) > history_before else None

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

    if tick_record.news:
        news_placeholder.info(f"\U0001F4F0 {tick_record.news}")
    else:
        news_placeholder.empty()

    flags = ""
    if tick_record.clamped:
        flags += " · clamped"
    if tick_record.llm_failed:
        flags += " · fell back to HOLD"
    agent_box.markdown(
        f"**{model_label}**\n\n"
        f"`{tick_record.action}` {tick_record.executed_amount:.3f} X{flags}\n\n"
        f"_{tick_record.reasoning or '—'}_"
    )

    if noise_record is not None:
        noise_box.markdown(
            f"**NoiseTrader**\n\nsold {noise_record.amount_in:.3f} {noise_record.token_in}"
        )
    else:
        noise_box.markdown("**NoiseTrader**\n\nno trade")

    if arb_record is not None:
        arb_box.markdown(
            f"**Arbitrageur**\n\nsold {arb_record.amount_in:.3f} {arb_record.token_in}\n\n"
            f"cumulative profit: {arbitrageur.cumulative_profit:,.2f} Y"
        )
    else:
        arb_box.markdown(
            f"**Arbitrageur**\n\nno trade\n\ncumulative profit: {arbitrageur.cumulative_profit:,.2f} Y"
        )

    log_rows.append(
        {
            "step": step,
            "agent": tick_record.action,
            "amount": round(tick_record.executed_amount, 3),
            "noise": "trade" if noise_record is not None else "-",
            "arb": "trade" if arb_record is not None else "-",
            "pool_price": round(pool_price, 4),
            "reference_price": round(reference_price, 4),
        }
    )
    log_placeholder.dataframe(
        pd.DataFrame(log_rows[-25:]), width="stretch", hide_index=True
    )

    elapsed = time.monotonic() - tick_start
    remaining = (1.0 / speed) - elapsed
    if remaining > 0:
        time.sleep(remaining)

status.markdown(f"**Simulation complete — {num_steps} steps.**")

report = build_report(
    agent=agent,
    pool=pool,
    step_logs=step_logs,
    scheduler=scheduler,
    model=model_label,
    initial_pool_x=reserve_x,
    initial_pool_y=reserve_y,
)
st.subheader("Safety report card")
st.code(report.render())
