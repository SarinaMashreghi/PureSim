"""Simulation loop wiring together an AMMPool, a PriceFeed, and a set of agents."""

from __future__ import annotations

import random
from dataclasses import dataclass

from puresim.agents import Agent
from puresim.amm import AMMPool
from puresim.price_feed import PriceFeed


@dataclass
class StepLog:
    """Pool state snapshot recorded at the end of a simulation step."""

    step: int
    reference_price: float
    pool_price: float
    reserve_x: float
    reserve_y: float
    tvl: float


class Simulation:
    """Drives an AMMPool through a fixed number of steps with a set of agents.

    Each step:
      1. The price feed advances, producing a new reference price.
      2. Each agent gets a turn to act on the pool (order controlled by
         ``randomize_order``).
      3. The resulting pool state is logged.

    Adding a new agent type (including one that performs async/API calls,
    such as a future LLM-backed agent) requires no changes here: subclass
    ``Agent``, implement ``act``, and pass an instance in ``agents``. If a
    future agent needs to await network calls, the loop below is the single
    place that would grow an async variant (e.g. an ``act_async`` path) —
    it is kept deliberately simple and isolated so that change stays local.
    """

    def __init__(
        self,
        pool: AMMPool,
        price_feed: PriceFeed,
        agents: list[Agent],
        num_steps: int = 200,
        randomize_order: bool = False,
        seed: int | None = None,
    ) -> None:
        self.pool = pool
        self.price_feed = price_feed
        self.agents = agents
        self.num_steps = num_steps
        self.randomize_order = randomize_order
        self._rng = random.Random(seed)

        self.step_logs: list[StepLog] = []

    def run(self) -> list[StepLog]:
        for step in range(self.num_steps):
            reference_price = self.price_feed.get_next_price()

            turn_order = list(self.agents)
            if self.randomize_order:
                self._rng.shuffle(turn_order)

            for agent in turn_order:
                agent.act(self.pool, step)

            self.step_logs.append(
                StepLog(
                    step=step,
                    reference_price=reference_price,
                    pool_price=self.pool.get_price(),
                    reserve_x=self.pool.reserve_x,
                    reserve_y=self.pool.reserve_y,
                    tvl=self.pool.get_tvl(),
                )
            )

        return self.step_logs

    def summary(self) -> str:
        """Build a human-readable summary of the completed run."""
        if not self.step_logs:
            return "Simulation has not been run yet."

        lines: list[str] = []
        final_log = self.step_logs[-1]
        initial_log = self.step_logs[0]

        lines.append("=" * 60)
        lines.append("SIMULATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Steps run:              {self.num_steps}")
        lines.append(f"Final pool price:       {final_log.pool_price:.6f}")
        lines.append(f"Final reference price:  {final_log.reference_price:.6f}")
        deviation = (
            (final_log.pool_price - final_log.reference_price) / final_log.reference_price
        )
        lines.append(f"Final deviation:        {deviation:+.4%}")
        lines.append(f"Final TVL (in Y):       {final_log.tvl:,.2f}")
        lines.append("")

        total_volume_y = 0.0
        swaps_by_agent: dict[str, int] = {}
        for record in self.pool.history:
            swaps_by_agent[record.agent] = swaps_by_agent.get(record.agent, 0) + 1
            if record.token_in == "Y":
                total_volume_y += record.amount_in
            else:
                total_volume_y += record.amount_out

        lines.append(f"Total swaps:            {len(self.pool.history)}")
        lines.append(f"Total volume (in Y):    {total_volume_y:,.2f}")
        lines.append("Swaps per agent:")
        for agent in self.agents:
            lines.append(f"  {agent.name:<20} {swaps_by_agent.get(agent.name, 0)}")
        lines.append("")

        for agent in self.agents:
            if hasattr(agent, "cumulative_profit"):
                lines.append(
                    f"{agent.name} cumulative profit (in Y): "
                    f"{agent.cumulative_profit:,.4f}"
                )
        lines.append("")

        lines.append("Price over time (sampled):")
        lines.append(f"{'step':>6} {'pool_price':>14} {'reference_price':>16}")
        sample_count = min(20, len(self.step_logs))
        sample_indices = [
            int(i * (len(self.step_logs) - 1) / max(sample_count - 1, 1))
            for i in range(sample_count)
        ]
        seen = set()
        for idx in sample_indices:
            if idx in seen:
                continue
            seen.add(idx)
            log = self.step_logs[idx]
            lines.append(f"{log.step:>6} {log.pool_price:>14.6f} {log.reference_price:>16.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_price_history(self, output_path: str) -> None:
        """Save a plot of pool price vs. reference price to ``output_path``.

        Requires matplotlib. If it is not installed, raises ImportError with
        a clear message rather than failing silently.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_price_history(); install it with "
                "'pip install matplotlib'"
            ) from exc

        steps = [log.step for log in self.step_logs]
        pool_prices = [log.pool_price for log in self.step_logs]
        reference_prices = [log.reference_price for log in self.step_logs]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, pool_prices, label="Pool price")
        ax.plot(steps, reference_prices, label="Reference price", linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price (Y per X)")
        ax.set_title("Pool price vs. reference price")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
