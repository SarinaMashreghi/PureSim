"""Background trading agents that act on an AMMPool each simulation step.

``Agent`` is the extensibility point: a future LLM-backed agent (which may
need to make async/API calls per step) can subclass it and slot into
``Simulation`` without any changes to the simulation loop, as long as it
implements ``act``. If/when an async agent is added, the simulation loop can
await ``act`` for agents that declare themselves async without disrupting
synchronous agents — see the note on ``Simulation`` in simulation.py.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from puresim.amm import AMMPool


class Agent(ABC):
    """Base class for all agents that trade against an AMMPool.

    Subclasses implement ``act`` to optionally perform one swap per
    simulation step. Agents are responsible for their own strategy state
    (e.g. reference prices, inventory) but should not mutate the pool except
    through ``AMMPool.swap``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.num_swaps = 0

    @abstractmethod
    def act(self, pool: "AMMPool", step: int) -> None:
        """Called once per simulation step; may execute zero or one swaps."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class NoiseTrader(Agent):
    """Trades randomly, independent of any price signal.

    Each step, with probability ``activity_prob``, buys or sells a random
    small amount of X (sized as a fraction of the current reserve, drawn
    uniformly from ``[min_size_frac, max_size_frac]``). Otherwise does
    nothing.
    """

    def __init__(
        self,
        name: str = "NoiseTrader",
        activity_prob: float = 0.3,
        min_size_frac: float = 0.0001,
        max_size_frac: float = 0.005,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        if not 0 <= activity_prob <= 1:
            raise ValueError("activity_prob must be in [0, 1]")
        if not 0 < min_size_frac <= max_size_frac:
            raise ValueError("min_size_frac must be positive and <= max_size_frac")

        self.activity_prob = activity_prob
        self.min_size_frac = min_size_frac
        self.max_size_frac = max_size_frac
        self._rng = random.Random(seed)

    def act(self, pool: "AMMPool", step: int) -> None:
        if self._rng.random() > self.activity_prob:
            return

        token_in = self._rng.choice(("X", "Y"))
        size_frac = self._rng.uniform(self.min_size_frac, self.max_size_frac)
        reserve = pool.reserve_x if token_in == "X" else pool.reserve_y
        amount_in = reserve * size_frac

        pool.swap(token_in, amount_in, agent=self.name, step=step)
        self.num_swaps += 1


class Arbitrageur(Agent):
    """Trades to correct the pool price toward an external reference price.

    Each step, compares the pool's marginal price to
    ``price_feed.current_price``. If the absolute relative deviation exceeds
    ``threshold``, executes a swap sized as ``correction_frac`` of the
    (approximate) amount needed to fully close the gap. Tracks cumulative
    profit in token Y, estimated as the value of the trade at the reference
    price minus the value actually paid/received from the pool.
    """

    def __init__(
        self,
        price_feed,
        name: str = "Arbitrageur",
        threshold: float = 0.005,
        correction_frac: float = 1.0,
    ) -> None:
        super().__init__(name)
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if not 0 < correction_frac <= 1:
            raise ValueError("correction_frac must be in (0, 1]")

        self.price_feed = price_feed
        self.threshold = threshold
        self.correction_frac = correction_frac
        self.cumulative_profit = 0.0

    def _target_swap_amount(self, pool: "AMMPool", reference_price: float) -> tuple[str, float]:
        """Compute the token to sell and amount that would move the pool's
        price to (approximately) ``reference_price``, assuming no fee.

        For constant product k = x * y, the reserve levels at which price
        reference_price = y' / x' with x' * y' = k are:
            x' = sqrt(k / reference_price)
            y' = sqrt(k * reference_price)
        """
        k = pool.reserve_x * pool.reserve_y
        target_x = (k / reference_price) ** 0.5
        target_y = (k * reference_price) ** 0.5

        if target_x < pool.reserve_x:
            # Pool has too much X (X is underpriced) -> sell Y for X
            amount_in = (target_y - pool.reserve_y) * self.correction_frac
            return "Y", amount_in
        else:
            # Pool has too little X (X is overpriced) -> sell X for Y
            amount_in = (target_x - pool.reserve_x) * self.correction_frac
            return "X", amount_in

    def act(self, pool: "AMMPool", step: int) -> None:
        reference_price = self.price_feed.current_price
        pool_price = pool.get_price()
        deviation = (pool_price - reference_price) / reference_price

        if abs(deviation) <= self.threshold:
            return

        token_in, amount_in = self._target_swap_amount(pool, reference_price)
        if amount_in <= 0:
            return

        amount_out, effective_price, _slippage = pool.swap(
            token_in, amount_in, agent=self.name, step=step
        )
        self.num_swaps += 1

        # Profit estimate: value of what was received at the reference price
        # minus value of what was given up, both expressed in Y.
        if token_in == "X":
            value_in_y = amount_in * reference_price
            value_out_y = amount_out
        else:
            value_in_y = amount_in
            value_out_y = amount_out * reference_price

        self.cumulative_profit += value_out_y - value_in_y
