"""External reference price feed for token X.

Structured behind a small interface (``get_next_price``) so the random-walk
implementation here can later be swapped for e.g. a feed that replays a
historical price CSV, without touching any calling code.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod


class PriceFeed(ABC):
    """Abstract interface for a reference price source."""

    @abstractmethod
    def get_next_price(self) -> float:
        """Advance the feed by one step and return the new reference price."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_price(self) -> float:
        """The most recently produced price, without advancing the feed."""
        raise NotImplementedError


class RandomWalkPriceFeed(PriceFeed):
    """Generates a reference price via geometric Brownian motion.

    Each step, the price is updated as::

        price *= exp((drift - 0.5 * volatility**2) + volatility * Z)

    where ``Z`` is a standard normal random variable. This is a discrete-time
    GBM step with unit time increments.
    """

    def __init__(
        self,
        initial_price: float,
        drift: float = 0.0,
        volatility: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        self.drift = drift
        self.volatility = volatility
        self._price = float(initial_price)
        self._rng = random.Random(seed)

    @property
    def current_price(self) -> float:
        return self._price

    def get_next_price(self) -> float:
        z = self._rng.gauss(0, 1)
        growth = (self.drift - 0.5 * self.volatility**2) + self.volatility * z
        self._price *= math.exp(growth)
        return self._price
