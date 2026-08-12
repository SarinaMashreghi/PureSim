"""Constant-product AMM pool (Uniswap V2 style, x * y = k)."""

from __future__ import annotations

from dataclasses import dataclass, field

FEE_RATE = 0.003  # 0.3% standard Uniswap trading fee


@dataclass
class SwapRecord:
    """A single logged swap event."""

    step: int
    agent: str
    token_in: str
    amount_in: float
    amount_out: float
    price_after: float
    effective_price: float
    slippage: float


class AMMPool:
    """A two-asset constant-product liquidity pool.

    Reserves follow the invariant ``reserve_x * reserve_y = k``, which is
    preserved (up to the trading fee, which grows k slightly) across swaps.
    """

    def __init__(self, reserve_x: float, reserve_y: float, fee_rate: float = FEE_RATE) -> None:
        if reserve_x <= 0 or reserve_y <= 0:
            raise ValueError("Initial reserves must be positive")
        if not 0 <= fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")

        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.fee_rate = fee_rate
        self.history: list[SwapRecord] = []
        self._step = 0

    def get_price(self) -> float:
        """Marginal price of X denominated in Y (i.e. Y per 1 X)."""
        return self.reserve_y / self.reserve_x

    def get_tvl(self) -> float:
        """Total value locked, expressed in units of token Y.

        Values the X reserve at the current marginal price and adds the Y
        reserve. For a constant-product pool this is simply ``2 * reserve_y``,
        but computing it via the price keeps the intent explicit and makes it
        robust to future changes in pricing (e.g. weighted pools).
        """
        return self.reserve_x * self.get_price() + self.reserve_y

    def swap(
        self,
        token_in: str,
        amount_in: float,
        agent: str = "unknown",
        step: int | None = None,
    ) -> tuple[float, float, float]:
        """Execute a swap of ``amount_in`` of ``token_in`` into the pool.

        Args:
            token_in: Either "X" or "Y" — the token being sold into the pool.
            amount_in: Amount of ``token_in`` being sold. Must be positive.
            agent: Name/identifier of the acting agent, for logging.
            step: Simulation step number, for logging. Defaults to an
                internally tracked counter if not supplied.

        Returns:
            A tuple of ``(amount_out, effective_price, slippage)`` where
            ``amount_out`` is the amount of the other token received,
            ``effective_price`` is amount_in / amount_out expressed in Y per X
            (i.e. the average price paid for the trade), and ``slippage`` is
            the fractional deviation of the effective price from the
            pre-trade marginal price.

        Raises:
            ValueError: If inputs are invalid or the swap would drain a
                reserve to zero or below (which is impossible under the
                constant-product formula but guarded explicitly for clarity).
        """
        if token_in not in ("X", "Y"):
            raise ValueError("token_in must be 'X' or 'Y'")
        if amount_in <= 0:
            raise ValueError("amount_in must be positive")

        pre_trade_price = self.get_price()
        amount_in_after_fee = amount_in * (1 - self.fee_rate)

        if token_in == "X":
            new_reserve_x = self.reserve_x + amount_in_after_fee
            new_reserve_y = (self.reserve_x * self.reserve_y) / new_reserve_x
            amount_out = self.reserve_y - new_reserve_y

            if amount_out <= 0 or amount_out >= self.reserve_y:
                raise ValueError("Swap would drain reserve_y to zero or below")

            self.reserve_x += amount_in
            self.reserve_y -= amount_out
            # amount_in of X buys amount_out of Y -> price of X in Y terms
            effective_price = amount_out / amount_in
        else:  # token_in == "Y"
            new_reserve_y = self.reserve_y + amount_in_after_fee
            new_reserve_x = (self.reserve_x * self.reserve_y) / new_reserve_y
            amount_out = self.reserve_x - new_reserve_x

            if amount_out <= 0 or amount_out >= self.reserve_x:
                raise ValueError("Swap would drain reserve_x to zero or below")

            self.reserve_y += amount_in
            self.reserve_x -= amount_out
            # amount_in of Y buys amount_out of X -> price of X in Y terms
            effective_price = amount_in / amount_out

        slippage = (effective_price - pre_trade_price) / pre_trade_price

        resolved_step = self._step if step is None else step
        self.history.append(
            SwapRecord(
                step=resolved_step,
                agent=agent,
                token_in=token_in,
                amount_in=amount_in,
                amount_out=amount_out,
                price_after=self.get_price(),
                effective_price=effective_price,
                slippage=slippage,
            )
        )
        self._step = resolved_step + 1

        return amount_out, effective_price, slippage

    def __repr__(self) -> str:
        return (
            f"AMMPool(reserve_x={self.reserve_x:.4f}, reserve_y={self.reserve_y:.4f}, "
            f"price={self.get_price():.6f})"
        )
