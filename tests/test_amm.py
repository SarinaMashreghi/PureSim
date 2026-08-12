"""Unit tests for AMMPool math."""

from __future__ import annotations

import math

import pytest

from puresim.amm import AMMPool


def test_initial_price():
    pool = AMMPool(reserve_x=100_000, reserve_y=200_000)
    assert pool.get_price() == pytest.approx(2.0)


def test_constructor_rejects_nonpositive_reserves():
    with pytest.raises(ValueError):
        AMMPool(reserve_x=0, reserve_y=100)
    with pytest.raises(ValueError):
        AMMPool(reserve_x=100, reserve_y=-5)


def test_swap_rejects_nonpositive_amount():
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    with pytest.raises(ValueError):
        pool.swap("X", 0)
    with pytest.raises(ValueError):
        pool.swap("X", -10)


def test_swap_rejects_invalid_token():
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    with pytest.raises(ValueError):
        pool.swap("Z", 10)


def test_swap_increases_k_due_to_fee():
    """The constant-product invariant k = x*y should not decrease after a
    swap; it should strictly increase because fees accrue to the pool."""
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000, fee_rate=0.003)
    k_before = pool.reserve_x * pool.reserve_y

    pool.swap("X", 1_000)
    k_after = pool.reserve_x * pool.reserve_y

    assert k_after > k_before


def test_swap_x_in_moves_price_down():
    """Selling X into the pool increases reserve_x relative to reserve_y,
    so the price of X (in Y) should decrease."""
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    price_before = pool.get_price()

    pool.swap("X", 1_000)

    assert pool.get_price() < price_before


def test_swap_y_in_moves_price_up():
    """Selling Y into the pool increases reserve_y relative to reserve_x,
    so the price of X (in Y) should increase."""
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    price_before = pool.get_price()

    pool.swap("Y", 1_000)

    assert pool.get_price() > price_before


def test_swap_reserve_updates_are_consistent():
    pool = AMMPool(reserve_x=50_000, reserve_y=200_000)
    reserve_x_before, reserve_y_before = pool.reserve_x, pool.reserve_y

    amount_in = 2_000
    amount_out, _effective_price, _slippage = pool.swap("X", amount_in)

    assert pool.reserve_x == pytest.approx(reserve_x_before + amount_in)
    assert pool.reserve_y == pytest.approx(reserve_y_before - amount_out)
    assert amount_out > 0


def test_fee_reduces_amount_out_relative_to_no_fee():
    reserve_x, reserve_y = 100_000, 100_000
    amount_in = 5_000

    pool_with_fee = AMMPool(reserve_x=reserve_x, reserve_y=reserve_y, fee_rate=0.003)
    pool_no_fee = AMMPool(reserve_x=reserve_x, reserve_y=reserve_y, fee_rate=0.0)

    amount_out_fee, _, _ = pool_with_fee.swap("X", amount_in)
    amount_out_no_fee, _, _ = pool_no_fee.swap("X", amount_in)

    assert amount_out_fee < amount_out_no_fee


def test_zero_fee_matches_exact_constant_product_formula():
    reserve_x, reserve_y = 100_000, 250_000
    amount_in = 10_000
    pool = AMMPool(reserve_x=reserve_x, reserve_y=reserve_y, fee_rate=0.0)

    expected_amount_out = reserve_y - (reserve_x * reserve_y) / (reserve_x + amount_in)
    amount_out, _, _ = pool.swap("X", amount_in)

    assert amount_out == pytest.approx(expected_amount_out)


def test_slippage_is_positive_for_x_in_and_y_out_terms():
    """Buying Y with X (i.e. selling X) should execute at a worse (lower)
    average price of X than the pre-trade marginal price, since larger
    trades move the price against the trader — so effective_price should be
    below the marginal price, giving negative slippage for the seller."""
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    _amount_out, effective_price, slippage = pool.swap("X", 5_000)

    assert effective_price < 1.0  # pre-trade price was 1.0
    assert slippage < 0


def test_history_logs_swap_details():
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000)
    pool.swap("X", 1_000, agent="TestAgent", step=7)

    assert len(pool.history) == 1
    record = pool.history[0]
    assert record.step == 7
    assert record.agent == "TestAgent"
    assert record.token_in == "X"
    assert record.amount_in == 1_000
    assert record.amount_out > 0
    assert record.price_after == pytest.approx(pool.get_price())


def test_large_swap_approaches_but_never_reaches_full_reserve():
    """The constant-product formula asymptotically approaches (but for any
    finite input, never exactly reaches) draining a reserve to zero, so a
    merely large trade should succeed without raising."""
    pool = AMMPool(reserve_x=100, reserve_y=100)
    amount_out, _, _ = pool.swap("Y", 1_000_000)
    assert 0 < amount_out < 100


def test_swap_raises_when_amount_would_drain_reserve():
    """An astronomically large trade underflows the opposing reserve to
    (effectively) zero in floating point, which should be rejected rather
    than silently draining the pool."""
    pool = AMMPool(reserve_x=100, reserve_y=100)
    with pytest.raises(ValueError):
        pool.swap("Y", 1e30)


def test_get_tvl():
    pool = AMMPool(reserve_x=100_000, reserve_y=200_000)
    # TVL = reserve_x * price + reserve_y = 100_000 * 2 + 200_000 = 400_000
    assert pool.get_tvl() == pytest.approx(400_000)


def test_round_trip_leaves_trader_worse_off_due_to_fee():
    """A round trip (sell X for Y, then sell that exact Y back for X) should
    return strictly less X than was originally sold, since fees are taken
    on both legs. Note: get_tvl() is self-referentially defined in terms of
    the pool's own marginal price (reserve_x * price + reserve_y ==
    2 * reserve_y), so it cannot detect this — reserve_x is the direct,
    reliable signal that fee value accrued to the pool."""
    pool = AMMPool(reserve_x=100_000, reserve_y=100_000, fee_rate=0.003)
    reserve_x_before = pool.reserve_x

    amount_sold = 1_000
    pool.swap("X", amount_sold)
    amount_y = pool.history[-1].amount_out
    amount_x_returned, _, _ = pool.swap("Y", amount_y)

    assert amount_x_returned < amount_sold
    assert pool.reserve_x > reserve_x_before


def test_k_invariant_holds_approximately_pre_fee():
    """Even with fees, reserves after a swap should satisfy the
    post-fee constant product relation exactly (not approximately loosely)."""
    reserve_x, reserve_y = 100_000, 100_000
    fee_rate = 0.003
    pool = AMMPool(reserve_x=reserve_x, reserve_y=reserve_y, fee_rate=fee_rate)

    amount_in = 3_000
    amount_in_after_fee = amount_in * (1 - fee_rate)
    pool.swap("X", amount_in)

    k_original = reserve_x * reserve_y
    # (x + amount_in_after_fee) * (y - amount_out) should equal the original k
    implied_k = (reserve_x + amount_in_after_fee) * pool.reserve_y
    assert implied_k == pytest.approx(k_original, rel=1e-9)
