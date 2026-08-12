"""CLI entry point for running a PureSim AMM simulation.

Example:
    python run.py --steps 200 --initial-reserve-x 100000 --initial-reserve-y 100000
"""

from __future__ import annotations

import argparse

from puresim.agents import Arbitrageur, NoiseTrader
from puresim.amm import AMMPool
from puresim.price_feed import RandomWalkPriceFeed
from puresim.simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PureSim AMM pool simulation.")

    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
    parser.add_argument(
        "--initial-reserve-x", type=float, default=100_000, help="Initial reserve of token X."
    )
    parser.add_argument(
        "--initial-reserve-y", type=float, default=100_000, help="Initial reserve of token Y."
    )
    parser.add_argument(
        "--fee-rate", type=float, default=0.003, help="Trading fee rate applied on swaps (e.g. 0.003 = 0.3%%)."
    )

    parser.add_argument(
        "--noise-activity-prob",
        type=float,
        default=0.3,
        help="Probability the NoiseTrader trades on a given step.",
    )
    parser.add_argument(
        "--noise-min-size-frac",
        type=float,
        default=0.0001,
        help="Minimum NoiseTrader swap size, as a fraction of the relevant reserve.",
    )
    parser.add_argument(
        "--noise-max-size-frac",
        type=float,
        default=0.005,
        help="Maximum NoiseTrader swap size, as a fraction of the relevant reserve.",
    )

    parser.add_argument(
        "--arb-threshold",
        type=float,
        default=0.005,
        help="Price deviation (fractional) beyond which the Arbitrageur corrects the pool price.",
    )
    parser.add_argument(
        "--arb-correction-frac",
        type=float,
        default=1.0,
        help="Fraction of the full price gap the Arbitrageur attempts to close per trade.",
    )

    parser.add_argument(
        "--price-drift", type=float, default=0.0, help="Drift (per-step mean log return) of the reference price."
    )
    parser.add_argument(
        "--price-volatility",
        type=float,
        default=0.01,
        help="Volatility (per-step log return std dev) of the reference price random walk.",
    )

    parser.add_argument(
        "--randomize-agent-order",
        action="store_true",
        help="Randomize the order in which agents act each step (default: fixed order).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="If set, save a price-history plot (requires matplotlib) to this file path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    initial_price = args.initial_reserve_y / args.initial_reserve_x

    pool = AMMPool(
        reserve_x=args.initial_reserve_x,
        reserve_y=args.initial_reserve_y,
        fee_rate=args.fee_rate,
    )
    price_feed = RandomWalkPriceFeed(
        initial_price=initial_price,
        drift=args.price_drift,
        volatility=args.price_volatility,
        seed=args.seed,
    )

    noise_trader = NoiseTrader(
        activity_prob=args.noise_activity_prob,
        min_size_frac=args.noise_min_size_frac,
        max_size_frac=args.noise_max_size_frac,
        seed=args.seed,
    )
    arbitrageur = Arbitrageur(
        price_feed=price_feed,
        threshold=args.arb_threshold,
        correction_frac=args.arb_correction_frac,
    )

    simulation = Simulation(
        pool=pool,
        price_feed=price_feed,
        agents=[noise_trader, arbitrageur],
        num_steps=args.steps,
        randomize_order=args.randomize_agent_order,
        seed=args.seed,
    )

    print(f"Starting simulation: {args.steps} steps, initial price = {initial_price:.6f}")
    simulation.run()
    print(simulation.summary())

    if args.plot_output:
        simulation.plot_price_history(args.plot_output)
        print(f"\nSaved price history plot to {args.plot_output}")


if __name__ == "__main__":
    main()
