"""CLI entry point for the agent safety sandbox.

Drops an agent under test into Sarina's simulation alongside the scripted
background agents, injects adversarial shocks, and prints a safety report card.

The harness is model-agnostic, so the same test suite can be pointed at models
from different labs and the results compared directly.

Examples:
    python run_safety.py --steps 60 --seed 7
    python run_safety.py --steps 60 --seed 7 --provider ollama
    python run_safety.py --steps 60 --seed 7 --compare gemini ollama claude-haiku
    python run_safety.py --steps 60 --seed 7 --compete groq-llama groq-qwen groq-gptoss

--compare runs each provider in its own isolated pool (same seed, so an
identical price path) — a controlled A/B test of one model against another.

--compete puts every named provider into ONE shared pool at the same time,
alongside the noise trader and arbitrageur, all fighting over the same
liquidity. This is a genuine market: one agent's trade moves the price the
next agent sees. Turn order is randomized each tick so no model gets a
permanent first-mover edge.
"""

from __future__ import annotations

import argparse

from puresim.agents import Arbitrageur, NoiseTrader
from puresim.amm import AMMPool
from puresim.metrics import ReportCard, build_report, render_comparison, save_tick_log
from puresim.price_feed import RandomWalkPriceFeed
from puresim.providers import PROVIDER_FACTORIES, ProviderError, build_provider
from puresim.shocks import default_schedule
from puresim.simulation import Simulation
from puresim.trading_agent import LLMAgent, StubAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an AI trading agent through the safety sandbox."
    )

    parser.add_argument("--steps", type=int, default=60, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=sorted(PROVIDER_FACTORIES),
        help="Model to put under test. Omit for the no-API stub agent.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER",
        help="Run several providers in isolated pools on an identical scenario.",
    )
    parser.add_argument(
        "--compete",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER",
        help="Run several providers as competing traders in ONE shared pool.",
    )

    parser.add_argument(
        "--initial-reserve-x", type=float, default=100_000, help="Initial pool reserve of X."
    )
    parser.add_argument(
        "--initial-reserve-y", type=float, default=100_000, help="Initial pool reserve of Y."
    )
    parser.add_argument(
        "--agent-x", type=float, default=500.0, help="Agent's starting balance of token X."
    )
    parser.add_argument(
        "--agent-y", type=float, default=500.0, help="Agent's starting balance of token Y."
    )

    parser.add_argument("--no-shocks", action="store_true", help="Disable shock injection.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tick output.")
    parser.add_argument(
        "--report-output",
        type=str,
        default="report_card.json",
        help="Where to write the report card JSON.",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default="tick_log.json",
        help="Where to write the raw per-tick log.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="If set, save a pool-vs-reference price plot here (requires matplotlib).",
    )

    return parser.parse_args()


def run_one(args: argparse.Namespace, provider_name: str | None) -> tuple[ReportCard, object]:
    """Run a single scenario end to end and build its report card.

    Constructs a fresh pool, price feed, and shock schedule each time, so that
    with a fixed seed every provider faces exactly the same scenario.
    """
    initial_price = args.initial_reserve_y / args.initial_reserve_x

    pool = AMMPool(reserve_x=args.initial_reserve_x, reserve_y=args.initial_reserve_y)
    price_feed = RandomWalkPriceFeed(initial_price=initial_price, seed=args.seed)

    scheduler = default_schedule(args.steps)
    if args.no_shocks:
        scheduler.shocks.clear()

    agent_kwargs = {
        "initial_x": args.agent_x,
        "initial_y": args.agent_y,
        "scheduler": scheduler,
        "price_feed": price_feed,
        "verbose": not args.quiet,
    }

    if provider_name is None:
        agent = StubAgent("StubAgent", **agent_kwargs)
        model_label = "none (stub)"
    else:
        provider = build_provider(provider_name)
        agent = LLMAgent("LLMAgent", provider=provider, **agent_kwargs)
        model_label = provider.label

    simulation = Simulation(
        pool=pool,
        price_feed=price_feed,
        # The agent under test acts first so that a shock lands before the
        # scripted agents get a chance to absorb it in the same step.
        agents=[agent, NoiseTrader(seed=args.seed), Arbitrageur(price_feed=price_feed)],
        num_steps=args.steps,
        seed=args.seed,
    )

    print(f"Running {args.steps} steps | agent under test: {model_label}")
    if scheduler.shocks:
        print("Scheduled shocks:")
        for shock in scheduler.shocks:
            print(f"  t={shock.step:<4} {shock.label}")
    print()

    simulation.run()
    print()

    report = build_report(
        agent=agent,
        pool=pool,
        step_logs=simulation.step_logs,
        scheduler=scheduler,
        model=model_label,
        initial_pool_x=args.initial_reserve_x,
        initial_pool_y=args.initial_reserve_y,
    )
    return report, (agent, simulation)


def run_market(
    args: argparse.Namespace, provider_names: list[str]
) -> tuple[list[ReportCard], Simulation]:
    """Run several providers as competing traders sharing one pool.

    Unlike ``run_one``, every competitor here acts on the *same* AMMPool,
    price feed, and shock scheduler in the same simulation. They are not
    measured in isolation — they trade against each other's price impact in
    real time. Providers that fail to even initialize (e.g. a missing API
    key) are skipped and simply don't join the market; the rest still compete.
    """
    initial_price = args.initial_reserve_y / args.initial_reserve_x

    pool = AMMPool(reserve_x=args.initial_reserve_x, reserve_y=args.initial_reserve_y)
    price_feed = RandomWalkPriceFeed(initial_price=initial_price, seed=args.seed)

    scheduler = default_schedule(args.steps)
    if args.no_shocks:
        scheduler.shocks.clear()

    competitors = []
    # Disambiguate if the same provider is entered twice, so pool.history and
    # the leaderboard can still tell the two instances apart.
    name_counts: dict[str, int] = {}
    for provider_name in provider_names:
        try:
            provider = build_provider(provider_name)
        except ProviderError as exc:
            print(f"  SKIPPED {provider_name}: {exc}")
            continue

        name_counts[provider_name] = name_counts.get(provider_name, 0) + 1
        suffix = "" if name_counts[provider_name] == 1 else f"-{name_counts[provider_name]}"
        agent_name = f"{provider_name}{suffix}"

        competitors.append(
            LLMAgent(
                agent_name,
                provider=provider,
                initial_x=args.agent_x,
                initial_y=args.agent_y,
                scheduler=scheduler,
                price_feed=price_feed,
                verbose=not args.quiet,
            )
        )

    if not competitors:
        raise ProviderError("no providers were runnable for --compete")

    # Turn order is randomized each tick so no competitor gets a permanent
    # first-mover edge on the shared pool.
    simulation = Simulation(
        pool=pool,
        price_feed=price_feed,
        agents=[*competitors, NoiseTrader(seed=args.seed), Arbitrageur(price_feed=price_feed)],
        num_steps=args.steps,
        randomize_order=True,
        seed=args.seed,
    )

    print(f"Running {args.steps} steps | {len(competitors)} competing agent(s):")
    for competitor in competitors:
        print(f"  {competitor.name:<20} {competitor.provider.label}")
    if scheduler.shocks:
        print("Scheduled shocks:")
        for shock in scheduler.shocks:
            print(f"  t={shock.step:<4} {shock.label}")
    print()

    simulation.run()
    print()

    reports = [
        build_report(
            agent=competitor,
            pool=pool,
            step_logs=simulation.step_logs,
            scheduler=scheduler,
            model=competitor.provider.label,
            initial_pool_x=args.initial_reserve_x,
            initial_pool_y=args.initial_reserve_y,
        )
        for competitor in competitors
    ]
    return reports, simulation


def main() -> None:
    args = parse_args()

    if args.compete:
        try:
            reports, simulation = run_market(args, args.compete)
        except ProviderError as exc:
            print(f"Could not start the market: {exc}")
            return

        for report in reports:
            print(report.render())
            print()

        print(render_comparison(reports, leaderboard=True))
        ReportCard.save_many(reports, args.report_output)
        print(f"\nWrote {args.report_output}")

        if args.plot_output:
            simulation.plot_price_history(args.plot_output)
            print(f"Saved price history plot to {args.plot_output}")
        return

    if args.compare:
        reports: list[ReportCard] = []
        for name in args.compare:
            print("=" * 66)
            try:
                report, _ = run_one(args, name)
            except ProviderError as exc:
                # A missing key for one vendor must not abort the whole
                # comparison — report it and move on to the next model.
                print(f"  SKIPPED {name}: {exc}\n")
                continue
            print(report.render())
            print()
            reports.append(report)

        if not reports:
            print("No providers were runnable. Check your API keys, or use --provider ollama.")
            return

        print(render_comparison(reports))
        ReportCard.save_many(reports, args.report_output)
        print(f"\nWrote {args.report_output}")
        return

    try:
        report, (agent, simulation) = run_one(args, args.provider)
    except ProviderError as exc:
        print(f"Could not start provider {args.provider!r}: {exc}")
        return

    print(report.render())
    report.save(args.report_output)
    save_tick_log(agent, args.log_output)
    print(f"\nWrote {args.report_output} and {args.log_output}")

    if args.plot_output:
        simulation.plot_price_history(args.plot_output)
        print(f"Saved price history plot to {args.plot_output}")


if __name__ == "__main__":
    main()
