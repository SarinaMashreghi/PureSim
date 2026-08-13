"""PureSim: AMM pool simulation with background trading agents."""

from puresim.agents import Agent, Arbitrageur, NoiseTrader
from puresim.amm import AMMPool
from puresim.metrics import ReportCard, build_report, render_comparison
from puresim.price_feed import PriceFeed, RandomWalkPriceFeed
from puresim.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    Provider,
    build_provider,
)
from puresim.shocks import FakeHeadline, PriceJump, ShockScheduler, WhaleTrade
from puresim.simulation import Simulation
from puresim.trading_agent import LLMAgent, StubAgent, TradingAgent

__all__ = [
    "AMMPool",
    "Agent",
    "AnthropicProvider",
    "Arbitrageur",
    "FakeHeadline",
    "GeminiProvider",
    "LLMAgent",
    "NoiseTrader",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "PriceFeed",
    "PriceJump",
    "Provider",
    "RandomWalkPriceFeed",
    "ReportCard",
    "ShockScheduler",
    "Simulation",
    "StubAgent",
    "TradingAgent",
    "WhaleTrade",
    "build_provider",
    "build_report",
    "render_comparison",
]
