"""
STRATEGY RESEARCH LAB
Research-only framework.
No broker connection. No orders.

Purpose:
- Keep every strategy experiment comparable.
- Record hypotheses and results.
- Enforce realistic costs and validation discipline.
- Provide a common structure for PBD, momentum, mean-reversion,
  breakout/retest, Market Profile, and future strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResearchConfig:
    starting_equity: float = 10_000.00
    risk_per_trade: float = 0.005
    max_open_positions: int = 5
    max_position_fraction: float = 0.20
    round_trip_cost: float = 0.001

    # Validation periods must remain separate.
    train_days: int = 40
    validation_days: int = 20
    test_days: int = 20

    # Robustness requirements.
    minimum_trades: int = 30
    minimum_profit_factor: float = 1.20
    maximum_drawdown: float = 0.10


@dataclass(frozen=True)
class StrategyExperiment:
    name: str
    version: str
    hypothesis: str
    entry_logic: str
    exit_logic: str
    market_regimes: str
    symbols: str
    data_source: str = "historical snapshot"
    status: str = "RESEARCH"


@dataclass
class ResearchResult:
    experiment: StrategyExperiment
    trades: int
    net_pnl: float
    net_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    estimated_costs: float
    passed_minimum_trades: bool
    passed_profit_factor: bool
    passed_drawdown: bool

    @property
    def robust_candidate(self) -> bool:
        return (
            self.passed_minimum_trades
            and self.passed_profit_factor
            and self.passed_drawdown
        )


LAB_DIR = Path("research_lab")
EXPERIMENTS_DIR = LAB_DIR / "experiments"
RESULTS_DIR = LAB_DIR / "results"

for directory in (LAB_DIR, EXPERIMENTS_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)


def print_config(config: ResearchConfig) -> None:
    print("=" * 60)
    print("STRATEGY RESEARCH LAB")
    print("=" * 60)
    print("Research only — no broker connection and no orders.")
    print()
    print("Risk configuration")
    print(f"Starting equity:        ${config.starting_equity:,.2f}")
    print(f"Risk per trade:         {config.risk_per_trade:.2%}")
    print(f"Max open positions:     {config.max_open_positions}")
    print(f"Max position fraction:  {config.max_position_fraction:.2%}")
    print(f"Round-trip cost:        {config.round_trip_cost:.2%}")
    print()
    print("Validation discipline")
    print(f"Training window:        {config.train_days} days")
    print(f"Validation window:      {config.validation_days} days")
    print(f"Out-of-sample window:   {config.test_days} days")
    print(f"Minimum trades:         {config.minimum_trades}")
    print(f"Minimum profit factor:  {config.minimum_profit_factor:.2f}")
    print(f"Maximum drawdown:       {config.maximum_drawdown:.2%}")
    print("=" * 60)


def print_experiment(experiment: StrategyExperiment) -> None:
    print()
    print("EXPERIMENT")
    print(f"Name:            {experiment.name}")
    print(f"Version:         {experiment.version}")
    print(f"Hypothesis:      {experiment.hypothesis}")
    print(f"Entry logic:     {experiment.entry_logic}")
    print(f"Exit logic:      {experiment.exit_logic}")
    print(f"Market regimes:  {experiment.market_regimes}")
    print(f"Symbols:         {experiment.symbols}")
    print(f"Data source:     {experiment.data_source}")
    print(f"Status:          {experiment.status}")


def main() -> None:
    config = ResearchConfig()

    experiment = StrategyExperiment(
        name="PBD Pullback",
        version="0.1",
        hypothesis=(
            "Waiting for a breakout followed by a controlled drift/pullback "
            "and confirmation may reduce false breakout entries."
        ),
        entry_logic=(
            "Push -> Break -> Drift/Pullback -> Confirmation -> Entry"
        ),
        exit_logic="Research placeholder; defined by each experiment.",
        market_regimes="Trending / expanding volatility",
        symbols="Current 20-symbol research universe",
    )

    print_config(config)
    print_experiment(experiment)

    print()
    print("Framework status: READY")
    print(f"Lab directory: {LAB_DIR.resolve()}")
    print()
    print("Next step:")
    print("Implement the PBD setup without changing the existing scanner.")
    print("The current -0.80% exact-scanner test remains the baseline.")


if __name__ == "__main__":
    main()
