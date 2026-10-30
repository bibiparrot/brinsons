"""User-selected column names for attribution input and result frames."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnNames:
    """Column names retained from input and generated summary columns.

    ``portfolio_return`` and ``benchmark_return`` describe input detail columns.
    The corresponding ``summary_*`` fields identify calculated summary columns.
    """
    period: str = "period"
    segment: str = "segment"
    portfolio_weight: str = "portfolio_weight"
    benchmark_weight: str = "benchmark_weight"
    portfolio_return: str = "portfolio_return"
    benchmark_return: str = "benchmark_return"
    local_return: str = "local_return"
    currency_return: str = "currency_return"
    allocation: str = "allocation"
    selection: str = "selection"
    interaction: str = "interaction"
    summary_portfolio_return: str = "portfolio_return"
    summary_benchmark_return: str = "benchmark_return"
    summary_allocation: str = "allocation"
    summary_selection: str = "selection"
    summary_interaction: str = "interaction"
    summary_excess_return: str = "excess_return"

    @property
    def keys(self) -> tuple[str, str]:
        return self.period, self.segment

    @property
    def weights(self) -> tuple[str, str]:
        return self.portfolio_weight, self.benchmark_weight

    @property
    def returns(self) -> tuple[str, str]:
        return self.portfolio_return, self.benchmark_return

    @property
    def effects(self) -> tuple[str, str, str]:
        return self.allocation, self.selection, self.interaction

    @property
    def summary_effects(self) -> tuple[str, str, str]:
        return self.summary_allocation, self.summary_selection, self.summary_interaction
