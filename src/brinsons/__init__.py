"""Expression-based Brinson attribution with Polars DataFrame inputs and outputs."""
from .columns import ColumnNames
from .core import (
    AttributionError, AttributionResult, bhb, fachler, currency_attribution,
    currency_premiums, link_carino, link_grap, link_frongello, link_menchero,
    link_geometric, prepare, roll_up, roll_up_currency_premiums,
)

__all__ = [
    "ColumnNames", "AttributionError", "AttributionResult", "bhb", "fachler",
    "currency_attribution", "currency_premiums", "link_carino", "link_grap",
    "link_frongello", "link_menchero", "link_geometric", "prepare", "roll_up",
    "roll_up_currency_premiums",
]
from .hentschel import (
    factor_attribution_restricted, brinson_interaction_restricted,
    brinson_interaction_reconciled, link_precision_weighted, link_entry_reconciled,
)

__all__ += ["factor_attribution_restricted", "brinson_interaction_restricted",
            "brinson_interaction_reconciled", "link_precision_weighted", "link_entry_reconciled"]
from .hentschel import brinson_group_covariance
__all__.append("brinson_group_covariance")
from .hentschel import benchmark_residual_covariance
__all__.append("benchmark_residual_covariance")
from .reporting import (
    print_category_weights, print_category_returns, print_category_attribution,
    print_total_attribution, print_attribution_summary,
    chart_category_returns, chart_category_weights, chart_category_attribution,
)
__all__ += ["print_category_weights", "print_category_returns", "print_category_attribution",
            "print_total_attribution", "print_attribution_summary", "chart_category_returns",
            "chart_category_weights", "chart_category_attribution"]

from .reporting import (
    print_single_period_category_weights,
    print_multiple_period_category_weights,
    print_single_period_category_returns,
    print_multiple_period_category_returns,
    print_single_period_category_attribution,
    print_multiple_period_category_attribution,
    print_single_period_total_attribution,
    print_multiple_period_total_attribution,
    print_single_period_attribution_summary,
    print_multiple_period_attribution_summary,
    chart_single_period_category_returns,
    chart_multiple_period_category_returns,
    chart_single_period_category_weights,
    chart_multiple_period_category_weights,
    chart_single_period_category_attribution,
    chart_multiple_period_category_attribution,
)
__all__ += [
    "print_single_period_category_weights",
    "print_multiple_period_category_weights",
    "print_single_period_category_returns",
    "print_multiple_period_category_returns",
    "print_single_period_category_attribution",
    "print_multiple_period_category_attribution",
    "print_single_period_total_attribution",
    "print_multiple_period_total_attribution",
    "print_single_period_attribution_summary",
    "print_multiple_period_attribution_summary",
    "chart_single_period_category_returns",
    "chart_multiple_period_category_returns",
    "chart_single_period_category_weights",
    "chart_multiple_period_category_weights",
    "chart_single_period_category_attribution",
    "chart_multiple_period_category_attribution",
]
from .reporting import chart_single_period_excess_return_waterfall, chart_multiple_period_excess_return_waterfall
__all__ += ["chart_single_period_excess_return_waterfall", "chart_multiple_period_excess_return_waterfall"]
from .workflow import MultiplePeriodAttribution, calculate_single_period, calculate_multiple_period
__all__ += ["MultiplePeriodAttribution", "calculate_single_period", "calculate_multiple_period"]

from .multi_account import (
    MultiAccountColumns, MacroMicroAttribution, HierarchicalDecisionAttribution,
    LinkedMultiAccountAttribution,
    macro_micro_attribution, value_weighted_total_attribution,
    hierarchical_decision_attribution,
    link_macro_micro_attribution, link_hierarchical_decision_attribution,
)

__all__ += [
    "MultiAccountColumns", "MacroMicroAttribution", "HierarchicalDecisionAttribution",
    "LinkedMultiAccountAttribution",
    "macro_micro_attribution", "value_weighted_total_attribution",
    "hierarchical_decision_attribution",
    "link_macro_micro_attribution", "link_hierarchical_decision_attribution",
]

from .baf import (
    BAFColumns, BAFError, BAFPrepared, BAFResult, prepare_baf,
    stock_picking_decisions, sizing_decisions, entry_timing_decisions,
    scaling_in_decisions, size_adjusting_decisions, scaling_out_decisions,
    exit_timing_decisions, payoff_scale, inverse_payoff_scale, normalized_edge,
    decision_metrics, calculate_baf, plot_baf,
    baf_change_table, chart_baf_change_table,
)

__all__ += [
    "BAFColumns", "BAFError", "BAFPrepared", "BAFResult", "prepare_baf",
    "stock_picking_decisions", "sizing_decisions", "entry_timing_decisions",
    "scaling_in_decisions", "size_adjusting_decisions", "scaling_out_decisions",
    "exit_timing_decisions", "payoff_scale", "inverse_payoff_scale", "normalized_edge",
    "decision_metrics", "calculate_baf", "plot_baf",
    "baf_change_table", "chart_baf_change_table",
]
