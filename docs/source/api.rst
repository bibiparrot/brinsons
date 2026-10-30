Python API
==========

All public calculation functions are separately named. The high-level
functions choose a single-period method or a multiple-period linker through
callable arguments. ``ColumnNames`` stores field roles and lets callers
control generated effect names without renaming source columns.

Calculation interfaces
----------------------

.. autofunction:: brinsons.calculate_single_period

.. autofunction:: brinsons.calculate_multiple_period

.. autoclass:: brinsons.MultiplePeriodAttribution
   :members:

.. autoclass:: brinsons.AttributionResult
   :members:

.. autoclass:: brinsons.ColumnNames
   :members:

Brinson and linking
-------------------

.. autofunction:: brinsons.bhb

.. autofunction:: brinsons.fachler

.. autofunction:: brinsons.link_carino

.. autofunction:: brinsons.link_grap

.. autofunction:: brinsons.link_frongello

.. autofunction:: brinsons.link_menchero

.. autofunction:: brinsons.link_geometric

.. autofunction:: brinsons.prepare

.. autofunction:: brinsons.roll_up

Currency
--------

.. autofunction:: brinsons.currency_attribution

.. autofunction:: brinsons.currency_premiums

.. autofunction:: brinsons.roll_up_currency_premiums

Hentschel methods
-----------------

.. autofunction:: brinsons.factor_attribution_restricted

.. autofunction:: brinsons.brinson_interaction_restricted

.. autofunction:: brinsons.brinson_interaction_reconciled

.. autofunction:: brinsons.benchmark_residual_covariance

.. autofunction:: brinsons.brinson_group_covariance

.. autofunction:: brinsons.link_precision_weighted

.. autofunction:: brinsons.link_entry_reconciled

Multi-account workflows
-----------------------

.. autoclass:: brinsons.MultiAccountColumns
   :members:

.. autoclass:: brinsons.MacroMicroAttribution
   :members:

.. autoclass:: brinsons.HierarchicalDecisionAttribution
   :members:

.. autoclass:: brinsons.LinkedMultiAccountAttribution
   :members:

.. autofunction:: brinsons.macro_micro_attribution

.. autofunction:: brinsons.value_weighted_total_attribution

.. autofunction:: brinsons.hierarchical_decision_attribution

.. autofunction:: brinsons.link_macro_micro_attribution

.. autofunction:: brinsons.link_hierarchical_decision_attribution

Behavioral Alpha Frontier research model
----------------------------------------

See :doc:`behavioral-alpha-frontier` for formulas and disclosure limits.

.. autoclass:: brinsons.BAFColumns
   :members:

.. autoclass:: brinsons.BAFPrepared
   :members:

.. autoclass:: brinsons.BAFResult
   :members:

.. autofunction:: brinsons.prepare_baf

.. autofunction:: brinsons.stock_picking_decisions

.. autofunction:: brinsons.sizing_decisions

.. autofunction:: brinsons.entry_timing_decisions

.. autofunction:: brinsons.scaling_in_decisions

.. autofunction:: brinsons.size_adjusting_decisions

.. autofunction:: brinsons.scaling_out_decisions

.. autofunction:: brinsons.exit_timing_decisions

.. autofunction:: brinsons.payoff_scale

.. autofunction:: brinsons.inverse_payoff_scale

.. autofunction:: brinsons.normalized_edge

.. autofunction:: brinsons.decision_metrics

.. autofunction:: brinsons.calculate_baf

.. autofunction:: brinsons.plot_baf

.. autofunction:: brinsons.baf_change_table

.. autofunction:: brinsons.chart_baf_change_table

Reporting
---------

The paired single-period and multiple-period report and chart functions
are listed in :doc:`reporting`. Their signatures and docstrings are generated
below from the source.

.. automodule:: brinsons.reporting
   :members:
