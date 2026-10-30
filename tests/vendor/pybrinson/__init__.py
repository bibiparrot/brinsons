"""Portfolio return attribution: BHB, Brinson-Fachler, multi-period linking.

Public API
----------
Single-period attribution:

- :func:`bhb` — Brinson-Hood-Beebower (3-effect)
- :func:`fachler` — Brinson-Fachler (3-effect)

Multi-period linking:

- :func:`link_carino` — Cariño (1999) log-smoothing
- :func:`link_grap` — GRAP (1997) factor linking
- :func:`link_frongello` — Frongello (2002) recursive linking
- :func:`link_menchero` — Menchero (2000, 2004) optimised linking
- :func:`link_geometric` — Bacon-style geometric linking

Data model:

- :class:`Segment` — input row
- :class:`SegmentAttribution`, :class:`PeriodAttribution`,
  :class:`LinkedAttribution` — result types

Errors:

- :class:`AttributionError` — raised on bad input or identity failure
  (subclass of :class:`ValueError`)
"""

from ._types import (
    CurrencyPeriodAttribution,
    CurrencySegmentAttribution,
    LinkedAttribution,
    PeriodAttribution,
    Segment,
    SegmentAttribution,
)
from .errors import AttributionError
from .linking import (
    link_carino,
    link_frongello,
    link_geometric,
    link_grap,
    link_menchero,
)
from .render import (
    linked_to_table,
    period_to_table,
)
from .single_period import bhb, currency_attribution, fachler

__version__ = "1.3.1"

__all__ = [
    "AttributionError",
    "CurrencyPeriodAttribution",
    "CurrencySegmentAttribution",
    "LinkedAttribution",
    "PeriodAttribution",
    "Segment",
    "SegmentAttribution",
    "__version__",
    "bhb",
    "currency_attribution",
    "fachler",
    "link_carino",
    "link_frongello",
    "link_geometric",
    "link_grap",
    "link_menchero",
    "linked_to_table",
    "period_to_table",
]
