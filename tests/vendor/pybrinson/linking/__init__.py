"""Multi-period linking methods (Cariño, GRAP, Frongello, Menchero, geometric).

All take a sequence of :class:`~pybrinson.PeriodAttribution` results
and return a :class:`~pybrinson.LinkedAttribution`. Cariño, GRAP,
Frongello and Menchero smooth arithmetic effects so that the additive
identity ``A + S + I == R_p − R_b`` (compounded) holds within ``1e-9``.
``link_geometric`` uses Bacon's geometric framework where effects
compound multiplicatively; see :func:`link_geometric` for the precise
identity satisfied.

Menchero linking ships in v1.2 after the US 7,249,082 B2 patent expired
(2024-02-18) — see :mod:`pybrinson.linking.menchero` for the patent
record and the primary references.
"""

from .carino import link_carino
from .frongello import link_frongello
from .geometric import link_geometric
from .grap import link_grap
from .menchero import link_menchero

__all__ = [
    "link_carino",
    "link_frongello",
    "link_geometric",
    "link_grap",
    "link_menchero",
]
