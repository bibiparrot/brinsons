"""Multi-level hierarchical roll-up of per-leaf attribution effects.

Used by ``single_period.bhb`` and ``single_period.fachler`` when the
caller passes segments with a non-``None`` ``parent`` field, optionally
together with a ``parents`` mapping describing the parent-of-parent
chain.

Hierarchy model
---------------
- The input :class:`~pybrinson.Segment` list contains **leaves only**.
  Each leaf may declare an immediate ``parent`` label (a string) or
  ``None`` (the leaf is itself a root).
- For chains deeper than one level, the caller passes a ``parents``
  mapping ``{name: parent_or_None}`` describing how each non-root parent
  label rolls up to its own parent. Roots map to ``None``.
- Roll-up is purely additive: a parent's allocation / selection /
  interaction is the sum of its descendant leaves' contributions, by
  the same construction that makes ``Σ leaves == period total``. The
  identity therefore holds at every level of the tree.

Sign / convention choice
------------------------
The aggregation is a strict additive sum with no sign flip per level.
Some practitioner write-ups (notably the GIPS *Guidance Statement on
Performance Attribution*) introduce a "child-to-parent active weight"
correction that allocates a portion of the parent-level allocation
effect down to children rather than up. pybrinson follows the simpler,
more common convention used by Bacon (2008) chap. 5 and the CFA
Institute Refresher Reading on attribution: per-level effects are
straight sums of children's effects. The choice is documented inline
here so the reader can audit the math without leaving the file.

References:
----------
- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley, chap. 5 (multi-level Brinson model).
  https://www.wiley.com/en-us/Practical+Portfolio+Performance+Measurement+and+Attribution%2C+2nd+Edition-p-9780470059289
- CFA Institute Refresher Reading "Portfolio Performance Evaluation":
  https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/performance-attribution
- Spaulding, D. (2003). *Investment Performance Attribution*, McGraw-Hill,
  chap. 7 (multi-level decomposition).
  https://www.spauldinggrp.com/product/investment-performance-attribution-book/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import Segment, SegmentAttribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
from .errors import AttributionError

_MAX_HIERARCHY_DEPTH = 32


def needs_hierarchy(
    segments: Sequence[Segment],
    parents: Mapping[str, str | None] | None,
) -> bool:
    """True if the input declares any hierarchical structure."""
    if parents:
        return True
    return any(s.parent is not None for s in segments)


def _walk_chain(
    immediate: str | None,
    parents: Mapping[str, str | None],
    *,
    leaf_name: str,
) -> list[str]:
    """Return the ancestor chain (immediate parent first) up to the root.

    Raises on cycles or chains exceeding ``_MAX_HIERARCHY_DEPTH``.
    """
    chain: list[str] = []
    seen: set[str] = set()
    current = immediate
    while current is not None:
        if current in seen:
            raise AttributionError(
                f"hierarchy cycle detected at parent label {current!r} "
                f"while walking chain from leaf {leaf_name!r}"
            )
        if len(chain) >= _MAX_HIERARCHY_DEPTH:
            raise AttributionError(
                f"hierarchy depth exceeds {_MAX_HIERARCHY_DEPTH} levels "
                f"while walking chain from leaf {leaf_name!r} — likely a "
                "missing terminator in the parents mapping"
            )
        chain.append(current)
        seen.add(current)
        # If a parent label has no entry in `parents`, it is a root.
        current = parents.get(current) if current in parents else None
    return chain


def _build_chain_cache(
    segments: Sequence[Segment],
    parents_map: Mapping[str, str | None],
) -> dict[str | None, list[str]]:
    """Walk every leaf chain once, caching by immediate-parent name."""
    cache: dict[str | None, list[str]] = {}
    for s in segments:
        if s.parent in cache:
            continue
        cache[s.parent] = (
            _walk_chain(s.parent, parents_map, leaf_name=s.name)
            if s.parent is not None
            else []
        )
    return cache


def _validate_reachability(
    chain_cache: dict[str | None, list[str]],
    parents_map: Mapping[str, str | None],
) -> None:
    """Ensure every parent label is reachable from at least one leaf."""
    reachable: set[str] = set()
    for chain in chain_cache.values():
        reachable.update(chain)
    for label in parents_map:
        if label not in reachable:
            raise AttributionError(
                f"parent label {label!r} in `parents` mapping is not reachable "
                "from any leaf segment — drop the dangling entry or fix the "
                "leaf classification"
            )


def roll_up(
    segments: Sequence[Segment],
    leaf_attrs: Sequence[SegmentAttribution],
    parents: Mapping[str, str | None] | None,
) -> tuple[SegmentAttribution, ...]:
    """Compute parent-level :class:`SegmentAttribution` rows.

    Returns the leaf rows (with ``parent`` / ``level`` populated) followed
    by every parent row in root-first depth order. Each parent row's
    allocation / selection / interaction is the sum of its descendant
    leaves' values, so the period-total identity is preserved at every
    level by construction.
    """
    if len(segments) != len(leaf_attrs):
        raise AttributionError(
            "internal: roll_up() called with mismatched segment/leaf lengths"
        )

    parents_map: Mapping[str, str | None] = parents or {}
    chain_cache = _build_chain_cache(segments, parents_map)
    _validate_reachability(chain_cache, parents_map)

    parent_alloc: dict[str, float] = {}
    parent_sel: dict[str, float] = {}
    parent_inter: dict[str, float] = {}
    parent_of: dict[str, str | None] = {}
    parent_level: dict[str, int] = {}

    leaves_with_chain: list[tuple[Segment, SegmentAttribution, list[str]]] = []
    for seg, leaf in zip(segments, leaf_attrs, strict=True):
        chain = chain_cache[seg.parent]
        leaves_with_chain.append((seg, leaf, chain))
        for ancestor in chain:
            parent_alloc[ancestor] = parent_alloc.get(ancestor, 0.0) + leaf.allocation
            parent_sel[ancestor] = parent_sel.get(ancestor, 0.0) + leaf.selection
            parent_inter[ancestor] = parent_inter.get(ancestor, 0.0) + leaf.interaction
        for depth_from_immediate, ancestor in enumerate(chain):
            ancestors_above = chain[depth_from_immediate + 1 :]
            level = len(ancestors_above)
            parent_level[ancestor] = level
            parent_of[ancestor] = ancestors_above[0] if ancestors_above else None

    leaf_rows = [
        SegmentAttribution(
            name=leaf.name,
            allocation=leaf.allocation,
            selection=leaf.selection,
            interaction=leaf.interaction,
            parent=seg.parent,
            level=len(chain),
            is_leaf=True,
        )
        for seg, leaf, chain in leaves_with_chain
    ]

    parent_rows: list[SegmentAttribution] = []
    for level in range(_MAX_HIERARCHY_DEPTH + 1):
        parent_rows.extend(
            SegmentAttribution(
                name=name,
                allocation=parent_alloc[name],
                selection=parent_sel[name],
                interaction=parent_inter[name],
                parent=parent_of[name],
                level=level,
                is_leaf=False,
            )
            for name in parent_alloc
            if parent_level[name] == level
        )

    return tuple(parent_rows + leaf_rows)
