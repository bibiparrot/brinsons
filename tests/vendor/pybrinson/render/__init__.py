"""Plain-text rendering of attribution results.

Pure stdlib formatting. Returns fixed-width tables with explicit columns
for allocation, selection, interaction and per-segment / total rows.

This subpackage sits **outside** the math core: it is excluded from the
100%-coverage gate enforced on the core (see ``pyproject.toml``
``[tool.coverage]`` configuration). Rendering bugs are aesthetic; math
bugs are not.
"""

from __future__ import annotations

from .._types import LinkedAttribution, PeriodAttribution

__all__ = ["linked_to_table", "period_to_table"]

_HEAD = ("Segment", "Allocation", "Selection", "Interaction", "Total")
_PCT_DECIMALS = 4


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.{_PCT_DECIMALS}f}%"


def _row(name: str, alloc: float, sel: float, inter: float) -> tuple[str, ...]:
    return (
        name,
        _fmt_pct(alloc),
        _fmt_pct(sel),
        _fmt_pct(inter),
        _fmt_pct(alloc + sel + inter),
    )


def _indent(name: str, level: int) -> str:
    """Indent hierarchical names by 2 spaces per level (root = no indent)."""
    return ("  " * level) + name


def _render(rows: list[tuple[str, ...]], header_lines: list[str]) -> str:
    widths = [max(len(row[col]) for row in (*rows, _HEAD)) for col in range(len(_HEAD))]
    sep = "  "

    def _line(row: tuple[str, ...]) -> str:
        return sep.join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    out: list[str] = list(header_lines)
    out.append(_line(_HEAD))
    out.append(sep.join("-" * w for w in widths))
    out.extend(_line(row) for row in rows)
    return "\n".join(out)


def period_to_table(result: PeriodAttribution) -> str:
    """Render a single-period attribution result as a fixed-width table.

    For hierarchical results, parent rows are emitted root-first with
    leaves indented under their immediate parent. The render order is:
    each parent followed immediately by its descendants in depth order.
    """
    label = result.period or "(unlabelled)"
    header_lines = [
        f"{result.method} attribution — period {label}",
        f"  R_p = {_fmt_pct(result.portfolio_return)}   "
        f"R_b = {_fmt_pct(result.benchmark_return)}   "
        f"excess = {_fmt_pct(result.excess_return)}",
        "",
    ]
    has_hierarchy = any(
        (s.parent is not None) or (not s.is_leaf) for s in result.by_segment
    )
    if has_hierarchy:
        rows = [
            _row(_indent(s.name, s.level), s.allocation, s.selection, s.interaction)
            for s in _hierarchy_render_order(result.by_segment)
        ]
    else:
        rows = [
            _row(s.name, s.allocation, s.selection, s.interaction)
            for s in result.by_segment
        ]
    rows.append(_row("Total", result.allocation, result.selection, result.interaction))
    return _render(rows, header_lines)


def _hierarchy_render_order(
    rows: tuple[PeriodAttribution, ...] | tuple,
) -> list:
    """Reorder hierarchical rows so each parent is followed by its children.

    Walks the tree breadth-first within each subtree to keep the output
    legible. Roots come first; for each root, its descendants follow in
    depth order.
    """
    children: dict[str | None, list] = {}
    for r in rows:
        children.setdefault(r.parent, []).append(r)

    out: list = []

    def walk(parent: str | None) -> None:
        for row in children.get(parent, []):
            out.append(row)
            walk(row.name)

    walk(None)
    return out


def linked_to_table(result: LinkedAttribution) -> str:
    """Render a multi-period linked attribution result."""
    header_lines = [
        f"{result.method}-linked attribution — {len(result.periods)} periods",
        f"  R_p = {_fmt_pct(result.portfolio_return)}   "
        f"R_b = {_fmt_pct(result.benchmark_return)}   "
        f"excess = {_fmt_pct(result.excess_return)}",
        "",
    ]
    rows = [_row("Total", result.allocation, result.selection, result.interaction)]
    return _render(rows, header_lines)


# Attach __repr__ overrides on the dataclasses. Done here (not in
# _types.py) to keep _types.py free of formatting concerns.
def _period_repr(self: PeriodAttribution) -> str:
    return period_to_table(self)


def _linked_repr(self: LinkedAttribution) -> str:
    return linked_to_table(self)


PeriodAttribution.__repr__ = _period_repr  # type: ignore[method-assign]
LinkedAttribution.__repr__ = _linked_repr  # type: ignore[method-assign]
