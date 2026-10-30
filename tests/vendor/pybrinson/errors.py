"""Exception types raised by pybrinson."""

from __future__ import annotations


class AttributionError(ValueError):
    """Raised when an attribution computation cannot be performed safely.

    Subclass of :class:`ValueError` so callers can catch via either the
    precise ``except AttributionError`` or the broad ``except ValueError``.
    pybrinson raises this — never returns a silent residual — when:

    - input weights or returns contain ``NaN`` / ``inf``;
    - portfolio or benchmark weights do not sum to 1 within tolerance;
    - the per-period attribution identity ``Σ(A + S + I) = R_p − R_b``
      fails to hold within ``1e-9`` (which would mean a bug or pathological
      input, not a data-cleaning question).
    """
