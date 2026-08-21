"""Fast-path deterministic execution constants.

Closed sets for outcome types, trigger types, trigger states, annotation
statuses, and narration sources.  These are static domain values — runtime
feature flags live in ``utils/gate_config.py``.

See: .kiro/specs/fast-path-deterministic-execution/design.md
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Outcome types — the six explicit results of trigger evaluation.
# (Requirement 3.1)
# ---------------------------------------------------------------------------

OUTCOME_TYPES: frozenset[str] = frozenset(
    {
        "watch_created",
        "watch_promoted",
        "pending_order_created",
        "trade_executed",
        "missed_move",
        "stand_down",
    }
)

# ---------------------------------------------------------------------------
# Trigger types — how a trigger condition is defined.
# ---------------------------------------------------------------------------

TRIGGER_TYPES: frozenset[str] = frozenset(
    {
        "entry_zone",
        "level_break",
        "level_reject",
        "vwap_cross",
        "price_target",
    }
)

# ---------------------------------------------------------------------------
# Trigger states — lifecycle of a registered trigger.
# FIRED is terminal; resolved_at timestamp marks completion (no separate
# RESOLVED state).
# ---------------------------------------------------------------------------

TRIGGER_STATES: frozenset[str] = frozenset(
    {
        "active",
        "fired",
        "expired",
        "invalidated",
    }
)

# ---------------------------------------------------------------------------
# Annotation statuses — tracks LLM annotation lifecycle per event.
# (Requirement 7.4)
# ---------------------------------------------------------------------------

ANNOTATION_STATUSES: frozenset[str] = frozenset(
    {
        "annotated",
        "annotation_pending",
        "annotation_failed",
        "annotation_skipped",
    }
)

# ---------------------------------------------------------------------------
# Narration sources — how the public-stream narration was produced.
# (Requirement 8.5)
# ---------------------------------------------------------------------------

NARRATION_SOURCES: frozenset[str] = frozenset(
    {
        "template",
        "llm_enriched",
    }
)

# ---------------------------------------------------------------------------
# Churn protection constants
# ---------------------------------------------------------------------------

# Rolling window (minutes) over which repeated stand_downs for the same
# symbol+setup_type are counted for churn detection.
CHURN_WINDOW_MINUTES: int = 10

# Maximum number of stand_down outcomes within the churn window before the
# cooldown system blocks further triggers for that symbol+setup_type.
CHURN_MAX_STANDDOWNS: int = 3
