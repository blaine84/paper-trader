"""Candidate Prompt Builder — constructs PM prompt and LLM schema for candidate selection.

Builds the PM prompt with a candidate summary table and decision instructions,
plus a dynamic LLM structured output schema that constrains candidate_ids to
the exact offered set.

See: design.md §PM Prompt Builder
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.1
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_candidate_pm_prompt(
    candidate_summaries: list[dict],
    portfolio_summary: dict,
    profile: dict,
    profile_id: str,
) -> str:
    """Build the PM prompt for candidate-ID selection.

    Args:
        candidate_summaries: List of summary dicts from registry.get_offered_summary()
            Each has: candidate_id, symbol, direction, setup_type, entry_price,
            stop_price, target_price, risk_reward, geometry_name, trigger,
            invalidation_basis, target_basis
        portfolio_summary: Dict with portfolio state: cash, total_equity, positions, daily_pnl
        profile: PM profile dict with personality, max_positions, etc.
        profile_id: Profile identifier

    Returns:
        Formatted prompt string for the PM LLM call.
    """
    # Build candidate table
    table_lines = []
    table_lines.append(
        "| # | candidate_id | Symbol | Dir | Entry | Stop | Target | R:R | Setup | Geometry | Trigger | Invalidation | Target Basis |"
    )
    table_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(candidate_summaries, 1):
        trigger_text = c.get("trigger", "") or ""
        invalidation_text = c.get("invalidation_basis", "") or ""
        target_basis_text = c.get("target_basis", "") or ""
        table_lines.append(
            f"| {i} | {c['candidate_id']} | {c['symbol']} | {c['direction']} "
            f"| ${c['entry_price']:.2f} | ${c['stop_price']:.2f} | ${c['target_price']:.2f} "
            f"| {c['risk_reward']:.1f}:1 | {c['setup_type']} | {c.get('geometry_name', '')} "
            f"| {trigger_text[:80]} | {invalidation_text[:80]} | {target_basis_text[:80]} |"
        )
    candidate_table = "\n".join(table_lines)

    # Build portfolio summary
    cash = portfolio_summary.get("cash", 0)
    equity = portfolio_summary.get("total_equity", 0)
    positions = portfolio_summary.get("positions", [])
    num_positions = len(positions) if isinstance(positions, (list, dict)) else 0
    max_positions = profile.get("max_positions", 3)

    portfolio_text = (
        f"Portfolio: ${equity:,.0f} equity, ${cash:,.0f} cash, "
        f"{num_positions}/{max_positions} positions open"
    )

    # Build full prompt
    prompt = f"""You are the {profile.get('name', profile_id)} Portfolio Manager.

{portfolio_text}

## Available Candidates

{candidate_table}

## Instructions

You are selecting from the candidates above by their candidate_id ONLY.

For each candidate, decide: accept (take the trade) or reject (pass on it).

Rules:
- Select by candidate_id only — do NOT specify symbols, prices, quantities, or sector labels
- A candidate not in this list cannot be traded
- Categories, themes, and sector concepts are for commentary only — they are not executable
- The table above is the complete executable candidate set; Entry, Stop, Target, R:R, Setup, Trigger, Invalidation, and Target Basis are already provided by the deterministic scaffold
- Do NOT reject a candidate because setup data, entry, stop, target, risk/reward, trigger, invalidation, or target basis is missing
- An empty accepted set is a valid response — passing on all candidates is acceptable
- If you accept a candidate, you may optionally specify a risk_multiplier (0.01 to 1.0) to reduce position size
- If you reject a candidate, cite concrete portfolio, timing, exposure, confidence, or market-quality criteria from your PM profile

Respond with your decisions in the required JSON format.
"""
    return prompt


def build_decision_schema(candidate_ids: set[str]) -> dict:
    """Build LLM structured output schema with dynamic candidate_id enum.

    The schema constrains the response to only valid candidate_ids from the
    current cycle, with additionalProperties: false at all levels.

    Args:
        candidate_ids: Set of valid candidate_ids for the current PM cycle.

    Returns:
        JSON schema dict suitable for LLM structured output constraints.
    """
    sorted_ids = sorted(candidate_ids)

    schema = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "enum": sorted_ids,
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["accept", "reject"],
                        },
                        "risk_multiplier": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 1.0,
                        },
                        "rationale": {
                            "type": "string",
                            "maxLength": 280,
                        },
                    },
                    "required": ["candidate_id", "decision"],
                    "additionalProperties": False,
                },
            },
            "portfolio_notes": {
                "type": "string",
                "maxLength": 420,
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }

    return schema
