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

from utils.gate_config import PM_PREFLIGHT_OBSERVE_MODE

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
            invalidation_basis, target_basis, multitimeframe_context
        portfolio_summary: Dict with portfolio state: cash, total_equity, positions, daily_pnl
        profile: PM profile dict with personality, max_positions, etc.
        profile_id: Profile identifier

    Returns:
        Formatted prompt string for the PM LLM call.
    """
    # Build candidate table
    table_lines = []
    table_lines.append(
        "| # | candidate_id | Symbol | Dir | Entry | Stop | Target | R:R | Setup | Geometry | MTF | Trigger | Invalidation | Target Basis | Horizon |"
    )
    table_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(candidate_summaries, 1):
        trigger_text = c.get("trigger", "") or ""
        invalidation_text = c.get("invalidation_basis", "") or ""
        target_basis_text = c.get("target_basis", "") or ""
        mtf_text = _format_mtf_summary(c.get("multitimeframe_context"))
        # Display holding horizon for swing candidates (integer days), blank for intraday
        holding_horizon = c.get("holding_horizon")
        horizon_text = f"{holding_horizon}d" if holding_horizon else ""
        table_lines.append(
            f"| {i} | {c['candidate_id']} | {c['symbol']} | {c['direction']} "
            f"| ${c['entry_price']:.2f} | ${c['stop_price']:.2f} | ${c['target_price']:.2f} "
            f"| {c['risk_reward']:.1f}:1 | {c['setup_type']} | {c.get('geometry_name', '')} "
            f"| {mtf_text[:180]} | {trigger_text[:80]} | {invalidation_text[:80]} "
            f"| {target_basis_text[:80]} | {horizon_text} |"
        )
    candidate_table = "\n".join(table_lines)
    candidate_details = "\n".join(
        _format_candidate_detail(i, c) for i, c in enumerate(candidate_summaries, 1)
    )

    # Build preflight attestation block (only when preflight is active)
    attestation_lines = []
    if PM_PREFLIGHT_OBSERVE_MODE != "observe":
        for c in candidate_summaries:
            entry = c.get("entry_price", 0)
            stop = c.get("stop_price", 0)
            target = c.get("target_price", 0)
            rr = c.get("risk_reward", 0)
            cid = c["candidate_id"]
            candidate_type = c.get("candidate_type", "intraday") or "intraday"
            setup_type = c.get("setup_type", "")
            horizon_label = "Swing" if candidate_type == "swing" else "Intraday"
            attestation_lines.append(
                f"✓ VALIDATED: Entry (${entry:.2f}), Stop (${stop:.2f}), "
                f"Target (${target:.2f}), R:R ({rr:.2f}), Candidate ID {cid} verified. "
                f"| Holding Horizon: {horizon_label} | Setup Type: {setup_type}"
            )

    attestation_block = ""
    if attestation_lines:
        attestation_block = (
            "\n\n## Preflight Attestation\n\n"
            + "\n".join(attestation_lines)
        )

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

    # Rejection guidance instruction (Requirements 4.1, 4.4)
    rejection_guidance = (
        "IMPORTANT: Valid reasons for rejection are market quality, profile fit, "
        "timing, exposure, or risk. "
        "Do NOT reject candidates for missing executable geometry — entry, stop, "
        "target, and R:R have been verified."
    )

    # Build full prompt
    prompt = f"""You are the {profile.get('name', profile_id)} Portfolio Manager.

{portfolio_text}

## Available Candidates

{candidate_table}{attestation_block}

## Candidate Details

{candidate_details}

## Instructions

You are selecting from the candidates above by their candidate_id ONLY.

For each candidate, decide: accept (take the trade) or reject (pass on it).

{rejection_guidance}

Rules:
- Select by candidate_id only — do NOT specify symbols, prices, quantities, or sector labels
- A candidate not in this list cannot be traded
- Categories, themes, and sector concepts are for commentary only — they are not executable
- The table above is the complete executable candidate set; Entry, Stop, Target, R:R, Setup, Trigger, Invalidation, and Target Basis are already provided by the deterministic scaffold
- Do NOT reject a candidate because setup data, entry, stop, target, risk/reward, trigger, invalidation, or target basis is missing
- Use the MTF column as shared Analyst context: it summarizes 5m/60m/daily trend, relative strength, volume, and directional alignment
- An empty accepted set is a valid response — passing on all candidates is acceptable
- If you accept a candidate, you may optionally specify a risk_multiplier (0.01 to 1.0) to reduce position size
- If you reject a candidate, cite concrete portfolio, timing, exposure, confidence, or market-quality criteria from your PM profile

Respond with your decisions in the required JSON format.
"""
    return prompt


def _format_candidate_detail(index: int, candidate: dict) -> str:
    """Repeat executable trade specs in model-friendly prose.

    The wide table is useful for operators, but local models occasionally miss
    individual columns. Keep a redundant plain-text block so entry geometry is
    hard to overlook during accept/reject decisions.
    """
    trigger_text = _clip_detail(candidate.get("trigger") or "n/a")
    invalidation_text = _clip_detail(candidate.get("invalidation_basis") or "n/a")
    target_basis_text = _clip_detail(candidate.get("target_basis") or "n/a")
    mtf_text = _format_mtf_summary(candidate.get("multitimeframe_context"))
    holding_horizon = candidate.get("holding_horizon")
    horizon_text = f"; horizon={holding_horizon}d" if holding_horizon else ""

    return (
        f"{index}. candidate_id={candidate['candidate_id']}; "
        f"symbol={candidate['symbol']}; direction={candidate['direction']}; "
        f"entry=${candidate['entry_price']:.2f}; stop=${candidate['stop_price']:.2f}; "
        f"target=${candidate['target_price']:.2f}; "
        f"risk_reward={candidate['risk_reward']:.1f}:1; "
        f"setup={candidate['setup_type']}; geometry={candidate.get('geometry_name', '')}; "
        f"trigger={trigger_text}; invalidation={invalidation_text}; "
        f"target_basis={target_basis_text}{horizon_text}; mtf={mtf_text}"
    )


def _clip_detail(value: Any, max_chars: int = 80) -> str:
    return str(value)[:max_chars]


def _format_mtf_summary(context: Any) -> str:
    """Compact shared multi-timeframe context for a candidate prompt row."""
    if not isinstance(context, dict):
        return "n/a"

    timeframes = context.get("timeframes") or {}
    alignment = context.get("directional_alignment") or {}
    relative_strength = context.get("relative_strength") or {}
    volume = context.get("volume_context") or {}
    same_time_volume = volume.get("same_time_of_day") or {}
    sector = context.get("sector_context") or {}
    breadth = context.get("breadth_proxy") or {}

    def trend(label: str) -> str:
        value = (timeframes.get(label) or {}).get("trend")
        return str(value) if value else "n/a"

    def fmt(value: Any) -> str:
        return "n/a" if value is None else str(value)

    return (
        f"bias={alignment.get('bias', 'n/a')} "
        f"agree={alignment.get('agreement', 'n/a')} "
        f"5m={trend('5m')} 60m={trend('60m')} D={trend('daily')} "
        f"rs_spy5={fmt(relative_strength.get('vs_spy_5d'))} "
        f"rs_sector5={fmt(relative_strength.get('vs_sector_5d'))} "
        f"vol_tod={fmt(same_time_volume.get('ratio'))} "
        f"sector_confirmed={fmt(sector.get('sector_confirmed'))} "
        f"breadth={breadth.get('bias', 'n/a')}"
    )


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
