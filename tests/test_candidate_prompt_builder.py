"""Unit tests for utils/candidate_prompt_builder.py.

Validates prompt construction and LLM schema generation for candidate-ID selection.
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.1
"""

import json
import uuid

import pytest

from utils.candidate_prompt_builder import (
    build_candidate_pm_prompt,
    build_decision_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate_summary(
    candidate_id: str | None = None,
    symbol: str = "AAPL",
    direction: str = "BUY",
    setup_type: str = "news_breakout",
    entry_price: float = 150.0,
    stop_price: float = 145.0,
    target_price: float = 160.0,
    risk_reward: float = 2.0,
    geometry_name: str = "primary",
    trigger: str = "Earnings beat with guidance raise",
    invalidation_basis: str = "Breaks below premarket VWAP",
    target_basis: str = "Prior day high retest",
    multitimeframe_context: dict | None = None,
) -> dict:
    """Create a candidate summary dict matching registry.get_offered_summary() output."""
    return {
        "candidate_id": candidate_id or str(uuid.uuid4()),
        "symbol": symbol,
        "direction": direction,
        "setup_type": setup_type,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_reward": risk_reward,
        "geometry_name": geometry_name,
        "trigger": trigger,
        "invalidation_basis": invalidation_basis,
        "target_basis": target_basis,
        "multitimeframe_context": multitimeframe_context,
    }


def _make_portfolio_summary(
    cash: float = 50000.0,
    total_equity: float = 100000.0,
    positions: list | None = None,
    daily_pnl: float = 0.0,
) -> dict:
    return {
        "cash": cash,
        "total_equity": total_equity,
        "positions": positions or [],
        "daily_pnl": daily_pnl,
    }


def _make_profile(
    name: str = "Aggressive Growth",
    max_positions: int = 5,
) -> dict:
    return {
        "name": name,
        "max_positions": max_positions,
    }


# ---------------------------------------------------------------------------
# build_candidate_pm_prompt — candidate table
# ---------------------------------------------------------------------------


class TestPromptCandidateTable:
    """Prompt includes a properly formatted candidate summary table."""

    def test_table_contains_all_candidates(self):
        summaries = [
            _make_candidate_summary(symbol="AAPL"),
            _make_candidate_summary(symbol="MSFT"),
            _make_candidate_summary(symbol="NVDA"),
        ]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "NVDA" in prompt

    def test_table_has_header_row(self):
        summaries = [_make_candidate_summary()]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert (
            "| # | candidate_id | Symbol | Dir | Entry | Stop | Target | R:R | Setup | Geometry | MTF | Trigger | Invalidation | Target Basis |"
            in prompt
        )

    def test_table_includes_full_candidate_id(self):
        cid = str(uuid.uuid4())
        summaries = [_make_candidate_summary(candidate_id=cid)]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert cid in prompt
        assert f"{cid[:8]}..." not in prompt

    def test_table_includes_direction(self):
        summaries = [_make_candidate_summary(direction="SHORT")]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "SHORT" in prompt

    def test_table_includes_formatted_prices(self):
        summaries = [
            _make_candidate_summary(
                entry_price=152.35, stop_price=148.00, target_price=165.50
            )
        ]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "$152.35" in prompt
        assert "$148.00" in prompt
        assert "$165.50" in prompt

    def test_table_includes_risk_reward_ratio(self):
        summaries = [_make_candidate_summary(risk_reward=3.5)]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "3.5:1" in prompt

    def test_table_includes_setup_type(self):
        summaries = [_make_candidate_summary(setup_type="momentum_continuation")]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "momentum_continuation" in prompt

    def test_table_includes_trigger_text(self):
        summaries = [_make_candidate_summary(trigger="Strong volume breakout above resistance")]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "Strong volume breakout above resistance" in prompt

    def test_trigger_truncated_at_80_chars(self):
        long_trigger = "A" * 100
        summaries = [_make_candidate_summary(trigger=long_trigger)]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "A" * 80 in prompt
        assert "A" * 81 not in prompt

    def test_table_includes_geometry_invalidation_and_target_basis(self):
        summaries = [
            _make_candidate_summary(
                geometry_name="gap_and_go",
                invalidation_basis="Fails opening range low",
                target_basis="Measured move to premarket high",
            )
        ]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "gap_and_go" in prompt
        assert "Fails opening range low" in prompt
        assert "Measured move to premarket high" in prompt

    def test_candidate_details_repeat_complete_trade_specs(self):
        cid = str(uuid.uuid4())
        summaries = [
            _make_candidate_summary(
                candidate_id=cid,
                symbol="XLE",
                direction="BUY",
                setup_type="sector_rotation_swing",
                entry_price=56.37,
                stop_price=55.24,
                target_price=58.62,
                risk_reward=2.0,
                geometry_name="swing_sector_rotation_swing",
                trigger="Energy rotation continuation above VWAP",
                invalidation_basis="Fails below swing support",
                target_basis="Sector rotation measured move",
            )
        ]

        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )

        assert "## Candidate Details" in prompt
        assert f"candidate_id={cid}" in prompt
        assert "symbol=XLE" in prompt
        assert "direction=BUY" in prompt
        assert "entry=$56.37" in prompt
        assert "stop=$55.24" in prompt
        assert "target=$58.62" in prompt
        assert "risk_reward=2.0:1" in prompt
        assert "setup=sector_rotation_swing" in prompt
        assert "geometry=swing_sector_rotation_swing" in prompt
        assert "trigger=Energy rotation continuation above VWAP" in prompt
        assert "invalidation=Fails below swing support" in prompt
        assert "target_basis=Sector rotation measured move" in prompt

    def test_table_includes_multitimeframe_context(self):
        summaries = [
            _make_candidate_summary(
                multitimeframe_context={
                    "timeframes": {
                        "5m": {"trend": "bullish"},
                        "60m": {"trend": "bullish"},
                        "daily": {"trend": "neutral"},
                    },
                    "directional_alignment": {
                        "bias": "bullish",
                        "agreement": "aligned",
                    },
                    "relative_strength": {
                        "vs_spy_5d": 1.8,
                        "vs_sector_5d": 0.7,
                    },
                    "volume_context": {
                        "intraday_vs_prior_session": 1.3,
                        "same_time_of_day": {"ratio": 1.6},
                    },
                    "sector_context": {"sector_confirmed": True},
                    "breadth_proxy": {"bias": "supportive"},
                }
            )
        ]
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )

        assert "bias=bullish" in prompt
        assert "5m=bullish" in prompt
        assert "60m=bullish" in prompt
        assert "D=neutral" in prompt
        assert "rs_spy5=1.8" in prompt
        assert "vol_tod=1.6" in prompt
        assert "sector_confirmed=True" in prompt
        assert "breadth=supportive" in prompt

    def test_none_trigger_handled_gracefully(self):
        summaries = [_make_candidate_summary(trigger=None)]
        # Override dict directly since helper sets a string
        summaries[0]["trigger"] = None
        prompt = build_candidate_pm_prompt(
            summaries, _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        # Should not raise
        assert "AAPL" in prompt


# ---------------------------------------------------------------------------
# build_candidate_pm_prompt — portfolio summary
# ---------------------------------------------------------------------------


class TestPromptPortfolioSummary:
    """Prompt includes portfolio state summary."""

    def test_includes_equity(self):
        portfolio = _make_portfolio_summary(total_equity=100000)
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], portfolio, _make_profile(), "prof-1"
        )
        assert "$100,000" in prompt

    def test_includes_cash(self):
        portfolio = _make_portfolio_summary(cash=42000)
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], portfolio, _make_profile(), "prof-1"
        )
        assert "$42,000" in prompt

    def test_includes_position_count(self):
        portfolio = _make_portfolio_summary(positions=[{"sym": "X"}, {"sym": "Y"}])
        profile = _make_profile(max_positions=5)
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], portfolio, profile, "prof-1"
        )
        assert "2/5 positions open" in prompt

    def test_zero_positions(self):
        portfolio = _make_portfolio_summary(positions=[])
        profile = _make_profile(max_positions=3)
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], portfolio, profile, "prof-1"
        )
        assert "0/3 positions open" in prompt


# ---------------------------------------------------------------------------
# build_candidate_pm_prompt — instructions (Requirements 7.2–7.5)
# ---------------------------------------------------------------------------


class TestPromptInstructions:
    """Prompt includes correct selection instructions per Requirements 7.2-7.5."""

    def test_instruct_select_by_candidate_id_only(self):
        """Requirement 7.2: PM prompt states selecting from supplied candidates only."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        assert "candidate_id" in prompt.lower()
        assert "select" in prompt.lower() or "selecting" in prompt.lower()

    def test_instruct_no_symbols_prices_quantities_sectors(self):
        """Requirement 3.1: No symbols, prices, quantities, or sector labels."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        # The instructions should explicitly say NOT to specify these
        lower = prompt.lower()
        assert "do not specify symbols" in lower or "not specify symbols" in lower

    def test_instruct_categories_not_executable(self):
        """Requirement 7.3: Categories/themes are commentary only."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        lower = prompt.lower()
        assert "categories" in lower or "themes" in lower
        assert "commentary" in lower or "not executable" in lower

    def test_instruct_cannot_trade_unlisted_candidate(self):
        """Requirement 7.4: A candidate not in the list cannot be traded."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        lower = prompt.lower()
        assert "not in this list cannot be traded" in lower

    def test_instruct_empty_set_valid(self):
        """Requirement 7.5: Empty accepted set is a valid response."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        lower = prompt.lower()
        assert "empty" in lower
        assert "valid" in lower

    def test_instruct_accept_reject_only(self):
        """Requirement 3.1: Decisions are accept or reject."""
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        lower = prompt.lower()
        assert "accept" in lower
        assert "reject" in lower

    def test_instruct_scaffold_geometry_is_complete(self):
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), _make_profile(), "prof-1"
        )
        lower = prompt.lower()
        assert "complete executable candidate set" in lower
        assert "do not reject" in lower
        assert "risk/reward" in lower


# ---------------------------------------------------------------------------
# build_candidate_pm_prompt — profile name
# ---------------------------------------------------------------------------


class TestPromptProfileName:
    """Prompt uses profile name for personality."""

    def test_uses_profile_name(self):
        profile = _make_profile(name="Conservative Value")
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), profile, "prof-1"
        )
        assert "Conservative Value" in prompt

    def test_falls_back_to_profile_id(self):
        profile = {}  # No name key
        prompt = build_candidate_pm_prompt(
            [_make_candidate_summary()], _make_portfolio_summary(), profile, "my-profile"
        )
        assert "my-profile" in prompt


# ---------------------------------------------------------------------------
# build_decision_schema — dynamic enum (Requirement 7.1)
# ---------------------------------------------------------------------------


class TestDecisionSchema:
    """Schema constrains candidate_ids to exact offered set."""

    def test_schema_has_decisions_array(self):
        schema = build_decision_schema({"id-1", "id-2"})
        assert schema["type"] == "object"
        assert "decisions" in schema["properties"]
        assert schema["properties"]["decisions"]["type"] == "array"

    def test_candidate_id_enum_matches_input(self):
        """Requirement 7.1: Schema constrains to exact current candidate IDs."""
        ids = {"abc-123", "def-456", "ghi-789"}
        schema = build_decision_schema(ids)
        enum_values = schema["properties"]["decisions"]["items"]["properties"]["candidate_id"]["enum"]
        assert set(enum_values) == ids

    def test_candidate_id_enum_is_sorted(self):
        ids = {"zzz", "aaa", "mmm"}
        schema = build_decision_schema(ids)
        enum_values = schema["properties"]["decisions"]["items"]["properties"]["candidate_id"]["enum"]
        assert enum_values == sorted(ids)

    def test_decision_enum_is_accept_reject(self):
        schema = build_decision_schema({"id-1"})
        decision_enum = schema["properties"]["decisions"]["items"]["properties"]["decision"]["enum"]
        assert decision_enum == ["accept", "reject"]

    def test_risk_multiplier_bounds(self):
        schema = build_decision_schema({"id-1"})
        rm = schema["properties"]["decisions"]["items"]["properties"]["risk_multiplier"]
        assert rm["minimum"] == 0.01
        assert rm["maximum"] == 1.0

    def test_rationale_max_length(self):
        schema = build_decision_schema({"id-1"})
        rationale = schema["properties"]["decisions"]["items"]["properties"]["rationale"]
        assert rationale["maxLength"] == 280

    def test_portfolio_notes_max_length(self):
        schema = build_decision_schema({"id-1"})
        notes = schema["properties"]["portfolio_notes"]
        assert notes["maxLength"] == 420

    def test_additional_properties_false_at_top_level(self):
        schema = build_decision_schema({"id-1"})
        assert schema["additionalProperties"] is False

    def test_additional_properties_false_at_item_level(self):
        schema = build_decision_schema({"id-1"})
        items = schema["properties"]["decisions"]["items"]
        assert items["additionalProperties"] is False

    def test_required_fields_at_top_level(self):
        schema = build_decision_schema({"id-1"})
        assert "decisions" in schema["required"]

    def test_required_fields_at_item_level(self):
        schema = build_decision_schema({"id-1"})
        items = schema["properties"]["decisions"]["items"]
        assert "candidate_id" in items["required"]
        assert "decision" in items["required"]

    def test_empty_candidate_set_produces_empty_enum(self):
        schema = build_decision_schema(set())
        enum_values = schema["properties"]["decisions"]["items"]["properties"]["candidate_id"]["enum"]
        assert enum_values == []

    def test_schema_is_json_serializable(self):
        ids = {str(uuid.uuid4()) for _ in range(5)}
        schema = build_decision_schema(ids)
        # Should not raise
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["additionalProperties"] is False
