"""
Lesson Registry
Stateless query layer over the cases table that aggregates recent review scores
by (symbol, bias, setup_type) and provides track-record verdicts for PM gating.
"""

from db.schema import get_session
from models.case import Case


def query_lessons(engine, symbol: str, bias: str, setup_type: str, limit: int = 10) -> list[dict]:
    """
    Query recent cases matching (symbol, bias, setup_type) from the case library.
    Returns cases ordered by created_at descending, filtered to non-null review_score.
    """
    db = get_session(engine)
    cases = (
        db.query(Case)
        .filter(
            Case.symbol == symbol,
            Case.bias == bias,
            Case.setup_type == setup_type,
            Case.review_score.isnot(None),
        )
        .order_by(Case.created_at.desc())
        .limit(limit)
        .all()
    )

    result = []
    for c in cases:
        result.append({
            "symbol": c.symbol,
            "bias": c.bias,
            "setup_type": c.setup_type,
            "review_score": c.review_score,
            "created_at": c.created_at,
        })

    db.close()
    return result


def _query_fallback(engine, bias: str, setup_type: str, limit: int = 10) -> list[dict]:
    """
    Fallback query: recent cases matching (bias, setup_type) only, ignoring symbol.
    Returns cases ordered by created_at descending, filtered to non-null review_score.
    """
    db = get_session(engine)
    cases = (
        db.query(Case)
        .filter(
            Case.bias == bias,
            Case.setup_type == setup_type,
            Case.review_score.isnot(None),
        )
        .order_by(Case.created_at.desc())
        .limit(limit)
        .all()
    )

    result = []
    for c in cases:
        result.append({
            "symbol": c.symbol,
            "bias": c.bias,
            "setup_type": c.setup_type,
            "review_score": c.review_score,
            "created_at": c.created_at,
        })

    db.close()
    return result


def check_track_record(engine, symbol: str, bias: str, setup_type: str) -> dict:
    """
    Evaluate the track record for a (symbol, bias, setup_type) combination.

    Returns a verdict dict:
    {
        "verdict": "OK" | "POOR_TRACK_RECORD" | "BLOCK" | "INSUFFICIENT_DATA",
        "avg_score_3": float | None,
        "avg_score_5": float | None,
        "sample_size": int,
        "size_multiplier": float,  # 1.0, 0.5, or 0.0
        "match_type": "exact" | "fallback",
    }
    """
    # Step 1: Try exact match on (symbol, bias, setup_type)
    cases = query_lessons(engine, symbol, bias, setup_type)
    scores = [c["review_score"] for c in cases]

    if len(scores) >= 3:
        return _evaluate_thresholds(scores, match_type="exact")

    # Step 2: Fallback query on (bias, setup_type) only with relaxed thresholds
    fallback_cases = _query_fallback(engine, bias, setup_type)
    fallback_scores = [c["review_score"] for c in fallback_cases]

    if len(fallback_scores) >= 3:
        return _evaluate_thresholds(
            fallback_scores,
            match_type="fallback",
            block_threshold=3.5,
            poor_threshold=4.0,
        )

    # Step 3: Neither query yielded 3+ cases
    return {
        "verdict": "INSUFFICIENT_DATA",
        "avg_score_3": None,
        "avg_score_5": None,
        "sample_size": len(fallback_scores),
        "size_multiplier": 1.0,
        "match_type": "fallback",
    }


def _evaluate_thresholds(
    scores: list[float],
    match_type: str,
    block_threshold: float = 4.0,
    poor_threshold: float = 4.5,
) -> dict:
    """
    Apply threshold logic to a list of review scores (most-recent first).

    Returns a verdict dict with all required keys.
    """
    sample_size = len(scores)

    if sample_size < 3:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "avg_score_3": None,
            "avg_score_5": None,
            "sample_size": sample_size,
            "size_multiplier": 1.0,
            "match_type": match_type,
        }

    avg_score_3 = sum(scores[:3]) / 3
    avg_score_5 = sum(scores[:5]) / 5 if sample_size >= 5 else None

    # BLOCK takes precedence over POOR_TRACK_RECORD
    if sample_size >= 5 and avg_score_5 < block_threshold:
        return {
            "verdict": "BLOCK",
            "avg_score_3": avg_score_3,
            "avg_score_5": avg_score_5,
            "sample_size": sample_size,
            "size_multiplier": 0.0,
            "match_type": match_type,
        }

    if avg_score_3 < poor_threshold:
        return {
            "verdict": "POOR_TRACK_RECORD",
            "avg_score_3": avg_score_3,
            "avg_score_5": avg_score_5,
            "sample_size": sample_size,
            "size_multiplier": 0.5,
            "match_type": match_type,
        }

    return {
        "verdict": "OK",
        "avg_score_3": avg_score_3,
        "avg_score_5": avg_score_5,
        "sample_size": sample_size,
        "size_multiplier": 1.0,
        "match_type": match_type,
    }
