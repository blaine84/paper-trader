"""Deterministic overlap candidate detection and severity scoring.

Analyzes a snapshot of trades to find same-symbol, same-direction overlap
across different trading profiles. Computes combined PnL, stop-loss outcomes,
and suggested severity for each overlap candidate group.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from zoneinfo import ZoneInfo


def _parse_datetime(dt_str, trading_tz):
    """Parse a datetime string as timezone-aware, normalizing to UTC.

    Naive timestamps (no timezone info) are interpreted as the configured
    trading timezone. Already-aware timestamps are converted to UTC.
    Returns a UTC-aware datetime.
    """
    if dt_str is None:
        return None

    # Try ISO-8601 with timezone offset first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    # Parse naive datetime formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(dt_str, fmt)
            # Interpret as trading timezone, then convert to UTC
            dt = dt.replace(tzinfo=trading_tz)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse datetime string: {dt_str!r}")


def _is_stop_loss(reason_exit):
    """Check if the exit reason indicates a stop-loss."""
    if not reason_exit:
        return False
    return "stop_loss" in reason_exit.lower()


def _compute_severity(num_profiles, combined_pnl, all_stopped_out):
    """Compute suggested_severity per design rules.

    - low: 2 profiles, combined PnL >= 0
    - medium: 2 profiles, combined PnL < 0 but not all stopped out
    - high: 2 profiles, combined PnL < 0 and all stopped out, OR 3+ profiles
    - critical: 3+ profiles all stopped out, or combined PnL loss exceeds -$500
    """
    # Critical conditions (checked first — highest priority)
    if combined_pnl < -500:
        return "critical"
    if num_profiles >= 3 and all_stopped_out:
        return "critical"

    # High conditions
    if num_profiles >= 3:
        return "high"
    if num_profiles == 2 and combined_pnl < 0 and all_stopped_out:
        return "high"

    # Medium conditions
    if num_profiles == 2 and combined_pnl < 0 and not all_stopped_out:
        return "medium"

    # Low: 2 profiles, combined PnL >= 0
    return "low"


def _build_evidence(trade):
    """Build a human-readable evidence string for a trade."""
    reason = trade.get("reason_exit") or "N/A"
    # Normalize reason_exit for evidence: extract the core reason
    reason_short = reason
    if "stop_loss" in reason.lower():
        reason_short = "stop_loss"
    elif "target_hit" in reason.lower():
        reason_short = "target_hit"

    return (
        f"trade_id={trade['id']} {trade['symbol']} {trade['profile']} "
        f"{trade['direction']} {trade['setup_type']} "
        f"pnl={trade['pnl']} reason_exit={reason_short}"
    )


def compute_overlap_candidates(
    snapshot,
    window_minutes=120,
    trading_timezone="America/New_York",
):
    """Compute overlap candidates from a snapshot of trades.

    Groups trades by (symbol, direction) and detects overlap from different
    profiles based on temporal proximity and open-position overlap.

    Args:
        snapshot: Dict with 'trades' array and other snapshot fields.
        window_minutes: Time window in minutes for overlap detection.
        trading_timezone: IANA timezone string for interpreting naive timestamps.

    Returns:
        Dict with keys:
          - diagnostic: "same_symbol_overlap"
          - window_minutes: int
          - overlap_candidates: list of candidate dicts
    """
    trading_tz = ZoneInfo(trading_timezone)
    trades = snapshot.get("trades", [])
    window = timedelta(minutes=window_minutes)

    # Parse datetimes for all trades upfront
    parsed_trades = []
    for t in trades:
        entry_dt = _parse_datetime(t.get("entry_time"), trading_tz)
        exit_dt = _parse_datetime(t.get("exit_time"), trading_tz)
        parsed_trades.append({
            **t,
            "_entry_dt": entry_dt,
            "_exit_dt": exit_dt,
        })

    # Group trades by (symbol, direction)
    groups = defaultdict(list)
    for t in parsed_trades:
        key = (t["symbol"], t["direction"])
        groups[key].append(t)

    # Find overlap pairs within each group
    # Each overlap is a frozenset of trade IDs
    raw_overlap_sets = []

    for (symbol, direction), group_trades in groups.items():
        if len(group_trades) < 2:
            continue

        # Check all pairs for overlap conditions
        for a, b in combinations(group_trades, 2):
            # Must be from different profiles
            if a["profile"] == b["profile"]:
                continue

            # Ensure a is the earlier entry
            if a["_entry_dt"] > b["_entry_dt"]:
                a, b = b, a

            overlap_found = False

            # Condition (a): entry times within window_minutes
            entry_diff_seconds = abs((b["_entry_dt"] - a["_entry_dt"]).total_seconds())
            if entry_diff_seconds <= window.total_seconds():
                overlap_found = True

            # Condition (b): trade B opens while trade A is still open
            if not overlap_found:
                a_is_open = (
                    a.get("status") == "open"
                    or (a["_exit_dt"] is not None and a["_exit_dt"] > b["_entry_dt"])
                )
                if a_is_open:
                    overlap_found = True

            # Condition (c): trade A closed within window_minutes before trade B opens
            if not overlap_found:
                if a["_exit_dt"] is not None:
                    time_since_exit = (b["_entry_dt"] - a["_exit_dt"]).total_seconds()
                    if 0 <= time_since_exit <= window.total_seconds():
                        overlap_found = True

            if overlap_found:
                raw_overlap_sets.append(frozenset([a["id"], b["id"]]))

    # Merge overlapping groups that share trade IDs
    merged = _merge_overlapping_sets(raw_overlap_sets)

    # Build candidate dicts from merged groups
    candidates = []
    for trade_id_set in merged:
        # Gather the trades in this candidate group
        trade_map = {t["id"]: t for t in parsed_trades}
        group_trades = [trade_map[tid] for tid in trade_id_set]

        # Sort trades by entry time for determinism
        group_trades.sort(key=lambda t: (t["_entry_dt"], t["id"]))

        symbol = group_trades[0]["symbol"]
        direction = group_trades[0]["direction"]

        # Determine setup_type: "mixed" if different, else the common value
        setup_types = set(t["setup_type"] for t in group_trades)
        setup_type = group_trades[0]["setup_type"] if len(setup_types) == 1 else "mixed"

        profiles = sorted(set(t["profile"] for t in group_trades))
        trade_ids = [t["id"] for t in group_trades]
        entry_times = [t.get("entry_time") for t in group_trades]

        # Compute outcome
        combined_pnl = round(sum(t.get("pnl", 0) for t in group_trades), 2)
        all_stopped_out = all(_is_stop_loss(t.get("reason_exit")) for t in group_trades)

        # Evidence strings
        evidence = [_build_evidence(t) for t in group_trades]

        # opened_while_existing_position_open
        opened_while_open = _check_opened_while_open(group_trades)

        # minutes_between_entries
        entry_dts = [t["_entry_dt"] for t in group_trades]
        earliest = min(entry_dts)
        latest = max(entry_dts)
        minutes_between = round(
            (latest - earliest).total_seconds() / 60, 1
        ) if len(entry_dts) > 1 else 0.0

        # minutes_since_prior_exit
        minutes_since_prior = _compute_minutes_since_prior_exit(group_trades)

        candidates.append({
            "symbol": symbol,
            "direction": direction,
            "setup_type": setup_type,
            "profiles": profiles,
            "trade_ids": trade_ids,
            "entry_times": entry_times,
            "outcome": {
                "combined_pnl": combined_pnl,
                "all_stopped_out": all_stopped_out,
            },
            "evidence": evidence,
            "suggested_severity": _compute_severity(
                len(profiles), combined_pnl, all_stopped_out
            ),
            "opened_while_existing_position_open": opened_while_open,
            "minutes_between_entries": minutes_between,
            "minutes_since_prior_exit": minutes_since_prior,
            # Sorting key for deterministic candidate_id assignment
            "_sort_key": (symbol, direction, min(entry_dts)),
        })

    # Sort candidates deterministically and assign candidate_ids
    candidates.sort(key=lambda c: c["_sort_key"])
    for i, c in enumerate(candidates, start=1):
        c["candidate_id"] = f"overlap_{i:03d}"
        del c["_sort_key"]

    return {
        "diagnostic": "same_symbol_overlap",
        "window_minutes": window_minutes,
        "overlap_candidates": candidates,
    }


def _merge_overlapping_sets(sets):
    """Merge frozensets that share any elements.

    Given a list of frozensets of trade IDs, merge any sets that share
    at least one trade ID. Returns a list of merged frozensets.
    """
    if not sets:
        return []

    merged = list(sets)
    changed = True
    while changed:
        changed = False
        new_merged = []
        while merged:
            current = merged.pop(0)
            found_overlap = False
            for i, existing in enumerate(new_merged):
                if current & existing:
                    new_merged[i] = existing | current
                    changed = True
                    found_overlap = True
                    break
            if not found_overlap:
                new_merged.append(current)
        merged = new_merged

    return merged


def _check_opened_while_open(group_trades):
    """Check if any trade in the group opened while another was still open.

    Returns True if trade B's entry time is before trade A's exit time
    (or trade A has status 'open') for any pair from different profiles.
    """
    for a, b in combinations(group_trades, 2):
        if a["profile"] == b["profile"]:
            continue

        # Ensure a is the earlier entry
        if a["_entry_dt"] > b["_entry_dt"]:
            a, b = b, a

        # Check if a was still open when b entered
        if a.get("status") == "open":
            return True
        if a["_exit_dt"] is not None and a["_exit_dt"] > b["_entry_dt"]:
            return True

    return False


def _compute_minutes_since_prior_exit(group_trades):
    """Compute minutes between the earliest prior exit and the latest entry.

    For a group of overlapping trades, this measures how long after the
    first trade closed before the last trade opened. Returns None if
    no trade has exited before another entered.
    """
    for a, b in combinations(group_trades, 2):
        if a["profile"] == b["profile"]:
            continue

        # Ensure a is the earlier entry
        if a["_entry_dt"] > b["_entry_dt"]:
            a, b = b, a

        # If a exited before b entered, compute the gap
        if a["_exit_dt"] is not None and a["_exit_dt"] <= b["_entry_dt"]:
            gap = (b["_entry_dt"] - a["_exit_dt"]).total_seconds() / 60
            return round(gap, 1)

    return None
