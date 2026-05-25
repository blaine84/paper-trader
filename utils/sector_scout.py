"""Core screening module for the Sector Scout pipeline.

Provides deterministic sector screening, hard gates, scoring, ranking,
and orchestration functions for multi-sector candidate discovery.

See: design.md §Components, requirements.md §1–§5
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from utils.finnhub_client import FinnhubClient
from utils.scout_logging import emit_scout_event
from utils.sector_scout_models import CandidateRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required top-level config sections
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_SECTIONS: set[str] = {
    "enabled",
    "sector_buckets",
    "hard_gates",
    "score_penalties",
    "scoring_weights",
    "scoring_caps",
    "budget_ceilings",
    "chief_scout",
    "reanalysis_cooldown",
}

# Default config path relative to project root
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sector_scout_config.yaml"


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------


def load_sector_scout_config(config_path: Path | None = None) -> dict:
    """Load and validate the sector scout configuration from YAML.

    Args:
        config_path: Optional override path to the config file.
            Defaults to ``config/sector_scout_config.yaml`` relative to
            the project root.

    Returns:
        Parsed configuration dictionary with all required sections.

    Raises:
        FileNotFoundError: If the config file does not exist at the
            expected path.
        ValueError: If the YAML is unparseable or required top-level
            sections are missing.
    """
    path = config_path or _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Sector scout config file not found: {path}. "
            "Ensure config/sector_scout_config.yaml exists in the project root."
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"Unable to read sector scout config file at {path}: {exc}"
        ) from exc

    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Sector scout config file is not valid YAML: {exc}"
        ) from exc

    if config is None:
        raise ValueError(
            "Sector scout config file is empty or contains only comments."
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Sector scout config must be a YAML mapping, got {type(config).__name__}."
        )

    missing = REQUIRED_CONFIG_SECTIONS - set(config.keys())
    if missing:
        raise ValueError(
            f"Sector scout config is missing required sections: {sorted(missing)}. "
            f"Expected all of: {sorted(REQUIRED_CONFIG_SECTIONS)}"
        )

    logger.info("Sector scout config loaded successfully from %s", path)
    return config


# ---------------------------------------------------------------------------
# Hard Gates — Binary rejection filters
# ---------------------------------------------------------------------------


def apply_hard_gates(row: CandidateRow, config: dict) -> tuple[bool, str | None]:
    """Apply binary rejection gates. Returns (passed, reason_code_or_none).

    Evaluates the candidate row against hard gate thresholds from config.
    Sets row.hard_gate_passed, appends reason_codes for each rejection,
    and returns a tuple of (passed, first_reason_code).

    Hard gate checks are applied in a fixed, deterministic order:
      1. Malformed/missing critical fields (symbol, sector)
      2. Missing or zero price
      3. Price below minimum threshold
      4. Spread too wide (when spread is known)
      5. Market cap below minimum (when available, or via proxy)

    Args:
        row: CandidateRow to evaluate.
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        Tuple of (passed: bool, first_reason_code: str | None).
        When passed is True, first_reason_code is None.
        When passed is False, first_reason_code is the first rejection reason.
    """
    hard_gates_cfg = config.get("hard_gates", {})
    min_price = hard_gates_cfg.get("min_price", 5.0)
    max_spread_pct = hard_gates_cfg.get("max_spread_pct", 5.0)
    min_market_cap = hard_gates_cfg.get("min_market_cap", 500_000_000)
    proxy_market_cap_enabled = hard_gates_cfg.get("proxy_market_cap_enabled", True)

    rejection_reasons: list[str] = []

    # Gate 1: Malformed/missing critical fields
    if not row.symbol or not row.sector:
        rejection_reasons.append("hard_gate:malformed_row")

    # Gate 2: Missing or zero price
    if row.current_price is None or row.current_price == 0:
        rejection_reasons.append("hard_gate:missing_or_zero_price")

    # Gate 3: Price below minimum (only if price is present and non-zero)
    if (
        row.current_price is not None
        and row.current_price != 0
        and row.current_price < min_price
    ):
        rejection_reasons.append("hard_gate:price_below_minimum")

    # Gate 4: Spread too wide (only when spread is known)
    if (
        row.spread_status == "known"
        and row.spread_pct is not None
        and row.spread_pct > max_spread_pct
    ):
        rejection_reasons.append("hard_gate:spread_too_wide")

    # Gate 5: Market cap check
    if row.market_cap is not None:
        # Market cap is available directly
        if row.market_cap < min_market_cap:
            rejection_reasons.append("hard_gate:below_min_market_cap")
    elif proxy_market_cap_enabled:
        # Market cap unavailable — attempt proxy using price * average_volume
        if (
            row.current_price is not None
            and row.current_price > 0
            and row.average_volume is not None
            and row.average_volume > 0
        ):
            proxy_market_cap = row.current_price * row.average_volume
            row.microcap_proxy_used = True
            row.market_cap_source = "proxy"

            if proxy_market_cap < min_market_cap:
                rejection_reasons.append("hard_gate:below_min_market_cap")
        else:
            # Proxy is inconclusive (missing data for proxy calculation).
            # Do NOT reject — flag for Score_Penalty later.
            # No hard gate rejection here per requirement 3.5 / 12.7.
            pass

    # Apply results
    if rejection_reasons:
        row.hard_gate_passed = False
        row.reason_codes.extend(rejection_reasons)
        return (False, rejection_reasons[0])
    else:
        row.hard_gate_passed = True
        return (True, None)


# ---------------------------------------------------------------------------
# Score Penalties — Soft deductions for quality issues
# ---------------------------------------------------------------------------


def apply_score_penalties(row: CandidateRow, config: dict) -> CandidateRow:
    """Apply soft deductions and append reason_codes. Returns modified row.

    Evaluates the candidate row against configurable penalty thresholds and
    applies deductions to scout_score for each triggered condition. Each
    applied penalty is recorded in row.penalties_applied and row.reason_codes.

    Penalty rules (applied in deterministic order):
      1. missing_news — news_freshness_minutes is None
      2. stale_news — news_freshness_minutes > threshold (mutually exclusive with missing_news)
      3. unknown_spread — spread_status == "unknown"
      4. weak_sector_confirmation — sector_confirmed is False or None
      5. low_rvol — relative_volume is not None AND < threshold
      6. low_dollar_volume — dollar_volume is not None AND < threshold

    missing_news and stale_news are MUTUALLY EXCLUSIVE:
      - If news_freshness_minutes is None → missing_news only
      - If news_freshness_minutes is not None and > threshold → stale_news only
      - If news_freshness_minutes is not None and <= threshold → no news penalty

    Args:
        row: CandidateRow to evaluate (must have passed hard gates).
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        The same CandidateRow with updated scout_score, penalties_applied,
        and reason_codes. Final scout_score is clamped to [0.0, 100.0].
    """
    penalties_cfg = config.get("score_penalties", {})

    # Thresholds
    stale_news_threshold = penalties_cfg.get("stale_news_threshold_minutes", 120)
    low_rvol_threshold = penalties_cfg.get("low_rvol_threshold", 1.2)
    low_dollar_volume_threshold = penalties_cfg.get("low_dollar_volume_threshold", 5_000_000)

    # Deduction magnitudes
    missing_news_deduction = penalties_cfg.get("missing_news_deduction", 20.0)
    stale_news_deduction = penalties_cfg.get("stale_news_deduction", 15.0)
    unknown_spread_deduction = penalties_cfg.get("unknown_spread_deduction", 10.0)
    weak_sector_deduction = penalties_cfg.get("weak_sector_confirmation_deduction", 10.0)
    low_rvol_deduction = penalties_cfg.get("low_rvol_deduction", 12.0)
    low_dollar_volume_deduction = penalties_cfg.get("low_dollar_volume_deduction", 8.0)

    total_deduction = 0.0

    # --- Penalty 1/2: News penalties (mutually exclusive) ---
    if row.news_freshness_minutes is None:
        # No news data available at all → missing_news penalty
        row.penalties_applied.append({"type": "missing_news", "deduction": missing_news_deduction})
        row.reason_codes.append(f"penalty:missing_news:{missing_news_deduction}")
        total_deduction += missing_news_deduction
    elif row.news_freshness_minutes > stale_news_threshold:
        # News exists but is stale → stale_news penalty
        row.penalties_applied.append({"type": "stale_news", "deduction": stale_news_deduction})
        row.reason_codes.append(f"penalty:stale_news:{stale_news_deduction}")
        total_deduction += stale_news_deduction
    # else: news is fresh enough — no news penalty

    # --- Penalty 3: Unknown spread ---
    if row.spread_status == "unknown":
        row.penalties_applied.append({"type": "unknown_spread", "deduction": unknown_spread_deduction})
        row.reason_codes.append(f"penalty:unknown_spread:{unknown_spread_deduction}")
        total_deduction += unknown_spread_deduction

    # --- Penalty 4: Weak or absent sector confirmation ---
    if row.sector_confirmed is False or row.sector_confirmed is None:
        row.penalties_applied.append({"type": "weak_sector_confirmation", "deduction": weak_sector_deduction})
        row.reason_codes.append(f"penalty:weak_sector_confirmation:{weak_sector_deduction}")
        total_deduction += weak_sector_deduction

    # --- Penalty 5: Low relative volume ---
    if row.relative_volume is not None and row.relative_volume < low_rvol_threshold:
        row.penalties_applied.append({"type": "low_rvol", "deduction": low_rvol_deduction})
        row.reason_codes.append(f"penalty:low_rvol:{low_rvol_deduction}")
        total_deduction += low_rvol_deduction

    # --- Penalty 6: Low dollar volume ---
    if row.dollar_volume is not None and row.dollar_volume < low_dollar_volume_threshold:
        row.penalties_applied.append({"type": "low_dollar_volume", "deduction": low_dollar_volume_deduction})
        row.reason_codes.append(f"penalty:low_dollar_volume:{low_dollar_volume_deduction}")
        total_deduction += low_dollar_volume_deduction

    # --- Apply total deduction and clamp ---
    row.scout_score = max(0.0, min(100.0, row.scout_score - total_deduction))

    return row


# ---------------------------------------------------------------------------
# Generic market commentary keywords for news filtering
# ---------------------------------------------------------------------------

_GENERIC_NEWS_KEYWORDS: set[str] = {
    "market wrap",
    "market recap",
    "futures",
    "s&p 500",
    "dow jones",
    "nasdaq composite",
    "wall street",
    "fed meeting",
    "fomc",
    "treasury yields",
    "economic data",
    "jobs report",
    "cpi report",
    "inflation data",
    "market outlook",
    "weekly roundup",
    "morning briefing",
    "premarket movers",
    "after-hours",
    "top gainers",
    "top losers",
}


def _is_generic_market_news(headline: str) -> bool:
    """Return True if headline is generic market commentary, not symbol-specific."""
    if not headline:
        return True
    lower = headline.lower()
    for keyword in _GENERIC_NEWS_KEYWORDS:
        if keyword in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Data Collection Helpers
# ---------------------------------------------------------------------------


def _get_quote_finnhub(symbol: str, fh: FinnhubClient) -> dict | None:
    """Attempt to get quote data from FinnhubClient. Returns None on failure."""
    try:
        quote = fh.get_quote(symbol)
        if quote and quote.get("price") is not None:
            return quote
    except Exception as exc:
        logger.warning("Finnhub quote failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_quote_yfinance(symbol: str) -> dict | None:
    """Attempt to get quote data from yfinance as fallback. Returns None on failure."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = info.get("lastPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if price is None or price == 0:
            return None
        return {
            "symbol": symbol,
            "price": float(price),
            "prev_close": float(prev_close) if prev_close else None,
            "open": info.get("open"),
            "high": info.get("dayHigh"),
            "low": info.get("dayLow"),
            "volume": info.get("lastVolume"),
            "bid": None,  # fast_info doesn't reliably provide bid/ask
            "ask": None,
        }
    except Exception as exc:
        logger.warning("yfinance quote fallback failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_average_volume_yfinance(symbol: str) -> float | None:
    """Get 20-day average volume from yfinance (primary source for avg volume)."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        # Try fast_info first for average volume
        avg_vol = ticker.fast_info.get("threeMonthAverageVolume")
        if avg_vol and avg_vol > 0:
            return float(avg_vol)
        # Fallback: compute from history
        hist = ticker.history(period="1mo")
        if hist is not None and not hist.empty and "Volume" in hist.columns:
            recent = hist["Volume"].tail(20)
            if len(recent) > 0:
                return float(recent.mean())
    except Exception as exc:
        logger.warning("yfinance avg volume failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_average_volume_finnhub(symbol: str, fh: FinnhubClient) -> float | None:
    """Get average volume from Finnhub candles as fallback."""
    try:
        candles = fh.get_candles(symbol, resolution="D", days=30)
        if candles and candles.get("volume"):
            volumes = candles["volume"]
            # Use last 20 trading days
            recent = volumes[-20:] if len(volumes) >= 20 else volumes
            if recent:
                return float(sum(recent) / len(recent))
    except Exception as exc:
        logger.warning("Finnhub avg volume fallback failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_market_cap_finnhub(symbol: str, fh: FinnhubClient) -> float | None:
    """Get market cap from Finnhub company profile (primary source)."""
    try:
        fh._rate_limit()
        profile = fh._call_with_retry(lambda: fh.client.company_profile2(symbol=symbol))
        if profile and profile.get("marketCapitalization"):
            # Finnhub returns market cap in millions
            return float(profile["marketCapitalization"]) * 1_000_000
    except Exception as exc:
        logger.warning("Finnhub market cap failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_market_cap_yfinance(symbol: str) -> float | None:
    """Get market cap from yfinance info (fallback source)."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        mcap = ticker.fast_info.get("marketCap")
        if mcap and mcap > 0:
            return float(mcap)
    except Exception as exc:
        logger.warning("yfinance market cap fallback failed for %s: %s", symbol, type(exc).__name__)
    return None


def _get_bid_ask_from_yfinance(symbol: str) -> tuple[float | None, float | None]:
    """Attempt to get bid/ask from yfinance Ticker info."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info
        bid = info.get("bid")
        ask = info.get("ask")
        if bid and ask and bid > 0 and ask > 0:
            return float(bid), float(ask)
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Sector Confirmation
# ---------------------------------------------------------------------------


def check_sector_confirmation(
    symbol: str, sector_etf: str | None, fh: FinnhubClient
) -> bool | None:
    """Compare symbol move direction with sector ETF.

    Returns True if both moved in the same direction, False if opposite,
    None if data is unavailable for either.
    """
    if not sector_etf:
        return None

    # Get symbol quote
    symbol_quote = _get_quote_finnhub(symbol, fh)
    if not symbol_quote:
        symbol_quote = _get_quote_yfinance(symbol)
    if not symbol_quote or symbol_quote.get("prev_close") is None:
        return None

    symbol_price = symbol_quote.get("price")
    symbol_prev = symbol_quote.get("prev_close")
    if not symbol_price or not symbol_prev or symbol_prev == 0:
        return None

    symbol_direction = symbol_price - symbol_prev  # positive = up

    # Get sector ETF quote
    etf_quote = _get_quote_finnhub(sector_etf, fh)
    if not etf_quote:
        etf_quote = _get_quote_yfinance(sector_etf)
    if not etf_quote or etf_quote.get("prev_close") is None:
        return None

    etf_price = etf_quote.get("price")
    etf_prev = etf_quote.get("prev_close")
    if not etf_price or not etf_prev or etf_prev == 0:
        return None

    etf_direction = etf_price - etf_prev  # positive = up

    # Same direction = confirmed (both positive or both negative)
    # If either is exactly zero, treat as not confirmed (no move to confirm)
    if symbol_direction == 0 or etf_direction == 0:
        return False

    return (symbol_direction > 0) == (etf_direction > 0)


# ---------------------------------------------------------------------------
# Main Data Collection
# ---------------------------------------------------------------------------


def collect_candidate_data(
    symbol: str, sector_key: str, config: dict, fh: FinnhubClient
) -> CandidateRow:
    """Collect quote, volume, news, spread data for a single symbol.

    Follows data source precedence:
    - Real-time quote: FinnhubClient primary, yfinance fallback
    - Current volume: FinnhubClient primary, yfinance fallback
    - Average volume (20-day): yfinance primary, FinnhubClient fallback
    - Market cap: FinnhubClient (profile) primary, yfinance fallback
    - News: FinnhubClient only, no fallback
    - Sector ETF quote: same as real-time quote precedence

    Never fabricates values. Sets missing_data_flags for unavailable fields.
    Excludes API secrets from log output.
    """
    sector_buckets = config.get("sector_buckets", {})
    bucket = sector_buckets.get(sector_key, {})
    sector_name = bucket.get("name", sector_key)
    sector_etf = bucket.get("sector_etf")

    missing_data_flags: list[str] = []
    now = datetime.now(timezone.utc)

    # --- Real-time quote (price, bid, ask) ---
    # Primary: FinnhubClient, Fallback: yfinance
    current_price: float | None = None
    prev_close: float | None = None
    bid: float | None = None
    ask: float | None = None
    current_volume: float | None = None

    quote = _get_quote_finnhub(symbol, fh)
    if quote:
        current_price = quote.get("price")
        prev_close = quote.get("prev_close")
        # Finnhub quote doesn't return bid/ask directly; we'll try yfinance for that
    else:
        # Fallback to yfinance
        quote = _get_quote_yfinance(symbol)
        if quote:
            current_price = quote.get("price")
            prev_close = quote.get("prev_close")
            current_volume = quote.get("volume")

    if current_price is None:
        missing_data_flags.append("price")
    if prev_close is None:
        missing_data_flags.append("prev_close")

    # --- Current volume ---
    # Primary: FinnhubClient (from candles for today), Fallback: yfinance
    if current_volume is None:
        # Try to get current volume from Finnhub intraday candles
        try:
            candles = fh.get_candles(symbol, resolution="D", days=1)
            if candles and candles.get("volume"):
                # Last volume entry is today's
                current_volume = float(candles["volume"][-1])
        except Exception as exc:
            logger.warning("Finnhub volume fetch failed for %s: %s", symbol, type(exc).__name__)

    if current_volume is None:
        # yfinance fallback for volume
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            vol = ticker.fast_info.get("lastVolume")
            if vol and vol > 0:
                current_volume = float(vol)
        except Exception as exc:
            logger.warning("yfinance volume fallback failed for %s: %s", symbol, type(exc).__name__)

    if current_volume is None:
        missing_data_flags.append("volume")

    # --- Average volume (20-day) ---
    # Primary: yfinance, Fallback: FinnhubClient
    average_volume = _get_average_volume_yfinance(symbol)
    if average_volume is None:
        average_volume = _get_average_volume_finnhub(symbol, fh)
    if average_volume is None:
        missing_data_flags.append("average_volume")

    # --- Bid/Ask spread ---
    # Try yfinance for bid/ask (Finnhub free tier doesn't provide bid/ask in quote)
    bid, ask = _get_bid_ask_from_yfinance(symbol)

    # --- Market cap ---
    # Primary: FinnhubClient (profile), Fallback: yfinance
    market_cap = _get_market_cap_finnhub(symbol, fh)
    market_cap_source: str | None = None
    microcap_proxy_used = False

    if market_cap is not None:
        market_cap_source = "api"
    else:
        market_cap = _get_market_cap_yfinance(symbol)
        if market_cap is not None:
            market_cap_source = "api"
        else:
            # Proxy: use price * average_volume as rough indicator
            if (
                current_price is not None
                and current_price > 0
                and average_volume is not None
                and average_volume > 0
            ):
                # Very rough proxy — not a real market cap, just for micro-cap detection
                proxy_val = current_price * average_volume * 20  # rough 20-day dollar volume
                market_cap = proxy_val
                market_cap_source = "proxy"
                microcap_proxy_used = True
            else:
                missing_data_flags.append("market_cap")

    # --- Compute derived metrics ---
    # move_pct = (current_price - prev_close) / prev_close * 100
    move_pct: float | None = None
    if current_price is not None and prev_close is not None and prev_close != 0:
        move_pct = (current_price - prev_close) / prev_close * 100.0

    # relative_volume = current_volume / average_volume
    relative_volume: float | None = None
    if current_volume is not None and average_volume is not None and average_volume > 0:
        relative_volume = current_volume / average_volume

    # dollar_volume = current_price * current_volume
    dollar_volume: float | None = None
    if current_price is not None and current_volume is not None:
        dollar_volume = current_price * current_volume

    # spread_pct = (ask - bid) / midpoint * 100
    spread_pct: float | None = None
    spread_status = "unknown"
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        midpoint = (ask + bid) / 2.0
        if midpoint > 0:
            spread_pct = (ask - bid) / midpoint * 100.0
            spread_status = "known"
    if spread_status == "unknown":
        missing_data_flags.append("spread")

    # --- News collection ---
    # Primary: FinnhubClient company news, no fallback
    news_headlines: list[dict] | None = None
    news_freshness_minutes: float | None = None

    try:
        raw_news = fh.get_news(symbol, days=2)
        if raw_news:
            # Filter out generic market commentary
            relevant_news = [
                n for n in raw_news
                if n.get("headline") and not _is_generic_market_news(n["headline"])
            ]
            if relevant_news:
                news_headlines = relevant_news
                # Compute freshness from newest relevant headline
                newest_dt_str = relevant_news[0].get("datetime")
                if newest_dt_str:
                    try:
                        newest_dt = datetime.fromisoformat(newest_dt_str)
                        # Make timezone-aware if naive
                        if newest_dt.tzinfo is None:
                            newest_dt = newest_dt.replace(tzinfo=timezone.utc)
                        delta = now - newest_dt
                        news_freshness_minutes = max(0.0, delta.total_seconds() / 60.0)
                    except (ValueError, TypeError):
                        pass
            else:
                # All news was generic — treat as no relevant news
                news_headlines = []
        else:
            news_headlines = []
    except Exception as exc:
        logger.warning("Finnhub news fetch failed for %s: %s", symbol, type(exc).__name__)
        missing_data_flags.append("news")

    if news_freshness_minutes is None and "news" not in missing_data_flags:
        # News API worked but no relevant headlines found
        pass  # news_freshness_minutes stays None → triggers missing_news penalty

    # --- Sector confirmation ---
    sector_confirmed: bool | None = None
    sector_etf_move_pct: float | None = None

    if sector_etf:
        sector_confirmed = check_sector_confirmation(symbol, sector_etf, fh)

        # Also get sector ETF move_pct for context
        etf_quote = _get_quote_finnhub(sector_etf, fh)
        if not etf_quote:
            etf_quote = _get_quote_yfinance(sector_etf)
        if etf_quote and etf_quote.get("prev_close") and etf_quote["prev_close"] != 0:
            etf_price = etf_quote.get("price")
            etf_prev = etf_quote.get("prev_close")
            if etf_price is not None and etf_prev is not None:
                sector_etf_move_pct = (etf_price - etf_prev) / etf_prev * 100.0
    else:
        missing_data_flags.append("sector_etf")

    # --- Sanitize NaN/Inf values (treat as missing) ---
    def _sanitize(val: float | None) -> float | None:
        if val is None:
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    move_pct = _sanitize(move_pct)
    relative_volume = _sanitize(relative_volume)
    dollar_volume = _sanitize(dollar_volume)
    spread_pct = _sanitize(spread_pct)
    sector_etf_move_pct = _sanitize(sector_etf_move_pct)
    news_freshness_minutes = _sanitize(news_freshness_minutes)
    market_cap = _sanitize(market_cap)

    # --- Build CandidateRow ---
    row = CandidateRow(
        symbol=symbol,
        sector=sector_key,
        sector_name=sector_name,
        current_price=current_price,
        prev_close=prev_close,
        move_pct=move_pct,
        current_volume=current_volume,
        average_volume=average_volume,
        relative_volume=relative_volume,
        dollar_volume=dollar_volume,
        news_headlines=news_headlines,
        news_freshness_minutes=news_freshness_minutes,
        sector_etf=sector_etf,
        sector_etf_move_pct=sector_etf_move_pct,
        sector_confirmed=sector_confirmed,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        spread_status=spread_status,
        market_cap=market_cap,
        market_cap_source=market_cap_source,
        microcap_proxy_used=microcap_proxy_used,
        missing_data_flags=missing_data_flags,
        collected_at=now.isoformat(),
    )

    logger.debug(
        "Collected data for %s [%s]: price=%s, move=%.2f%%, rvol=%s, news_fresh=%s min",
        symbol,
        sector_key,
        current_price,
        move_pct if move_pct is not None else 0.0,
        f"{relative_volume:.2f}" if relative_volume is not None else "N/A",
        f"{news_freshness_minutes:.0f}" if news_freshness_minutes is not None else "N/A",
    )

    return row


# ---------------------------------------------------------------------------
# Scout Score Computation
# ---------------------------------------------------------------------------

# Total possible data fields for data_completeness calculation.
# These are the fields that can appear in missing_data_flags.
_TOTAL_POSSIBLE_DATA_FIELDS: int = 7  # price, prev_close, volume, average_volume, spread, news, market_cap, sector_etf
# Note: We count the fields that collect_candidate_data can flag as missing:
# "price", "prev_close", "volume", "average_volume", "spread", "news", "market_cap", "sector_etf"
_ALL_DATA_FIELDS: set[str] = {
    "price",
    "prev_close",
    "volume",
    "average_volume",
    "spread",
    "news",
    "market_cap",
    "sector_etf",
}


def compute_scout_score(row: CandidateRow, config: dict) -> CandidateRow:
    """Compute weighted scout_score with component scores. Returns modified row.

    Scoring logic:
    1. Read scoring_weights and scoring_caps from config.
    2. For each component, compute a normalized 0–1 score.
    3. Multiply each normalized score by its weight.
    4. Store component_scores dict on the row for auditability.
    5. Set scout_score = sum of weighted component scores (before penalties).
    6. Clamp scout_score to [0.0, 100.0].

    Penalties are subtracted separately by apply_score_penalties().
    The function is deterministic for identical inputs.

    Args:
        row: CandidateRow that has passed hard gates.
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        The modified CandidateRow with scout_score and component_scores set.
    """
    scoring_weights = config.get("scoring_weights", {})
    scoring_caps = config.get("scoring_caps", {})
    score_penalties_cfg = config.get("score_penalties", {})

    # Weights (defaults match config/sector_scout_config.yaml)
    w_move_pct = float(scoring_weights.get("move_pct", 25.0))
    w_relative_volume = float(scoring_weights.get("relative_volume", 20.0))
    w_dollar_volume = float(scoring_weights.get("dollar_volume", 15.0))
    w_news_freshness = float(scoring_weights.get("news_freshness", 20.0))
    w_sector_confirmation = float(scoring_weights.get("sector_confirmation", 10.0))
    w_spread_sanity = float(scoring_weights.get("spread_sanity", 5.0))
    w_data_completeness = float(scoring_weights.get("data_completeness", 5.0))

    # Caps
    move_pct_max = float(scoring_caps.get("move_pct_max", 15.0))
    relative_volume_max = float(scoring_caps.get("relative_volume_max", 10.0))

    # Stale news threshold for news_freshness normalization
    stale_news_threshold = float(
        score_penalties_cfg.get("stale_news_threshold_minutes", 120)
    )

    # --- Component 1: move_pct ---
    # Normalized: abs(move_pct) capped at move_pct_max, then / move_pct_max
    if row.move_pct is not None and move_pct_max > 0:
        capped_move = min(abs(row.move_pct), move_pct_max)
        norm_move_pct = capped_move / move_pct_max
    else:
        norm_move_pct = 0.0

    # --- Component 2: relative_volume ---
    # Normalized: capped at relative_volume_max, then / relative_volume_max
    if row.relative_volume is not None and relative_volume_max > 0:
        capped_rvol = min(row.relative_volume, relative_volume_max)
        norm_relative_volume = capped_rvol / relative_volume_max
    else:
        norm_relative_volume = 0.0

    # --- Component 3: dollar_volume ---
    # Log scale normalization against range $1M to $1B
    # log10(1_000_000) = 6, log10(1_000_000_000) = 9 → range of 3
    _DOLLAR_VOL_MIN = 1_000_000.0   # $1M floor
    _DOLLAR_VOL_MAX = 1_000_000_000.0  # $1B ceiling
    if row.dollar_volume is not None and row.dollar_volume > 0:
        # Clamp to range before log
        clamped_dv = max(_DOLLAR_VOL_MIN, min(row.dollar_volume, _DOLLAR_VOL_MAX))
        log_min = math.log10(_DOLLAR_VOL_MIN)  # 6.0
        log_max = math.log10(_DOLLAR_VOL_MAX)  # 9.0
        log_val = math.log10(clamped_dv)
        norm_dollar_volume = (log_val - log_min) / (log_max - log_min)
    else:
        norm_dollar_volume = 0.0

    # --- Component 4: news_freshness ---
    # Fresher is better. None → 0. Otherwise: max(0, 1 - minutes/threshold)
    if row.news_freshness_minutes is not None and stale_news_threshold > 0:
        norm_news_freshness = max(
            0.0, 1.0 - (row.news_freshness_minutes / stale_news_threshold)
        )
    else:
        norm_news_freshness = 0.0

    # --- Component 5: sector_confirmation ---
    # 1.0 if True, 0.0 if False/None
    if row.sector_confirmed is True:
        norm_sector_confirmation = 1.0
    else:
        norm_sector_confirmation = 0.0

    # --- Component 6: spread_sanity ---
    # 1.0 if spread_status == "known" and spread_pct is low (≤ 1%), scaled down for wider spreads
    # 0.0 if unknown
    if row.spread_status == "known" and row.spread_pct is not None:
        # Spread quality: perfect at 0%, degrades linearly to 0 at 5% (max_spread_pct hard gate)
        # Since hard gate already rejects > 5%, we normalize within 0-5% range
        max_spread_for_scoring = 5.0  # matches hard_gates.max_spread_pct default
        norm_spread_sanity = max(0.0, 1.0 - (row.spread_pct / max_spread_for_scoring))
    else:
        norm_spread_sanity = 0.0

    # --- Component 7: data_completeness ---
    # 1.0 - (len(missing_data_flags) / total_possible_fields)
    total_fields = len(_ALL_DATA_FIELDS)
    if total_fields > 0:
        missing_count = len(row.missing_data_flags) if row.missing_data_flags else 0
        # Cap missing_count at total_fields to avoid negative scores
        missing_count = min(missing_count, total_fields)
        norm_data_completeness = 1.0 - (missing_count / total_fields)
    else:
        norm_data_completeness = 1.0

    # --- Compute weighted component scores ---
    component_scores: dict[str, float] = {
        "move_pct": norm_move_pct * w_move_pct,
        "relative_volume": norm_relative_volume * w_relative_volume,
        "dollar_volume": norm_dollar_volume * w_dollar_volume,
        "news_freshness": norm_news_freshness * w_news_freshness,
        "sector_confirmation": norm_sector_confirmation * w_sector_confirmation,
        "spread_sanity": norm_spread_sanity * w_spread_sanity,
        "data_completeness": norm_data_completeness * w_data_completeness,
    }

    # --- Raw score = sum of weighted components ---
    raw_score = sum(component_scores.values())

    # --- Clamp to [0.0, 100.0] ---
    scout_score = max(0.0, min(100.0, raw_score))

    # --- Store results on row ---
    row.component_scores = component_scores
    row.scout_score = scout_score

    return row


# ---------------------------------------------------------------------------
# Ranking — Stable tie-breaking and finalist selection
# ---------------------------------------------------------------------------


def _sort_key(row: CandidateRow) -> tuple:
    """Return a composite sort key for stable, deterministic ordering.

    Sort order:
      1. scout_score descending
      2. relative_volume descending (None treated as 0)
      3. dollar_volume descending (None treated as 0)
      4. symbol ascending (lexicographic)

    Python's sort is stable, so equal keys preserve insertion order.
    We negate numeric values to achieve descending sort with a single
    ascending sorted() call.
    """
    return (
        -(row.scout_score),
        -(row.relative_volume if row.relative_volume is not None else 0.0),
        -(row.dollar_volume if row.dollar_volume is not None else 0.0),
        row.symbol,
    )


def rank_candidates(
    candidates_by_sector: dict[str, list[CandidateRow]],
    config: dict,
) -> tuple[dict[str, list[CandidateRow]], list[CandidateRow]]:
    """Rank candidates within each sector and globally with stable tie-breaking.

    Args:
        candidates_by_sector: Dict mapping sector_key to list of scored CandidateRows.
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        Tuple of (finalists_by_sector, global_finalists):
        - finalists_by_sector: Top N per sector (N = max_finalists_per_sector)
        - global_finalists: All finalists merged and re-sorted globally
    """
    budget_ceilings = config.get("budget_ceilings", {})
    max_finalists_per_sector: int = int(
        budget_ceilings.get("max_finalists_per_sector", 5)
    )

    finalists_by_sector: dict[str, list[CandidateRow]] = {}

    for sector_key, candidates in candidates_by_sector.items():
        # Sort within sector using stable tie-breaking key
        sorted_candidates = sorted(candidates, key=_sort_key)
        # Truncate to max_finalists_per_sector
        finalists_by_sector[sector_key] = sorted_candidates[:max_finalists_per_sector]

    # Merge all sector finalists into a global list
    global_finalists: list[CandidateRow] = []
    for sector_finalists in finalists_by_sector.values():
        global_finalists.extend(sector_finalists)

    # Sort the global list using the same stable tie-breaking key
    global_finalists.sort(key=_sort_key)

    return finalists_by_sector, global_finalists


# ---------------------------------------------------------------------------
# Sector Screener Orchestration
# ---------------------------------------------------------------------------


def run_sector_screeners(
    config: dict,
    core_watchlist: list[str],
    fh: FinnhubClient,
) -> dict:
    """Execute all enabled sector screeners and return ranked candidates.

    Iterates enabled sector buckets (up to max_sectors_per_run ceiling),
    collects data, applies hard gates, scores, ranks, and returns a
    structured result dict.

    Args:
        config: Parsed sector_scout_config.yaml dict.
        core_watchlist: Current Core_Watchlist symbols to exclude.
        fh: FinnhubClient instance for data collection.

    Returns:
        {
            "sectors_scanned": int,
            "candidates_by_sector": {sector_key: [CandidateRow, ...]},
            "finalists_by_sector": {sector_key: [CandidateRow, ...]},
            "global_finalists": [CandidateRow, ...],
            "rejections": [{"symbol": str, "sector": str, "reason_code": str}, ...],
            "reason_counts": {reason_code: int, ...},
            "budget_hits": [str, ...],
        }
    """
    budget_ceilings = config.get("budget_ceilings", {})
    max_sectors_per_run: int = int(budget_ceilings.get("max_sectors_per_run", 7))
    max_candidates_per_sector: int = int(budget_ceilings.get("max_candidates_per_sector", 20))

    sector_buckets = config.get("sector_buckets", {})
    core_set = set(core_watchlist)

    # Result accumulators
    candidates_by_sector: dict[str, list[CandidateRow]] = {}
    rejections: list[dict] = []
    budget_hits: list[str] = []
    sectors_scanned: int = 0

    for sector_key, bucket in sector_buckets.items():
        # Only process enabled buckets
        if not bucket.get("enabled", False):
            continue

        # Enforce max_sectors_per_run ceiling
        if sectors_scanned >= max_sectors_per_run:
            budget_hits.append(f"max_sectors_per_run:{max_sectors_per_run}")
            emit_scout_event("BUDGET_CEILING_HIT", {
                "ceiling_type": "max_sectors_per_run",
                "limit_value": max_sectors_per_run,
                "context": f"Reached {max_sectors_per_run} sectors, stopping iteration",
            })
            break

        sectors_scanned += 1

        # Load symbol universe from bucket config
        symbols: list[str] = bucket.get("symbols", [])
        core_re_ranking: bool = bucket.get("core_re_ranking", False)

        # Exclude Core_Watchlist symbols unless core_re_ranking is enabled
        if not core_re_ranking:
            symbols = [s for s in symbols if s not in core_set]

        # Enforce max_candidates_per_sector ceiling
        if len(symbols) > max_candidates_per_sector:
            symbols = symbols[:max_candidates_per_sector]
            budget_hits.append(
                f"max_candidates_per_sector:{sector_key}:{max_candidates_per_sector}"
            )
            emit_scout_event("BUDGET_CEILING_HIT", {
                "ceiling_type": "max_candidates_per_sector",
                "limit_value": max_candidates_per_sector,
                "context": f"Sector {sector_key} truncated to {max_candidates_per_sector} symbols",
            })

        # Process each symbol in the sector
        scored_candidates: list[CandidateRow] = []

        for symbol in symbols:
            # Collect candidate data
            row = collect_candidate_data(symbol, sector_key, config, fh)

            # Apply hard gates
            passed, reason_code = apply_hard_gates(row, config)

            if not passed:
                # Track rejection
                rejections.append({
                    "symbol": symbol,
                    "sector": sector_key,
                    "reason_code": reason_code or "hard_gate:unknown",
                })
                continue

            # Compute scout score
            row = compute_scout_score(row, config)

            # Apply score penalties
            row = apply_score_penalties(row, config)

            scored_candidates.append(row)

        # Store scored candidates for this sector
        candidates_by_sector[sector_key] = scored_candidates

        # Log sector screen event
        emit_scout_event("SECTOR_SCREEN", {
            "sector_key": sector_key,
            "candidates_found": len(symbols),
            "passed_gates": len(scored_candidates),
            "top_score": scored_candidates[0].scout_score if scored_candidates else 0.0,
        })

    # Rank candidates across all sectors and select finalists
    finalists_by_sector, global_finalists = rank_candidates(candidates_by_sector, config)

    # Build reason_counts from all rejections
    reason_counts: dict[str, int] = {}
    for rejection in rejections:
        rc = rejection["reason_code"]
        reason_counts[rc] = reason_counts.get(rc, 0) + 1

    return {
        "sectors_scanned": sectors_scanned,
        "candidates_by_sector": candidates_by_sector,
        "finalists_by_sector": finalists_by_sector,
        "global_finalists": global_finalists,
        "rejections": rejections,
        "reason_counts": reason_counts,
        "budget_hits": budget_hits,
    }
