"""
Finnhub API wrapper.
Handles quotes, candles, news, and basic technicals.
Free tier limit: 60 calls/minute — rate limiting is built in.
"""

import os
import time
import logging
import threading
import finnhub
import requests
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


class FinnhubClient:
    CALLS_PER_MINUTE = 55  # stay under the 60 limit with a small buffer
    _shared_call_times = []  # shared across all instances
    _shared_lock = None
    _yfinance_lock = threading.Lock()

    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in environment")
        self.client = finnhub.Client(api_key=api_key)

    def _rate_limit(self):
        """Block if we're approaching the per-minute call limit. Shared across all instances."""
        now = time.time()
        FinnhubClient._shared_call_times = [t for t in FinnhubClient._shared_call_times if now - t < 60]
        if len(FinnhubClient._shared_call_times) >= self.CALLS_PER_MINUTE:
            wait = 60 - (now - FinnhubClient._shared_call_times[0]) + 0.5
            if wait > 0:
                log.info(f"Finnhub rate limit: pausing {wait:.0f}s")
                time.sleep(wait)
            FinnhubClient._shared_call_times = []
        FinnhubClient._shared_call_times.append(time.time())

    def _call_with_retry(self, fn, retries=2):
        """Retry on 429/502 with a 60s pause."""
        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as e:
                err = str(e)
                if ("429" in err or "502" in err) and attempt < retries:
                    log.warning(f"Finnhub {err[:60]}... retrying in 60s")
                    time.sleep(60)
                    continue
                raise

    def get_quote(self, symbol: str) -> dict:
        """Current price quote."""
        self._rate_limit()
        q = self._call_with_retry(lambda: self.client.quote(symbol))
        return {
            "symbol": symbol,
            "price": q["c"],        # current
            "open": q["o"],
            "high": q["h"],
            "low": q["l"],
            "prev_close": q["pc"],
            "change_pct": round((q["c"] - q["pc"]) / q["pc"] * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_candles(self, symbol: str, resolution: str = "5", days: int = 2) -> dict:
        """
        OHLCV candles.
        resolution: 1, 5, 15, 30, 60, D, W, M

        For sub-daily resolutions (1, 5, 15, 30, 60): Alpaca first, yfinance fallback, Finnhub fallback.
        For daily+ resolutions (D, W, M): Finnhub first, yfinance fallback (unchanged).

        Returns dict with keys: symbol, resolution, timestamps, open, high, low, close, volume, source.
        The 'source' field identifies which provider supplied the data.
        Returns empty dict {} when both sources fail.
        """
        SUB_DAILY = {"1", "5", "15", "30", "60"}

        if resolution in SUB_DAILY:
            # Alpaca primary for intraday aggregate bars.
            result = self._get_candles_alpaca(symbol, resolution, days)
            if result:
                result["source"] = "alpaca"
                return result
            # yfinance fallback for intraday
            result = self._get_candles_yfinance(symbol, resolution, days)
            if result:
                result["source"] = "yfinance"
                return result
            # Finnhub fallback
            result = self._get_candles_finnhub(symbol, resolution, days)
            if result:
                result["source"] = "finnhub"
                return result
            return {}
        else:
            # Daily+ unchanged: Finnhub primary
            result = self._get_candles_finnhub(symbol, resolution, days)
            if result:
                result["source"] = "finnhub"
                return result
            result = self._get_candles_yfinance(symbol, resolution, days)
            if result:
                result["source"] = "yfinance"
                return result
            return {}

    def _get_candles_alpaca(self, symbol: str, resolution: str, days: int) -> dict:
        """Alpaca Market Data candle fetch for intraday aggregate bars."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return {}

        timeframe_map = {
            "1": "1Min",
            "5": "5Min",
            "15": "15Min",
            "30": "30Min",
            "60": "1Hour",
        }
        timeframe = timeframe_map.get(resolution)
        if not timeframe:
            return {}

        end = datetime.utcnow()
        start = end - timedelta(days=days)
        base_url = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start.replace(microsecond=0).isoformat() + "Z",
            "end": end.replace(microsecond=0).isoformat() + "Z",
            "limit": 10000,
            "adjustment": os.getenv("ALPACA_ADJUSTMENT", "raw"),
            "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
            "sort": "asc",
        }
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        try:
            resp = requests.get(
                f"{base_url}/v2/stocks/bars",
                params=params,
                headers=headers,
                timeout=float(os.getenv("ALPACA_TIMEOUT_SECONDS", "10")),
            )
            resp.raise_for_status()
            payload = resp.json()
            bars = (payload.get("bars") or {}).get(symbol) or []
            if not bars:
                return {}

            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            for bar in bars:
                ts_raw = bar.get("t")
                if not ts_raw:
                    continue
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                timestamps.append(int(ts.timestamp()))
                opens.append(bar.get("o"))
                highs.append(bar.get("h"))
                lows.append(bar.get("l"))
                closes.append(bar.get("c"))
                volumes.append(bar.get("v", 0))

            if not timestamps:
                return {}

            return {
                "symbol": symbol,
                "resolution": resolution,
                "timestamps": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        except Exception as e:
            log.warning(f"Alpaca candle fetch failed for {symbol}: {e}")
            return {}

    def _get_candles_finnhub(self, symbol: str, resolution: str, days: int) -> dict:
        """Finnhub API candle fetch with rate limiting and retry logic."""
        now = int(time.time())
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        try:
            self._rate_limit()
            c = self._call_with_retry(lambda: self.client.stock_candles(symbol, resolution, since, now))
            if c.get("s") == "ok":
                return {
                    "symbol": symbol,
                    "resolution": resolution,
                    "timestamps": c["t"],
                    "open": c["o"],
                    "high": c["h"],
                    "low": c["l"],
                    "close": c["c"],
                    "volume": c["v"],
                }
        except Exception as e:
            if "403" not in str(e):
                log.warning(f"Finnhub candle fetch failed for {symbol}: {e}")
            return {}
        return {}

    def _get_candles_yfinance(self, symbol: str, resolution: str, days: int) -> dict:
        """yfinance fallback for intraday candles."""
        try:
            import yfinance as yf
            # Map Finnhub resolution to yfinance interval
            interval_map = {
                "1": "1m", "5": "5m", "15": "15m",
                "30": "30m", "60": "1h", "D": "1d",
            }
            interval = interval_map.get(resolution, "5m")
            # yfinance limits: 1m = 7 days, 5m/15m/30m = 60 days
            period = f"{min(days, 7)}d" if resolution == "1" else f"{min(days, 59)}d"
            # yfinance uses module-level/shared state internally. Concurrent
            # downloads from Analyst worker threads have produced cross-symbol
            # candle contamination (e.g. IWM receiving SPY-priced candles).
            # Serialize downloads so each symbol's OHLCV frame is isolated.
            with FinnhubClient._yfinance_lock:
                df = yf.download(symbol, period=period, interval=interval,
                                 progress=False, auto_adjust=True, threads=False)
            if df.empty:
                return {}

            import pandas as pd

            # Newer yfinance versions can return a MultiIndex even for a
            # single ticker, e.g. ("Open", "SPY"). Select the requested
            # ticker level first, then normalize to flat OHLCV columns.
            if isinstance(df.columns, pd.MultiIndex):
                ticker_level = df.columns.names.index("Ticker") if "Ticker" in df.columns.names else -1
                if symbol in df.columns.get_level_values(ticker_level):
                    df = df.xs(symbol, axis=1, level=ticker_level, drop_level=True)
                else:
                    df.columns = df.columns.get_level_values(0)

            def series_values(column: str) -> list:
                values = df[column]
                # Defensive fallback: duplicated/tuple columns can make pandas
                # return a DataFrame here; candles need a one-dimensional list.
                if isinstance(values, pd.DataFrame):
                    values = values.iloc[:, 0]
                return values.tolist()

            return {
                "symbol": symbol,
                "resolution": resolution,
                "timestamps": [int(t.timestamp()) for t in df.index],
                "open": series_values("Open"),
                "high": series_values("High"),
                "low": series_values("Low"),
                "close": series_values("Close"),
                "volume": series_values("Volume"),
            }
        except Exception as e:
            log.warning(f"yfinance fallback failed for {symbol}: {e}")
            return {}

    def get_news(self, symbol: str, days: int = 1) -> list:
        """Recent company news."""
        self._rate_limit()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        news = self._call_with_retry(lambda: self.client.company_news(symbol, _from=since, to=today))
        return [
            {
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "datetime": datetime.fromtimestamp(n.get("datetime", 0)).isoformat(),
                "url": n.get("url"),
            }
            for n in (news or [])[:10]  # cap at 10
        ]

    def get_market_news(self, category: str = "general") -> list:
        """General market news."""
        self._rate_limit()
        news = self._call_with_retry(lambda: self.client.general_news(category))
        return [
            {
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "datetime": datetime.fromtimestamp(n.get("datetime", 0)).isoformat(),
            }
            for n in (news or [])[:10]
        ]

    def get_basic_financials(self, symbol: str) -> dict:
        """Key financial metrics."""
        self._rate_limit()
        data = self._call_with_retry(lambda: self.client.company_basic_financials(symbol, "all"))
        metrics = data.get("metric", {})
        return {
            "symbol": symbol,
            "52w_high": metrics.get("52WeekHigh"),
            "52w_low": metrics.get("52WeekLow"),
            "beta": metrics.get("beta"),
            "pe_ratio": metrics.get("peBasicExclExtraTTM"),
            "eps": metrics.get("epsBasicExclExtraAnnual"),
            "rsi": metrics.get("rsi14D"),
        }

    def get_recommendation_trends(self, symbol: str) -> list:
        """Analyst recommendations."""
        self._rate_limit()
        return self._call_with_retry(lambda: self.client.recommendation_trends(symbol)[:3] or [])

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        self._rate_limit()
        status = self._call_with_retry(lambda: self.client.market_status(exchange="US"))
        return status.get("isOpen", False)
