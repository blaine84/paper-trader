"""
Finnhub API wrapper.
Handles quotes, candles, news, and basic technicals.
Free tier limit: 60 calls/minute — rate limiting is built in.
"""

import os
import time
import finnhub
from datetime import datetime, timedelta


class FinnhubClient:
    CALLS_PER_MINUTE = 55  # stay under the 60 limit with a small buffer

    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in environment")
        self.client = finnhub.Client(api_key=api_key)
        self._call_times = []

    def _rate_limit(self):
        """Block if we're approaching the per-minute call limit."""
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.CALLS_PER_MINUTE:
            wait = 60 - (now - self._call_times[0]) + 0.5
            if wait > 0:
                time.sleep(wait)
            self._call_times = []
        self._call_times.append(time.time())

    def get_quote(self, symbol: str) -> dict:
        """Current price quote."""
        self._rate_limit()
        q = self.client.quote(symbol)
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
        Free Finnhub tier only supports daily (D) candles.
        Intraday resolutions require a paid plan — falls back to daily if 403.
        """
        now = int(time.time())
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        try:
            self._rate_limit()
            c = self.client.stock_candles(symbol, resolution, since, now)
        except Exception as e:
            if "403" in str(e) and resolution != "D":
                since = int((datetime.utcnow() - timedelta(days=30)).timestamp())
                try:
                    self._rate_limit()
                    c = self.client.stock_candles(symbol, "D", since, now)
                except Exception:
                    return {}
            else:
                return {}
        if c.get("s") != "ok":
            return {}
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

    def get_news(self, symbol: str, days: int = 1) -> list:
        """Recent company news."""
        self._rate_limit()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        news = self.client.company_news(symbol, _from=since, to=today)
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
        news = self.client.general_news(category)
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
        data = self.client.company_basic_financials(symbol, "all")
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
        return self.client.recommendation_trends(symbol)[:3] or []

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        self._rate_limit()
        status = self.client.market_status(exchange="US")
        return status.get("isOpen", False)
