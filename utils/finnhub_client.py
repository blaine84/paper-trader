"""
Finnhub API wrapper.
Handles quotes, candles, news, and basic technicals.
"""

import os
import finnhub
from datetime import datetime, timedelta
import time


class FinnhubClient:
    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in environment")
        self.client = finnhub.Client(api_key=api_key)

    def get_quote(self, symbol: str) -> dict:
        """Current price quote."""
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
        """
        now = int(time.time())
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        c = self.client.stock_candles(symbol, resolution, since, now)
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
        return self.client.recommendation_trends(symbol)[:3] or []

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        status = self.client.market_status(exchange="US")
        return status.get("isOpen", False)
