"""
LSEG Workspace news client with caching + graceful fallbacks.

This module fetches real-time news headlines from LSEG/Refinitiv,
filtered by ticker symbol for brand-specific news feeds.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import refinitiv.data as rd
    from refinitiv.data.content import news
except Exception:  # pragma: no cover - optional dependency
    rd = None
    news = None

try:
    from ingest.ticker_mapping import brand_to_ticker, supported_tickers
except ImportError:  # Allow running from inside the ingest/ folder
    from ticker_mapping import brand_to_ticker, supported_tickers


# Connection throttle: don't retry LSEG connection more than once per N seconds
_last_connection_attempt: Optional[datetime] = None
_last_connection_success: bool = False
CONNECTION_THROTTLE_SECONDS = 30  # Wait 30s between failed connection attempts


class LSEGNewsClient:
    """Fetches LSEG news headlines with caching and fallback behavior."""

    def __init__(
        self,
        cache_path: str = "data/news_cache.json",
        cache_ttl_minutes: Optional[int] = None,
        fallback_enabled: Optional[bool] = None,
        app_key: Optional[str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        cache_path_obj = Path(cache_path)
        self.cache_path = cache_path_obj if cache_path_obj.is_absolute() else base_dir / cache_path_obj

        # News refreshes more frequently than consensus (2 min default vs 15 min)
        self.cache_ttl_minutes = cache_ttl_minutes or int(os.getenv("LSEG_NEWS_CACHE_TTL_MINUTES", "2"))
        self.fallback_enabled = (
            fallback_enabled
            if fallback_enabled is not None
            else os.getenv("LSEG_NEWS_FALLBACK_ENABLED", "true").lower() == "true"
        )
        self.app_key = app_key or os.getenv("LSEG_APP_KEY")
        self._session_open = False
        self.last_error: Optional[str] = None

        # News older than 6 hours is too stale to display
        self.stale_threshold = timedelta(hours=6)

    def _should_attempt_connection(self) -> bool:
        """Check if we should attempt LSEG connection (throttle failed attempts)."""
        global _last_connection_attempt, _last_connection_success

        # Always allow if last attempt succeeded
        if _last_connection_success:
            return True

        # If never attempted, allow
        if _last_connection_attempt is None:
            return True

        # Throttle failed attempts
        elapsed = (datetime.now(timezone.utc) - _last_connection_attempt).total_seconds()
        return elapsed >= CONNECTION_THROTTLE_SECONDS

    def connect(self, retries: int = 1, backoff_seconds: int = 1) -> bool:
        """Try to open a session to LSEG Workspace with retry + backoff."""
        global _last_connection_attempt, _last_connection_success

        if rd is None:
            self.last_error = "refinitiv-data is not installed"
            return False

        # Check throttle
        if not self._should_attempt_connection():
            self.last_error = "Connection throttled - waiting before retry"
            return False

        _last_connection_attempt = datetime.now(timezone.utc)

        for attempt in range(retries):
            try:
                if self.app_key:
                    rd.open_session(app_key=self.app_key)
                else:
                    rd.open_session()
                self._session_open = True
                _last_connection_success = True
                return True
            except Exception as exc:
                self.last_error = str(exc)
                if attempt < retries - 1:
                    time.sleep(backoff_seconds * (2**attempt))

        _last_connection_success = False
        return False

    def disconnect(self) -> None:
        """Close the LSEG session cleanly."""
        if rd is None or not self._session_open:
            return
        try:
            rd.close_session()
        finally:
            self._session_open = False

    def is_connected(self) -> bool:
        """Lightweight health check for session state."""
        return self._session_open

    def get_headlines(self, ticker: str, count: int = 15) -> List[Dict]:
        """
        Public entry point used by dashboard.
        Tries live LSEG, then cache, then returns empty list.

        Args:
            ticker: Refinitiv ticker (e.g., "SBUX.O")
            count: Number of headlines to fetch (default 15)

        Returns:
            List of headline dictionaries with keys:
            - headline: The news headline text
            - timestamp: ISO timestamp when the story was published
            - story_id: Unique identifier for the story
            - source: News source (e.g., "Reuters", "Dow Jones")
            - image_url: URL to news image (if available)
            - language: Language code (e.g., "en", "ja")
            - is_translated: Whether headline was translated
            - story_url: URL to full story (if available)
            - is_stale: Whether this came from stale cache
        """
        # 1) Prefer cache if it's still fresh (keeps dashboard fast).
        cached, cache_age = self._read_cache(ticker)

        if cached and cache_age <= timedelta(minutes=self.cache_ttl_minutes):
            for item in cached:
                item["is_stale"] = False
            return cached

        # 2) Attempt live LSEG fetch (with throttling).
        headlines = None
        if self._should_attempt_connection() and self.connect():
            try:
                headlines = self._fetch_live_headlines(ticker, count)
            finally:
                self.disconnect()

        if headlines:
            self._write_cache(ticker, headlines)
            return headlines

        # 3) Fallback to cache if enabled, even if it's older.
        if self.fallback_enabled and cached:
            is_stale = cache_age > self.stale_threshold
            for item in cached:
                item["is_stale"] = is_stale
            return cached

        return []

    def get_headlines_for_brand(self, brand: str, count: int = 15) -> List[Dict]:
        """
        Convenience method that takes a brand name instead of ticker.

        Args:
            brand: Brand name (e.g., "STARBUCKS")
            count: Number of headlines to fetch

        Returns:
            List of headline dictionaries
        """
        ticker = brand_to_ticker(brand)
        if not ticker:
            return []
        return self.get_headlines(ticker, count)

    def get_story(self, story_id: str) -> Optional[Dict]:
        """
        Fetch full story content by story ID.

        Args:
            story_id: The story identifier from headlines

        Returns:
            Dictionary with story content, or None if unavailable
        """
        if rd is None or not story_id:
            return None

        if not self._should_attempt_connection():
            return None

        if not self.connect():
            return None

        try:
            story = rd.news.get_story(story_id)
            if story is None:
                return None

            return {
                "story_id": story_id,
                "content": str(story) if story else "",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            self.last_error = str(exc)
            return None
        finally:
            self.disconnect()

    def refresh_cache(self, tickers: Optional[List[str]] = None, count: int = 15) -> Dict[str, List[Dict]]:
        """
        Refresh cached news for a list of tickers.
        Intended for cron/scheduled runs to keep cache warm.

        Args:
            tickers: List of tickers to refresh (defaults to all supported)
            count: Number of headlines per ticker

        Returns:
            Dictionary mapping ticker to list of headlines
        """
        tickers = tickers or supported_tickers()
        results: Dict[str, List[Dict]] = {}

        if not self.connect():
            return results

        try:
            for ticker in tickers:
                headlines = self._fetch_live_headlines(ticker, count)
                if not headlines:
                    continue
                self._write_cache(ticker, headlines)
                results[ticker] = headlines
        finally:
            self.disconnect()

        return results

    def _fetch_live_headlines(self, ticker: str, count: int = 15) -> List[Dict]:
        """Pull news headlines from LSEG and normalize the response."""
        if rd is None:
            return []

        try:
            # Query LSEG news API for headlines related to this ticker
            # The query format "R:{ticker}" searches for news mentioning the RIC
            response = rd.news.get_headlines(
                query=f"R:{ticker}",
                count=count,
            )

            if response is None or response.empty:
                return []

            headlines = []
            for _, row in response.iterrows():
                headline_text = self._safe_str(row.get("headline", row.get("text", "")))
                language = self._detect_language(headline_text, row)
                is_translated = False

                # Translate non-English headlines
                if language != "en" and headline_text:
                    headline_text, is_translated = self._translate_headline(headline_text, language)

                headline_data = {
                    "headline": headline_text,
                    "timestamp": self._to_iso(row.get("versionCreated", row.get("pubDate"))),
                    "story_id": self._safe_str(row.get("storyId", row.get("id", ""))),
                    "source": self._extract_source(row),
                    "image_url": self._extract_image_url(row),
                    "language": language,
                    "is_translated": is_translated,
                    "story_url": self._extract_story_url(row),
                    "is_stale": False,
                }
                if headline_data["headline"]:  # Only include if we have a headline
                    headlines.append(headline_data)

            return headlines

        except Exception as exc:
            self.last_error = str(exc)
            return []

    def _detect_language(self, text: str, row) -> str:
        """Detect the language of headline text."""
        # First check if LSEG provides language metadata
        for col in ["language", "lang", "languageCode"]:
            if col in row.index and row[col]:
                lang = str(row[col]).lower()[:2]
                if lang in ["en", "ja", "zh", "ko", "de", "fr", "es", "pt", "it", "ru"]:
                    return lang

        # Simple heuristic detection based on character sets
        if not text:
            return "en"

        # Japanese (Hiragana, Katakana, or CJK with Japanese context)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return "ja"

        # Chinese (CJK without Japanese kana)
        if re.search(r'[\u4E00-\u9FFF]', text) and not re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return "zh"

        # Korean (Hangul)
        if re.search(r'[\uAC00-\uD7AF\u1100-\u11FF]', text):
            return "ko"

        # Cyrillic (Russian, etc.)
        if re.search(r'[\u0400-\u04FF]', text):
            return "ru"

        # Arabic
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar"

        # Default to English
        return "en"

    def _translate_headline(self, text: str, source_lang: str) -> Tuple[str, bool]:
        """
        Translate a headline to English using LibreTranslate.

        Args:
            text: The text to translate
            source_lang: Source language code (e.g., "ja", "zh")

        Returns:
            Tuple of (translated_text, was_translated)
        """
        if requests is None:
            return text, False

        # LibreTranslate public instance (free, no API key needed)
        # You can also self-host or use other instances
        translate_url = os.getenv(
            "LIBRETRANSLATE_URL",
            "https://libretranslate.com/translate"
        )

        try:
            response = requests.post(
                translate_url,
                json={
                    "q": text,
                    "source": source_lang,
                    "target": "en",
                    "format": "text",
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )

            if response.ok:
                result = response.json()
                translated = result.get("translatedText", "")
                if translated and translated != text:
                    return translated, True

        except Exception as e:
            # Translation failed - return original
            self.last_error = f"Translation error: {e}"

        return text, False

    def _extract_source(self, row) -> str:
        """Extract the news source from a row, with fallbacks."""
        # Try different possible column names
        for col in ["sourceCode", "source", "provider", "newsSrc"]:
            if col in row.index and row[col]:
                source = str(row[col])
                # Clean up common source codes
                source_map = {
                    "NS:RTRS": "Reuters",
                    "NS:DJN": "Dow Jones",
                    "NS:BW": "Business Wire",
                    "NS:PRN": "PR Newswire",
                    "NS:GNW": "GlobeNewswire",
                    "NS:MKW": "MarketWatch",
                    "NS:AFX": "AFX News",
                    "NS:NIKKEI": "Nikkei",
                }
                return source_map.get(source, source.replace("NS:", ""))
        return "Unknown"

    def _extract_image_url(self, row) -> Optional[str]:
        """Extract image URL from news row if available."""
        for col in ["thumbnailUrl", "imageUrl", "thumbnail", "image", "mediaUrl"]:
            if col in row.index and row[col]:
                url = str(row[col])
                if url.startswith("http"):
                    return url
        return None

    def _extract_story_url(self, row) -> Optional[str]:
        """Extract URL to full story if available."""
        for col in ["newsLink", "storyUrl", "url", "link"]:
            if col in row.index and row[col]:
                url = str(row[col])
                if url.startswith("http"):
                    return url
        return None

    def _safe_str(self, value) -> str:
        """Safely convert a value to string."""
        if value is None:
            return ""
        return str(value).strip()

    def _to_iso(self, value) -> Optional[str]:
        """Convert a datetime value to ISO format string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        try:
            # Try parsing as ISO string
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt.isoformat()
        except Exception:
            # Return as-is if we can't parse it
            return str(value) if value else None

    def _load_cache(self) -> Dict:
        """Load the cache from disk."""
        if not self.cache_path.exists():
            return {"items": {}, "updated_at": None}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {"items": {}, "updated_at": None}

    def _save_cache(self, cache: Dict) -> None:
        """Save the cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)

    def _read_cache(self, ticker: str) -> Tuple[Optional[List[Dict]], timedelta]:
        """
        Read cached headlines for a ticker.

        Returns:
            Tuple of (headlines list, cache age)
        """
        cache = self._load_cache()
        item = cache.get("items", {}).get(ticker)
        if not item:
            return None, timedelta.max

        headlines = item.get("headlines", [])
        fetched_at = item.get("fetched_at")

        if not fetched_at:
            return headlines, timedelta.max

        try:
            fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        except Exception:
            return headlines, timedelta.max

        age = datetime.now(timezone.utc) - fetched_dt
        return headlines, age

    def _write_cache(self, ticker: str, headlines: List[Dict]) -> None:
        """Write headlines to the cache."""
        cache = self._load_cache()
        cache.setdefault("items", {})[ticker] = {
            "headlines": headlines,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_cache(cache)
