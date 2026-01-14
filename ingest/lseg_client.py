"""
LSEG Workspace consensus client with caching + graceful fallbacks.

This module is used by both the pipeline (scheduled refresh) and the predictor
to avoid hitting the LSEG API on every dashboard load.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import refinitiv.data as rd
except Exception:  # pragma: no cover - optional dependency
    rd = None

try:
    from ingest.ticker_mapping import brand_to_ticker, get_fiscal_period_params, supported_tickers
except ImportError:  # Allow running from inside the ingest/ folder
    from ticker_mapping import brand_to_ticker, get_fiscal_period_params, supported_tickers


LSEG_FIELDS = [
    "TR.RevenueSmartEst",
    "TR.RevenueMean",
    "TR.RevenueHigh",
    "TR.RevenueLow",
    "TR.EPSSmartEst",
    "TR.NumOfEst",
    "TR.OriginalAnnouncementDate",
]

# Column name fallbacks for common LSEG display outputs
FIELD_ALIASES = {
    "revenue_smart_estimate": ["TR.RevenueSmartEst", "Revenue - SmartEstimate"],
    "revenue_mean": ["TR.RevenueMean", "Revenue - Mean"],
    "revenue_high": ["TR.RevenueHigh", "Revenue - High"],
    "revenue_low": ["TR.RevenueLow", "Revenue - Low"],
    "eps_smart_estimate": ["TR.EPSSmartEst", "Earnings Per Share - SmartEstimate", "EPS - SmartEstimate"],
    "num_analysts": ["TR.NumOfEst", "Number of Analysts", "Num of Estimates"],
    "earnings_date": [
        "TR.OriginalAnnouncementDate",
        "Original Announcement Date",
        "Announcement Date",
        "Earnings Announcement Date",
    ],
    "period_label": ["TR.FiscalPeriod", "Fiscal Period", "Period"],
    "fiscal_year": ["TR.FiscalYear", "Fiscal Year", "Year"],
}

# Circuit Breaker: Stop trying if we fail too many times
_consecutive_failures: int = 0
_circuit_break_until: Optional[datetime] = None
MAX_CONSECUTIVE_FAILURES = 3
CIRCUIT_BREAK_DURATION = 300  # 5 minutes break after max failures


class LSEGConsensusClient:
    """Fetches LSEG consensus data with caching and fallback behavior."""

    def __init__(
        self,
        cache_path: str = "data/consensus_cache.json",
        cache_ttl_minutes: Optional[int] = None,
        fallback_enabled: Optional[bool] = None,
        app_key: Optional[str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        cache_path = Path(cache_path)
        self.cache_path = cache_path if cache_path.is_absolute() else base_dir / cache_path

        self.cache_ttl_minutes = cache_ttl_minutes or int(os.getenv("LSEG_CACHE_TTL_MINUTES", "15"))
        self.fallback_enabled = (
            fallback_enabled
            if fallback_enabled is not None
            else os.getenv("LSEG_FALLBACK_ENABLED", "true").lower() == "true"
        )
        self.app_key = app_key or os.getenv("LSEG_APP_KEY")
        self._session_open = False
        self.last_error: Optional[str] = None

        # Treat anything older than 24h as too stale to trust.
        self.stale_threshold = timedelta(hours=24)

    def _should_attempt_connection(self) -> bool:
        """Check if we should attempt LSEG connection (circuit breaker)."""
        global _circuit_break_until

        if _circuit_break_until:
            if datetime.now(timezone.utc) < _circuit_break_until:
                self.last_error = "Circuit breaker active - LSEG connection suspended"
                return False
            else:
                _circuit_break_until = None
        return True

    def connect(self, retries: int = 3, backoff_seconds: int = 1) -> bool:
        """Try to open a session to LSEG Workspace with retry + backoff + circuit breaker."""
        global _consecutive_failures, _circuit_break_until

        if rd is None:
            self.last_error = "refinitiv-data is not installed"
            return False

        if not self._should_attempt_connection():
            return False

        for attempt in range(retries):
            try:
                if self.app_key:
                    rd.open_session(app_key=self.app_key)
                else:
                    rd.open_session()
                
                # Success
                self._session_open = True
                _consecutive_failures = 0
                return True
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(backoff_seconds * (2**attempt))
        
        # All retries failed
        _consecutive_failures += 1
        if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            _circuit_break_until = datetime.now(timezone.utc) + timedelta(seconds=CIRCUIT_BREAK_DURATION)
            print(f"LSEG Consensus: Circuit breaker tripped. Pausing for {CIRCUIT_BREAK_DURATION}s.")
            
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

    def get_consensus(self, brand: str, quarter: Optional[str]) -> Optional[Dict]:
        """
        Public entry point used by predictor/dashboard.
        Tries live LSEG, then cache, then returns None.
        """
        ticker = brand_to_ticker(brand)
        if not ticker:
            return None

        period_info = get_fiscal_period_params(ticker, quarter)
        period_label = period_info["period_label"] or "FQ0"

        # 1) Prefer cache if it's still fresh (keeps dashboard fast).
        cached, cache_age = self._read_cache(ticker, period_label)
        if not cached and period_label != "FQ0":
            cached, cache_age = self._read_cache(ticker, "FQ0")

        if cached and cache_age <= timedelta(minutes=self.cache_ttl_minutes):
            cached["source"] = "lseg_cache"
            cached["is_stale"] = False
            return cached

        # 2) Attempt live LSEG fetch.
        data = None
        if self.connect():
            try:
                data = self._fetch_live(ticker, period_info)
            finally:
                self.disconnect()

        if data:
            data["source"] = "lseg_live"
            self._write_cache(ticker, period_label, data)
            return data

        # 3) Fallback to cache if enabled, even if it's older.
        if self.fallback_enabled and cached:
            is_stale = cache_age > self.stale_threshold
            cached["source"] = "lseg_cache_stale" if is_stale else "lseg_cache"
            cached["is_stale"] = is_stale
            return cached

        return None

    def refresh_cache(self, tickers: Optional[list[str]] = None, quarter: Optional[str] = None) -> Dict[str, Dict]:
        """
        Refresh cached consensus for a list of tickers.
        Intended for cron/scheduled runs to keep cache warm.
        """
        tickers = tickers or supported_tickers()
        results: Dict[str, Dict] = {}

        if not self.connect():
            return results

        try:
            for ticker in tickers:
                period_info = get_fiscal_period_params(ticker, quarter)
                period_label = period_info["period_label"] or "FQ0"
                data = self._fetch_live(ticker, period_info)
                if not data:
                    continue
                data["source"] = "lseg_live"
                self._write_cache(ticker, period_label, data)
                results[ticker] = data
        finally:
            self.disconnect()

        return results

    def _fetch_live(self, ticker: str, period_info: Dict) -> Optional[Dict]:
        """Pull consensus data from LSEG and normalize column names."""
        if rd is None:
            return None

        params = period_info["parameters"]
        period_label = period_info["period_label"]

        try:
            df = rd.get_data(universe=[ticker], fields=LSEG_FIELDS, parameters=params)
        except Exception:
            # If the param set fails, fall back to a minimal request.
            try:
                df = rd.get_data(universe=[ticker], fields=LSEG_FIELDS)
            except Exception as exc:
                self.last_error = str(exc)
                return None

        if df is None or df.empty:
            return None

        # If a period label exists in the response, filter to the target quarter.
        row = self._select_period_row(df, period_label)

        period_value = self._pick_value(row, FIELD_ALIASES["period_label"]) or period_label

        # LSEG revenue fields return ANNUAL consensus, not quarterly.
        # Divide by 4 to get approximate quarterly estimates.
        annual_smart = self._to_float(self._pick_value(row, FIELD_ALIASES["revenue_smart_estimate"]))
        annual_mean = self._to_float(self._pick_value(row, FIELD_ALIASES["revenue_mean"]))
        annual_high = self._to_float(self._pick_value(row, FIELD_ALIASES["revenue_high"]))
        annual_low = self._to_float(self._pick_value(row, FIELD_ALIASES["revenue_low"]))

        return {
            "ticker": ticker,
            "period": period_value,
            # Quarterly estimates (annual / 4)
            "revenue_smart_estimate": annual_smart / 4 if annual_smart else None,
            "revenue_mean": annual_mean / 4 if annual_mean else None,
            "revenue_high": annual_high / 4 if annual_high else None,
            "revenue_low": annual_low / 4 if annual_low else None,
            # Annual values preserved for reference
            "revenue_annual_smart": annual_smart,
            "revenue_annual_mean": annual_mean,
            "eps_smart_estimate": self._to_float(self._pick_value(row, FIELD_ALIASES["eps_smart_estimate"])),
            "num_analysts": self._to_int(self._pick_value(row, FIELD_ALIASES["num_analysts"])),
            "earnings_date": self._to_iso(self._pick_value(row, FIELD_ALIASES["earnings_date"])),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def _select_period_row(self, df, period_label: Optional[str]):
        """Pick the row matching the fiscal quarter label when possible."""
        if not period_label:
            return df.iloc[0]

        for col in FIELD_ALIASES["period_label"]:
            if col in df.columns:
                mask = df[col].astype(str).str.contains(period_label, na=False)
                if mask.any():
                    return df[mask].iloc[0]

        # Default to the first row (usually the most recent period).
        return df.iloc[0]

    def _pick_value(self, row, columns: list[str]):
        for col in columns:
            if col in row.index and row[col] is not None:
                return row[col]

        # Fuzzy match by stripping non-alphanumerics (handles symbols like (R)).
        normalized_index = {self._normalize_key(col): col for col in row.index}
        for col in columns:
            normalized = self._normalize_key(col)
            if normalized in normalized_index:
                return row[normalized_index[normalized]]

        return None

    def _normalize_key(self, value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    def _to_float(self, value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _to_int(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except Exception:
            return None

    def _to_iso(self, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc).isoformat()
        try:
            return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            return None

    def _cache_key(self, ticker: str, period: str) -> str:
        return f"{ticker}:{period}"

    def _load_cache(self) -> Dict:
        if not self.cache_path.exists():
            return {"items": {}, "updated_at": None}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {"items": {}, "updated_at": None}

    def _save_cache(self, cache: Dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)

    def _read_cache(self, ticker: str, period: str) -> Tuple[Optional[Dict], timedelta]:
        cache = self._load_cache()
        key = self._cache_key(ticker, period)
        item = cache.get("items", {}).get(key)
        if not item:
            return None, timedelta.max

        fetched_at = item.get("fetched_at")
        if not fetched_at:
            return item, timedelta.max

        try:
            fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        except Exception:
            return item, timedelta.max

        age = datetime.now(timezone.utc) - fetched_dt
        return item, age

    def _write_cache(self, ticker: str, period: str, data: Dict) -> None:
        cache = self._load_cache()
        cache.setdefault("items", {})[self._cache_key(ticker, period)] = data
        cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_cache(cache)
