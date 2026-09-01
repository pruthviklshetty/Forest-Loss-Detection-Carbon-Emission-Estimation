"""Place-name -> coordinates via Nominatim (OpenStreetMap), no API key.

Respects Nominatim's usage policy: a real identifying User-Agent, an in-process
result cache, and a minimum interval between upstream calls. Ambiguous queries
return every candidate - the caller (frontend) picks; nothing is auto-selected.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import requests

from .config import (GEOCODE_CACHE_MAX, GEOCODE_MIN_INTERVAL_S,
                     GEOCODE_USER_AGENT, NOMINATIM_URL)

_cache: "OrderedDict[str, list[dict]]" = OrderedDict()
_lock = threading.Lock()
_last_call = [0.0]


class GeocodeError(RuntimeError):
    pass


def _norm(q: str) -> str:
    return " ".join(q.strip().lower().split())


def geocode(query: str, limit: int = 5) -> list[dict]:
    q = _norm(query)
    if not q:
        raise GeocodeError("empty query")
    if len(q) < 2:
        raise GeocodeError("query too short")

    with _lock:
        if q in _cache:
            _cache.move_to_end(q)
            return _cache[q]

    with _lock:
        wait = GEOCODE_MIN_INTERVAL_S - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": limit,
                    "addressdetails": 0},
            headers={"User-Agent": GEOCODE_USER_AGENT,
                     "Accept-Language": "en"},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        raise GeocodeError(f"geocoding service unavailable: {exc}") from exc
    except ValueError as exc:
        raise GeocodeError(f"geocoding service returned invalid data: {exc}") from exc

    results = []
    for r in raw:
        try:
            results.append({
                "display_name": r["display_name"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "type": r.get("type"),
                "category": r.get("category") or r.get("class"),
                "importance": r.get("importance"),
            })
        except (KeyError, TypeError, ValueError):
            continue

    with _lock:
        _cache[q] = results
        _cache.move_to_end(q)
        while len(_cache) > GEOCODE_CACHE_MAX:
            _cache.popitem(last=False)
    return results
