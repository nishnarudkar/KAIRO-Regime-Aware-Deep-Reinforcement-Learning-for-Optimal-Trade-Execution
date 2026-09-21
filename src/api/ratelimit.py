"""
src/api/ratelimit.py — Per-client rate limiting for the CPU-heavy routes.

A sliding one-minute window per client IP, held in process memory. ``KAIRO_RATE_LIMIT`` is the number of
requests per minute allowed to /simulate and /backtest (default 60; 0 disables). With several worker
processes each keeps its own counters, so treat the limit as per-worker.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

_WINDOW_SECONDS = 60.0
_hits: dict = defaultdict(deque)
_lock = threading.Lock()


def reset() -> None:
    with _lock:
        _hits.clear()


def rate_limit(request: Request) -> None:
    limit = int(os.environ.get("KAIRO_RATE_LIMIT", "60"))
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        q = _hits[client]
        while q and now - q[0] > _WINDOW_SECONDS:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(_WINDOW_SECONDS - (now - q[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit} requests/minute). Retry in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        q.append(now)
