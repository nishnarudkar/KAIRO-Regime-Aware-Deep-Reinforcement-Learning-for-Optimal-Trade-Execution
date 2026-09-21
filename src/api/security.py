"""
src/api/security.py — API-key protection for state-changing / order-routing routes.

Behaviour:
  * ``KAIRO_API_KEY`` set   -> protected routes require a matching ``X-API-Key`` header.
  * not set                 -> protected routes are refused (403), unless
                               ``KAIRO_ALLOW_UNAUTHENTICATED=1`` is set (local development).

Protected routes: ``POST /api/execution/paper`` and ``POST /api/execution/kill-switch``.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    configured = os.environ.get("KAIRO_API_KEY", "")
    if configured:
        if not x_api_key or not hmac.compare_digest(x_api_key, configured):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
        return
    if os.environ.get("KAIRO_ALLOW_UNAUTHENTICATED") == "1":
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This route is disabled: set KAIRO_API_KEY (or KAIRO_ALLOW_UNAUTHENTICATED=1 for local development).",
    )
