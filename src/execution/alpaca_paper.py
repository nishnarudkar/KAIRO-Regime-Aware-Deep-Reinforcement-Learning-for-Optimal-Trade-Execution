"""
src/execution/alpaca_paper.py — Alpaca Paper Trading Integration & Execution Engine

Provides paper trading execution capabilities:
  - Safety check verifying paper endpoint (https://paper-api.alpaca.markets)
  - Pre-trade validation via ExecutionRiskGate
  - Synchronous / Mock paper order execution
  - Real-time fill summary response
"""

from __future__ import annotations

import os
import uuid
import datetime
from typing import Any, Dict, Optional, Tuple

import requests
from src.execution.risk_gates import ExecutionRiskGate, RiskGateConfig


class AlpacaPaperExecutor:
    """
    Paper Trading Execution Engine interfacing with Alpaca Paper API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
        risk_gate: Optional[ExecutionRiskGate] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.risk_gate = risk_gate or ExecutionRiskGate()
        self.mock_mode = mock_mode or (not self.api_key or not self.secret_key)

        # Enforce safety check: verify base URL is paper trading endpoint
        if "paper" not in self.base_url.lower() and not self.mock_mode:
            raise ValueError(f"Safety constraint violated: base_url '{self.base_url}' is not Alpaca Paper trading.")

    def execute_slice(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        arrival_price: float,
        target_inventory: float,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """
        Execute a single order slice on Alpaca Paper API after validating pre-trade risk gates.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            side: Order side ('BUY' or 'SELL')
            quantity: Shares to execute
            current_price: Current market price ($)
            arrival_price: Initial decision price ($)
            target_inventory: Total target order inventory
            order_type: Order type ('market' or 'limit')

        Returns:
            Dictionary with order status, fill price, executed shares, and risk gate results.
        """
        # Step 1: Pre-trade Risk Gate Check
        passed, reason = self.risk_gate.validate_order(
            symbol=symbol,
            side=side,
            order_quantity=quantity,
            current_price=current_price,
            arrival_price=arrival_price,
            target_inventory=target_inventory,
        )

        if not passed:
            return {
                "order_id": None,
                "status": "rejected",
                "reason": reason,
                "executed_quantity": 0.0,
                "fill_price": 0.0,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

        # Step 2: Route Order to Paper API or Mock Sandbox
        if self.mock_mode:
            return self._execute_mock_order(symbol, side, quantity, current_price)
        else:
            return self._execute_alpaca_api_order(symbol, side, quantity, order_type, current_price)

    def _execute_mock_order(
        self, symbol: str, side: str, quantity: float, current_price: float
    ) -> Dict[str, Any]:
        """Simulate a paper fill in offline mock mode."""
        # Add small random slippage modeling market fill
        slippage = current_price * 0.0002 * (1.0 if side.upper() == "BUY" else -1.0)
        fill_price = current_price + slippage

        return {
            "order_id": f"paper-mock-{uuid.uuid4().hex[:8]}",
            "status": "filled",
            "mode": "paper_mock",
            "symbol": symbol,
            "side": side.upper(),
            "executed_quantity": float(quantity),
            "fill_price": float(fill_price),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": "Order passed risk gates and executed in Alpaca Paper Sandbox.",
        }

    def _execute_alpaca_api_order(
        self, symbol: str, side: str, quantity: float, order_type: str, current_price: float
    ) -> Dict[str, Any]:
        """Submit order to live Alpaca Paper REST endpoint."""
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }
        payload = {
            "symbol": symbol.upper(),
            "qty": str(int(quantity)),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "gtc",
        }

        try:
            resp = requests.post(f"{self.base_url}/v2/orders", json=payload, headers=headers, timeout=5)
            if resp.status_code in (200, 201):
                data = resp.json()
                return {
                    "order_id": data.get("id"),
                    "status": data.get("status", "submitted"),
                    "mode": "alpaca_paper_api",
                    "symbol": symbol,
                    "side": side.upper(),
                    "executed_quantity": float(data.get("qty", quantity)),
                    "fill_price": float(data.get("filled_avg_price") or current_price),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "reason": "Order submitted successfully to Alpaca Paper API.",
                }
            else:
                return {
                    "order_id": None,
                    "status": "rejected",
                    "reason": f"Alpaca API HTTP {resp.status_code}: {resp.text}",
                    "executed_quantity": 0.0,
                    "fill_price": 0.0,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
        except Exception as err:
            return {
                "order_id": None,
                "status": "rejected",
                "reason": f"Alpaca connection error: {str(err)}",
                "executed_quantity": 0.0,
                "fill_price": 0.0,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
