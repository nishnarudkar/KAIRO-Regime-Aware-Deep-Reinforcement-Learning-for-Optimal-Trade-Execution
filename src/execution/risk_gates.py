"""
src/execution/risk_gates.py — Pre-Trade Risk Gates & Safety Controls

Enforces institutional safety controls prior to sending orders to paper or live markets:
  - Max Notional Order Value limit ($)
  - Max Single Order Quantity limit (% of target inventory)
  - Price Collar Limit (% deviation from decision arrival price)
  - Kill Switch Emergency Halt Flag
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskGateConfig:
    """Pre-trade risk control parameter limits."""
    max_notional_value: float = 1_000_000.0  # $1M max per order
    max_single_order_pct: float = 0.50        # 50% max of target inventory per step
    price_collar_pct: float = 0.03            # 3% max price deviation from arrival price
    kill_switch_active: bool = False          # Emergency kill switch


class RiskCheckViolation(Exception):
    """Exception raised when an order violates pre-trade risk gates."""
    pass


class ExecutionRiskGate:
    """
    Validates execution orders against configurable risk constraints.
    """

    def __init__(self, config: Optional[RiskGateConfig] = None):
        self.config = config or RiskGateConfig()

    def validate_order(
        self,
        symbol: str,
        side: str,
        order_quantity: float,
        current_price: float,
        arrival_price: float,
        target_inventory: float,
    ) -> Tuple[bool, str]:
        """
        Validate an execution slice order against all pre-trade risk gates.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            side: Order side ('BUY' or 'SELL')
            order_quantity: Shares requested to trade in this step
            current_price: Current market price ($)
            arrival_price: Initial decision arrival price ($)
            target_inventory: Total initial target order inventory

        Returns:
            Tuple of (is_passed: bool, reason_message: str)
        """
        # 1. Kill Switch Check
        if self.config.kill_switch_active:
            return False, "Order rejected: Emergency Kill Switch is ACTIVE."

        # 2. Max Single Order Percentage Check
        max_allowed_shares = target_inventory * self.config.max_single_order_pct
        if order_quantity > max_allowed_shares + 1e-5:
            return (
                False,
                f"Order rejected: Quantity {order_quantity:,.1f} exceeds max single order limit "
                f"({self.config.max_single_order_pct*100:.0f}% of target = {max_allowed_shares:,.1f} shares).",
            )

        # 3. Max Notional Value Check
        notional_value = order_quantity * current_price
        if notional_value > self.config.max_notional_value + 1e-5:
            return (
                False,
                f"Order rejected: Notional value ${notional_value:,.2f} exceeds limit "
                f"(${self.config.max_notional_value:,.2f}).",
            )

        # 4. Price Collar Deviation Check
        if arrival_price > 0:
            price_dev = abs(current_price - arrival_price) / arrival_price
            if price_dev > self.config.price_collar_pct:
                return (
                    False,
                    f"Order rejected: Price collar breach ({price_dev*100:.2f}% deviation vs arrival ${arrival_price:.2f} "
                    f"exceeds limit {self.config.price_collar_pct*100:.1f}%).",
                )

        return True, "Order passed all pre-trade risk gates."

    def set_kill_switch(self, active: bool) -> None:
        """Activate or deactivate emergency kill switch."""
        self.config.kill_switch_active = active

    def get_status(self) -> Dict[str, Any]:
        """Return active risk gate parameters and status."""
        return {
            "max_notional_value": self.config.max_notional_value,
            "max_single_order_pct": self.config.max_single_order_pct,
            "price_collar_pct": self.config.price_collar_pct,
            "kill_switch_active": self.config.kill_switch_active,
        }
