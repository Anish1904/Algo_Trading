"""
Renko + EMA Strategy Engine
============================
Fetches live price from Binance, builds Renko bricks, applies EMA21/EMA50
signals, manages risk, and POSTs webhook payloads on signal events.

Usage:
    pip install requests pandas numpy
    python renko_strategy.py

Configuration:
    Edit the CONFIG block below to set your symbol, brick size, webhook URL, etc.
"""

import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

# ─────────────────────────── CONFIGURATION ────────────────────────────────────

CONFIG = {
    # Binance symbol (must match Binance ticker format)
    # Note: Binance does not have a native BTCUSD spot pair.
    # BTCUSDT is the equivalent — BTC priced against Tether (USD-pegged stablecoin).
    "symbol": "BTCUSDT",  # ← this is Binance's BTCUSD equivalent

    # Renko brick size in price units (e.g. $100 for BTC)
    "brick_size": 100.0,

    # EMA periods
    "ema_fast": 21,
    "ema_slow": 50,

    # Webhook endpoint — replace with your actual URL
    "webhook_url": "https://your-webhook-url.com/webhook",

    # Interval label sent in the payload (informational)
    "interval": "renko",

    # Polling interval in seconds (how often to fetch price from Binance)
    "poll_seconds": 5,

    # Binance price endpoint
    "binance_url": "https://api.binance.com/api/v3/ticker/price",

    # Consecutive green bricks before 50% profit take
    "profit_take_bricks": 5,

    # Stop-loss multiplier (entry - N × brick_size)
    "sl_multiplier": 2,
}

# ──────────────────────────── LOGGING SETUP ───────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("renko_strategy")

# ────────────────────────── DATA STRUCTURES ───────────────────────────────────

@dataclass
class RenkoBrick:
    direction: str          # "green" | "red"
    open_price: float
    close_price: float
    timestamp: datetime


@dataclass
class SignalEvent:
    """
    Emitted for every actionable signal.
    ML layer can gate execution via ml_filter() before the webhook is called.
    """
    ticker: str
    action: str             # "buy" | "sell"
    price: float
    timestamp: datetime
    interval: str
    quantity: str = "1"
    stop_loss: Optional[float] = None
    take_profit_trigger: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyState:
    bricks: List[RenkoBrick] = field(default_factory=list)
    last_close: Optional[float] = None     # last confirmed Renko close
    in_long: bool = False
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    consecutive_green: int = 0
    partial_exit_done: bool = False
    last_signal: Optional[str] = None      # avoid duplicate signals
    prices_for_ema: deque = field(default_factory=lambda: deque(maxlen=200))
    prev_ema_fast: Optional[float] = None
    prev_ema_slow: Optional[float] = None

# ──────────────────────────── EMA UTILITIES ───────────────────────────────────

def compute_ema(prices: list, period: int) -> Optional[float]:
    """Compute EMA for the given list of prices and period."""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema

# ───────────────────────── RENKO BUILDER ──────────────────────────────────────

def update_renko(state: StrategyState, current_price: float, brick_size: float) -> List[RenkoBrick]:
    """
    Given a new market price, generate any new Renko bricks.
    Returns a list of newly formed bricks (may be empty, or multiple).
    """
    new_bricks: List[RenkoBrick] = []

    if state.last_close is None:
        # Bootstrap: anchor the first brick level to the current price
        state.last_close = current_price
        log.info(f"Renko anchored at {current_price}")
        return new_bricks

    last = state.last_close

    # How many bricks up or down?
    diff = current_price - last
    bricks_up = int(diff / brick_size)
    bricks_down = int((-diff) / brick_size)

    if bricks_up >= 1:
        for i in range(bricks_up):
            brick = RenkoBrick(
                direction="green",
                open_price=last + i * brick_size,
                close_price=last + (i + 1) * brick_size,
                timestamp=datetime.now(timezone.utc),
            )
            new_bricks.append(brick)
        state.last_close = last + bricks_up * brick_size

    elif bricks_down >= 1:
        for i in range(bricks_down):
            brick = RenkoBrick(
                direction="red",
                open_price=last - i * brick_size,
                close_price=last - (i + 1) * brick_size,
                timestamp=datetime.now(timezone.utc),
            )
            new_bricks.append(brick)
        state.last_close = last - bricks_down * brick_size

    if new_bricks:
        state.bricks.extend(new_bricks)
        # Keep only last 500 bricks in memory
        if len(state.bricks) > 500:
            state.bricks = state.bricks[-500:]

    return new_bricks

# ───────────────────────── STRATEGY ENGINE ────────────────────────────────────

def evaluate_strategy(
    state: StrategyState,
    new_bricks: List[RenkoBrick],
    current_price: float,
    cfg: dict,
) -> Optional[SignalEvent]:
    """
    Core strategy logic.  Returns a SignalEvent when a new signal fires,
    otherwise None.  Duplicate signals (same action as last) are suppressed.
    """
    if not new_bricks:
        return None

    prices = list(state.prices_for_ema)
    ema_fast = compute_ema(prices, cfg["ema_fast"])
    ema_slow = compute_ema(prices, cfg["ema_slow"])

    signal: Optional[SignalEvent] = None

    for brick in new_bricks:
        # ── EXIT CONDITIONS (checked first) ───────────────────────────────────
        if state.in_long:
            # Exit on red brick
            if brick.direction == "red":
                if state.last_signal != "sell":
                    log.info("EXIT SIGNAL — red brick appeared")
                    signal = _make_signal("sell", brick.close_price, state, cfg)
                    _reset_long(state)
                continue

            # Exit if price drops below EMA21
            if ema_fast is not None and current_price < ema_fast:
                if state.last_signal != "sell":
                    log.info("EXIT SIGNAL — price crossed below EMA21")
                    signal = _make_signal("sell", current_price, state, cfg)
                    _reset_long(state)
                continue

            # Count consecutive green bricks for partial exit
            if brick.direction == "green":
                state.consecutive_green += 1
                if (
                    state.consecutive_green >= cfg["profit_take_bricks"]
                    and not state.partial_exit_done
                    and brick.close_price > state.entry_price  # in profit
                ):
                    log.info(
                        f"PARTIAL EXIT — {cfg['profit_take_bricks']} consecutive green bricks in profit"
                    )
                    state.partial_exit_done = True
                    # Emit a partial sell signal (quantity 0.5 handled downstream)
                    signal = _make_signal(
                        "sell", brick.close_price, state, cfg,
                        metadata={"partial": True, "quantity": "0.5"},
                    )
                    # Stay in trade (don't reset)

        # ── ENTRY CONDITIONS ──────────────────────────────────────────────────
        if not state.in_long and brick.direction == "green":
            if ema_fast is None or ema_slow is None:
                continue

            above_both_emas = brick.close_price > ema_fast and brick.close_price > ema_slow

            # Golden cross: EMA21 crosses above EMA50 on this brick
            golden_cross = (
                state.prev_ema_fast is not None
                and state.prev_ema_slow is not None
                and state.prev_ema_fast <= state.prev_ema_slow   # was below
                and ema_fast > ema_slow                           # now above
            )

            if above_both_emas and golden_cross:
                if state.last_signal != "buy":
                    log.info(
                        f"ENTRY SIGNAL — green brick above EMA21 & EMA50 + golden cross "
                        f"| EMA21={ema_fast:.2f} EMA50={ema_slow:.2f}"
                    )
                    signal = _make_signal("buy", brick.close_price, state, cfg)
                    state.in_long = True
                    state.entry_price = brick.close_price
                    state.stop_loss = brick.close_price - cfg["sl_multiplier"] * cfg["brick_size"]
                    state.consecutive_green = 1
                    state.partial_exit_done = False
                    log.info(
                        f"  Stop-loss set at {state.stop_loss:.2f}  "
                        f"(entry {state.entry_price:.2f} - {cfg['sl_multiplier']}×{cfg['brick_size']})"
                    )

    # Update previous EMA values for next iteration
    state.prev_ema_fast = ema_fast
    state.prev_ema_slow = ema_slow

    return signal


def _make_signal(
    action: str,
    price: float,
    state: StrategyState,
    cfg: dict,
    metadata: dict = None,
) -> SignalEvent:
    state.last_signal = action
    return SignalEvent(
        ticker=cfg["symbol"],
        action=action,
        price=price,
        timestamp=datetime.now(timezone.utc),
        interval=cfg["interval"],
        stop_loss=state.stop_loss if action == "buy" else None,
        metadata=metadata or {},
    )


def _reset_long(state: StrategyState):
    state.in_long = False
    state.entry_price = None
    state.stop_loss = None
    state.consecutive_green = 0
    state.partial_exit_done = False

# ─────────────────────── ML FILTER SLOT ───────────────────────────────────────

def ml_filter(signal: SignalEvent) -> bool:
    """
    Gate signals through an ML model before execution.
    Return True  → allow the signal through.
    Return False → suppress the signal.

    Replace this stub with your trained model inference logic.
    """
    # Stub: always allow (no ML model loaded yet)
    return True

# ─────────────────────────── WEBHOOK ──────────────────────────────────────────

def build_payload(signal: SignalEvent) -> dict:
    """Build the webhook JSON payload matching the required schema."""
    quantity = signal.metadata.get("quantity", signal.quantity)
    return {
        "ticker":   signal.ticker,
        "action":   signal.action,
        "quantity": quantity,
        "price":    str(round(signal.price, 8)),
        "time":     signal.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": signal.interval,
    }


def send_webhook(signal: SignalEvent, cfg: dict) -> bool:
    """POST the signal payload to the configured webhook URL."""
    payload = build_payload(signal)
    log.info(f"Sending webhook → {cfg['webhook_url']}  payload={payload}")
    try:
        resp = requests.post(cfg["webhook_url"], json=payload, timeout=10)
        if resp.ok:
            log.info(f"Webhook accepted  [{resp.status_code}]")
            return True
        else:
            log.error(f"Webhook rejected  [{resp.status_code}] {resp.text[:200]}")
            return False
    except requests.RequestException as exc:
        log.error(f"Webhook request failed: {exc}")
        return False

# ──────────────────────── BINANCE PRICE FETCH ─────────────────────────────────

def fetch_price(symbol: str, binance_url: str) -> Optional[float]:
    """Fetch current price from Binance REST API."""
    try:
        resp = requests.get(
            binance_url,
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as exc:
        log.warning(f"Price fetch error: {exc}")
        return None

# ─────────────────────────── MAIN LOOP ────────────────────────────────────────

def run():
    cfg = CONFIG
    state = StrategyState()

    log.info("=" * 60)
    log.info("Renko + EMA Strategy Engine  —  Starting")
    log.info(f"  Symbol      : {cfg['symbol']}")
    log.info(f"  Brick size  : {cfg['brick_size']}")
    log.info(f"  EMA periods : {cfg['ema_fast']} / {cfg['ema_slow']}")
    log.info(f"  Webhook     : {cfg['webhook_url']}")
    log.info(f"  Poll every  : {cfg['poll_seconds']}s")
    log.info("=" * 60)

    while True:
        price = fetch_price(cfg["symbol"], cfg["binance_url"])

        if price is not None:
            last = state.last_close or 0
            distance_up   = round((last + cfg["brick_size"]) - price, 2)
            distance_down = round(price - (last - cfg["brick_size"]), 2)
            log.info(
                f"Price={price:.2f}  |  Renko anchor={last:.2f}  |  "
                f"+brick in ${distance_up:.2f}  |  -brick in ${distance_down:.2f}"
            )

            # Feed price into EMA window on every tick
            state.prices_for_ema.append(price)

            # Build new Renko bricks (if any)
            new_bricks = update_renko(state, price, cfg["brick_size"])

            if new_bricks:
                for b in new_bricks:
                    log.info(
                        f"NEW BRICK  {b.direction.upper():5s}  "
                        f"{b.open_price:.2f} → {b.close_price:.2f}"
                    )

            # Evaluate strategy
            signal = evaluate_strategy(state, new_bricks, price, cfg)

            # ML gate + webhook dispatch
            if signal:
                if ml_filter(signal):
                    send_webhook(signal, cfg)
                else:
                    log.info(f"Signal {signal.action.upper()} suppressed by ml_filter()")

        time.sleep(cfg["poll_seconds"])


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Strategy engine stopped by user.")