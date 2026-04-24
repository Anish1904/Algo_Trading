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
import pandas as pd
import joblib
from pathlib import Path
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

    # ── ML Filter ─────────────────────────────────────────────────────────────
    # Path to the folder containing the 3 pkl files produced by ml_train.py
    # If ml_train.py is in the same folder as this file, leave as "."
    "ml_model_dir": ".",

    # Minimum probability for a BUY signal to be approved by the ML model
    # Range 0.0–1.0 — raise this to be more selective, lower to be more permissive
    "ml_threshold": 0.55,

    # Set to False to bypass ML filter entirely (useful for debugging)
    "ml_enabled": True,
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
    # Rolling OHLCV window for ML feature computation (keeps last 100 ticks)
    price_history: deque = field(default_factory=lambda: deque(maxlen=100))

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

# ─────────────────────── ML INTEGRATION ──────────────────────────────────────

def load_ml_artifacts(model_dir: str):
    """
    Load the 3 pkl files produced by ml_train.py.
    Returns (model, scaler, feature_cols) or (None, None, None) if files missing.
    """
    d = Path(model_dir)
    model_path   = d / "ml_model.pkl"
    scaler_path  = d / "ml_scaler.pkl"
    feature_path = d / "ml_feature_cols.pkl"

    missing = [str(p) for p in [model_path, scaler_path, feature_path] if not p.exists()]
    if missing:
        log.warning(f"ML artifacts not found — filter disabled: {missing}")
        return None, None, None

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_path)
    log.info(f"ML model loaded  ({len(feature_cols)} features)")
    return model, scaler, feature_cols


def build_ml_features(state: StrategyState, cfg: dict) -> Optional[np.ndarray]:
    """
    Compute the same 30 features used during training from live price history.
    Returns a (1, n_features) array ready for model.predict_proba(), or None
    if there isn't enough history yet.
    """
    if len(state.price_history) < 30:
        return None   # not enough data yet — wait for warmup

    prices = pd.Series(list(state.price_history), dtype=float)
    close  = prices   # single price feed; open/high/low approximated

    def ema_series(s, period):
        return s.ewm(span=period, adjust=False).mean()

    ema_fast_s = ema_series(close, cfg["ema_fast"])
    ema_slow_s = ema_series(close, cfg["ema_slow"])
    ma_short_s = ema_series(close, 9)
    ma_long_s  = ema_series(close, 21)

    ema_f = ema_fast_s.iloc[-1]
    ema_s = ema_slow_s.iloc[-1]
    ma_sh = ma_short_s.iloc[-1]
    ma_lo = ma_long_s.iloc[-1]

    # Rolling volatility
    returns = close.pct_change()
    vol_10  = returns.rolling(10).std().iloc[-1]
    vol_20  = returns.rolling(20).std().iloc[-1]

    # ATR approximation (single price feed — uses rolling high-low range proxy)
    hl_proxy = close.rolling(2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]), raw=False)
    atr_val  = hl_proxy.rolling(14).mean().iloc[-1]
    atr_pct  = atr_val / close.iloc[-1] if close.iloc[-1] != 0 else 0

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi   = 100 - 100 / (1 + rs)

    # MACD
    macd_line   = ema_series(close, 12) - ema_series(close, 26)
    macd_sig    = ema_series(macd_line, 9)
    macd_val    = macd_line.iloc[-1]
    macd_sig_v  = macd_sig.iloc[-1]
    macd_hist_v = macd_val - macd_sig_v

    # Renko features from brick history
    recent_dirs = [1 if b.direction == "green" else -1
                   for b in list(state.bricks)[-5:]] if state.bricks else [0]
    renko_dir   = recent_dirs[-1] if recent_dirs else 0
    green_streak = sum(1 for d in recent_dirs if d == 1)
    red_streak   = sum(1 for d in recent_dirs if d == -1)

    now = datetime.now(timezone.utc)

    feat = {
        "returns_1":        returns.iloc[-1]  if len(returns) > 1  else 0,
        "returns_5":        (close.iloc[-1] / close.iloc[-6]  - 1) if len(close) > 5  else 0,
        "returns_10":       (close.iloc[-1] / close.iloc[-11] - 1) if len(close) > 10 else 0,
        "returns_20":       (close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else 0,
        "hl_ratio":         atr_pct,                          # proxy: ATR% ≈ HL range %
        "co_ratio":         returns.iloc[-1] if len(returns) > 1 else 0,
        "price_vs_ema_fast":(close.iloc[-1] - ema_f) / ema_f if ema_f else 0,
        "price_vs_ema_slow":(close.iloc[-1] - ema_s) / ema_s if ema_s else 0,
        "ema_spread":       (ema_f - ema_s) / ema_s           if ema_s else 0,
        "ema_slope_fast":   ema_fast_s.pct_change(3).iloc[-1],
        "ema_slope_slow":   ema_slow_s.pct_change(3).iloc[-1],
        "ma_regime":        1 if ma_sh > ma_lo else -1,
        "ma_spread":        (ma_sh - ma_lo) / ma_lo           if ma_lo else 0,
        "renko_direction":  renko_dir,
        "green_streak":     green_streak,
        "red_streak":       red_streak,
        "volatility_10":    vol_10,
        "volatility_20":    vol_20,
        "atr":              atr_val,
        "atr_pct":          atr_pct,
        "volume_usd":       0,   # not available from ticker endpoint
        "volume_btc":       0,   # not available from ticker endpoint
        "vol_ratio":        1,   # neutral default
        "rsi_14":           rsi,
        "macd":             macd_val,
        "macd_signal":      macd_sig_v,
        "macd_hist":        macd_hist_v,
        "hour":             now.hour,
        "day_of_week":      now.weekday(),
        "month":            now.month,
    }

    return np.array([[feat[col] for col in ML_FEATURE_COLS]], dtype=float)


def ml_filter(signal: SignalEvent, state: StrategyState, cfg: dict) -> bool:
    """
    Gate BUY signals through the trained ML model.
    SELL / partial exit signals always pass through (risk management must not be blocked).
    Returns True → allow, False → reject.
    """
    # Always let exits through — never block risk management
    if signal.action == "sell":
        return True

    if not cfg.get("ml_enabled", True) or ML_MODEL is None:
        return True   # ML disabled or model not loaded

    feat_arr = build_ml_features(state, cfg)
    if feat_arr is None:
        log.info("ML filter — insufficient history, allowing signal through")
        return True

    # Handle NaN/inf that can appear during live warmup
    if not np.isfinite(feat_arr).all():
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    feat_scaled = ML_SCALER.transform(feat_arr)
    proba       = ML_MODEL.predict_proba(feat_scaled)[0][1]   # P(good trade)
    threshold   = cfg.get("ml_threshold", 0.55)
    approved    = proba >= threshold

    if approved:
        log.info(f"ML APPROVED  {signal.action.upper()}  probability={proba:.4f}  "
                 f"(threshold={threshold})")
    else:
        log.warning(f"ML REJECTED  {signal.action.upper()}  probability={proba:.4f}  "
                    f"(threshold={threshold}) — signal suppressed")

    return approved

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

# ── Module-level ML artifacts (loaded once at startup) ────────────────────────
ML_MODEL        = None
ML_SCALER       = None
ML_FEATURE_COLS = None


def run():
    global ML_MODEL, ML_SCALER, ML_FEATURE_COLS

    cfg   = CONFIG
    state = StrategyState()

    # Load ML artifacts
    ML_MODEL, ML_SCALER, ML_FEATURE_COLS = load_ml_artifacts(cfg["ml_model_dir"])

    log.info("=" * 60)
    log.info("Renko + EMA Strategy Engine  —  Starting")
    log.info(f"  Symbol      : {cfg['symbol']}")
    log.info(f"  Brick size  : {cfg['brick_size']}")
    log.info(f"  EMA periods : {cfg['ema_fast']} / {cfg['ema_slow']}")
    log.info(f"  Webhook     : {cfg['webhook_url']}")
    log.info(f"  Poll every  : {cfg['poll_seconds']}s")
    log.info(f"  ML filter   : {'ENABLED' if cfg['ml_enabled'] and ML_MODEL else 'DISABLED'}")
    log.info(f"  ML threshold: {cfg['ml_threshold']}")
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

            # Feed price into EMA window and ML history on every tick
            state.prices_for_ema.append(price)
            state.price_history.append(price)

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
                if ml_filter(signal, state, cfg):
                    send_webhook(signal, cfg)

        time.sleep(cfg["poll_seconds"])


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Strategy engine stopped by user.")