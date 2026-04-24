"""
Renko + EMA + ML Strategy Backtester
======================================
Replays historical OHLCV data through the exact same strategy + ML logic
used in renko_strategy.py and produces a full performance report.

Usage:
    python backtest.py --data ../mldata/2023.xlsx
    python backtest.py --data ../mldata/2024.xlsx --no-ml
    python backtest.py --data ../mldata/2023.xlsx ../mldata/2024.xlsx ../mldata/2025.xlsx
    python backtest.py --data ../mldata/2023.xlsx --capital 50000 --threshold 0.60

Dependencies:
    pip install pandas numpy scikit-learn xgboost matplotlib openpyxl joblib
"""

import argparse
import warnings
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

warnings.filterwarnings("ignore")

CONFIG = {
    "symbol":             "BTCUSDT",
    "brick_size":         500.0,   # larger = fewer bricks/trades; smaller = more trades
    "ema_fast":           21,
    "ema_slow":           50,
    "profit_take_bricks": 5,
    "sl_multiplier":      2,
    "ml_model_dir":       ".",
    "ml_threshold":       0.55,
    "ml_enabled":         True,
}

@dataclass
class RenkoBrick:
    direction:   str
    open_price:  float
    close_price: float
    timestamp:   datetime

@dataclass
class StrategyState:
    bricks:            List[RenkoBrick] = field(default_factory=list)
    last_close:        Optional[float]  = None
    in_long:           bool             = False
    entry_price:       Optional[float]  = None
    stop_loss:         Optional[float]  = None
    consecutive_green: int              = 0
    partial_exit_done: bool             = False
    last_signal:       Optional[str]    = None
    prices_for_ema:    deque = field(default_factory=lambda: deque(maxlen=200))
    prev_ema_fast:     Optional[float]  = None
    prev_ema_slow:     Optional[float]  = None
    price_history:     deque = field(default_factory=lambda: deque(maxlen=200))

@dataclass
class Trade:
    entry_time:    datetime
    exit_time:     Optional[datetime]
    entry_price:   float
    exit_price:    Optional[float]
    qty:           float
    stop_loss:     float
    pnl:           float  = 0.0
    pnl_pct:       float  = 0.0
    exit_reason:   str    = ""
    ml_prob:       float  = -1.0
    ml_approved:   bool   = True

def compute_ema(prices, period):
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema

def update_renko(state, price, brick_size, ts):
    new_bricks = []
    if state.last_close is None:
        state.last_close = price
        return new_bricks
    last = state.last_close
    diff = price - last
    bricks_up   = int( diff / brick_size)
    bricks_down = int(-diff / brick_size)
    if bricks_up >= 1:
        for i in range(bricks_up):
            new_bricks.append(RenkoBrick("green", last+i*brick_size, last+(i+1)*brick_size, ts))
        state.last_close = last + bricks_up * brick_size
    elif bricks_down >= 1:
        for i in range(bricks_down):
            new_bricks.append(RenkoBrick("red", last-i*brick_size, last-(i+1)*brick_size, ts))
        state.last_close = last - bricks_down * brick_size
    if new_bricks:
        state.bricks.extend(new_bricks)
        if len(state.bricks) > 500:
            state.bricks = state.bricks[-500:]
    return new_bricks

def _ema_s(s, p):
    return s.ewm(span=p, adjust=False).mean()

def build_ml_features(state, cfg, feature_cols):
    if len(state.price_history) < 50:
        return None
    close = pd.Series(list(state.price_history), dtype=float)
    ema_f_s = _ema_s(close, cfg["ema_fast"])
    ema_s_s = _ema_s(close, cfg["ema_slow"])
    ma_sh_s = _ema_s(close, 9)
    ma_lo_s = _ema_s(close, 21)
    ema_f = ema_f_s.iloc[-1]; ema_s = ema_s_s.iloc[-1]
    ma_sh = ma_sh_s.iloc[-1]; ma_lo = ma_lo_s.iloc[-1]
    returns = close.pct_change()
    vol_10  = returns.rolling(10).std().iloc[-1]
    vol_20  = returns.rolling(20).std().iloc[-1]
    hl_p    = close.rolling(2).apply(lambda x: abs(x.iloc[1]-x.iloc[0]), raw=False)
    atr_val = hl_p.rolling(14).mean().iloc[-1]
    atr_pct = atr_val / close.iloc[-1] if close.iloc[-1] != 0 else 0
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi     = 100 - 100 / (1 + rs)
    macd_l  = _ema_s(close, 12) - _ema_s(close, 26)
    macd_sg = _ema_s(macd_l, 9)
    mv      = macd_l.iloc[-1]; msv = macd_sg.iloc[-1]
    rdirs   = [1 if b.direction=="green" else -1 for b in list(state.bricks)[-5:]] if state.bricks else [0]
    rdir    = rdirs[-1] if rdirs else 0
    gs      = sum(1 for d in rdirs if d==1)
    rs2     = sum(1 for d in rdirs if d==-1)
    now     = datetime.now(timezone.utc)
    feat = {
        "returns_1":         returns.iloc[-1] if len(returns)>1 else 0,
        "returns_5":         (close.iloc[-1]/close.iloc[-6]-1) if len(close)>5 else 0,
        "returns_10":        (close.iloc[-1]/close.iloc[-11]-1) if len(close)>10 else 0,
        "returns_20":        (close.iloc[-1]/close.iloc[-21]-1) if len(close)>20 else 0,
        "hl_ratio":          atr_pct,
        "co_ratio":          returns.iloc[-1] if len(returns)>1 else 0,
        "price_vs_ema_fast": (close.iloc[-1]-ema_f)/ema_f if ema_f else 0,
        "price_vs_ema_slow": (close.iloc[-1]-ema_s)/ema_s if ema_s else 0,
        "ema_spread":        (ema_f-ema_s)/ema_s if ema_s else 0,
        "ema_slope_fast":    ema_f_s.pct_change(3).iloc[-1],
        "ema_slope_slow":    ema_s_s.pct_change(3).iloc[-1],
        "ma_regime":         1 if ma_sh>ma_lo else -1,
        "ma_spread":         (ma_sh-ma_lo)/ma_lo if ma_lo else 0,
        "renko_direction":   rdir,
        "green_streak":      gs,
        "red_streak":        rs2,
        "volatility_10":     vol_10,
        "volatility_20":     vol_20,
        "atr":               atr_val,
        "atr_pct":           atr_pct,
        "volume_usd":        0,
        "volume_btc":        0,
        "vol_ratio":         1,
        "rsi_14":            rsi,
        "macd":              mv,
        "macd_signal":       msv,
        "macd_hist":         mv - msv,
        "hour":              now.hour,
        "day_of_week":       now.weekday(),
        "month":             now.month,
    }
    arr = np.array([[feat[col] for col in feature_cols]], dtype=float)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def run_backtest(df, cfg, initial_capital, ml_model, ml_scaler, ml_feature_cols, use_ml):
    capital      = initial_capital
    position_qty = 0.0
    entry_price  = 0.0
    state        = StrategyState()
    trades       = []
    equity_curve = []
    equity_times = []
    ml_rejections= []
    open_trade   = None

    # Pre-warm price_history so ML features are valid from bar 1
    warmup_rows = min(200, len(df))
    for i in range(warmup_rows):
        state.price_history.append(float(df["CLOSE"].iloc[i]))
        state.prices_for_ema.append(float(df["CLOSE"].iloc[i]))
    print(f"\n  Pre-warmed with {warmup_rows} bars. Replaying {len(df):,} bars ...")

    for _, row in df.iterrows():
        price = float(row["CLOSE"])
        ts    = row.get("DATETIME", datetime.now(timezone.utc))
        if not isinstance(ts, datetime):
            try:    ts = pd.to_datetime(ts).to_pydatetime()
            except: ts = datetime.now(timezone.utc)

        state.prices_for_ema.append(price)
        state.price_history.append(price)

        # Stop-loss check
        if state.in_long and state.stop_loss is not None and price <= state.stop_loss:
            pnl     = (price - open_trade.entry_price) * position_qty
            pnl_pct = pnl / (open_trade.entry_price * position_qty) * 100
            capital += open_trade.entry_price * position_qty + pnl
            open_trade.exit_price  = price
            open_trade.exit_time   = ts
            open_trade.pnl         = pnl
            open_trade.pnl_pct     = pnl_pct
            open_trade.exit_reason = "stop_loss"
            trades.append(open_trade)
            open_trade = None; position_qty = 0.0
            state.in_long = False; state.entry_price = None; state.stop_loss = None
            state.consecutive_green = 0; state.partial_exit_done = False
            state.last_signal = "sell"
            equity_curve.append(capital); equity_times.append(ts)
            continue

        new_bricks = update_renko(state, price, cfg["brick_size"], ts)
        if not new_bricks:
            mtm = capital + position_qty * price if position_qty else capital
            equity_curve.append(mtm); equity_times.append(ts)
            continue

        prices_list = list(state.prices_for_ema)
        ema_fast    = compute_ema(prices_list, cfg["ema_fast"])
        ema_slow    = compute_ema(prices_list, cfg["ema_slow"])

        for brick in new_bricks:
            # EXIT
            if state.in_long and open_trade is not None:
                if brick.direction == "red":
                    ep  = brick.close_price
                    pnl = (ep - open_trade.entry_price) * position_qty
                    pnl_pct = pnl / (open_trade.entry_price * position_qty) * 100
                    capital += open_trade.entry_price * position_qty + pnl
                    open_trade.exit_price=ep; open_trade.exit_time=ts
                    open_trade.pnl=pnl; open_trade.pnl_pct=pnl_pct
                    open_trade.exit_reason="red_brick"
                    trades.append(open_trade)
                    open_trade=None; position_qty=0.0; state.in_long=False
                    state.entry_price=None; state.stop_loss=None
                    state.consecutive_green=0; state.partial_exit_done=False
                    state.last_signal="sell"; continue

                if ema_fast is not None and price < ema_fast:
                    ep  = price
                    pnl = (ep - open_trade.entry_price) * position_qty
                    pnl_pct = pnl / (open_trade.entry_price * position_qty) * 100
                    capital += open_trade.entry_price * position_qty + pnl
                    open_trade.exit_price=ep; open_trade.exit_time=ts
                    open_trade.pnl=pnl; open_trade.pnl_pct=pnl_pct
                    open_trade.exit_reason="below_ema21"
                    trades.append(open_trade)
                    open_trade=None; position_qty=0.0; state.in_long=False
                    state.entry_price=None; state.stop_loss=None
                    state.consecutive_green=0; state.partial_exit_done=False
                    state.last_signal="sell"; continue

                if brick.direction == "green":
                    state.consecutive_green += 1
                    if (state.consecutive_green >= cfg["profit_take_bricks"]
                            and not state.partial_exit_done
                            and brick.close_price > open_trade.entry_price):
                        pq  = position_qty * 0.5; ep = brick.close_price
                        pnl = (ep - open_trade.entry_price) * pq
                        pnl_pct = pnl / (open_trade.entry_price * pq) * 100
                        capital += open_trade.entry_price * pq + pnl
                        position_qty -= pq
                        trades.append(Trade(
                            entry_time=open_trade.entry_time, exit_time=ts,
                            entry_price=open_trade.entry_price, exit_price=ep,
                            qty=pq, stop_loss=open_trade.stop_loss,
                            pnl=pnl, pnl_pct=pnl_pct,
                            exit_reason="partial_profit",
                            ml_prob=open_trade.ml_prob, ml_approved=open_trade.ml_approved,
                        ))
                        state.partial_exit_done = True

            # ENTRY
            if not state.in_long and brick.direction == "green":
                if ema_fast is None or ema_slow is None: continue
                above_both    = brick.close_price > ema_fast and brick.close_price > ema_slow
                ema_bullish   = ema_fast is not None and ema_slow is not None and ema_fast > ema_slow
                # Two consecutive green bricks = confirmation (doc strategy rule)
                last_brick_green = (len(state.bricks) >= 2 and
                                    state.bricks[-2].direction == "green" and
                                    state.bricks[-1].direction == "green")
                if above_both and ema_bullish and last_brick_green and state.last_signal != "buy":
                    ml_prob = -1.0; ml_approved = True
                    if use_ml and ml_model is not None:
                        fa = build_ml_features(state, cfg, ml_feature_cols)
                        if fa is not None:
                            scaled  = ml_scaler.transform(fa)
                            ml_prob = float(ml_model.predict_proba(scaled)[0][1])
                            ml_approved = ml_prob >= cfg["ml_threshold"]
                            if not ml_approved:
                                ml_rejections.append({"time": ts, "price": brick.close_price, "prob": ml_prob})
                                state.last_signal = "buy"; continue
                        else:
                            # Not enough history yet — allow through
                            ml_approved = True
                    ep  = brick.close_price
                    sl  = ep - cfg["sl_multiplier"] * cfg["brick_size"]
                    qty = capital / ep
                    capital -= qty * ep; position_qty = qty; entry_price = ep
                    state.in_long=True; state.entry_price=ep; state.stop_loss=sl
                    state.consecutive_green=1; state.partial_exit_done=False
                    state.last_signal="buy"
                    open_trade = Trade(entry_time=ts, exit_time=None,
                                       entry_price=ep, exit_price=None,
                                       qty=qty, stop_loss=sl,
                                       ml_prob=ml_prob, ml_approved=ml_approved)

        state.prev_ema_fast = ema_fast; state.prev_ema_slow = ema_slow
        mtm = capital + position_qty * price if position_qty else capital
        equity_curve.append(mtm); equity_times.append(ts)

    # Close open position at end
    if open_trade is not None and position_qty > 0:
        lp  = float(df["CLOSE"].iloc[-1])
        pnl = (lp - open_trade.entry_price) * position_qty
        pnl_pct = pnl / (open_trade.entry_price * position_qty) * 100
        capital += open_trade.entry_price * position_qty + pnl
        open_trade.exit_price=lp; open_trade.exit_time=equity_times[-1]
        open_trade.pnl=pnl; open_trade.pnl_pct=pnl_pct
        open_trade.exit_reason="end_of_data"
        trades.append(open_trade); equity_curve[-1]=capital

    return {"trades": trades, "equity_curve": equity_curve,
            "equity_times": equity_times, "ml_rejections": ml_rejections,
            "final_capital": capital}

def compute_metrics(results, initial_capital):
    trades = results["trades"]; eq = results["equity_curve"]
    final  = results["final_capital"]
    if not trades: return {"error": "No trades executed"}
    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    eq_arr = np.array(eq)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak * 100
    daily  = pd.Series(eq).pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    hold   = [(t.exit_time - t.entry_time).total_seconds()/3600
              for t in trades if t.exit_time and t.entry_time]
    exits  = {}
    for t in trades: exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
    return {
        "total_trades":     len(trades),
        "winning_trades":   len(wins),
        "losing_trades":    len(losses),
        "win_rate_pct":     len(wins)/len(trades)*100,
        "total_pnl":        final - initial_capital,
        "total_return_pct": (final - initial_capital) / initial_capital * 100,
        "avg_win_usd":      np.mean(wins)   if wins   else 0,
        "avg_loss_usd":     np.mean(losses) if losses else 0,
        "profit_factor":    abs(sum(wins)/sum(losses)) if losses else float("inf"),
        "max_drawdown_pct": float(dd.min()),
        "sharpe_ratio":     sharpe,
        "avg_hold_hours":   np.mean(hold) if hold else 0,
        "final_capital":    final,
        "initial_capital":  initial_capital,
        "exit_reasons":     exits,
        "ml_rejections":    len(results["ml_rejections"]),
    }

def print_report(m, label, use_ml):
    print("\n" + "="*60)
    print(f"  BACKTEST RESULTS  —  {label}  |  ML {'ON' if use_ml else 'OFF'}")
    print("="*60)
    if "error" in m: print(f"  {m['error']}"); return
    print(f"  Starting capital  : ${m['initial_capital']:>12,.2f}")
    print(f"  Final capital     : ${m['final_capital']:>12,.2f}")
    print(f"  Total P&L         : ${m['total_pnl']:>+12,.2f}")
    print(f"  Total return      : {m['total_return_pct']:>+.2f}%")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:.3f}")
    print("-"*60)
    print(f"  Total trades      : {m['total_trades']}")
    print(f"  Winning trades    : {m['winning_trades']}  ({m['win_rate_pct']:.1f}%)")
    print(f"  Losing trades     : {m['losing_trades']}")
    print(f"  Avg win           : ${m['avg_win_usd']:>+,.2f}")
    print(f"  Avg loss          : ${m['avg_loss_usd']:>+,.2f}")
    print(f"  Profit factor     : {m['profit_factor']:.2f}")
    print(f"  Avg hold time     : {m['avg_hold_hours']:.1f} hours")
    print("-"*60)
    print(f"  Exit breakdown:")
    for r, c in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {r:<22}: {c}")
    if use_ml: print(f"  ML rejected       : {m['ml_rejections']} signals")
    print("="*60)

def save_charts(results, metrics, label, use_ml, out_dir):
    trades = results["trades"]; eq = results["equity_curve"]
    if not trades or not eq: return
    m   = metrics
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Backtest — {label}  |  ML {'ON' if use_ml else 'OFF'}",
                 fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq, color="#2ecc71", linewidth=1.5, label="Equity")
    ax1.axhline(m["initial_capital"], color="gray", linestyle="--", alpha=0.6, label="Start")
    ax1.fill_between(range(len(eq)), m["initial_capital"], eq,
                     where=[e >= m["initial_capital"] for e in eq], alpha=0.15, color="#2ecc71")
    ax1.fill_between(range(len(eq)), m["initial_capital"], eq,
                     where=[e <  m["initial_capital"] for e in eq], alpha=0.15, color="#e74c3c")
    ax1.set_title("Equity Curve"); ax1.set_ylabel("Capital (USD)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Drawdown
    ax2  = fig.add_subplot(gs[1, :])
    eqa  = np.array(eq); peak = np.maximum.accumulate(eqa)
    dd   = (eqa - peak) / peak * 100
    ax2.fill_between(range(len(dd)), dd, 0, color="#e74c3c", alpha=0.6)
    ax2.set_title("Drawdown (%)"); ax2.set_ylabel("Drawdown %"); ax2.grid(alpha=0.3)

    # Trade P&L bars
    ax3   = fig.add_subplot(gs[2, 0])
    pnls  = [t.pnl for t in trades]
    cols  = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
    ax3.bar(range(len(pnls)), pnls, color=cols, alpha=0.8, width=0.8)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Trade P&L (USD)"); ax3.set_xlabel("Trade #"); ax3.set_ylabel("P&L")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax3.grid(alpha=0.3, axis="y")

    # Exit pie
    ax4  = fig.add_subplot(gs[2, 1])
    wcol = {"red_brick":"#e74c3c","stop_loss":"#c0392b","below_ema21":"#e67e22",
             "partial_profit":"#2ecc71","end_of_data":"#95a5a6"}
    reasons = m["exit_reasons"]
    ax4.pie(reasons.values(), labels=reasons.keys(), autopct="%1.1f%%",
            colors=[wcol.get(r,"#3498db") for r in reasons], startangle=90,
            textprops={"fontsize":9})
    ax4.set_title("Exit Reasons")

    summary = (f"Return: {m['total_return_pct']:+.2f}%  |  Win Rate: {m['win_rate_pct']:.1f}%  |  "
               f"Trades: {m['total_trades']}  |  Max DD: {m['max_drawdown_pct']:.2f}%  |  "
               f"Sharpe: {m['sharpe_ratio']:.2f}  |  PF: {m['profit_factor']:.2f}")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", alpha=0.8))

    safe = label.replace(" ","_").replace("/","-")
    ml_t = "ml_on" if use_ml else "ml_off"
    out  = out_dir / f"backtest_{safe}_{ml_t}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Chart saved → {out}")

def save_trade_log(results, label, use_ml, out_dir):
    trades = results["trades"]
    if not trades: return
    rows = [{"entry_time": t.entry_time, "exit_time": t.exit_time,
              "entry_price": t.entry_price, "exit_price": t.exit_price,
              "qty_btc": round(t.qty,6), "pnl_usd": round(t.pnl,2),
              "pnl_pct": round(t.pnl_pct,4), "exit_reason": t.exit_reason,
              "ml_prob": round(t.ml_prob,4) if t.ml_prob>=0 else "N/A",
              "ml_approved": t.ml_approved} for t in trades]
    safe = label.replace(" ","_").replace("/","-")
    ml_t = "ml_on" if use_ml else "ml_off"
    out  = out_dir / f"trades_{safe}_{ml_t}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Trade log saved → {out}")

def load_file(path):
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):  df = pd.read_excel(path)
    elif ext == ".csv":            df = pd.read_csv(path)
    else: raise ValueError(f"Unsupported: {path}")
    df.columns = [c.strip().upper() for c in df.columns]
    missing = {"OPEN","HIGH","LOW","CLOSE"} - set(df.columns)
    if missing: raise ValueError(f"Missing columns in {path.name}: {missing}")
    if "UNIX_TIMESTAMP" in df.columns: df = df.sort_values("UNIX_TIMESTAMP").reset_index(drop=True)
    elif "DATETIME" in df.columns:     df = df.sort_values("DATETIME").reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Renko + EMA + ML Backtester")
    parser.add_argument("--data",       nargs="+", required=True,
                        help="xlsx file(s) to backtest  e.g. --data ../mldata/2023.xlsx ../mldata/2024.xlsx")
    parser.add_argument("--capital",    type=float, default=100000.0,
                        help="Starting capital USD (default: 100000)")
    parser.add_argument("--threshold",  type=float, default=CONFIG["ml_threshold"],
                        help=f"ML threshold (default: {CONFIG['ml_threshold']})")
    parser.add_argument("--brick-size", type=float, default=CONFIG["brick_size"],
                        help=f"Renko brick size (default: {CONFIG['brick_size']}). "
                             "Smaller = more trades e.g. 200 or 300 for BTC")
    parser.add_argument("--no-ml",      action="store_true", help="Disable ML filter")
    parser.add_argument("--ml-dir",     type=str, default=".", help="Folder with pkl files")
    parser.add_argument("--out",        type=str, default=".", help="Output folder")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["ml_threshold"] = args.threshold
    cfg["ml_enabled"]   = not args.no_ml
    cfg["brick_size"]   = args.brick_size
    use_ml = cfg["ml_enabled"]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ml_model = ml_scaler = ml_feature_cols = None
    if use_ml:
        d = Path(args.ml_dir)
        mp,sp,fp = d/"ml_model.pkl", d/"ml_scaler.pkl", d/"ml_feature_cols.pkl"
        if mp.exists() and sp.exists() and fp.exists():
            ml_model = joblib.load(mp); ml_scaler = joblib.load(sp)
            ml_feature_cols = joblib.load(fp)
            print(f"ML model loaded  ({len(ml_feature_cols)} features, threshold={args.threshold})")
        else:
            print("WARNING: pkl files not found — running without ML filter")
            use_ml = False

    print(f"\nStarting capital : ${args.capital:,.2f}")
    print(f"ML filter        : {'ON' if use_ml else 'OFF'}")
    print(f"Brick size       : ${cfg['brick_size']:.0f}")
    print(f"ML threshold     : {cfg['ml_threshold']}")

    all_metrics = []

    for data_path in args.data:
        p = Path(data_path)
        if not p.exists(): print(f"\nERROR: File not found — {data_path}"); continue
        print(f"\n{'='*60}\n  FILE: {p.name}\n{'='*60}")
        try:
            df = load_file(p)
        except Exception as e:
            print(f"  ERROR: {e}"); continue
        print(f"  Rows: {len(df):,}")

        results = run_backtest(df, cfg, args.capital, ml_model, ml_scaler, ml_feature_cols, use_ml)
        metrics = compute_metrics(results, args.capital)
        label   = p.stem
        all_metrics.append((label, metrics))
        print_report(metrics, label, use_ml)
        save_charts(results, metrics, label, use_ml, out_dir)
        save_trade_log(results, label, use_ml, out_dir)

    if len(all_metrics) > 1:
        print("\n" + "="*60 + "\n  COMBINED SUMMARY\n" + "="*60)
        print(f"  {'File':<10} {'Return':>10} {'Win%':>8} {'Trades':>7} {'MaxDD':>9} {'Sharpe':>8} {'PF':>6}")
        print("-"*60)
        for lbl, m in all_metrics:
            if "error" in m: print(f"  {lbl:<10}  NO TRADES"); continue
            print(f"  {lbl:<10} {m['total_return_pct']:>+9.2f}% {m['win_rate_pct']:>7.1f}% "
                  f"{m['total_trades']:>7} {m['max_drawdown_pct']:>8.2f}% "
                  f"{m['sharpe_ratio']:>8.3f} {m['profit_factor']:>6.2f}")
        print("="*60)

    print("\nDone.")

if __name__ == "__main__":
    main()