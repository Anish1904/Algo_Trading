# Algo_Trading
A hybrid algorithmic trading system that combines Renko price action, EMA trend signals, and a machine learning filter to generate high-probability trades. The system supports live trading via webhook, ML-based signal filtering, and historical backtesting with performance analytics.

# 📈 Renko + EMA + ML Algo Trading System

A complete algorithmic trading pipeline that combines **Renko charts**, **EMA-based trend detection**, and a **Machine Learning model** to filter trades and improve accuracy.

This project includes:
- Live trading engine (Binance data + webhook execution)
- ML training pipeline
- Backtesting system with performance analytics

---

## 🚀 Features

- 🔷 Renko-based price action strategy
- 📊 EMA21 & EMA50 trend confirmation
- 🤖 Machine Learning trade filtering
- 📈 Backtesting with full performance metrics
- ⚡ Real-time signal generation (via webhook)
- 📉 Risk management (Stop-loss + Partial exits)

---

## 🧠 Strategy Overview

### Entry Conditions
- Green Renko bricks
- Price above EMA21 & EMA50
- EMA21 > EMA50 (bullish trend)
- Optional ML confirmation

### Exit Conditions
- Red Renko brick
- Price drops below EMA21
- Stop-loss hit
- Partial profit after consecutive green bricks

---

## 📂 Project Structure
├── renko_strategy.py # Live trading engine
├── backtester.py # Backtesting system
├── ml_train.py # ML training pipeline
├── strategy.py # Core strategy logic
├── ml_model.pkl # Trained ML model (generated)
├── ml_scaler.pkl # Feature scaler (generated)
└── data/ # Historical OHLCV data


