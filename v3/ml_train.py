"""
ML Trade Filter — Training Pipeline
=====================================
Trains a binary classifier to approve/reject Renko trade signals.

Input  : OHLCV Excel file with columns:
         UNIX_TIMESTAMP, DATETIME, Time, OPEN, HIGH, CLOSE, LOW,
         VOLUME_USD, VOLUME_BTC

Output : ml_model.pkl        — trained model
         ml_scaler.pkl       — feature scaler
         ml_report.txt       — classification report + metrics
         ml_feature_importance.png — feature importance chart

Usage:
    pip install pandas numpy scikit-learn xgboost matplotlib openpyxl joblib
    python ml_train.py --data your_data.xlsx
"""

import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────── CONFIG ──────────────────────────────────────────

BRICK_SIZE      = 100.0   # must match renko_strategy.py
EMA_FAST        = 21
EMA_SLOW        = 50
MA_SHORT        = 9
MA_LONG         = 21
FUTURE_WINDOW   = 10      # bars ahead to measure if trade was profitable
PROFIT_THRESH   = 0.003   # 0.3% minimum move to label as "good trade"
TEST_SIZE       = 0.20    # 20% held out for final test
VAL_SIZE        = 0.10    # 10% of train used as validation during CV
RANDOM_STATE    = 42

OUTPUT_DIR      = Path(".")

# ─────────────────────── FEATURE ENGINEERING ──────────────────────────────────

def compute_ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_renko_bricks(close: pd.Series, brick_size: float) -> pd.Series:
    """Vectorised Renko direction: +1 green, -1 red, 0 no new brick."""
    directions = pd.Series(0, index=close.index)
    base = close.iloc[0]
    for i, price in enumerate(close):
        diff = price - base
        if diff >= brick_size:
            n = int(diff / brick_size)
            directions.iloc[i] = 1
            base += n * brick_size
        elif diff <= -brick_size:
            n = int(-diff / brick_size)
            directions.iloc[i] = -1
            base -= n * brick_size
    return directions


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from raw OHLCV data.
    Features mirror what the live strategy sees at signal time.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Price-based features ──────────────────────────────────────────────────
    feat["close"]        = df["CLOSE"]
    feat["returns_1"]    = df["CLOSE"].pct_change(1)
    feat["returns_5"]    = df["CLOSE"].pct_change(5)
    feat["returns_10"]   = df["CLOSE"].pct_change(10)
    feat["returns_20"]   = df["CLOSE"].pct_change(20)

    feat["hl_ratio"]     = (df["HIGH"] - df["LOW"]) / df["CLOSE"]   # candle range %
    feat["co_ratio"]     = (df["CLOSE"] - df["OPEN"]) / df["OPEN"]  # candle body %

    # ── EMA features ─────────────────────────────────────────────────────────
    ema_fast = compute_ema_series(df["CLOSE"], EMA_FAST)
    ema_slow = compute_ema_series(df["CLOSE"], EMA_SLOW)
    ma_short = compute_ema_series(df["CLOSE"], MA_SHORT)
    ma_long  = compute_ema_series(df["CLOSE"], MA_LONG)

    feat["ema_fast"]          = ema_fast
    feat["ema_slow"]          = ema_slow
    feat["price_vs_ema_fast"] = (df["CLOSE"] - ema_fast) / ema_fast
    feat["price_vs_ema_slow"] = (df["CLOSE"] - ema_slow) / ema_slow
    feat["ema_spread"]        = (ema_fast - ema_slow) / ema_slow     # golden/death cross proximity
    feat["ema_slope_fast"]    = ema_fast.pct_change(3)               # EMA21 slope
    feat["ema_slope_slow"]    = ema_slow.pct_change(3)               # EMA50 slope

    # MA crossover regime: +1 bullish, -1 bearish
    feat["ma_regime"]         = np.where(ma_short > ma_long, 1, -1)
    feat["ma_spread"]         = (ma_short - ma_long) / ma_long

    # ── Renko features ───────────────────────────────────────────────────────
    renko_dir = compute_renko_bricks(df["CLOSE"], BRICK_SIZE)
    feat["renko_direction"]   = renko_dir

    # rolling count of green/red bricks in last 5 brick events
    feat["green_streak"]      = (
        renko_dir.rolling(5).apply(lambda x: (x == 1).sum(), raw=True)
    )
    feat["red_streak"]        = (
        renko_dir.rolling(5).apply(lambda x: (x == -1).sum(), raw=True)
    )

    # ── Volatility features ───────────────────────────────────────────────────
    feat["volatility_10"]     = df["CLOSE"].pct_change().rolling(10).std()
    feat["volatility_20"]     = df["CLOSE"].pct_change().rolling(20).std()
    feat["atr"]               = (
        pd.concat([
            df["HIGH"] - df["LOW"],
            (df["HIGH"] - df["CLOSE"].shift()).abs(),
            (df["LOW"]  - df["CLOSE"].shift()).abs(),
        ], axis=1).max(axis=1).rolling(14).mean()
    )
    feat["atr_pct"]           = feat["atr"] / df["CLOSE"]

    # ── Volume features ───────────────────────────────────────────────────────
    feat["volume_usd"]        = df["VOLUME_USD"]
    feat["volume_btc"]        = df["VOLUME_BTC"]
    feat["vol_usd_ma"]        = df["VOLUME_USD"].rolling(20).mean()
    feat["vol_ratio"]         = df["VOLUME_USD"] / feat["vol_usd_ma"]   # volume surge

    # ── Momentum / oscillators ────────────────────────────────────────────────
    delta = df["CLOSE"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    feat["rsi_14"]            = 100 - 100 / (1 + rs)

    # MACD
    macd_line   = compute_ema_series(df["CLOSE"], 12) - compute_ema_series(df["CLOSE"], 26)
    macd_signal = compute_ema_series(macd_line, 9)
    feat["macd"]              = macd_line
    feat["macd_signal"]       = macd_signal
    feat["macd_hist"]         = macd_line - macd_signal

    # ── Time features ─────────────────────────────────────────────────────────
    if "DATETIME" in df.columns:
        dt = pd.to_datetime(df["DATETIME"], errors="coerce")
        feat["hour"]          = dt.dt.hour
        feat["day_of_week"]   = dt.dt.dayofweek
        feat["month"]         = dt.dt.month

    return feat


def create_labels(df: pd.DataFrame, future_window: int, profit_thresh: float) -> pd.Series:
    """
    Binary label: 1 = "good trade" (price moved up ≥ profit_thresh in next N bars)
                  0 = "bad trade"

    Logic mirrors what a long entry from the Renko strategy would experience:
    - Look FUTURE_WINDOW bars ahead
    - Max favourable move vs entry close
    - Label 1 if max move ≥ PROFIT_THRESH
    """
    close = df["CLOSE"]
    labels = pd.Series(0, index=df.index)
    for i in range(len(close) - future_window):
        entry = close.iloc[i]
        future_slice = close.iloc[i + 1: i + 1 + future_window]
        max_gain = (future_slice.max() - entry) / entry
        if max_gain >= profit_thresh:
            labels.iloc[i] = 1
    return labels

# ─────────────────────────── LOAD DATA ────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/6] Loading data from: {path}")
    p = Path(path)

    # If a folder is passed, load ALL xlsx/xls/csv files and concatenate them
    if p.is_dir():
        all_files = sorted(p.glob("*.xlsx")) + sorted(p.glob("*.xls")) + sorted(p.glob("*.csv"))
        if not all_files:
            raise ValueError(f"No .xlsx / .xls / .csv files found in folder: {path}")
        print(f"    Found {len(all_files)} file(s): {[f.name for f in all_files]}")
        frames = []
        for f in all_files:
            print(f"    Reading {f.name} ...")
            ext = f.suffix.lower()
            if ext in (".xlsx", ".xls"):
                frames.append(pd.read_excel(f))
            else:
                frames.append(pd.read_csv(f))
        df = pd.concat(frames, ignore_index=True)
        print(f"    Combined {len(all_files)} files → {len(df):,} total rows")
    else:
        ext = p.suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif ext == ".csv":
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_USD", "VOLUME_BTC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort chronologically
    if "UNIX_TIMESTAMP" in df.columns:
        df = df.sort_values("UNIX_TIMESTAMP").reset_index(drop=True)
    elif "DATETIME" in df.columns:
        df = df.sort_values("DATETIME").reset_index(drop=True)

    print(f"    Rows loaded : {len(df):,}")
    print(f"    Date range  : {df.get('DATETIME', pd.Series(['?'])).iloc[0]}  →  "
          f"{df.get('DATETIME', pd.Series(['?'])).iloc[-1]}")
    return df

# ────────────────────────── TRAIN / EVALUATE ──────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, name, model):
    """Fit, cross-validate, and report one model."""
    print(f"\n  ── {name} ──")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"    CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    pr  = precision_score(y_test, y_pred)
    rc  = recall_score(y_test, y_pred)

    print(f"    Test ROC-AUC: {auc:.4f}")
    print(f"    Precision   : {pr:.4f}")
    print(f"    Recall      : {rc:.4f}")
    print(f"    F1          : {f1:.4f}")

    return model, auc, y_pred, y_proba

# ─────────────────────────── MAIN ─────────────────────────────────────────────

def main(data_path: str):

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_data(data_path)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/6] Engineering features ...")
    features = engineer_features(df)
    labels   = create_labels(df, FUTURE_WINDOW, PROFIT_THRESH)

    # Drop warmup rows that have NaN from rolling windows
    valid_idx = features.dropna().index
    features  = features.loc[valid_idx]
    labels    = labels.loc[valid_idx]

    # Drop raw price cols not needed in model (leave indicator-derived ones)
    drop_cols = ["close", "ema_fast", "ema_slow", "vol_usd_ma"]
    feature_cols = [c for c in features.columns if c not in drop_cols]
    X = features[feature_cols].values
    y = labels.values

    print(f"    Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    Label balance  : {y.mean()*100:.1f}% positive (good trades)")

    # ── 3. Train / Test split  (chronological — NO shuffle to avoid leakage) ──
    print("\n[3/6] Splitting data (chronological) ...")
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_end = int(len(df) * (1 - TEST_SIZE))
    print(f"    Train : rows 0 → {split_idx:,}  ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"    Test  : rows {split_idx:,} → {len(X):,}  ({TEST_SIZE*100:.0f}%)")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\n[4/6] Training models ...")

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in candidates.items():
        m, auc, y_pred, y_proba = train_and_evaluate(
            X_train, X_test, y_train, y_test, name, model
        )
        results[name] = {
            "model": m, "auc": auc,
            "y_pred": y_pred, "y_proba": y_proba,
        }

    # ── 5. Select best model ──────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    best_model = best["model"]
    print(f"\n[5/6] Best model → {best_name}  (AUC={best['auc']:.4f})")

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    print("\n[6/6] Saving artefacts ...")

    joblib.dump(best_model, OUTPUT_DIR / "ml_model.pkl")
    joblib.dump(scaler,     OUTPUT_DIR / "ml_scaler.pkl")
    joblib.dump(feature_cols, OUTPUT_DIR / "ml_feature_cols.pkl")
    print("    Saved: ml_model.pkl  ml_scaler.pkl  ml_feature_cols.pkl")

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "ML TRADE FILTER — TRAINING REPORT",
        "=" * 60,
        f"Best model     : {best_name}",
        f"Test ROC-AUC   : {best['auc']:.4f}",
        f"",
        "Classification Report (test set):",
        classification_report(y_test, best["y_pred"],
                              target_names=["Bad Trade", "Good Trade"]),
        "",
        "Confusion Matrix:",
        str(confusion_matrix(y_test, best["y_pred"])),
        "",
        "Feature columns used:",
        *[f"  {i+1:2d}. {c}" for i, c in enumerate(feature_cols)],
        "",
        "Training config:",
        f"  Brick size      : {BRICK_SIZE}",
        f"  EMA fast/slow   : {EMA_FAST} / {EMA_SLOW}",
        f"  MA short/long   : {MA_SHORT} / {MA_LONG}",
        f"  Future window   : {FUTURE_WINDOW} bars",
        f"  Profit threshold: {PROFIT_THRESH*100:.1f}%",
        f"  Train/Test split: {(1-TEST_SIZE)*100:.0f}% / {TEST_SIZE*100:.0f}%",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    (OUTPUT_DIR / "ml_report.txt").write_text(report_text)
    print("    Saved: ml_report.txt")
    print("\n" + report_text)

    # ── Feature importance chart ──────────────────────────────────────────────
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        else:
            importances = np.zeros(len(feature_cols))

        sorted_idx = np.argsort(importances)[::-1][:20]
        top_features = [feature_cols[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#2ecc71" if imp > np.median(top_importances) else "#3498db"
                  for imp in top_importances]
        bars = ax.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top 20 Feature Importances — {best_name}", fontsize=14, fontweight="bold")
        ax.axvline(np.median(top_importances), color="red", linestyle="--",
                   alpha=0.7, label="Median importance")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "ml_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("    Saved: ml_feature_importance.png")
    except Exception as e:
        print(f"    Warning: could not generate feature chart — {e}")

    print("\n✅ Training complete. Files ready for integration:")
    print("   ml_model.pkl          — trained classifier")
    print("   ml_scaler.pkl         — feature scaler")
    print("   ml_feature_cols.pkl   — ordered feature list")
    print("   ml_report.txt         — full evaluation report")
    print("   ml_feature_importance.png — top features")


# ──────────────────────────── ENTRY POINT ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Renko ML trade filter")
    parser.add_argument(
        "--data", required=True,
        help="Path to your OHLCV Excel/CSV file (e.g. btc_data.xlsx)"
    )
    args = parser.parse_args()
    main(args.data)