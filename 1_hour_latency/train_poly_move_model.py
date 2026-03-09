#!/usr/bin/env python3
"""
train_poly_move_model.py
------------------------
Load a dataset, select a model by validation MAE (regression) or Brier (classification), then fit/export artifacts.

Modes:
- regression (default): predict delta; select by validation MAE.
- classification: predict P(adverse move >= 1 cent in window); label = (delta <= -0.01); select by Brier.

Outputs (under --out-dir) are target-specific:
- poly_move_model_<target>.joblib
- poly_move_scaler_<target>.joblib
- poly_move_imputer_<target>.joblib
- poly_move_model_report_<target>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False
    HistGradientBoostingClassifier = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS_BASELINE = ["binance_move", "binance_vol_ema", "poly_down_bid", "poly_up_bid", "secs_left"]
ENGINEERED_FEATURE_COLS = ["binance_move_secs_left", "poly_spread", "secs_left_inv"]
TARGET_DOWN = "delta_down_bid"
TARGET_UP = "delta_up_bid"
CENTS_PER_DOLLAR = 100.0
ADVERSE_THRESHOLD_DOLLARS = 0.01  # 1 cent; label = (delta <= -ADVERSE_THRESHOLD_DOLLARS)


def _resolve_feature_cols(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Resolve feature columns from dataset and --feature-set. Baseline cols must exist."""
    for c in FEATURE_COLS_BASELINE:
        if c not in df.columns:
            raise SystemExit(f"Missing baseline column: {c}")
    if feature_set == "baseline":
        return list(FEATURE_COLS_BASELINE)
    # engineered or auto: baseline + any engineered columns present
    out = list(FEATURE_COLS_BASELINE)
    for c in ENGINEERED_FEATURE_COLS:
        if c in df.columns and c not in out:
            out.append(c)
    return out


def _market_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by market: first train_frac of markets (by first appearance order) = train,
    rest = test. Markets are ordered by min(epoch_ns) per market to preserve time order.
    """
    market_order = (
        df.groupby("market_id")["epoch_ns"]
        .min()
        .sort_values()
        .index.tolist()
    )
    n_markets = len(market_order)
    n_train_markets = max(1, int(n_markets * train_frac))
    if n_train_markets >= n_markets:
        n_train_markets = max(1, n_markets - 1)
    train_markets = set(market_order[:n_train_markets])
    test_markets = set(market_order[n_train_markets:])

    train_df = df[df["market_id"].isin(train_markets)].copy()
    test_df = df[df["market_id"].isin(test_markets)].copy()
    return train_df, test_df


def _market_train_val_split(
    train_df: pd.DataFrame,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a train-only dataframe by market chronology into train_core and validation.
    Uses the most recent val_frac markets as validation.
    """
    market_order = (
        train_df.groupby("market_id")["epoch_ns"]
        .min()
        .sort_values()
        .index.tolist()
    )
    n_markets = len(market_order)
    if n_markets <= 1:
        return train_df.copy(), train_df.iloc[0:0].copy()

    n_val_markets = max(1, int(round(n_markets * val_frac)))
    if n_val_markets >= n_markets:
        n_val_markets = n_markets - 1

    core_markets = set(market_order[:-n_val_markets])
    val_markets = set(market_order[-n_val_markets:])
    core_df = train_df[train_df["market_id"].isin(core_markets)].copy()
    val_df = train_df[train_df["market_id"].isin(val_markets)].copy()
    return core_df, val_df


def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    imputer: Optional[SimpleImputer] = None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    """Return (X, y, imputer). Drops rows with missing target; imputes features (median)."""
    use = df[feature_cols + [target]].dropna(subset=[target])
    X = use[feature_cols].values
    y = use[target].values
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
    else:
        X = imputer.transform(X)
    return X, y, imputer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train poly move regression model from poly_move_dataset.csv; split by market (train 3/4, test 1/4)."
    )
    _out_default = Path(__file__).resolve().parent / "out"
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(_out_default / "poly_move_dataset.csv"),
        help="Path to poly_move_dataset.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_out_default),
        help="Output directory for model, scaler, report, plots",
    )
    ap.add_argument(
        "--train-frac",
        type=float,
        default=0.75,
        help="Fraction of markets (by order) for train (default 0.75); rest = test",
    )
    ap.add_argument(
        "--train-all-markets",
        action="store_true",
        help="Train deploy model on 100%% of markets (no held-out test split).",
    )
    ap.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to held-out test dataset CSV (e.g. one market). When set, --dataset is used in full as train and this file as test.",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Task: regression (delta) or classification (P(adverse move >= 1c in window)).",
    )
    ap.add_argument(
        "--target",
        type=str,
        choices=["delta_down_bid", "delta_up_bid"],
        default="delta_down_bid",
        help="Target column (default delta_down_bid)",
    )
    ap.add_argument(
        "--model",
        type=str,
        choices=["auto", "linear", "ridge", "tree"],
        default="auto",
        help="Model family: auto (select best), or force linear/ridge/tree.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Fraction of train markets used for validation model selection (default 0.20).",
    )
    ap.add_argument(
        "--ridge-alphas",
        type=str,
        default="0.1,1.0,10.0,25.0",
        help="Comma-separated alpha grid for ridge (default: 0.1,1.0,10.0,25.0).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Write predicted_vs_actual.png",
    )
    ap.add_argument(
        "--export-test-preds",
        action="store_true",
        help="Export held-out test predictions with features and errors to CSV.",
    )
    ap.add_argument(
        "--feature-set",
        type=str,
        choices=["baseline", "engineered", "auto"],
        default="auto",
        help="Feature set: baseline only, baseline+engineered (if present), or auto (default).",
    )
    ap.add_argument(
        "--target-clip-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Winsorize target to P and 100-P percentiles (e.g. 2.5). Off by default.",
    )
    args = ap.parse_args()

    ds_path = Path(args.dataset).expanduser().resolve()
    if not ds_path.is_file():
        raise SystemExit(f"Dataset not found: {ds_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ds_path)
    if df.empty:
        raise SystemExit("Dataset is empty.")

    feature_cols = _resolve_feature_cols(df, args.feature_set)
    for c in [args.target]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    test_dataset_path = getattr(args, "test_dataset", None)
    if test_dataset_path:
        test_path = Path(test_dataset_path).expanduser().resolve()
        if not test_path.is_file():
            raise SystemExit(f"Test dataset not found: {test_path}")
        train_df = df.copy()
        test_df = pd.read_csv(test_path)
        if test_df.empty:
            raise SystemExit("Test dataset is empty.")
        for c in feature_cols + [args.target]:
            if c not in test_df.columns:
                raise SystemExit(f"Test dataset missing column: {c}")
    elif args.train_all_markets:
        train_df = df.copy()
        test_df = df.iloc[0:0].copy()
    else:
        train_df, test_df = _market_train_test_split(df, train_frac=args.train_frac)
        if train_df.empty or test_df.empty:
            raise SystemExit("Train or test set is empty after market split.")

    target_clip_low: Optional[float] = None
    target_clip_high: Optional[float] = None
    target_n_clipped_low = 0
    target_n_clipped_high = 0
    if args.task == "regression" and args.target_clip_percentile is not None and args.target_clip_percentile > 0:
        p = float(args.target_clip_percentile)
        target_clip_low, target_clip_high = np.nanpercentile(
            train_df[args.target].dropna(), [p, 100.0 - p]
        )
        target_n_clipped_low = int((train_df[args.target] < target_clip_low).sum())
        target_n_clipped_high = int((train_df[args.target] > target_clip_high).sum())

    def _clip_y_if(y_arr: np.ndarray) -> np.ndarray:
        if target_clip_low is None:
            return y_arr
        return np.clip(y_arr, target_clip_low, target_clip_high)

    def _adverse_label(delta_arr: np.ndarray) -> np.ndarray:
        """Binary: 1 if delta <= -1 cent (adverse move)."""
        return (delta_arr <= -ADVERSE_THRESHOLD_DOLLARS).astype(np.int64)

    train_core_df, val_df = _market_train_val_split(train_df, val_frac=args.val_frac)
    if train_core_df.empty:
        raise SystemExit("Train-core set is empty after market validation split.")

    X_core, y_core, imputer_core = _prepare_xy(train_core_df, args.target, feature_cols)
    if args.task == "classification":
        y_core = _adverse_label(y_core)
    else:
        y_core = _clip_y_if(y_core)
    if val_df.empty:
        X_val, y_val, _ = _prepare_xy(train_core_df, args.target, feature_cols, imputer=imputer_core)
    else:
        X_val, y_val, _ = _prepare_xy(val_df, args.target, feature_cols, imputer=imputer_core)
    if args.task == "classification":
        y_val = _adverse_label(y_val)
    else:
        y_val = _clip_y_if(y_val)

    X_train_all, y_train_all, _ = _prepare_xy(train_df, args.target, feature_cols, imputer=imputer_core)
    if args.task == "classification":
        y_train_all = _adverse_label(y_train_all)
    else:
        y_train_all = _clip_y_if(y_train_all)
    if args.train_all_markets:
        X_test = np.empty((0, len(feature_cols)), dtype=np.float64)
        y_test = np.empty((0,), dtype=np.float64)
        test_use_df = test_df.iloc[0:0].copy()
    else:
        X_test, y_test, _ = _prepare_xy(test_df, args.target, feature_cols, imputer=imputer_core)
        test_use_df = test_df[feature_cols + [args.target]].dropna(subset=[args.target]).copy()
    if args.task == "classification" and len(y_test):
        y_test = _adverse_label(y_test)

    if args.task == "classification":
        # Classifier candidates: logistic + HistGradientBoostingClassifier
        candidates: list[tuple[str, Any, bool, dict[str, Any]]] = []
        if args.model in {"auto", "linear"}:
            candidates.append(("logistic", LogisticRegression(max_iter=500, random_state=42), True, {}))
        if args.model in {"auto", "tree"} and HAS_HGB and HistGradientBoostingClassifier is not None:
            for cfg in [
                {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 10},
                {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 15},
                {"max_depth": 6, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 10},
            ]:
                name = f"tree_d{cfg['max_depth']}_lr{cfg['learning_rate']}_it{cfg['max_iter']}_leaf{cfg['min_samples_leaf']}"
                candidates.append((
                    name,
                    HistGradientBoostingClassifier(
                        max_depth=cfg["max_depth"],
                        learning_rate=cfg["learning_rate"],
                        max_iter=cfg["max_iter"],
                        min_samples_leaf=cfg["min_samples_leaf"],
                        random_state=42,
                    ),
                    False,
                    cfg,
                ))
        if not candidates:
            raise SystemExit("No classifier candidates; need sklearn and model auto or tree.")
        candidate_scores = []
        best = None
        best_brier = float("inf")
        best_roc = -1.0
        for name, model, use_scaling, params in candidates:
            if use_scaling:
                scaler_sel = StandardScaler(with_mean=True, with_std=True)
                X_core_fit = scaler_sel.fit_transform(X_core)
                X_val_fit = scaler_sel.transform(X_val)
            else:
                scaler_sel = None
                X_core_fit = X_core
                X_val_fit = X_val
            model.fit(X_core_fit, y_core)
            p_val = model.predict_proba(X_val_fit)[:, 1]
            brier = float(brier_score_loss(y_val, p_val))
            roc = -1.0
            if len(np.unique(y_val)) == 2:
                roc = float(roc_auc_score(y_val, p_val))
            candidate_scores.append({"name": name, "params": params, "val_brier": brier, "val_roc_auc": roc})
            if brier < best_brier or (brier == best_brier and roc > best_roc):
                best_brier = brier
                best_roc = roc
                best = (name, model, use_scaling, params)
        assert best is not None
        best_name, _best_model_obj, best_use_scaling, best_params = best
        if best_name == "logistic":
            model = LogisticRegression(max_iter=500, random_state=42)
        elif best_name.startswith("tree"):
            model = HistGradientBoostingClassifier(
                max_depth=int(best_params["max_depth"]),
                learning_rate=float(best_params["learning_rate"]),
                max_iter=int(best_params["max_iter"]),
                min_samples_leaf=int(best_params["min_samples_leaf"]),
                random_state=42,
            )
        else:
            raise SystemExit(f"Unknown best classifier: {best_name}")
        if best_use_scaling:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_fit = scaler.fit_transform(X_train_all)
            X_test_fit = scaler.transform(X_test) if len(X_test) else X_test
        else:
            scaler = None
            X_train_fit = X_train_all
            X_test_fit = X_test
        model.fit(X_train_fit, y_train_all)
        p_train = model.predict_proba(X_train_fit)[:, 1]
        p_test = model.predict_proba(X_test_fit)[:, 1] if len(X_test_fit) else np.empty((0,), dtype=np.float64)
        brier_train = float(brier_score_loss(y_train_all, p_train))
        brier_test = float(brier_score_loss(y_test, p_test)) if len(y_test) else None
        log_loss_train = float(log_loss(y_train_all, np.column_stack([1 - p_train, p_train])))
        log_loss_test = float(log_loss(y_test, np.column_stack([1 - p_test, p_test]))) if len(y_test) else None
        roc_train = float(roc_auc_score(y_train_all, p_train)) if len(np.unique(y_train_all)) == 2 else None
        roc_test = float(roc_auc_score(y_test, p_test)) if len(y_test) and len(np.unique(y_test)) == 2 else None
        pr_train = float(average_precision_score(y_train_all, p_train))
        pr_test = float(average_precision_score(y_test, p_test)) if len(y_test) else None
        pred_train_05 = (p_train >= 0.5).astype(np.int64)
        pred_test_05 = (p_test >= 0.5).astype(np.int64) if len(p_test) else np.array([], dtype=np.int64)
        prec_05_train = float(precision_score(y_train_all, pred_train_05, zero_division=0))
        rec_05_train = float(recall_score(y_train_all, pred_train_05, zero_division=0))
        cm = confusion_matrix(y_train_all, pred_train_05)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        fpr_05_train = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_05_train = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        prec_05_test = float(precision_score(y_test, pred_test_05, zero_division=0)) if len(y_test) else None
        rec_05_test = float(recall_score(y_test, pred_test_05, zero_division=0)) if len(y_test) else None
        label_window_sec = None
        if "next_tick_window_sec" in df.columns:
            vals = pd.to_numeric(df["next_tick_window_sec"], errors="coerce").dropna().unique()
            if len(vals) > 0:
                label_window_sec = float(vals[0])
        report = {
            "target": args.target,
            "task_type": "classification",
            "model": best_name,
            "feature_cols": feature_cols,
            "evaluation_mode": "train_all_no_holdout" if args.train_all_markets else "market_holdout",
            "train_all_markets": bool(args.train_all_markets),
            "n_rows": int(len(df)),
            "n_markets": int(df["market_id"].nunique()),
            "n_train": int(len(y_train_all)),
            "n_test": int(len(y_test)) if len(y_test) else 0,
            "n_train_markets": int(train_df["market_id"].nunique()),
            "n_test_markets": int(test_df["market_id"].nunique()) if not args.train_all_markets else 0,
            "n_val_markets": int(val_df["market_id"].nunique()) if not val_df.empty else 0,
            "label_definition": "adverse_move_ge_1c_in_window",
            "label_condition": "delta <= -0.01",
            "label_window_sec": label_window_sec,
            "selected_by": "val_brier",
            "val_frac_markets": float(args.val_frac),
            "candidate_scores": candidate_scores,
            "best_params": best_params,
            "brier_train": brier_train,
            "brier_test": brier_test,
            "log_loss_train": log_loss_train,
            "log_loss_test": log_loss_test,
            "roc_auc_train": roc_train,
            "roc_auc_test": roc_test,
            "pr_auc_train": pr_train,
            "pr_auc_test": pr_test,
            "threshold_0_5_precision_train": prec_05_train,
            "threshold_0_5_recall_train": rec_05_train,
            "threshold_0_5_fpr_train": fpr_05_train,
            "threshold_0_5_fnr_train": fnr_05_train,
            "threshold_0_5_precision_test": prec_05_test,
            "threshold_0_5_recall_test": rec_05_test,
        }
        y_pred_test = p_test
        y_pred_train = p_train
        rmse_test = None
        mae_test = None
        rmse_train = None
        mae_train = None
    else:
        # Regression path
        ridge_alphas: list[float] = []
        for tok in str(args.ridge_alphas).split(","):
            tok = tok.strip()
            if tok:
                ridge_alphas.append(float(tok))
        if not ridge_alphas:
            ridge_alphas = [1.0]

        candidates = []
        if args.model in {"auto", "linear"}:
            candidates.append(("linear", LinearRegression(), True, {}))
        if args.model in {"auto", "ridge"}:
            for alpha in ridge_alphas:
                candidates.append((f"ridge_alpha_{alpha:g}", Ridge(alpha=float(alpha)), True, {"alpha": float(alpha)}))
        if args.model in {"auto", "tree"}:
            if not HAS_HGB:
                raise SystemExit("HistGradientBoostingRegressor not available; use linear or ridge.")
            tree_grid = [
                {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 10, "loss": "squared_error"},
                {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 10, "loss": "absolute_error"},
                {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 15, "loss": "squared_error"},
                {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 15, "loss": "absolute_error"},
                {"max_depth": 6, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 10, "loss": "squared_error"},
                {"max_depth": 6, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 10, "loss": "absolute_error"},
                {"max_depth": 6, "learning_rate": 0.10, "max_iter": 200, "min_samples_leaf": 10, "loss": "squared_error"},
                {"max_depth": 6, "learning_rate": 0.10, "max_iter": 200, "min_samples_leaf": 10, "loss": "absolute_error"},
                {"max_depth": 8, "learning_rate": 0.05, "max_iter": 400, "min_samples_leaf": 20, "loss": "absolute_error"},
            ]
            for cfg in tree_grid:
                loss_name = cfg.get("loss", "squared_error")
                leaf = cfg.get("min_samples_leaf", 10)
                name = f"tree_d{cfg['max_depth']}_lr{cfg['learning_rate']}_it{cfg['max_iter']}_leaf{leaf}_{loss_name.replace('_', '')}"
                candidates.append((
                    name,
                    HistGradientBoostingRegressor(
                        max_depth=cfg["max_depth"],
                        learning_rate=cfg["learning_rate"],
                        max_iter=cfg["max_iter"],
                        min_samples_leaf=cfg.get("min_samples_leaf", 10),
                        loss=cfg.get("loss", "squared_error"),
                        random_state=42,
                    ),
                    False,
                    cfg,
                ))

        if not candidates:
            raise SystemExit("No model candidates configured.")

        candidate_scores = []
        best = None
        best_val_mae = float("inf")
        best_val_rmse_tie = float("inf")

        for name, model, use_scaling, params in candidates:
            if use_scaling:
                scaler_sel = StandardScaler(with_mean=True, with_std=True)
                X_core_fit = scaler_sel.fit_transform(X_core)
                X_val_fit = scaler_sel.transform(X_val)
            else:
                scaler_sel = None
                X_core_fit = X_core
                X_val_fit = X_val
            model.fit(X_core_fit, y_core)
            y_val_pred = model.predict(X_val_fit)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            val_mae = float(mean_absolute_error(y_val, y_val_pred))
            candidate_scores.append({
                "name": name,
                "family": "tree" if not use_scaling else ("ridge" if name.startswith("ridge") else "linear"),
                "params": params,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            })
            if val_mae < best_val_mae or (val_mae == best_val_mae and val_rmse < best_val_rmse_tie):
                best_val_mae = val_mae
                best_val_rmse_tie = val_rmse
                best = (name, model, use_scaling, params, val_rmse, val_mae)

        assert best is not None
        best_name, _best_model_obj, best_use_scaling, best_params, best_val_rmse, best_val_mae = best

        if best_name == "linear":
            model = LinearRegression()
        elif best_name.startswith("ridge"):
            model = Ridge(alpha=float(best_params["alpha"]))
        elif best_name.startswith("tree"):
            if not HAS_HGB:
                raise SystemExit("HistGradientBoostingRegressor not available; use linear or ridge.")
            model = HistGradientBoostingRegressor(
                max_depth=int(best_params["max_depth"]),
                learning_rate=float(best_params["learning_rate"]),
                max_iter=int(best_params["max_iter"]),
                min_samples_leaf=int(best_params.get("min_samples_leaf", 10)),
                loss=str(best_params.get("loss", "squared_error")),
                random_state=42,
            )
        else:
            raise SystemExit(f"Unknown best model: {best_name}")

        if best_use_scaling:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_fit = scaler.fit_transform(X_train_all)
            X_test_fit = scaler.transform(X_test) if len(X_test) else X_test
        else:
            scaler = None
            X_train_fit = X_train_all
            X_test_fit = X_test

        model.fit(X_train_fit, y_train_all)
        y_pred_test = model.predict(X_test_fit) if len(X_test_fit) else np.empty((0,), dtype=np.float64)
        y_pred_train = model.predict(X_train_fit)

        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test))) if len(y_test) else None
        mae_test = float(mean_absolute_error(y_test, y_pred_test)) if len(y_test) else None
        rmse_train = float(np.sqrt(mean_squared_error(y_train_all, y_pred_train)))
        mae_train = float(mean_absolute_error(y_train_all, y_pred_train))

        label_window_sec = None
        if "next_tick_window_sec" in df.columns:
            vals = pd.to_numeric(df["next_tick_window_sec"], errors="coerce").dropna().unique()
            if len(vals) > 0:
                label_window_sec = float(vals[0])

        report = {
            "target": args.target,
            "model": best_name,
            "feature_cols": feature_cols,
            "evaluation_mode": "train_all_no_holdout" if args.train_all_markets else "market_holdout",
            "train_all_markets": bool(args.train_all_markets),
            "n_rows": int(len(df)),
            "n_markets": int(df["market_id"].nunique()),
            "n_train": int(len(y_train_all)),
            "n_test": int(len(y_test)) if len(y_test) else 0,
            "n_train_markets": int(train_df["market_id"].nunique()),
            "n_test_markets": int(test_df["market_id"].nunique()) if not args.train_all_markets else 0,
            "n_val_markets": int(val_df["market_id"].nunique()) if not val_df.empty else 0,
            "selected_by": "val_mae",
            "val_frac_markets": float(args.val_frac),
            "val_rmse_selected": float(best_val_rmse),
            "val_mae_selected": float(best_val_mae),
            "rmse_train": rmse_train,
            "mae_train": mae_train,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "mae_train_cents": float(mae_train * CENTS_PER_DOLLAR),
            "rmse_train_cents": float(rmse_train * CENTS_PER_DOLLAR),
            "mae_test_cents": float(mae_test * CENTS_PER_DOLLAR) if mae_test is not None else None,
            "rmse_test_cents": float(rmse_test * CENTS_PER_DOLLAR) if rmse_test is not None else None,
            "target_clip_percentile": args.target_clip_percentile,
            "target_clip_low": target_clip_low,
            "target_clip_high": target_clip_high,
            "target_n_clipped_low": target_n_clipped_low,
            "target_n_clipped_high": target_n_clipped_high,
            "target_n_train_total": int(len(y_train_all)),
            "candidate_scores": candidate_scores,
            "best_params": best_params,
            "label_definition": "next_poly_tick_change_within_window_else_zero",
            "label_window_sec": label_window_sec,
        }
        if best_name == "linear" and hasattr(model, "coef_"):
            report["coef"] = {c: float(v) for c, v in zip(feature_cols, model.coef_)}
            report["intercept"] = float(model.intercept_)

    report_path = out_dir / f"poly_move_model_report_{args.target}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_path}")

    model_path = out_dir / f"poly_move_model_{args.target}.joblib"
    joblib.dump(model, model_path)
    scaler_path = out_dir / f"poly_move_scaler_{args.target}.joblib"
    joblib.dump(scaler, scaler_path)
    imputer_path = out_dir / f"poly_move_imputer_{args.target}.joblib"
    joblib.dump(imputer_core, imputer_path)
    print(f"Saved model to {model_path}, scaler to {scaler_path}, imputer to {imputer_path}")

    if args.plot and len(y_test) and args.task == "regression":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=10, label="test")
        mn = min(y_test.min(), y_pred_test.min())
        mx = max(y_test.max(), y_pred_test.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel(f"Actual {args.target}")
        ax.set_ylabel(f"Predicted {args.target}")
        ax.set_title("Predicted vs actual (test set)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plot_path = out_dir / f"poly_move_predicted_vs_actual_{args.target}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {plot_path}")

    if args.export_test_preds and len(y_test):
        preds_df = test_use_df.copy()
        preds_df["target_col"] = str(args.target)
        preds_df["actual"] = y_test
        preds_df["predicted"] = y_pred_test
        if args.task == "classification":
            preds_df["pred_label"] = (np.asarray(y_pred_test) >= 0.5).astype(np.int64)
        else:
            preds_df["residual"] = preds_df["predicted"] - preds_df["actual"]
            preds_df["abs_error"] = np.abs(preds_df["residual"])
            preds_df["sq_error"] = preds_df["residual"] ** 2

        # Keep common diagnostic columns when present.
        diagnostic_cols = [
            "step_id",
            "epoch_ns",
            "market_id",
            "next_tick_window_sec",
            "label_reason_down",
            "label_reason_up",
            "target_delay_sec_down",
            "target_delay_sec_up",
            "target_epoch_ns_down",
            "target_epoch_ns_up",
        ]
        for c in diagnostic_cols:
            if c in test_df.columns and c not in preds_df.columns:
                preds_df[c] = test_df.loc[preds_df.index, c]

        pred_path = out_dir / f"poly_move_test_predictions_{args.target}.csv"
        preds_df.to_csv(pred_path, index=False)
        print(f"Wrote {pred_path}")

    if args.task == "classification":
        if report.get("brier_test") is not None:
            print(f"Classification: Brier_test={report['brier_test']:.6f} log_loss_test={report['log_loss_test']:.6f} ROC_AUC_test={report.get('roc_auc_test')}")
        else:
            print(f"Classification: Brier_train={report['brier_train']:.6f} (no held-out test)")
    elif rmse_test is None or mae_test is None:
        print(f"Train-only mode: RMSE_train={rmse_train:.6f} MAE_train={mae_train:.6f} (no held-out test)")
    else:
        print(f"Test RMSE={rmse_test:.6f} MAE={mae_test:.6f}")


if __name__ == "__main__":
    main()
