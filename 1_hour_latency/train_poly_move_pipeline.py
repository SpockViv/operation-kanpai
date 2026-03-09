#!/usr/bin/env python3
"""
train_poly_move_pipeline.py
---------------------------
Single entrypoint pipeline that:
1) Reads mm_rl_log.sqlite and builds the poly move dataset.
2) Trains target-specific models for DOWN and UP bid deltas.
3) Exports a single deploy bundle used by market_maker_v2.py.

This consolidates extraction + training + artifact export into one command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib


BUNDLE_SCHEMA_VERSION = "poly_move_bundle_v1"


def _run(cmd: list[str]) -> None:
    print(f"[pipeline] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_joblib(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def _load_target_artifacts(out_dir: Path, target: str) -> dict[str, Any]:
    report_path = out_dir / f"poly_move_model_report_{target}.json"
    model_path = out_dir / f"poly_move_model_{target}.joblib"
    scaler_path = out_dir / f"poly_move_scaler_{target}.joblib"
    imputer_path = out_dir / f"poly_move_imputer_{target}.joblib"
    report = _load_json(report_path)
    if not report:
        raise FileNotFoundError(f"Missing/invalid report: {report_path}")
    return {
        "target": target,
        "model": _load_joblib(model_path),
        "scaler": _load_joblib(scaler_path),
        "imputer": _load_joblib(imputer_path),
        "report": report,
        "feature_cols": list(report.get("feature_cols", [])),
        "artifact_paths": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "imputer": str(imputer_path),
            "report": str(report_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dataset + train UP/DOWN poly move models from SQLite in one run."
    )
    ap.add_argument(
        "--db",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "mm_rl_log.sqlite"),
        help="Path to mm_rl_log.sqlite",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="Output directory for dataset and trained artifacts",
    )
    ap.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Optional dataset CSV path. Default: <out-dir>/poly_move_dataset_nexttick2s.csv",
    )
    ap.add_argument(
        "--next-tick-window-sec",
        type=float,
        default=2.0,
        help="Window for next-tick-within-window labels (default 2.0s; use 1.0 for classifier).",
    )
    ap.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task: regression (delta) or classification (P(adverse move >= 1c in window)).",
    )
    ap.add_argument(
        "--vol-half-life-ticks",
        type=int,
        default=50,
        help="EMA half-life in Binance moves for binance_vol_ema",
    )
    ap.add_argument(
        "--include-zero-moves",
        action="store_true",
        help="Include zero Binance moves. Default is non-zero moves only.",
    )
    ap.add_argument(
        "--min-secs-left",
        type=float,
        default=0.0,
        help="Minimum secs_left to include in dataset",
    )
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "linear", "ridge", "tree"],
        help="Model family selection passed to train_poly_move_model.py",
    )
    ap.add_argument(
        "--ridge-alphas",
        type=str,
        default="0.1,1.0,10.0,25.0",
        help="Ridge alpha grid passed to train_poly_move_model.py",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Export predicted-vs-actual plots for each target.",
    )
    ap.add_argument(
        "--bundle-file",
        type=str,
        default="poly_move_bundle.joblib",
        help="Bundle filename (or absolute path) for single deploy artifact.",
    )
    ap.add_argument(
        "--add-engineered-features",
        action="store_true",
        help="Add engineered columns when building dataset (binance_move_secs_left, poly_spread, secs_left_inv).",
    )
    ap.add_argument(
        "--feature-set",
        type=str,
        default="auto",
        choices=["baseline", "engineered", "auto"],
        help="Feature set for training: baseline, engineered, or auto (default).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    build_script = script_dir / "build_poly_move_dataset.py"
    train_script = script_dir / "train_poly_move_model.py"
    if not build_script.is_file():
        raise SystemExit(f"Missing script: {build_script}")
    if not train_script.is_file():
        raise SystemExit(f"Missing script: {train_script}")

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    window_sec = float(args.next_tick_window_sec)
    default_dataset_name = "poly_move_dataset_nexttick1s.csv" if args.task == "classification" else "poly_move_dataset_nexttick2s.csv"
    dataset_path = (
        Path(args.dataset_path).expanduser().resolve()
        if args.dataset_path
        else out_dir / default_dataset_name
    )
    if args.task == "classification" and not args.dataset_path:
        window_sec = 1.0

    # 1) Build dataset from SQLite
    build_cmd = [
        sys.executable,
        str(build_script),
        "--db",
        str(db_path),
        "--out",
        str(dataset_path),
        "--next-tick-window-sec",
        str(window_sec),
        "--vol-half-life-ticks",
        str(int(args.vol_half_life_ticks)),
        "--min-secs-left",
        str(float(args.min_secs_left)),
    ]
    if args.include_zero_moves:
        build_cmd.append("--include-zero-moves")
    if args.add_engineered_features:
        build_cmd.append("--add-engineered-features")
    _run(build_cmd)

    # 2) Train both targets
    train_common = [
        sys.executable,
        str(train_script),
        "--dataset",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--train-frac",
        str(float(args.train_frac)),
        "--val-frac",
        str(float(args.val_frac)),
        "--model",
        str(args.model),
        "--ridge-alphas",
        str(args.ridge_alphas),
        "--train-all-markets",
        "--feature-set",
        str(args.feature_set),
        "--task",
        str(args.task),
    ]
    if args.plot:
        train_common.append("--plot")

    _run(train_common + ["--target", "delta_down_bid"])
    _run(train_common + ["--target", "delta_up_bid"])

    # 3) Build single deploy bundle
    down_target = "delta_down_bid"
    up_target = "delta_up_bid"
    down_art = _load_target_artifacts(out_dir, down_target)
    up_art = _load_target_artifacts(out_dir, up_target)

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "source_db": str(db_path),
            "dataset_path": str(dataset_path),
            "out_dir": str(out_dir),
            "train_scope": "train_all_markets",
            "next_tick_window_sec": window_sec,
            "task": str(args.task),
            "vol_half_life_ticks": int(args.vol_half_life_ticks),
            "model_family": str(args.model),
            "ridge_alphas": str(args.ridge_alphas),
        },
        "up": up_art,
        "down": down_art,
    }
    bundle_path = Path(args.bundle_file).expanduser()
    if not bundle_path.is_absolute():
        bundle_path = (out_dir / bundle_path).resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(f"[pipeline] wrote single deploy bundle: {bundle_path}")

    # 4) Emit pipeline summary
    down_report_path = out_dir / "poly_move_model_report_delta_down_bid.json"
    up_report_path = out_dir / "poly_move_model_report_delta_up_bid.json"
    down_report = _load_json(down_report_path)
    up_report = _load_json(up_report_path)
    summary = {
        "db": str(db_path),
        "dataset": str(dataset_path),
        "out_dir": str(out_dir),
        "bundle_file": str(bundle_path),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "task": str(args.task),
        "targets": {
            "delta_down_bid": {
                "report_path": str(down_report_path),
                "rmse_test": down_report.get("rmse_test"),
                "mae_test": down_report.get("mae_test"),
                "rmse_train": down_report.get("rmse_train"),
                "mae_train": down_report.get("mae_train"),
                "brier_test": down_report.get("brier_test"),
                "roc_auc_test": down_report.get("roc_auc_test"),
                "model_selected": down_report.get("model_selected", down_report.get("model")),
            },
            "delta_up_bid": {
                "report_path": str(up_report_path),
                "rmse_test": up_report.get("rmse_test"),
                "mae_test": up_report.get("mae_test"),
                "rmse_train": up_report.get("rmse_train"),
                "mae_train": up_report.get("mae_train"),
                "brier_test": up_report.get("brier_test"),
                "roc_auc_test": up_report.get("roc_auc_test"),
                "model_selected": up_report.get("model_selected", up_report.get("model")),
            },
        },
    }
    summary_path = out_dir / "poly_move_pipeline_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pipeline] done. summary: {summary_path}")


if __name__ == "__main__":
    main()
