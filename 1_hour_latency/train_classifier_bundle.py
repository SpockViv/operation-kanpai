#!/usr/bin/env python3
"""
Train full-data classifier bundle on mm7 + mm3_8 and export one deploy artifact.

Train scope:
- mm_rl_log_7: all markets
- mm_rl_log_3_8: all markets

Output:
- Single bundle joblib containing UP/DOWN models + scaler + imputer + reports.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from build_poly_move_dataset import _connect_ro, build_dataset
from train_poly_move_pipeline import BUNDLE_SCHEMA_VERSION

WINDOW_SEC = 1.0
VOL_HALF_LIFE = 50


def _find_first(paths: list[Path]) -> Path:
    for p in paths:
        if p.is_file():
            return p
    raise FileNotFoundError(f"None of these paths exist: {[str(p) for p in paths]}")


def _run(cmd: list[str]) -> None:
    print(f"[full_bundle] running: {' '.join(cmd)}")
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
    return json.loads(path.read_text(encoding="utf-8"))


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
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "imputer": joblib.load(imputer_path),
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
    ap = argparse.ArgumentParser(description="Train full-data classifier bundle (mm7 + mm3_8).")
    ap.add_argument("--out-dir", type=str, default=str(_script_dir / "out"))
    ap.add_argument(
        "--bundle-out",
        type=str,
        default=str(_root / "src" / "poly_move_bundle.joblib"),
        help="Absolute or relative output path for single deploy bundle.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_out = Path(args.bundle_out).expanduser()
    if not bundle_out.is_absolute():
        bundle_out = (_root / bundle_out).resolve()
    bundle_out.parent.mkdir(parents=True, exist_ok=True)

    db_7 = _find_first(
        [
            _root / "mm_rl_log_7_recovered.sqlite",
            _root / "mm_rl_log_7.sqlite",
            _root / "training_data" / "mm_rl_log_7_recovered.sqlite",
            _root / "training_data" / "mm_rl_log_7.sqlite",
        ]
    )
    db_38 = _find_first(
        [
            _root / "mm_rl_log_3_8_recovered.sqlite",
            _root / "training_data" / "mm_rl_log_3_8_recovered.sqlite",
        ]
    )
    print(f"[full_bundle] mm7 db:  {db_7}")
    print(f"[full_bundle] mm38 db: {db_38}")

    conn7 = _connect_ro(db_7)
    try:
        df_7 = build_dataset(
            conn7,
            next_tick_window_sec=WINDOW_SEC,
            vol_half_life_ticks=VOL_HALF_LIFE,
            only_nonzero_moves=True,
        )
    finally:
        conn7.close()

    conn38 = _connect_ro(db_38)
    try:
        df_38 = build_dataset(
            conn38,
            next_tick_window_sec=WINDOW_SEC,
            vol_half_life_ticks=VOL_HALF_LIFE,
            only_nonzero_moves=True,
        )
    finally:
        conn38.close()

    if df_7.empty or df_38.empty:
        raise SystemExit("One or both datasets are empty; cannot train full-data bundle.")

    for df in (df_7, df_38):
        df["binance_move_secs_left"] = df["binance_move"] * df["secs_left"]
        df["poly_spread"] = df["poly_down_bid"] - df["poly_up_bid"].fillna(df["poly_down_bid"])
        df["secs_left_inv"] = 1.0 / (1.0 + df["secs_left"].clip(lower=0))

    full_df = pd.concat([df_7, df_38], ignore_index=True)
    dataset_csv = out_dir / "poly_move_dataset_mm7_mm38_full_classifier.csv"
    full_df.to_csv(dataset_csv, index=False)
    print(f"[full_bundle] wrote dataset: {dataset_csv} ({len(full_df)} rows)")

    train_script = _script_dir / "train_poly_move_model.py"
    common = [
        sys.executable,
        str(train_script),
        "--dataset",
        str(dataset_csv),
        "--out-dir",
        str(out_dir),
        "--task",
        "classification",
        "--model",
        "auto",
        "--feature-set",
        "auto",
        "--train-all-markets",
    ]
    _run(common + ["--target", "delta_down_bid"])
    _run(common + ["--target", "delta_up_bid"])

    down_art = _load_target_artifacts(out_dir, "delta_down_bid")
    up_art = _load_target_artifacts(out_dir, "delta_up_bid")
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "source_dbs": [str(db_7), str(db_38)],
            "dataset_path": str(dataset_csv),
            "task": "classification",
            "train_scope": "train_all_markets_full_mm7_plus_mm3_8",
            "next_tick_window_sec": WINDOW_SEC,
            "vol_half_life_ticks": VOL_HALF_LIFE,
        },
        "up": up_art,
        "down": down_art,
    }
    joblib.dump(bundle, bundle_out)
    print(f"[full_bundle] wrote bundle: {bundle_out}")


if __name__ == "__main__":
    main()
