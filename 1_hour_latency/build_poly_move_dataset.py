#!/usr/bin/env python3
"""
build_poly_move_dataset.py
--------------------------
Build a dataset from mm_rl_log.sqlite for predicting Polymarket bid change
after Binance moves.

Anchor:
- Binance tick-to-tick moves (all non-zero moves by default).

State at anchor time T:
- binance_move          : current move (mid_t - mid_{t-1})
- binance_vol_ema       : EMA of absolute Binance moves
- poly_down_bid         : latest Poly DOWN bid at or before T
- poly_up_bid           : latest Poly UP bid at or before T
- secs_left             : seconds until market roll

Target definition (next-tick-within-window):
- Look for the first Polymarket bid change within (T, T+W].
- If a changed bid is observed, use that changed bid as target.
- If no changed bid is observed by T+W, assume unchanged (delta = 0).

This matches sampled-tick logging where a logged tick is not guaranteed
to be a price change event.

Safety: opens SQLite read-only (mode=ro). Never writes to DB.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except Exception:
        return sqlite3.connect(str(db_path))


def _load_markets(conn: sqlite3.Connection) -> list[tuple[int, str, int, int]]:
    """Return (market_id, slug, created_epoch_ns, roll_epoch_ns) ordered by created_epoch_ns ASC."""
    rows = conn.execute(
        "SELECT market_id, COALESCE(slug,''), COALESCE(created_epoch_ns,0), COALESCE(roll_iso,'') "
        "FROM markets ORDER BY created_epoch_ns ASC"
    ).fetchall()
    out: list[tuple[int, str, int, int]] = []
    for mid, slug, created_ns, roll_iso in rows:
        try:
            roll_epoch_ns = int(pd.Timestamp(str(roll_iso)).tz_convert("UTC").value)
        except Exception:
            # If roll time is malformed, skip market.
            continue
        out.append((int(mid), str(slug), int(created_ns), roll_epoch_ns))
    return out


def _load_binance_ticks_for_market(conn: sqlite3.Connection, market_id: int) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT epoch_ns, mid FROM ticks "
        "WHERE market_id = ? AND source = 'binance' AND symbol = 'btcusdt' "
        "AND epoch_ns IS NOT NULL AND mid IS NOT NULL "
        "ORDER BY epoch_ns ASC",
        (market_id,),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["epoch_ns", "mid"])
    return pd.DataFrame(rows, columns=["epoch_ns", "mid"]).astype(
        {"epoch_ns": "int64", "mid": "float64"}
    )


def _load_poly_ticks_for_market(conn: sqlite3.Connection, market_id: int, symbol: str) -> pd.DataFrame:
    """Load Poly ticks (epoch_ns, bid) for one market and symbol, ordered by epoch_ns."""
    rows = conn.execute(
        "SELECT epoch_ns, bid FROM ticks "
        "WHERE market_id = ? AND source = 'poly' AND symbol = ? "
        "AND epoch_ns IS NOT NULL AND bid IS NOT NULL "
        "ORDER BY epoch_ns ASC",
        (market_id, symbol),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["epoch_ns", "bid"])
    return pd.DataFrame(rows, columns=["epoch_ns", "bid"]).astype({"epoch_ns": "int64", "bid": "float64"})


def _bid_at_or_before_vec(tick_times: np.ndarray, tick_bids: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Vectorized latest bid at/before each t. Returns np.nan when unavailable."""
    idx = np.searchsorted(tick_times, t_arr, side="right") - 1
    out = np.full(len(t_arr), np.nan, dtype=np.float64)
    good = idx >= 0
    out[good] = tick_bids[idx[good]]
    return out


def _next_change_within_window_vec(
    tick_times: np.ndarray,
    tick_bids: np.ndarray,
    t_arr: np.ndarray,
    window_ns: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each anchor T in t_arr:
      - current bid = latest bid at/before T
      - target bid  = first changed bid within (T, T+window_ns]
      - if no changed bid in window, target bid = current bid (unchanged)

    Returns (target_bid, target_epoch_ns, reason_code):
      reason_code:
        1 -> changed_within_window
        2 -> no_change_within_window (ticks seen, no bid change)
        3 -> no_tick_within_window
    """
    out_target_bid = np.full(len(t_arr), np.nan, dtype=np.float64)
    out_target_t = np.full(len(t_arr), -1, dtype=np.int64)
    out_reason = np.zeros(len(t_arr), dtype=np.int8)

    if len(tick_times) == 0:
        return out_target_bid, out_target_t, out_reason

    idx_cur = np.searchsorted(tick_times, t_arr, side="right") - 1
    idx_next = np.searchsorted(tick_times, t_arr, side="right")

    for i in range(len(t_arr)):
        c_idx = int(idx_cur[i])
        if c_idx < 0:
            continue
        cur_bid = float(tick_bids[c_idx])
        limit = int(t_arr[i]) + int(window_ns)

        j = int(idx_next[i])
        saw_tick = False
        found_change = False
        while j < len(tick_times) and int(tick_times[j]) <= limit:
            saw_tick = True
            nxt_bid = float(tick_bids[j])
            if abs(nxt_bid - cur_bid) > eps:
                out_target_bid[i] = nxt_bid
                out_target_t[i] = int(tick_times[j])
                out_reason[i] = 1
                found_change = True
                break
            j += 1

        if not found_change:
            out_target_bid[i] = cur_bid
            out_target_t[i] = limit
            out_reason[i] = 2 if saw_tick else 3

    return out_target_bid, out_target_t, out_reason


def build_dataset(
    conn: sqlite3.Connection,
    next_tick_window_sec: float,
    vol_half_life_ticks: int = 50,
    only_nonzero_moves: bool = True,
    min_secs_left: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build dataset: one row per Binance move anchor with a next-tick-style label.
    """
    window_ns = int(float(next_tick_window_sec) * 1e9)
    min_sl = min_secs_left if min_secs_left is not None else 0.0

    markets = _load_markets(conn)
    if not markets:
        return pd.DataFrame()

    # EMA alpha for absolute move volatility proxy
    alpha = 2.0 / (float(max(1, int(vol_half_life_ticks))) + 1.0)
    rows_out = []
    for market_id, _slug, _created, roll_epoch_ns in markets:
        ticks_bin = _load_binance_ticks_for_market(conn, market_id)
        ticks_up = _load_poly_ticks_for_market(conn, market_id, "UP")
        ticks_dn = _load_poly_ticks_for_market(conn, market_id, "DOWN")
        if ticks_bin.empty or ticks_dn.empty:
            continue

        bin_t = ticks_bin["epoch_ns"].to_numpy(dtype=np.int64)
        bin_m = ticks_bin["mid"].to_numpy(dtype=np.float64)
        if len(bin_t) < 2:
            continue

        # Move-anchored samples use t_i (i>=1), move_i = mid_i - mid_{i-1}
        t_anchor = bin_t[1:]
        bin_move = bin_m[1:] - bin_m[:-1]
        abs_move = np.abs(bin_move)

        # EMA(abs(move)) across Binance move sequence
        vol_ema = np.empty_like(abs_move)
        ema = float(abs_move[0])
        vol_ema[0] = ema
        for i in range(1, len(abs_move)):
            ema = alpha * float(abs_move[i]) + (1.0 - alpha) * ema
            vol_ema[i] = ema

        if only_nonzero_moves:
            mask = np.abs(bin_move) > 0.0
            t_anchor = t_anchor[mask]
            bin_move = bin_move[mask]
            vol_ema = vol_ema[mask]

        if len(t_anchor) == 0:
            continue

        secs_left = (float(roll_epoch_ns) - t_anchor.astype(np.float64)) / 1e9
        mask_sl = secs_left >= float(min_sl)
        t_anchor = t_anchor[mask_sl]
        bin_move = bin_move[mask_sl]
        vol_ema = vol_ema[mask_sl]
        secs_left = secs_left[mask_sl]
        if len(t_anchor) == 0:
            continue

        dn_t = ticks_dn["epoch_ns"].to_numpy(dtype=np.int64)
        dn_b = ticks_dn["bid"].to_numpy(dtype=np.float64)
        up_t = ticks_up["epoch_ns"].to_numpy(dtype=np.int64) if not ticks_up.empty else np.array([], dtype=np.int64)
        up_b = ticks_up["bid"].to_numpy(dtype=np.float64) if not ticks_up.empty else np.array([], dtype=np.float64)

        down_bid_t = _bid_at_or_before_vec(dn_t, dn_b, t_anchor)
        down_bid_f, down_target_t, down_reason = _next_change_within_window_vec(
            dn_t, dn_b, t_anchor, window_ns
        )
        up_bid_t = _bid_at_or_before_vec(up_t, up_b, t_anchor) if len(up_t) else np.full(len(t_anchor), np.nan)
        if len(up_t):
            up_bid_f, up_target_t, up_reason = _next_change_within_window_vec(
                up_t, up_b, t_anchor, window_ns
            )
        else:
            up_bid_f = np.full(len(t_anchor), np.nan)
            up_target_t = np.full(len(t_anchor), -1, dtype=np.int64)
            up_reason = np.zeros(len(t_anchor), dtype=np.int8)

        valid = np.isfinite(down_bid_t) & np.isfinite(down_bid_f) & (down_target_t > 0)
        if not np.any(valid):
            continue

        t_anchor = t_anchor[valid]
        bin_move = bin_move[valid]
        vol_ema = vol_ema[valid]
        secs_left = secs_left[valid]
        down_bid_t = down_bid_t[valid]
        down_bid_f = down_bid_f[valid]
        down_target_t = down_target_t[valid]
        down_reason = down_reason[valid]
        up_bid_t = up_bid_t[valid]
        up_bid_f = up_bid_f[valid]
        up_target_t = up_target_t[valid]
        up_reason = up_reason[valid]

        delta_dn = down_bid_f - down_bid_t
        delta_up = np.where(np.isfinite(up_bid_t) & np.isfinite(up_bid_f), up_bid_f - up_bid_t, np.nan)
        down_delay_sec = (down_target_t.astype(np.float64) - t_anchor.astype(np.float64)) / 1e9
        up_delay_sec = np.where(up_target_t > 0, (up_target_t.astype(np.float64) - t_anchor.astype(np.float64)) / 1e9, np.nan)

        reason_map = {
            1: "changed_within_window",
            2: "no_change_within_window",
            3: "no_tick_within_window",
        }

        for i in range(len(t_anchor)):
            rows_out.append(
                {
                    "step_id": int(i + 1),
                    "epoch_ns": int(t_anchor[i]),
                    "market_id": int(market_id),
                    "binance_move": float(bin_move[i]),
                    "binance_vol_ema": float(vol_ema[i]),
                    "poly_down_bid": float(down_bid_t[i]),
                    "poly_up_bid": float(up_bid_t[i]) if np.isfinite(up_bid_t[i]) else None,
                    "secs_left": float(secs_left[i]),
                    "delta_down_bid": float(delta_dn[i]),
                    "delta_up_bid": float(delta_up[i]) if np.isfinite(delta_up[i]) else None,
                    "target_epoch_ns_down": int(down_target_t[i]),
                    "target_delay_sec_down": float(down_delay_sec[i]),
                    "label_reason_down": reason_map.get(int(down_reason[i]), "unknown"),
                    "target_epoch_ns_up": int(up_target_t[i]) if int(up_target_t[i]) > 0 else None,
                    "target_delay_sec_up": float(up_delay_sec[i]) if np.isfinite(up_delay_sec[i]) else None,
                    "label_reason_up": reason_map.get(int(up_reason[i]), "unknown") if int(up_reason[i]) else None,
                    "next_tick_window_sec": float(next_tick_window_sec),
                }
            )

    return pd.DataFrame(rows_out)


def main() -> None:
    _script_dir = Path(__file__).resolve().parent
    _root = _script_dir.parent
    ap = argparse.ArgumentParser(
        description="Build poly move dataset from mm_rl_log.sqlite (read-only)."
    )
    ap.add_argument(
        "--db",
        type=str,
        default=str(_root / "mm_rl_log.sqlite"),
        help="Path to SQLite DB (default: repo root mm_rl_log.sqlite)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_script_dir / "out" / "poly_move_dataset.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--next-tick-window-sec",
        type=float,
        default=2.0,
        help="Window (seconds) to look for the next Poly bid change (default 2.0).",
    )
    ap.add_argument(
        "--vol-half-life-ticks",
        type=int,
        default=50,
        help="Half-life (in Binance moves) for binance_vol_ema (default 50)",
    )
    ap.add_argument(
        "--include-zero-moves",
        action="store_true",
        help="Include zero Binance moves. Default is to keep only non-zero moves.",
    )
    ap.add_argument(
        "--min-secs-left",
        type=float,
        default=None,
        help="Minimum secs_left to include (default 0.0 for next-tick labels)",
    )
    ap.add_argument(
        "--add-engineered-features",
        action="store_true",
        help="Add optional engineered columns: binance_move_secs_left, poly_spread, secs_left_inv.",
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect_ro(db_path)
    try:
        df = build_dataset(
            conn,
            next_tick_window_sec=args.next_tick_window_sec,
            vol_half_life_ticks=args.vol_half_life_ticks,
            only_nonzero_moves=(not args.include_zero_moves),
            min_secs_left=args.min_secs_left,
        )
    finally:
        conn.close()

    if df.empty:
        print("No rows produced; check DB has steps and Poly ticks (source=poly, symbol=UP/DOWN).")
        return

    if args.add_engineered_features:
        df["binance_move_secs_left"] = df["binance_move"] * df["secs_left"]
        df["poly_spread"] = df["poly_down_bid"] - df["poly_up_bid"].fillna(df["poly_down_bid"])
        df["secs_left_inv"] = 1.0 / (1.0 + df["secs_left"].clip(lower=0))

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
