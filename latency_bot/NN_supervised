#!/usr/bin/env python3
"""
REG_1_hour.py — Offline regression trainer & backtester for the 1-hour latency trader.

WHY REGRESSION (instead of RL):
- You already have a "counterfactual" label: pnl_1s_{yes,no} (1-second markout per share).
- That makes this a supervised learning problem:
    predict expected markout given state features, then trade only when predicted edge > cost.

THIS VERSION:
- Train/Test split is by market interval (slug) in time order:
    * train = first half of slugs
    * test  = second half of slugs
  (ensures we move market-to-market as your interval unit)
- Predicts expected 1s PnL (DOLLARS) for ENTER decision for the chosen side.
- Decision rule:
    enter if predicted_pnl_usd >= min_pred_pnl_usd
  (you can sweep this threshold to control trade count)
- Fill assumed at ASK WE SEE (no +0.01).
- ZERO inventory penalization in the objective.
  (Inventory is tracked optionally for state consistency, but not penalized.)

Run examples:
  python REG_1_hour.py --db trader_log.snapshot.db --epochs 25 --device cpu
  python REG_1_hour.py --db trader_log.snapshot.db --epochs 25 --min-pred-pnl-usd 0.01
  python REG_1_hour.py --db trader_log.snapshot.db --epochs 50 --huber-delta 1.0 --lr 1e-3
  python REG_1_hour.py --db trader_log.snapshot.db --sweep-thresholds
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterator, Tuple

import numpy as np

# ---- torch ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise SystemExit("ERROR: PyTorch is required. Install with: pip install torch") from e

# ---- matplotlib (optional) ----
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    HAS_MPL = False

EPS = 1e-9


# ============================================================
# Utils
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def softsign(x: float) -> float:
    return x / (1.0 + abs(x))

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


# ============================================================
# Rows / Data loading
# ============================================================

@dataclass
class Row:
    opp_id: int
    t_ms: int
    slug: str

    bin_mid: Optional[float]
    fair_yes: Optional[float]

    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    yes_mid: Optional[float]
    no_mid: Optional[float]
    spread_yes_c: Optional[float]
    spread_no_c: Optional[float]

    dp_fair_cents: Optional[float]
    dP_yes_cents: Optional[float]
    dP_no_cents: Optional[float]

    edge_yes_pp: Optional[float]
    edge_no_pp: Optional[float]

    pnl_1s_yes: Optional[float]   # per-share PnL label
    pnl_1s_no: Optional[float]    # per-share PnL label
    best_pnl_1s: Optional[float]

    passed_filter: Optional[int]


def load_rows(db_path: Path) -> list[Row]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    q = """
    WITH latest_outcomes AS (
      SELECT opp_id, MAX(label_t_ms) AS max_label_t_ms
      FROM outcomes
      GROUP BY opp_id
    ),
    u AS (
      SELECT o2.*
      FROM outcomes o2
      JOIN latest_outcomes lo
        ON lo.opp_id = o2.opp_id
       AND lo.max_label_t_ms = o2.label_t_ms
    )
    SELECT
      o.opp_id,
      o.t_ms,
      o.slug,
      o.bin_mid,
      o.fair_yes,
      o.yes_bid, o.yes_ask, o.no_bid, o.no_ask,
      o.yes_mid, o.no_mid,
      o.spread_yes_c, o.spread_no_c,
      o.dp_fair_cents,
      o.dP_yes_cents,
      o.dP_no_cents,
      o.edge_yes_pp,
      o.edge_no_pp,
      o.passed_filter,
      u.pnl_1s_yes,
      u.pnl_1s_no,
      u.best_pnl_1s
    FROM opportunities o
    LEFT JOIN u
      ON u.opp_id = o.opp_id
    ORDER BY o.t_ms ASC;
    """

    rows: list[Row] = []
    for r in cur.execute(q):
        rows.append(
            Row(
                opp_id=int(r["opp_id"]),
                t_ms=int(r["t_ms"]),
                slug=str(r["slug"] or ""),

                bin_mid=safe_float(r["bin_mid"]),
                fair_yes=safe_float(r["fair_yes"]),

                yes_bid=safe_float(r["yes_bid"]),
                yes_ask=safe_float(r["yes_ask"]),
                no_bid=safe_float(r["no_bid"]),
                no_ask=safe_float(r["no_ask"]),
                yes_mid=safe_float(r["yes_mid"]),
                no_mid=safe_float(r["no_mid"]),
                spread_yes_c=safe_float(r["spread_yes_c"]),
                spread_no_c=safe_float(r["spread_no_c"]),

                dp_fair_cents=safe_float(r["dp_fair_cents"]),
                dP_yes_cents=safe_float(r["dP_yes_cents"]),
                dP_no_cents=safe_float(r["dP_no_cents"]),

                edge_yes_pp=safe_float(r["edge_yes_pp"]),
                edge_no_pp=safe_float(r["edge_no_pp"]),

                passed_filter=int(r["passed_filter"]) if r["passed_filter"] is not None else None,

                pnl_1s_yes=safe_float(r["pnl_1s_yes"]),
                pnl_1s_no=safe_float(r["pnl_1s_no"]),
                best_pnl_1s=safe_float(r["best_pnl_1s"]),
            )
        )

    conn.close()
    return rows


def load_exit_realized_pnl(db_path: Path) -> dict[int, float]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
      SELECT opp_id, realized_pnl
      FROM exits
      WHERE opp_id IS NOT NULL
    """)
    out: dict[int, float] = {}
    for opp_id, rpnl in cur.fetchall():
        if opp_id is None:
            continue
        try:
            out[int(opp_id)] = out.get(int(opp_id), 0.0) + float(rpnl or 0.0)
        except Exception:
            continue
    conn.close()
    return out


# ============================================================
# Train/test split by slug (market interval)
# ============================================================

def split_by_slug_first_half(rows: list[Row]) -> tuple[list[Row], list[Row], list[str], list[str]]:
    """Train on 2nd & 3rd hour (slugs index 1-2), test on next 5 full hours (slugs index 3-7)."""
    seen = set()
    ordered_slugs: list[str] = []
    for r in rows:
        s = r.slug
        if s and s not in seen:
            seen.add(s)
            ordered_slugs.append(s)

    if not ordered_slugs:
        return [], [], [], []

    # Train = 2nd hour only (index 1: 9am — the only market that finished down)
    # Test  = next 5 full markets (indices 3 through 7)
    train_slugs = ordered_slugs[1:6]   # 9am only
    test_slugs = ordered_slugs[7:8]    # 11am, 12pm, 1pm, 2pm, 3pm

    train_set = set(train_slugs)
    test_set = set(test_slugs)

    train_rows = [r for r in rows if r.slug in train_set]
    test_rows = [r for r in rows if r.slug in test_set]

    return train_rows, test_rows, train_slugs, test_slugs


# ============================================================
# Trading helpers
# ============================================================

def best_side_for_row(row: Row) -> Optional[str]:
    ey = row.edge_yes_pp
    en = row.edge_no_pp
    if ey is not None and en is not None:
        return "YES" if ey >= en else "NO"
    if row.fair_yes is not None:
        return "YES" if row.fair_yes >= 0.5 else "NO"
    return None

def ask_for_side(row: Row, side: str) -> Optional[float]:
    base_ask = row.yes_ask if side == "YES" else row.no_ask
    if base_ask is None:
        return None
    return float(base_ask)

def mid_for_side(row: Row, side: str) -> Optional[float]:
    return row.yes_mid if side == "YES" else row.no_mid

def spread_for_side_cents(row: Row, side: str) -> Optional[float]:
    return row.spread_yes_c if side == "YES" else row.spread_no_c

def edge_for_side_pp(row: Row, side: str) -> Optional[float]:
    return row.edge_yes_pp if side == "YES" else row.edge_no_pp

def pnl_per_share_for_side(row: Row, side: str) -> Optional[float]:
    return row.pnl_1s_yes if side == "YES" else row.pnl_1s_no

def qty_from_notional(notional: float, ask: float) -> float:
    ask = max(EPS, float(ask))
    return float(notional) / ask


# ============================================================
# Filters
# ============================================================

def row_is_usable(row: Row, args: argparse.Namespace) -> bool:
    # Need outcomes (at least one label)
    if (row.pnl_1s_yes is None) and (row.pnl_1s_no is None) and (row.best_pnl_1s is None):
        return False

    # Need core features
    if row.fair_yes is None or row.bin_mid is None:
        return False
    if row.yes_mid is None or row.no_mid is None:
        return False
    if row.yes_ask is None or row.no_ask is None:
        return False
    if row.edge_yes_pp is None or row.edge_no_pp is None:
        return False

    # Require some fair movement
    df = abs(float(row.dp_fair_cents)) if row.dp_fair_cents is not None else 0.0
    if df < float(args.fair_move_min_cents):
        return False

    # Reject if Poly fully moved on BOTH sides
    dP_yes = abs(float(row.dP_yes_cents)) if row.dP_yes_cents is not None else 999.0
    dP_no = abs(float(row.dP_no_cents)) if row.dP_no_cents is not None else 999.0
    if dP_yes > float(args.poly_both_moved_cents) and dP_no > float(args.poly_both_moved_cents):
        return False

    # Optional extra cap (reject only if BOTH sides exceed)
    if float(args.max_poly_move_cents) > 0:
        if dP_yes > float(args.max_poly_move_cents) and dP_no > float(args.max_poly_move_cents):
            return False

    return True


# ============================================================
# Feature engineering
# ============================================================

def build_state_from_row(
    *,
    row: Row,
    side: str,
    edge_pp: float,
    spread_c: float,
    dP_cents: float,
    dF_cents: float,
    dS: float,
    poly_reaction: float,
    inv_yes: float,
    inv_no: float,
    args: argparse.Namespace,
) -> np.ndarray:
    mid = mid_for_side(row, side)
    if mid is None or row.fair_yes is None:
        mid = 0.5
        fair_yes = 0.5
    else:
        fair_yes = float(row.fair_yes)

    inv_net = float(inv_yes - inv_no)
    inv_gross = float(inv_yes + inv_no)

    cap_net = max(1.0, float(args.max_net))
    cap_yes = max(1.0, float(args.max_yes))
    cap_no = max(1.0, float(args.max_no))

    inv_side = inv_yes if side == "YES" else inv_no
    inv_other = inv_no if side == "YES" else inv_yes

    inv_scaled = float(inv_side) / (cap_yes if side == "YES" else cap_no)
    inv_other_scaled = float(inv_other) / (cap_no if side == "YES" else cap_yes)
    inv_net_scaled = float(inv_net) / cap_net
    inv_gross_scaled = float(inv_gross) / max(1.0, cap_yes + cap_no)

    svec = np.asarray(
        [
            float(mid),                              # 0  poly mid
            float(fair_yes),                         # 1  fair YES
            float(edge_pp) / 10.0,                   # 2  edge in 10pp units
            float(spread_c) / 10.0,                  # 3  spread in 10c units
            softsign(float(dP_cents) / 10.0),        # 4  poly delta
            softsign(float(dF_cents) / 10.0),        # 5  fair delta
            softsign(float(dS) / max(EPS, float(args.bin_dS_scale))),  # 6 bin delta
            softsign(float(poly_reaction)),          # 7 reaction ratio |dP|/|dF|
            float(inv_scaled),                       # 8 side inv
            float(inv_other_scaled),                 # 9 other inv
            float(inv_net_scaled),                   # 10 net inv
            float(inv_gross_scaled),                 # 11 gross inv
        ],
        dtype=np.float32,
    )
    return svec


# ============================================================
# Supervised dataset builder
# ============================================================

@dataclass
class Sample:
    s: np.ndarray   # features
    y: float        # label = expected pnl in USD if we ENTER now (for chosen side)


def iter_samples(rows: list[Row], *, args: argparse.Namespace) -> Iterator[Sample]:
    usable = [r for r in rows if row_is_usable(r, args)]
    if args.max_usables is not None and args.max_usables > 0:
        usable = usable[: int(args.max_usables)]
    if len(usable) < 2:
        return

    inv_yes = 0.0
    inv_no = 0.0

    prev_bin = None
    prev_yes_mid = None
    prev_no_mid = None
    prev_fair = None
    prev_slug = usable[0].slug

    for i in range(len(usable)):
        row = usable[i]

        # Reset at new market interval
        if row.slug != prev_slug:
            inv_yes = 0.0
            inv_no = 0.0
            prev_bin = None
            prev_yes_mid = None
            prev_no_mid = None
            prev_fair = None
            prev_slug = row.slug

        side = best_side_for_row(row)
        if side is None:
            continue

        # deltas (within slug)
        dS = 0.0 if prev_bin is None else (float(row.bin_mid) - float(prev_bin))
        dP_yes = 0.0 if prev_yes_mid is None else (float(row.yes_mid) - float(prev_yes_mid)) * 100.0
        dP_no = 0.0 if prev_no_mid is None else (float(row.no_mid) - float(prev_no_mid)) * 100.0
        dF = 0.0 if prev_fair is None else (float(row.fair_yes) - float(prev_fair)) * 100.0

        edge_pp = float(edge_for_side_pp(row, side) or 0.0)
        spread_c = float(spread_for_side_cents(row, side) or 0.0)
        dP_cents = dP_yes if side == "YES" else dP_no
        poly_reaction = abs(float(dP_cents)) / (abs(float(dF)) + 1e-6)

        s = build_state_from_row(
            row=row, side=side,
            edge_pp=edge_pp, spread_c=spread_c,
            dP_cents=dP_cents, dF_cents=dF, dS=dS,
            poly_reaction=poly_reaction,
            inv_yes=inv_yes, inv_no=inv_no,
            args=args
        )

        # Label: did Polymarket move UP on the next tick for our side?
        #   +1 if poly mid moved favourably (price went up for our side)
        #   -1 if poly mid moved against us
        #    0 if no change
        # We use pnl_1s (per-share markout) as the directional proxy.
        ask = ask_for_side(row, side)
        if ask is None or ask <= 0:
            continue

        pnl_ps = pnl_per_share_for_side(row, side)
        if pnl_ps is None:
            pnl_ps = row.best_pnl_1s
        if pnl_ps is None:
            continue

        if float(pnl_ps) > 0:
            y = 1.0
        elif float(pnl_ps) < 0:
            y = -1.0
        else:
            y = 0.0

        yield Sample(s=s, y=float(y))

        prev_bin = row.bin_mid
        prev_yes_mid = row.yes_mid
        prev_no_mid = row.no_mid
        prev_fair = row.fair_yes
        prev_slug = row.slug


def materialize_samples(rows: list[Row], *, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for sm in iter_samples(rows, args=args):
        xs.append(sm.s)
        ys.append(sm.y)
    if not xs:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return X, y


# ============================================================
# Regressor
# ============================================================

class RegNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield X[j], y[j]


# ============================================================
# Backtest using regressor (TEST HALF ONLY)
# ============================================================

@dataclass
class BacktestResult:
    timestamps: list[int]
    slugs: list[str]

    orig_pnl_per_step: list[float]
    orig_cum_pnl: list[float]
    orig_trade_count: int

    pred_pnl_per_step: list[float]
    traded_flag: list[int]
    traded_raw_pnl_per_step: list[float]
    traded_cum_pnl: list[float]
    trade_count: int


def backtest_regressor(
    rows: list[Row],
    model: RegNet,
    *,
    args: argparse.Namespace,
    exit_realized_pnl: dict[int, float],
) -> BacktestResult:
    model.eval()
    device = torch.device(args.device)

    usable = [r for r in rows if row_is_usable(r, args)]
    if args.max_usables is not None and args.max_usables > 0:
        usable = usable[: int(args.max_usables)]

    timestamps: list[int] = []
    slugs: list[str] = []

    orig_pnl: list[float] = []
    orig_cum: list[float] = []
    orig_trade_count = 0
    orig_c = 0.0

    preds: list[float] = []
    traded_flag: list[int] = []
    traded_pnl: list[float] = []
    traded_cum: list[float] = []
    trade_count = 0
    c = 0.0

    inv_yes = 0.0
    inv_no = 0.0
    prev_bin = None
    prev_yes_mid = None
    prev_no_mid = None
    prev_fair = None
    prev_slug = usable[0].slug if usable else ""

    for row in usable:
        if row.slug != prev_slug:
            inv_yes = 0.0
            inv_no = 0.0
            prev_bin = None
            prev_yes_mid = None
            prev_no_mid = None
            prev_fair = None
            prev_slug = row.slug

        side = best_side_for_row(row)
        if side is None:
            continue

        dS = 0.0 if prev_bin is None else (float(row.bin_mid) - float(prev_bin))
        dP_yes = 0.0 if prev_yes_mid is None else (float(row.yes_mid) - float(prev_yes_mid)) * 100.0
        dP_no = 0.0 if prev_no_mid is None else (float(row.no_mid) - float(prev_no_mid)) * 100.0
        dF = 0.0 if prev_fair is None else (float(row.fair_yes) - float(prev_fair)) * 100.0

        edge_pp = float(edge_for_side_pp(row, side) or 0.0)
        spread_c = float(spread_for_side_cents(row, side) or 0.0)
        dP_cents = dP_yes if side == "YES" else dP_no
        poly_reaction = abs(float(dP_cents)) / (abs(float(dF)) + 1e-6)

        s = build_state_from_row(
            row=row, side=side,
            edge_pp=edge_pp, spread_c=spread_c,
            dP_cents=dP_cents, dF_cents=dF, dS=dS,
            poly_reaction=poly_reaction,
            inv_yes=inv_yes, inv_no=inv_no,
            args=args
        )

        timestamps.append(row.t_ms)
        slugs.append(row.slug)

        # Baseline original realized (optional)
        op = float(exit_realized_pnl.get(row.opp_id, 0.0))
        orig_c += op
        orig_pnl.append(op)
        orig_cum.append(orig_c)
        if abs(op) > 0:
            orig_trade_count += 1

        with torch.no_grad():
            x = torch.from_numpy(s.reshape(1, -1)).to(device)
            pred = float(model(x).detach().cpu().item())
        preds.append(pred)

        do_trade = 1 if pred >= float(args.min_pred_pnl_usd) else 0
        traded_flag.append(do_trade)

        rawp = 0.0
        if do_trade:
            ask = ask_for_side(row, side)
            if ask is not None and ask > 0:
                qty = qty_from_notional(float(args.entry_notional), float(ask))
                pnl_ps = pnl_per_share_for_side(row, side) or row.best_pnl_1s
                if pnl_ps is not None:
                    rawp = float(pnl_ps) * float(qty) - float(args.entry_cross_cost_cents) / 100.0
                    trade_count += 1

                    # inventory dynamics (not penalized)
                    if side == "YES":
                        inv_yes += qty
                    else:
                        inv_no += qty

        c += rawp
        traded_pnl.append(rawp)
        traded_cum.append(c)

        prev_bin = row.bin_mid
        prev_yes_mid = row.yes_mid
        prev_no_mid = row.no_mid
        prev_fair = row.fair_yes
        prev_slug = row.slug

    return BacktestResult(
        timestamps=timestamps,
        slugs=slugs,
        orig_pnl_per_step=orig_pnl,
        orig_cum_pnl=orig_cum,
        orig_trade_count=orig_trade_count,
        pred_pnl_per_step=preds,
        traded_flag=traded_flag,
        traded_raw_pnl_per_step=traded_pnl,
        traded_cum_pnl=traded_cum,
        trade_count=trade_count,
    )


# ============================================================
# Plotting
# ============================================================

def plot_results(result: BacktestResult, out_path: str) -> None:
    if not HAS_MPL:
        print("matplotlib not available -- skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if result.timestamps:
        t0 = result.timestamps[0]
        hours = [(t - t0) / 3_600_000 for t in result.timestamps]
    else:
        hours = list(range(len(result.traded_cum_pnl)))

    ax.plot(hours, result.orig_cum_pnl, label=f"Original realized ({result.orig_trade_count} trades)", linewidth=2, alpha=0.85)
    ax.plot(hours, result.traded_cum_pnl, label=f"Regressor strategy ({result.trade_count} trades)", linewidth=2, alpha=0.85)
    ax.axhline(y=0, linestyle="--", alpha=0.5)
    ax.set_xlabel("Hours from test start")
    ax.set_ylabel("Cumulative P&L (USD-ish units)")
    ax.set_title("TEST HALF ONLY: Original (realized) vs Regressor Trading Strategy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {out_path}")


# ============================================================
# Threshold sweep (trade frequency knob)
# ============================================================

def sweep_thresholds(
    result: BacktestResult,
    thresholds: list[float],
) -> None:
    preds = np.asarray(result.pred_pnl_per_step, dtype=np.float64)
    realized = np.asarray(result.traded_raw_pnl_per_step, dtype=np.float64)  # NOTE: this is pnl under chosen threshold, not universal

    # For proper sweep, we need per-step "true pnl if traded" regardless of threshold.
    # We can reconstruct it by noting: traded_raw_pnl_per_step is already computed when do_trade,
    # else 0. But for sweep, we need the counterfactual pnl for every step.
    #
    # So we will re-use a trick: compute "true_pnl_if_trade" as:
    #   true = traded_raw_pnl_per_step + (1 - traded_flag)*0
    # This only works if the backtest was run with min_pred_pnl_usd <= -inf.
    #
    # Instead: we implement sweep properly inside main by re-running decisions.
    # Here we just provide a safety message if user calls this without proper prep.
    print("\n[WARN] sweep_thresholds() requires per-step true_pnl_if_trade to be computed for every step.")
    print("       Use --sweep-thresholds (implemented in main) to do a proper sweep.\n")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline regressor trainer & backtester (train first half slugs, test second half)"
    )

    # Data
    p.add_argument("--db", type=str, default="trader_log.snapshot.db", help="Path to SQLite DB")

    # Training
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)

    # Loss
    p.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"], help="Training loss")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta (only if --loss huber)")

    # Backtest / sizing
    p.add_argument("--entry-notional", type=float, default=5.0, help="Notional per trade (USD)")
    p.add_argument("--max-qty-per-entry", type=int, default=20, help="Max contracts per single entry")
    p.add_argument("--max-inventory-per-side", type=float, default=1000.0,
                   help="Max total contracts per side (YES or NO). Trades that would exceed this are skipped.")

    # Inventory caps (hard constraints only; not used in loss)
    p.add_argument("--max-yes", type=float, default=500.0)
    p.add_argument("--max-no", type=float, default=500.0)
    p.add_argument("--max-net", type=float, default=250.0)
    
    # Inventory penalty (for accumulation strategy)
    p.add_argument("--inv-penalty-net", type=float, default=0.0000002, help="Penalty per unit of net inventory imbalance (absolute value)")
    p.add_argument("--inv-penalty-scale", type=float, default=0.0001, help="Scale factor for inventory penalty (multiplies the penalty)")

    # Feature scaling
    p.add_argument("--bin-dS-scale", type=float, default=50.0, help="USD scale for bin dS softsign")

    # Friction / label engineering
    p.add_argument("--entry-cross-cost-cents", type=float, default=0.0, help="Crossing+fees proxy in cents (deducted from label)")
    p.add_argument("--label-clip-usd", type=float, default=0.0, help="If >0, clip label to [-clip, +clip] USD for stability")

    # Dislocation filter knobs
    p.add_argument("--fair-move-min-cents", type=float, default=0.10, help="Require |dp_fair_cents| >= this")
    p.add_argument("--max-poly-move-cents", type=float, default=0.0, help="If >0: reject rows where BOTH sides moved > this many cents")
    p.add_argument("--poly-both-moved-cents", type=float, default=1.0, help="Reject if BOTH dP_yes and dP_no exceed this")
    p.add_argument("--max-usables", type=int, default=0, help="Cap usable rows for faster dev runs (0=all)")

    # Trading decision threshold
    p.add_argument("--min-pred-pnl-usd", type=float, default=0.0, help="Trade if predicted pnl_usd >= this threshold")
    p.add_argument("--reverse-strategy", action="store_true", help="Reverse strategy: trade when prediction is negative instead of positive")

    # Optional sweep
    p.add_argument("--sweep-thresholds", action="store_true", help="Sweep thresholds on TEST set and print trade/PnL curve")
    p.add_argument("--sweep-n", type=int, default=25, help="How many thresholds to try in sweep")

    # Output
    p.add_argument("--out-plot", type=str, default="reg_backtest_test_half.png")
    p.add_argument("--save-model", type=str, default="reg_model_state.pt")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_usables is not None and args.max_usables <= 0:
        args.max_usables = None

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    device = torch.device(args.device)

    db_path = Path(args.db)

    print("\n" + "=" * 70)
    print(f"Loading DB: {db_path}")
    print("=" * 70)

    rows_all = load_rows(db_path)
    print(f"Loaded opportunities/outcomes rows: {len(rows_all):,}")

    exit_realized = load_exit_realized_pnl(db_path)
    print(f"Loaded exits realized_pnl rows: {len(exit_realized):,}")

    # Split by slug (market interval)
    train_rows, test_rows, train_slugs, test_slugs = split_by_slug_first_half(rows_all)
    print("\n" + "=" * 70)
    print("SPLIT BY MARKET (SLUG)")
    print("=" * 70)
    print(f"Unique slugs total: {len(train_slugs) + len(test_slugs):,}")
    print(f"Train slugs: {len(train_slugs):,} | Test slugs: {len(test_slugs):,}")
    if train_slugs:
        print(f"Train slug range: {train_slugs[0]}  ...  {train_slugs[-1]}")
    if test_slugs:
        print(f"Test  slug range: {test_slugs[0]}  ...  {test_slugs[-1]}")
    print(f"Rows in train half: {len(train_rows):,} | Rows in test half: {len(test_rows):,}")

    # Build supervised datasets
    print("\n" + "=" * 70)
    print("BUILDING SUPERVISED DATASET")
    print("=" * 70)

    Xtr, ytr = materialize_samples(train_rows, args=args)
    Xte, yte = materialize_samples(test_rows, args=args)

    print(f"Train samples: {Xtr.shape[0]:,} | Test samples: {Xte.shape[0]:,}")
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print("No samples built. Check row_is_usable() filters or DB contents.")
        return

    state_dim = int(Xtr.shape[1])

    # Quick label stats
    def stats(name: str, y: np.ndarray) -> None:
        y_ = y.astype(np.float64)
        print(
            f"{name} y: mean={y_.mean():.6f}  std={y_.std():.6f}  "
            f"p50={np.percentile(y_,50):.6f}  p90={np.percentile(y_,90):.6f}  p99={np.percentile(y_,99):.6f}  "
            f"pos%={(y_>0).mean()*100:.1f}%"
        )
    stats("TRAIN", ytr)
    stats("TEST ", yte)

    # Model
    model = RegNet(state_dim, hidden=int(args.hidden)).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.HuberLoss(delta=float(args.huber_delta))

    print("\n" + "=" * 70)
    print("TRAINING REGRESSOR")
    print("=" * 70)

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []

        for xb, yb in batch_iter(Xtr, ytr, int(args.batch_size), rng):
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        # Eval
        model.eval()
        with torch.no_grad():
            xte = torch.from_numpy(Xte).to(device)
            yte_t = torch.from_numpy(yte).to(device)
            p = model(xte)
            te_loss = float(loss_fn(p, yte_t).detach().cpu().item())

        print(f"Epoch {ep:>3d}/{int(args.epochs)} | train_loss={np.mean(losses):.6f} | test_loss={te_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"\nModel saved to: {args.save_model}")

    # Backtest on TEST half using threshold
    print("\n" + "=" * 70)
    print("BACKTEST (TEST HALF ONLY)")
    print("=" * 70)
    if args.reverse_strategy:
        print(f"Decision: trade OPPOSITE side if predicted_pnl_usd >= {args.min_pred_pnl_usd:.6f} (REVERSED STRATEGY)")
    else:
        print(f"Decision: trade if predicted_pnl_usd >= {args.min_pred_pnl_usd:.6f}")

    # Walk ALL test rows for MtM tracking; only trade on usable rows.
    # Filter for rows that have the minimum fields for MtM (yes_mid, no_mid).
    all_test = [r for r in test_rows
                if r.yes_mid is not None and r.no_mid is not None]
    if not all_test:
        print("No test rows with mid prices. Aborting backtest.")
        return

    test_start_t_ms = all_test[0].t_ms if all_test else 0
    MAX_QTY_PER_ENTRY = int(args.max_qty_per_entry)  # hard cap on contracts per single entry
    MAX_INV_PER_SIDE = float(args.max_inventory_per_side)  # total cap per side

    preds_all: list[float] = []
    true_all: list[float] = []
    ts_all: list[int] = []
    slug_all: list[str] = []
    inv_yes_all: list[float] = []  # track inventory over time
    inv_no_all: list[float] = []   # track inventory over time

    # ---- Inventory: track average entry price per side ----
    inv_yes_qty = 0.0      # total YES contracts held
    inv_yes_cost = 0.0     # cumulative (entry_px * qty) for YES
    inv_no_qty = 0.0       # total NO contracts held
    inv_no_cost = 0.0      # cumulative (entry_px * qty) for NO

    prev_unrealized = 0.0  # unrealized P&L at the end of the previous step
    prev_bin: Optional[float] = None
    prev_yes_mid: Optional[float] = None
    prev_no_mid: Optional[float] = None
    prev_fair: Optional[float] = None
    prev_slug: str = all_test[0].slug if all_test else ""

    usable_count = 0  # count how many rows pass the usable filter (for reporting)

    def _unrealized(yes_mid: Optional[float], no_mid: Optional[float]) -> float:
        """Compute total unrealized P&L from current inventory & prices."""
        u = 0.0
        if inv_yes_qty > 0 and yes_mid is not None:
            avg_px = inv_yes_cost / inv_yes_qty
            u += (yes_mid - avg_px) * inv_yes_qty
        if inv_no_qty > 0 and no_mid is not None:
            avg_px = inv_no_cost / inv_no_qty
            u += (no_mid - avg_px) * inv_no_qty
        return u

    def _reset_inventory() -> None:
        """Clear all inventory for a new market hour."""
        nonlocal inv_yes_qty, inv_yes_cost, inv_no_qty, inv_no_cost, prev_unrealized
        nonlocal prev_bin, prev_yes_mid, prev_no_mid, prev_fair
        inv_yes_qty = 0.0
        inv_yes_cost = 0.0
        inv_no_qty = 0.0
        inv_no_cost = 0.0
        prev_unrealized = 0.0
        prev_bin = None
        prev_yes_mid = None
        prev_no_mid = None
        prev_fair = None

    model.eval()
    for row in all_test:
        # ---- Market transition: reset inventory when we START processing a new slug ----
        if row.slug != prev_slug and prev_slug:
            _reset_inventory()

        # ---- Mark-to-market P&L (always, even on non-tradeable rows) ----
        current_unrealized = _unrealized(row.yes_mid, row.no_mid)
        step_pnl = current_unrealized - prev_unrealized
        prev_unrealized = current_unrealized

        # ---- Check if this row is usable for trading ----
        is_usable = row_is_usable(row, args)
        pred = 0.0
        do_trade = False

        if is_usable:
            usable_count += 1
            best_side = best_side_for_row(row)

            if best_side is not None:
                trade_side = best_side
                if args.reverse_strategy:
                    trade_side = "NO" if best_side == "YES" else "YES"

                # Deltas (0 at first row of each market because prev_* are None)
                dS = 0.0 if prev_bin is None or row.bin_mid is None else (float(row.bin_mid) - float(prev_bin))
                dP_yes = 0.0 if prev_yes_mid is None or row.yes_mid is None else (float(row.yes_mid) - float(prev_yes_mid)) * 100.0
                dP_no = 0.0 if prev_no_mid is None or row.no_mid is None else (float(row.no_mid) - float(prev_no_mid)) * 100.0
                dF = 0.0 if prev_fair is None or row.fair_yes is None else (float(row.fair_yes) - float(prev_fair)) * 100.0

                edge_pp = float(edge_for_side_pp(row, best_side) or 0.0)
                spread_c = float(spread_for_side_cents(row, best_side) or 0.0)
                dP_cents = dP_yes if best_side == "YES" else dP_no
                poly_reaction = abs(float(dP_cents)) / (abs(float(dF)) + 1e-6) if abs(float(dF)) > 1e-6 else 0.0

                s = build_state_from_row(
                    row=row, side=best_side,
                    edge_pp=edge_pp, spread_c=spread_c,
                    dP_cents=dP_cents, dF_cents=dF, dS=dS,
                    poly_reaction=poly_reaction,
                    inv_yes=inv_yes_qty, inv_no=inv_no_qty,
                    args=args
                )

                with torch.no_grad():
                    x = torch.from_numpy(s.reshape(1, -1)).to(device)
                    pred = float(model(x).detach().cpu().item())

                # ---- Trade decision ----
                ask = ask_for_side(row, trade_side)
                if ask is not None and ask > 0 and pred >= float(args.min_pred_pnl_usd):
                    qty = qty_from_notional(float(args.entry_notional), float(ask))
                    qty = min(qty, MAX_QTY_PER_ENTRY)
                    if trade_side == "YES":
                        room = max(0.0, MAX_INV_PER_SIDE - inv_yes_qty)
                        qty = min(qty, room)
                    else:
                        room = max(0.0, MAX_INV_PER_SIDE - inv_no_qty)
                        qty = min(qty, room)
                    if qty > 0:
                        do_trade = True
                        if trade_side == "YES":
                            inv_yes_cost += float(ask) * qty
                            inv_yes_qty += qty
                        else:
                            inv_no_cost += float(ask) * qty
                            inv_no_qty += qty

        # ---- Record this step ----
        preds_all.append(pred)
        true_all.append(step_pnl)
        ts_all.append(row.t_ms)
        slug_all.append(row.slug)
        inv_yes_all.append(inv_yes_qty)
        inv_no_all.append(inv_no_qty)

        # Carry forward prices for next iteration's deltas
        prev_bin = row.bin_mid
        prev_yes_mid = row.yes_mid
        prev_no_mid = row.no_mid
        prev_fair = row.fair_yes
        prev_slug = row.slug

    # ---- End-of-data: no extra settlement needed.
    #      The last step already marked-to-market at the final prices. ----

    preds_arr = np.asarray(preds_all, dtype=np.float64)
    true_arr = np.asarray(true_all, dtype=np.float64)

    # Count actual trades: only rows that were usable AND had pred >= threshold AND qty > 0
    # We use a simple heuristic: pred != 0 means the row was usable and the model ran
    usable_mask = preds_arr != 0.0
    trades = int((usable_mask & (preds_arr >= float(args.min_pred_pnl_usd))).sum())
    total_rows = len(preds_all)
    trade_percent = (trades / usable_count * 100.0) if usable_count > 0 else 0.0
    cum_pnl = float(true_arr.sum())
    avg_pnl = cum_pnl / trades if trades > 0 else 0.0

    mode_str = "REVERSED" if args.reverse_strategy else "NORMAL"
    print(f"\n+------------------------------------------------------------+")
    print(f"| {mode_str} STRATEGY RESULTS (raw mark-to-market):{'':>14}|")
    print(f"|   Total rows:          {total_rows:>10,}                       |")
    print(f"|   Usable opportunities:{usable_count:>10,}                       |")
    print(f"|   Trades taken:         {trades:>10,} ({trade_percent:.1f}%)               |")
    print(f"|   Max qty/entry:       {MAX_QTY_PER_ENTRY:>10}                       |")
    print(f"|   Max inv/side:        {MAX_INV_PER_SIDE:>10.0f}                       |")
    print(f"|   Notional/entry:      ${float(args.entry_notional):>9.2f}                       |")
    print(f"|   Total MtM P&L:       ${cum_pnl:>11.2f}                       |")
    print(f"|   Avg P&L/trade:       ${avg_pnl:>11.4f}                       |")
    print(f"+------------------------------------------------------------+")

    # Per-market breakdown
    print("\n  Per-market breakdown:")
    unique_slugs = []
    seen = set()
    for sl in slug_all:
        if sl not in seen:
            unique_slugs.append(sl)
            seen.add(sl)
    for sl in unique_slugs:
        mask = np.array([s == sl for s in slug_all])
        sl_pnl = float(true_arr[mask].sum())
        sl_steps = int(mask.sum())
        sl_usable = usable_mask[mask]
        sl_trades = int((sl_usable & (preds_arr[mask] >= float(args.min_pred_pnl_usd))).sum())
        print(f"    {sl}: {sl_steps} steps, {sl_trades} trades, P&L=${sl_pnl:.2f}")

    # Plot
    if HAS_MPL:
        cum = np.cumsum(true_arr)

        # Debug: large single-step P&L
        large_steps = np.where(np.abs(true_arr) > 1.0)[0]
        if len(large_steps) > 0:
            print(f"\n[DEBUG] {len(large_steps)} steps with |step P&L| > $1:")
            for idx in large_steps[:10]:
                print(f"  Step {idx}: t_ms={ts_all[idx]}, slug={slug_all[idx]}, step_pnl={true_arr[idx]:.4f}")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1 = axes[0]  # P&L plot
        ax2 = axes[1]  # Inventory plot

        if ts_all:
            times = [datetime.fromtimestamp(t / 1000.0) for t in ts_all]
            # Top plot: P&L
            ax1.plot(times, cum,
                    label=f"{mode_str} raw MtM ({trades}/{usable_count} usable, {trade_percent:.1f}%)",
                    linewidth=2, alpha=0.85)

            # Bottom plot: Inventory
            ax2.plot(times, inv_yes_all,
                    label="YES inventory", linewidth=1.5, alpha=0.8, color='green')
            ax2.plot(times, inv_no_all,
                    label="NO inventory", linewidth=1.5, alpha=0.8, color='red')
            ax2.axhline(y=MAX_INV_PER_SIDE, linestyle='--', color='gray', alpha=0.5, label=f"Cap ({MAX_INV_PER_SIDE:.0f})")
        else:
            # Fallback if no timestamps
            ax1.plot(cum,
                    label=f"{mode_str} raw MtM ({trades}/{usable_count} usable, {trade_percent:.1f}%)",
                    linewidth=2, alpha=0.85)
            ax2.plot(inv_yes_all, label="YES inventory", linewidth=1.5, alpha=0.8, color='green')
            ax2.plot(inv_no_all, label="NO inventory", linewidth=1.5, alpha=0.8, color='red')
            ax2.axhline(y=MAX_INV_PER_SIDE, linestyle='--', color='gray', alpha=0.5, label=f"Cap ({MAX_INV_PER_SIDE:.0f})")

        # Mark market boundaries with vertical lines on both plots
        for i in range(1, len(slug_all)):
            if slug_all[i] != slug_all[i-1] and ts_all:
                x_val = datetime.fromtimestamp(ts_all[i] / 1000.0) if ts_all else i
                ax1.axvline(x=x_val, color='red', linestyle=':', alpha=0.4, linewidth=1)
                ax2.axvline(x=x_val, color='red', linestyle=':', alpha=0.4, linewidth=1)

        # Top plot formatting
        ax1.axhline(y=0, linestyle="--", color='gray', alpha=0.5)
        ax1.set_ylabel("Cumulative MtM P&L (USD)")
        ax1.set_title(f"TEST: {mode_str} Accumulation (max {MAX_QTY_PER_ENTRY}/entry, {MAX_INV_PER_SIDE:.0f}/side)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Bottom plot formatting
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Inventory (contracts)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        if ts_all:
            fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150)
        plt.close()
        print(f"\nPlot saved to: {args.out_plot}")

    # ---- Additional overview plot: BTC price + Polymarket odds for ALL hours ----
    if HAS_MPL:
        plot_market_overview(rows_all, out_path="market_overview.png")

    print("\nDone.\n")


def plot_market_overview(rows_all: list[Row], out_path: str = "market_overview.png") -> None:
    """Plot BTC price and Polymarket YES/NO odds across ALL hours (train + test)."""
    if not HAS_MPL:
        return

    ts_list: list[datetime] = []
    btc_list: list[float] = []
    yes_list: list[float] = []
    no_list: list[float] = []
    slug_list: list[str] = []

    for r in rows_all:
        if r.t_ms is None:
            continue
        if r.bin_mid is None and r.yes_mid is None:
            continue
        dt = datetime.fromtimestamp(r.t_ms / 1000.0)
        ts_list.append(dt)
        btc_list.append(float(r.bin_mid) if r.bin_mid is not None else float('nan'))
        yes_list.append(float(r.yes_mid) if r.yes_mid is not None else float('nan'))
        no_list.append(float(r.no_mid) if r.no_mid is not None else float('nan'))
        slug_list.append(r.slug)

    if not ts_list:
        print("No data for market overview plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax_btc = axes[0]
    ax_poly = axes[1]

    # ---- Top: BTC price ----
    ax_btc.plot(ts_list, btc_list, linewidth=1.0, alpha=0.85, color='orange', label='BTC Mid')
    ax_btc.set_ylabel("BTC Price (USD)")
    ax_btc.set_title("Bitcoin Price (All Hours: Train + Test)")
    ax_btc.grid(True, alpha=0.3)
    ax_btc.legend(loc="best")

    # ---- Bottom: Polymarket odds ----
    ax_poly.plot(ts_list, yes_list, linewidth=1.0, alpha=0.8, color='green', label='YES mid')
    ax_poly.plot(ts_list, no_list, linewidth=1.0, alpha=0.8, color='red', label='NO mid')
    ax_poly.set_ylabel("Probability")
    ax_poly.set_xlabel("Time")
    ax_poly.set_title("Polymarket Odds (All Hours: Train + Test)")
    ax_poly.set_ylim(-0.05, 1.05)
    ax_poly.grid(True, alpha=0.3)
    ax_poly.legend(loc="best")

    # Mark slug boundaries on both
    for i in range(1, len(slug_list)):
        if slug_list[i] != slug_list[i - 1]:
            ax_btc.axvline(x=ts_list[i], color='blue', linestyle=':', alpha=0.5, linewidth=1)
            ax_poly.axvline(x=ts_list[i], color='blue', linestyle=':', alpha=0.5, linewidth=1)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nMarket overview plot saved to: {out_path}")


if __name__ == "__main__":
    main()
