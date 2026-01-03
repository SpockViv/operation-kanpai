#!/usr/bin/env python3
"""
arb_backtest.py (LESS STRICT VERSION, ONE-SIDE-PER-SECOND)

Goal:
  Buy-only backtest for Polymarket "BTC Up/Down" 15-minute markets using a pricing engine.

Key features / changes:
  1) Trades primarily on EV edge = (fair_prob - ask). Default z-filter OFF (z_threshold=0.0).
  2) RELAXED hedge/completion rule to reduce imbalance (does NOT require instant lock):
       - If net UP > DOWN, consider buying DOWN when:
           down_ask <= (1 - avg_cost_up) + hedge_slack
       - If net DOWN > UP, consider buying UP when:
           up_ask <= (1 - avg_cost_down) + hedge_slack
  3) IMPORTANT FIX: Can buy ONLY ONE side per second:
       - If both sides are attractive, choose exactly one side using a deterministic rule:
           (a) Prefer the side that reduces imbalance (risk-first)
           (b) If balanced, choose higher edge
           (c) Tie-breaker: higher |z|, then UP

Assumptions (as requested):
  - Bot can ONLY BUY inventory (no sells).
  - Fills are instant at ASK up to available ask_depth.
  - If desired size > depth => partial fill (min(desired, depth)).
  - Conservative timestamp alignment:
      - market timestamps floored to second, keep earliest quote in that second.
      - join on epoch_sec (no look-ahead).
  - Hold to settlement and compute payoff vs BTC start/end for the interval.

Inputs:
  --market-csv : per-second snapshot CSV (NO HEADER):
      ts, up_bid, up_ask, up_bid_depth, up_ask_depth, down_bid, down_ask, down_bid_depth, down_ask_depth

  --engine-csv : pricing engine output CSV (NO HEADER), first 14 cols:
      0  epoch_sec
      1  time_local
      2  slot_start_epoch
      3  slot_start_time
      4  sec_into_slot
      5  sec_remaining
      6  start_price
      7  spot_price
      8  log_return
      9  mu_hat
      10 sigma_hat
      11 fair_up
      12 fair_down
      13 ticks

Optional:
  --btc-csv : 1s BTC bar CSV (NO HEADER):
      epoch_sec, time, open, high, low, close, ...

Output:
  --out : actions log CSV with per-second decisions and inventory stats.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def safe_float(x, default=0.0) -> float:
    """Parse float safely; return default on error."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for an output path if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def bernoulli_z(p: float, q: float, eps: float = 1e-9) -> float:
    """
    z-score for probability edge using Bernoulli standard deviation:
      z = (p - q) / sqrt(p(1-p))
    where:
      p = model fair probability
      q = market implied probability (we use ask when buying)

    NOTE: Often small in practice, so default z_threshold=0.0 (OFF).
    """
    p = max(eps, min(1.0 - eps, float(p)))
    q = max(0.0, min(1.0, float(q)))
    sd = math.sqrt(max(eps, p * (1.0 - p)))
    return (p - q) / sd


# -----------------------------
# Lot tracking (buy-only)
# -----------------------------
@dataclass
class Lot:
    qty: int
    cost: float  # per-share price paid


def total_qty(lots: List[Lot]) -> int:
    return sum(l.qty for l in lots)


def avg_cost(lots: List[Lot]) -> Optional[float]:
    q = total_qty(lots)
    if q <= 0:
        return None
    tot = sum(l.qty * l.cost for l in lots)
    return tot / q


# -----------------------------
# CSV loaders (conservative)
# -----------------------------
def load_market_csv(path: str) -> pd.DataFrame:
    """
    Market CSV expected columns (NO HEADER):
      ts, up_bid, up_ask, up_bid_depth, up_ask_depth, down_bid, down_ask, down_bid_depth, down_ask_depth

    Conservative alignment:
      - parse ts as utc-aware
      - floor to second -> epoch_sec
      - keep earliest snapshot per second
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 9:
        raise ValueError(f"Market CSV must have 9 columns; found {df.shape[1]}")

    df.columns = [
        "ts",
        "up_bid", "up_ask", "up_bid_depth", "up_ask_depth",
        "down_bid", "down_ask", "down_bid_depth", "down_ask_depth",
    ]

    df["ts_dt"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_dt"]).copy()

    # floor to second (no look-ahead)
    df["epoch_sec"] = (df["ts_dt"].astype("int64") // 10**9).astype("int64")

    # keep earliest quote within each second
    df = df.sort_values("ts_dt").drop_duplicates(subset=["epoch_sec"], keep="first")

    # numeric coercion
    for c in [
        "up_bid", "up_ask", "up_bid_depth", "up_ask_depth",
        "down_bid", "down_ask", "down_bid_depth", "down_ask_depth"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[[
        "epoch_sec", "ts_dt",
        "up_bid", "up_ask", "up_bid_depth", "up_ask_depth",
        "down_bid", "down_ask", "down_bid_depth", "down_ask_depth",
    ]].copy()


def load_engine_csv(path: str) -> pd.DataFrame:
    """
    Engine CSV assumed positional columns (NO HEADER), from your samples.
    We take first 14 columns to avoid accidental shift if extra columns are appended later.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 12:
        raise ValueError(f"Engine CSV seems too few columns ({df.shape[1]}).")

    df = df.iloc[:, :14].copy()
    df.columns = [
        "epoch_sec",
        "time_local",
        "slot_start_epoch",
        "slot_start_time",
        "sec_into_slot",
        "sec_remaining",
        "start_price",
        "spot_price",
        "log_return",
        "mu_hat",
        "sigma_hat",
        "fair_up",
        "fair_down",
        "ticks",
    ]

    for c in [
        "epoch_sec", "slot_start_epoch", "sec_into_slot", "sec_remaining",
        "start_price", "spot_price", "log_return", "mu_hat", "sigma_hat",
        "fair_up", "fair_down", "ticks"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["epoch_sec", "fair_up"]).copy()
    df["epoch_sec"] = df["epoch_sec"].astype("int64")

    # clamp fair probs
    df["fair_up"] = df["fair_up"].clip(0.0, 1.0)
    df["fair_down"] = df["fair_down"].fillna(1.0 - df["fair_up"]).clip(0.0, 1.0)

    return df


def load_btc_csv_optional(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Optional BTC 1s bars (NO HEADER):
      epoch_sec, time, open, high, low, close, ...
    Only need epoch_sec + close.
    """
    if not path:
        return None
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        raise ValueError("BTC CSV must have at least 6 columns (epoch_sec,...,close,...)")
    df = df.iloc[:, :6].copy()
    df.columns = ["epoch_sec", "time", "open", "high", "low", "close"]
    df["epoch_sec"] = pd.to_numeric(df["epoch_sec"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["epoch_sec", "close"]).copy()
    df["epoch_sec"] = df["epoch_sec"].astype("int64")
    df = df.sort_values("epoch_sec")
    return df[["epoch_sec", "close"]].copy()


def btc_price_at_or_before(btc_df: pd.DataFrame, t_epoch: int) -> Optional[float]:
    """Conservative lookup: last BTC close at or BEFORE t_epoch."""
    s = btc_df[btc_df["epoch_sec"] <= t_epoch]
    if s.empty:
        return None
    return float(s.iloc[-1]["close"])


# -----------------------------
# Backtest
# -----------------------------
@dataclass
class Trade:
    epoch_sec: int
    side: str     # "UP" or "DOWN"
    qty: int
    price: float
    reason: str


def backtest(
    market_df: pd.DataFrame,
    engine_df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame],
    out_actions_csv: str,
    z_threshold: float,
    edge_threshold: float,
    trade_size: int,
    max_pos_per_side: int,
    max_net_imbalance: int,
    hedge_slack: float,
) -> None:
    """
    Strategy (less strict, one-side-per-second):

    Per second:
      1) Compute edge_up = fair_up - up_ask, edge_down = fair_down - down_ask.
      2) Candidate signals:
           - Undervaluation: edge >= edge_threshold (and optional z filter)
           - Relaxed hedge: if imbalanced and price is "reasonable" relative to avg other cost
      3) If BOTH sides are candidates, choose ONE:
           (a) Reduce imbalance first
           (b) Else bigger edge
           (c) Else bigger |z|
           (d) Else UP
      4) Execute chosen side at ask with partial fills allowed; apply position + imbalance limits.
    """

    # Conservative join: only seconds that exist in BOTH datasets
    df = pd.merge(
        market_df,
        engine_df[["epoch_sec", "slot_start_epoch", "start_price", "spot_price", "fair_up", "fair_down"]],
        on="epoch_sec",
        how="inner",
    ).sort_values("epoch_sec").reset_index(drop=True)

    if df.empty:
        raise ValueError("No overlapping seconds between market and engine data after conservative merge.")

    # Interval boundaries
    slot_start = int(df["slot_start_epoch"].dropna().iloc[0])
    slot_end = slot_start + 900

    # Settlement (conservative)
    if btc_df is not None:
        btc_start = btc_price_at_or_before(btc_df, slot_start)
        btc_end = btc_price_at_or_before(btc_df, slot_end)
        if btc_start is None or btc_end is None:
            # fallback to engine values (still conservative-ish, but not as good as BTC file)
            btc_start = float(df["start_price"].dropna().iloc[0])
            btc_end = float(df["spot_price"].dropna().iloc[-1])
    else:
        btc_start = float(df["start_price"].dropna().iloc[0])
        btc_end = float(df["spot_price"].dropna().iloc[-1])

    resolves_up = 1 if btc_end >= btc_start else 0

    # Inventory lots
    up_lots: List[Lot] = []
    down_lots: List[Lot] = []
    trades: List[Trade] = []

    ensure_parent_dir(out_actions_csv)

    def apply_limits(side: str, qty: int, inv_up: int, inv_down: int) -> int:
        """
        Apply:
          - max_pos_per_side
          - max_net_imbalance (abs(up - down))
        """
        if qty <= 0:
            return 0

        net = inv_up - inv_down

        if side == "UP":
            qty = min(qty, max(0, max_pos_per_side - inv_up))
            # Buying UP increases net by qty => need abs(net + qty) <= max_net_imbalance
            if net >= 0:
                qty = min(qty, max(0, max_net_imbalance - net))
            # if net < 0, buying UP reduces imbalance, usually allowed
            return max(0, qty)

        if side == "DOWN":
            qty = min(qty, max(0, max_pos_per_side - inv_down))
            # Buying DOWN decreases net by qty => net' = net - qty
            if net <= 0:
                # net negative; buying DOWN makes it more negative
                # Need abs(net - qty) <= max_net_imbalance
                # For net<=0: abs(net - qty) = -(net - qty) = -net + qty
                # => -net + qty <= max => qty <= max + net
                qty = min(qty, max(0, max_net_imbalance + net))
            # if net > 0, buying DOWN reduces imbalance, usually allowed
            return max(0, qty)

        return 0

    with open(out_actions_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch_sec", "ts_utc",
            "up_bid", "up_ask", "up_ask_depth",
            "down_bid", "down_ask", "down_ask_depth",
            "fair_up", "fair_down",
            "edge_up", "edge_down",
            "z_up", "z_down",
            "chosen_side",
            "buy_qty", "buy_price",
            "reasons",
            "inv_up", "inv_down", "net_imbalance",
            "avg_cost_up", "avg_cost_down",
            "paired_units_est", "avg_pair_cost_est",
        ])

        for _, row in df.iterrows():
            epoch = int(row["epoch_sec"])
            ts_utc = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

            # Market
            up_ask = safe_float(row["up_ask"], default=float("nan"))
            down_ask = safe_float(row["down_ask"], default=float("nan"))
            up_ask_depth = int(safe_float(row["up_ask_depth"], default=0.0))
            down_ask_depth = int(safe_float(row["down_ask_depth"], default=0.0))

            # Engine fair probs
            fair_up = safe_float(row["fair_up"], default=float("nan"))
            fair_down = safe_float(row["fair_down"], default=float("nan"))

            # Skip if missing critical values
            if not (math.isfinite(up_ask) and math.isfinite(down_ask) and math.isfinite(fair_up) and math.isfinite(fair_down)):
                continue

            # Current inventory
            inv_up = total_qty(up_lots)
            inv_down = total_qty(down_lots)
            net = inv_up - inv_down
            au = avg_cost(up_lots)
            ad = avg_cost(down_lots)

            # Edges and z-scores vs ASK (buying cost)
            edge_up = fair_up - up_ask
            edge_down = fair_down - down_ask
            z_up = bernoulli_z(fair_up, up_ask)
            z_down = bernoulli_z(fair_down, down_ask)

            # Optional z-filter
            pass_z_up = True if z_threshold <= 0 else (z_up >= z_threshold)
            pass_z_down = True if z_threshold <= 0 else (z_down >= z_threshold)

            # Candidate desires (do NOT execute yet)
            want_up = 0
            want_down = 0
            reasons = []

            # (A) Undervaluation buys
            if edge_up >= edge_threshold and pass_z_up:
                want_up = trade_size
                reasons.append(f"UNDER_UP edge={edge_up:.4f} z={z_up:.2f}")
            if edge_down >= edge_threshold and pass_z_down:
                want_down = trade_size
                reasons.append(f"UNDER_DN edge={edge_down:.4f} z={z_down:.2f}")

            # (B) Relaxed hedge buys to reduce imbalance
            # net = inv_up - inv_down
            if net > 0 and au is not None and down_ask_depth > 0:
                # Need more DOWN
                if down_ask <= (1.0 - au) + hedge_slack:
                    # cap hedge desire by imbalance size (don’t over-hedge in one shot)
                    want_down = max(want_down, min(trade_size, net))
                    reasons.append("HEDGE_DN (down_ask <= (1-avg_up)+slack)")

            if net < 0 and ad is not None and up_ask_depth > 0:
                # Need more UP
                if up_ask <= (1.0 - ad) + hedge_slack:
                    want_up = max(want_up, min(trade_size, -net))
                    reasons.append("HEDGE_UP (up_ask <= (1-avg_dn)+slack)")

            # -----------------------------
            # CHOOSE ONE SIDE ONLY (the requested fix)
            # -----------------------------
            chosen_side = None

            if want_up > 0 or want_down > 0:
                if want_up > 0 and want_down > 0:
                    # (a) Reduce imbalance first
                    if net > 0:
                        chosen_side = "DOWN"
                    elif net < 0:
                        chosen_side = "UP"
                    else:
                        # (b) Bigger edge
                        if edge_up > edge_down:
                            chosen_side = "UP"
                        elif edge_down > edge_up:
                            chosen_side = "DOWN"
                        else:
                            # (c) Bigger |z|
                            if abs(z_up) >= abs(z_down):
                                chosen_side = "UP"
                            else:
                                chosen_side = "DOWN"
                else:
                    chosen_side = "UP" if want_up > 0 else "DOWN"

            # -----------------------------
            # EXECUTE ONLY THE CHOSEN SIDE
            # -----------------------------
            buy_qty = 0
            buy_price = ""

            if chosen_side == "UP" and up_ask_depth > 0:
                inv_up = total_qty(up_lots)
                inv_down = total_qty(down_lots)

                q = min(want_up, up_ask_depth)              # partial fill
                q = apply_limits("UP", q, inv_up, inv_down) # caps
                if q > 0:
                    up_lots.append(Lot(qty=q, cost=up_ask))
                    trades.append(Trade(epoch_sec=epoch, side="UP", qty=q, price=up_ask, reason=";".join(reasons)))
                    buy_qty = q
                    buy_price = f"{up_ask:.6f}"

            elif chosen_side == "DOWN" and down_ask_depth > 0:
                inv_up = total_qty(up_lots)
                inv_down = total_qty(down_lots)

                q = min(want_down, down_ask_depth)
                q = apply_limits("DOWN", q, inv_up, inv_down)
                if q > 0:
                    down_lots.append(Lot(qty=q, cost=down_ask))
                    trades.append(Trade(epoch_sec=epoch, side="DOWN", qty=q, price=down_ask, reason=";".join(reasons)))
                    buy_qty = q
                    buy_price = f"{down_ask:.6f}"

            # Refresh stats after trade
            inv_up = total_qty(up_lots)
            inv_down = total_qty(down_lots)
            net = inv_up - inv_down
            au = avg_cost(up_lots)
            ad = avg_cost(down_lots)

            paired_units_est = min(inv_up, inv_down)
            avg_pair_cost_est = ""
            if paired_units_est > 0 and au is not None and ad is not None:
                avg_pair_cost_est = f"{(au + ad):.6f}"

            w.writerow([
                epoch, ts_utc,
                safe_float(row["up_bid"], float("nan")),
                up_ask, up_ask_depth,
                safe_float(row["down_bid"], float("nan")),
                down_ask, down_ask_depth,
                fair_up, fair_down,
                f"{edge_up:.6f}", f"{edge_down:.6f}",
                f"{z_up:.4f}", f"{z_down:.4f}",
                chosen_side or "",
                buy_qty, buy_price,
                " | ".join(reasons),
                inv_up, inv_down, net,
                "" if au is None else f"{au:.6f}",
                "" if ad is None else f"{ad:.6f}",
                paired_units_est, avg_pair_cost_est,
            ])

    # -----------------------------
    # Settlement + Summary
    # -----------------------------
    inv_up = total_qty(up_lots)
    inv_down = total_qty(down_lots)

    cost_up = sum(l.qty * l.cost for l in up_lots)
    cost_down = sum(l.qty * l.cost for l in down_lots)
    total_cost = cost_up + cost_down

    payoff = inv_up * resolves_up + inv_down * (1 - resolves_up)
    pnl = payoff - total_cost

    au = avg_cost(up_lots)
    ad = avg_cost(down_lots)

    print("\n========== BACKTEST SUMMARY ==========")
    print(f"Slot start epoch: {slot_start}  ({datetime.fromtimestamp(slot_start).isoformat()})")
    print(f"Slot end   epoch: {slot_end}    ({datetime.fromtimestamp(slot_end).isoformat()})")

    print("\n-- Settlement (conservative) --")
    print(f"BTC_start (<= slot_start): {btc_start}")
    print(f"BTC_end   (<= slot_end):   {btc_end}")
    print(f"Resolves: {'UP' if resolves_up == 1 else 'DOWN'}")

    print("\n-- Inventory --")
    print(f"UP qty:   {inv_up}   avg_cost_up:   {au if au is not None else 'n/a'}   total_cost_up:   {cost_up:.6f}")
    print(f"DOWN qty: {inv_down} avg_cost_down: {ad if ad is not None else 'n/a'}   total_cost_down: {cost_down:.6f}")
    print(f"Paired units (min): {min(inv_up, inv_down)}")
    if au is not None and ad is not None:
        print(f"Avg cost (UP+DOWN): {(au + ad):.6f}  (if < 1.0, avg pair is guaranteed profitable)")

    print("\n-- PnL (hold to settlement) --")
    print(f"Total cost:   {total_cost:.6f}")
    print(f"Total payoff: {payoff:.6f}")
    print(f"Total PnL:    {pnl:.6f}")

    print(f"\n[done] Wrote action log to: {out_actions_csv}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Less-strict buy-only arb/MM backtester using fair probabilities + market quotes (ONE SIDE per second)."
    )
    p.add_argument("--market-csv", required=True, help="Market snapshot CSV for ONE 15-min interval (no header).")
    p.add_argument("--engine-csv", required=True, help="Engine per-second fair price CSV (no header).")
    p.add_argument("--btc-csv", default="", help="Optional BTC 1s bars CSV (no header) to compute settlement conservatively.")
    p.add_argument("--out", default="out/actions_log.csv", help="Output CSV for action log.")

    # Defaults tuned to be less strict (more trades):
    p.add_argument("--z", type=float, default=1.75, help="Z-score threshold (default 0.0 = OFF).")
    p.add_argument("--edge", type=float, default=0.005, help="Probability edge threshold vs ask (default 0.005).")

    p.add_argument("--size", type=int, default=5, help="Shares per trade attempt (default 10).")
    p.add_argument("--max-pos", type=int, default=5000, help="Max shares per side (inventory cap).")
    p.add_argument("--max-imbalance", type=int, default=100, help="Max abs(up - down) allowed.")

    p.add_argument("--hedge-slack", type=float, default=0.1,
                   help="Relaxed hedge slack added to (1 - avg_cost_other). Default 0.03.")
    return p.parse_args()


def main():
    args = parse_args()

    market_df = load_market_csv(args.market_csv)
    engine_df = load_engine_csv(args.engine_csv)
    btc_df = load_btc_csv_optional(args.btc_csv.strip() or None)

    backtest(
        market_df=market_df,
        engine_df=engine_df,
        btc_df=btc_df,
        out_actions_csv=args.out,
        z_threshold=args.z,
        edge_threshold=args.edge,
        trade_size=args.size,
        max_pos_per_side=args.max_pos,
        max_net_imbalance=args.max_imbalance,
        hedge_slack=args.hedge_slack,
    )


if __name__ == "__main__":
    main()
