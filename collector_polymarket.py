#!/usr/bin/env python3
"""
Polymarket BTC Up/Down collector (1-second snapshots, beginner-friendly).

Records every ~1 second:
- token0 + token1:
  - best ask price, best bid price
  - best ask size, best bid size  (from /book)
  - mid, spread, top-of-book notional (price * size)
- BTC spot reference price (Binance):
  - btc_price, btc_source, btc_payload_ts_ms

Output:
- data/raw/<slug>_snapshots.csv

Run (PowerShell, from project root):
  python src/collector_polymarket.py --event-url "https://polymarket.com/event/<slug>" --seconds 60 --interval 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------
# Endpoints
# -----------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"


# -----------------------
# Helpers: paths
# -----------------------
def project_root() -> Path:
    # assumes this file is <root>/src/collector_polymarket.py
    return Path(__file__).resolve().parents[1]


# -----------------------
# Utilities: parsing
# -----------------------
def event_slug_from_url(url: str) -> str:
    m = re.search(r"/event/([^/?#]+)", url)
    if not m:
        raise ValueError(f"Could not extract slug from URL: {url}")
    return m.group(1)


def sanitize_token_id(token_id: Any) -> str:
    s = str(token_id).strip().strip('"').strip("'")
    s_digits = re.sub(r"[^\d]", "", s)
    if not s_digits:
        raise ValueError(f"Token id did not contain digits: {token_id}")
    return s_digits


def parse_token_id_list(maybe_list: Any) -> List[str]:
    """
    Accepts either:
      - a real list: ["123","456"]
      - a JSON string: '["123","456"]'
      - a single token as string/int
    Returns a list of digit-only token ids.
    """
    if isinstance(maybe_list, list):
        raw = maybe_list
    elif isinstance(maybe_list, str):
        s = maybe_list.strip()
        if s.startswith("[") and s.endswith("]"):
            raw = json.loads(s)
        else:
            raw = [s]
    else:
        raw = [maybe_list]

    return [sanitize_token_id(x) for x in raw]


# -----------------------
# Gamma: fetch event JSON
# -----------------------
def fetch_event_by_slug(slug: str, timeout: float = 15.0) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/events/slug/{slug}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_clob_token_ids_from_event(event_json: Dict[str, Any]) -> List[str]:
    markets = event_json.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("Event JSON missing 'markets' list or it's empty.")

    candidate_keys = ("clobTokenIds", "clob_token_ids", "tokenIds", "token_ids")
    for market in markets:
        for key in candidate_keys:
            if key in market:
                token_ids = parse_token_id_list(market[key])
                if len(token_ids) >= 2:
                    return token_ids[:2]

    raise ValueError("Could not find CLOB token IDs in event JSON.")


# -----------------------
# CLOB: /book parsing
# -----------------------
def _parse_level(level: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Polymarket /book formats vary. This safely parses:
      - {"price":"0.51","size":"10"}
      - ["0.51","10"]
      - {"p":"0.51","s":"10"}  (just in case)
    Returns (price, size) floats or (None, None).
    """
    if isinstance(level, dict):
        p = level.get("price", level.get("p"))
        s = level.get("size", level.get("s"))
        try:
            return (float(p) if p is not None else None,
                    float(s) if s is not None else None)
        except Exception:
            return (None, None)

    if isinstance(level, (list, tuple)) and len(level) >= 2:
        try:
            return float(level[0]), float(level[1])
        except Exception:
            return (None, None)

    return (None, None)


def get_book_top(
    session: requests.Session,
    token_id: str,
    timeout: float = 5.0,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    GET /book?token_id=...
    Returns:
      best_ask_price, best_ask_size, best_bid_price, best_bid_size
    """
    token_id = sanitize_token_id(token_id)
    url = f"{CLOB_BASE}/book"
    r = session.get(url, params={"token_id": token_id}, timeout=timeout)
    r.raise_for_status()
    book = r.json()

    asks = book.get("asks") or []
    bids = book.get("bids") or []

    best_ask_p = best_ask_s = None
    best_bid_p = best_bid_s = None

    if isinstance(asks, list) and asks:
        best_ask_p, best_ask_s = _parse_level(asks[0])

    if isinstance(bids, list) and bids:
        best_bid_p, best_bid_s = _parse_level(bids[0])

    return best_ask_p, best_ask_s, best_bid_p, best_bid_s


# -----------------------
# BTC: Binance spot price
# -----------------------
def get_btc_price_binance(session: requests.Session, timeout: float = 3.0) -> Tuple[Optional[float], str, Optional[int]]:
    """
    Returns (btc_price, source, payload_ts_ms).
    Binance endpoint doesn't always include a timestamp, so payload_ts_ms is None.
    """
    try:
        r = session.get(BINANCE_PRICE_URL, params={"symbol": "BTCUSDT"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        price = float(data["price"])
        return price, "binance_spot", None
    except Exception:
        return None, "binance_spot", None


# -----------------------
# Collector
# -----------------------
def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))


def mid_and_spread(a: Optional[float], b: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if a is None or b is None:
        return None, None
    return (a + b) / 2.0, (a - b)


def notional(p: Optional[float], s: Optional[float]) -> Optional[float]:
    if p is None or s is None:
        return None
    return p * s


def collect_every_second(
    event_slug: str,
    seconds: int = 60,
    interval: float = 1.0,
    out_dir: str = "data/raw",
) -> Path:
    root = project_root()
    out_path = (root / out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Fetch event + tokens
    event_json = fetch_event_by_slug(event_slug)
    token0, token1 = extract_clob_token_ids_from_event(event_json)
    token0 = sanitize_token_id(token0)
    token1 = sanitize_token_id(token1)

    print(f"Event slug: {event_slug}")
    print(f"Token0: {token0}")
    print(f"Token1: {token1}")

    out_file = out_path / f"{event_slug}_snapshots.csv"

    session = requests.Session()

    start_mono = time.monotonic()
    end_mono = start_mono + float(seconds)
    next_tick = start_mono

    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "unix_ts", "iso_time", "event_slug",
            "btc_price", "btc_source", "btc_payload_ts_ms",

            "token0_id", "token0_ask", "token0_bid", "token0_ask_size", "token0_bid_size",
            "token0_mid", "token0_spread", "token0_ask_notional", "token0_bid_notional",

            "token1_id", "token1_ask", "token1_bid", "token1_ask_size", "token1_bid_size",
            "token1_mid", "token1_spread", "token1_ask_notional", "token1_bid_notional",
        ])
        f.flush()

        while time.monotonic() < end_mono:
            now_mono = time.monotonic()
            if now_mono < next_tick:
                time.sleep(next_tick - now_mono)
            next_tick += float(interval)

            unix_ts = time.time()
            iso = iso_now()

            # BTC price (never None unless Binance request fails)
            btc_price, btc_source, btc_ts_ms = get_btc_price_binance(session)

            # Book tops (prices + sizes)
            try:
                t0_ask, t0_ask_s, t0_bid, t0_bid_s = get_book_top(session, token0)
            except Exception:
                t0_ask = t0_ask_s = t0_bid = t0_bid_s = None

            try:
                t1_ask, t1_ask_s, t1_bid, t1_bid_s = get_book_top(session, token1)
            except Exception:
                t1_ask = t1_ask_s = t1_bid = t1_bid_s = None

            t0_mid, t0_spread = mid_and_spread(t0_ask, t0_bid)
            t1_mid, t1_spread = mid_and_spread(t1_ask, t1_bid)

            t0_ask_not = notional(t0_ask, t0_ask_s)
            t0_bid_not = notional(t0_bid, t0_bid_s)
            t1_ask_not = notional(t1_ask, t1_ask_s)
            t1_bid_not = notional(t1_bid, t1_bid_s)

            w.writerow([
                f"{unix_ts:.3f}", iso, event_slug,
                btc_price, btc_source, btc_ts_ms,

                token0, t0_ask, t0_bid, t0_ask_s, t0_bid_s,
                t0_mid, t0_spread, t0_ask_not, t0_bid_not,

                token1, t1_ask, t1_bid, t1_ask_s, t1_bid_s,
                t1_mid, t1_spread, t1_ask_not, t1_bid_not,
            ])
            f.flush()

            print(
                f"[{iso}] "
                f"t0 ask={t0_ask} bid={t0_bid} (askSz={t0_ask_s} bidSz={t0_bid_s}) | "
                f"t1 ask={t1_ask} bid={t1_bid} (askSz={t1_ask_s} bidSz={t1_bid_s}) | "
                f"btc={btc_price} ({btc_source})"
            )

    return out_file


# -----------------------
# CLI
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Polymarket book snapshots at ~1 Hz.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--event-url", type=str, help="Polymarket event URL")
    g.add_argument("--slug", type=str, help="Event slug, e.g. btc-updown-15m-1766514600")

    parser.add_argument("--seconds", type=int, default=60, help="How long to collect")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples (target)")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Output dir (relative to project root)")

    args = parser.parse_args()
    slug = args.slug if args.slug else event_slug_from_url(args.event_url)

    out = collect_every_second(
        event_slug=slug,
        seconds=args.seconds,
        interval=args.interval,
        out_dir=args.out_dir,
    )
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
