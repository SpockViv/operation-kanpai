#!/usr/bin/env python3
"""
Polymarket BTC Up/Down 15-minute orderbook collector (self updating).

- Paste ONE starting EVENT_SLUG (e.g. "btc-updown-15m-1766907900")
- Script automatically rolls to the next slug every 15 minutes (+900s)
- For each 15-minute event:
  - looks up YES/NO token IDs via Gamma
  - subscribes to Polymarket CLOB market websocket
  - writes 1-second snapshots to: <slug>_depth.csv (yes_bid, yes_ask, yes_bid_depth, yes_ask_depth, no_bid, no_ask, no_bid_depth, no_ask_depth)
"""

import asyncio
import csv
import json
import time
from datetime import datetime

import requests
import websockets

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    NY_TZ = None  # fallback below


# ======================
# CONFIG (only change EVENT_SLUG)
# ======================
EVENT_SLUG = "btc-updown-15m-1766941200"
DEPTH_LEVELS = 10
SNAPSHOT_INTERVAL_SECONDS = 1.0

GAMMA_API = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


# ======================
# TIME + SLUG HELPERS
# ======================
def split_slug(slug: str):
    """
    "btc-updown-15m-1766907900" -> ("btc-updown-15m", 1766907900)
    """
    prefix, ts_str = slug.rsplit("-", 1)
    return prefix, int(ts_str)


def slug_for_timestamp(prefix: str, ts: int) -> str:
    return f"{prefix}-{ts}"


def next_15m_timestamp(ts: int) -> int:  ##  adds 900 seconds to timestamp
    return ts + 900


def now_ny_iso() -> str:
    if NY_TZ is not None:
        return datetime.now(NY_TZ).isoformat()
    # fallback: UTC iso
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ======================
# GAMMA HELPERS
# ======================
def fetch_event_by_slug(slug: str) -> dict:
    r = requests.get(f"{GAMMA_API}/events/slug/{slug}", timeout=15)
    r.raise_for_status()
    return r.json()  ##  converts JSON response into Python dict
'''
What this returns
{
  "id": 126418,
  "slug": "btc-updown-15m-1766907900",
  "title": "Bitcoin Up or Down - December 28, 2:00AM-2:15AM ET",
  "markets": [
      {
        "clobTokenIds": ["7773...", "11036..."],
        ...
      }
  ],
  ...
}
'''

def parse_clob_token_ids(val):
    if val is None:
        return []
    if isinstance(val, list):  ##  class stuff. Hopefully you understand.
        return [str(x) for x in val]
    if isinstance(val, str):
        return [str(x) for x in json.loads(val)]
    return []


def extract_yes_no_tokens(event: dict) -> tuple[str, str]:
    markets = event.get("markets", [])
    if not markets:
        raise ValueError("No markets in event")

    for m in markets:
        if "clobTokenIds" in m:
            ids = parse_clob_token_ids(m["clobTokenIds"])
            if len(ids) >= 2:
                return ids[0], ids[1]

    raise ValueError("Could not find YES/NO token IDs")


# ======================
# ORDERBOOK HELPERS
# ======================
def parse_level(lvl):
    if isinstance(lvl, dict):
        p = lvl.get("price") or lvl.get("p")
        s = lvl.get("size") or lvl.get("s")
    elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
        p, s = lvl[0], lvl[1]
    else:
        return None, None

    try:
        return float(p), float(s)
    except Exception:
        return None, None


def normalize_levels(levels, side: str):
    """
    side: "bids" -> sort desc (highest first)
          "asks" -> sort asc  (lowest first)
    """
    out = []
    for lvl in levels or []:
        p, s = parse_level(lvl)
        if p is not None and s is not None:
            out.append((p, s))

    out.sort(key=lambda x: x[0], reverse=(side == "bids"))
    return out


def best_price(levels):
    return levels[0][0] if levels else None


def depth_sum(levels):
    return sum(s for _, s in levels)


# ======================
# COLLECT ONE EVENT (15 minutes)
# ======================
async def collect_one_event(slug: str, end_unix_ts: int):
    event = fetch_event_by_slug(slug)
    yes_token, no_token = extract_yes_no_tokens(event)

    print(f"\nEvent title: {event.get('title')}")
    print(f"Event slug:  {slug}")
    print(f"YES token:   {yes_token}")
    print(f"NO token:    {no_token}")

    out_csv = f"{slug}_depth.csv"
    print(f"Writing CSV: {out_csv}")

    books = {
        yes_token: {"bids": [], "asks": []},
        no_token:  {"bids": [], "asks": []},
    }

    # open file (normal context manager), ws (async context manager)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ts_ny",
            "yes_bid", "yes_ask", "yes_bid_depth", "yes_ask_depth",
            "no_bid",  "no_ask",  "no_bid_depth",  "no_ask_depth",
        ])
        f.flush()

        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({
                "type": "market",
                "assets_ids": [yes_token, no_token]
            }))

            last_snapshot = time.monotonic()

            while True:
                # roll when the 15-minute interval ends
                if time.time() >= end_unix_ts:
                    break

                raw = await ws.recv()
                if raw == "PONG":
                    continue

                msg = json.loads(raw)
                updates = msg if isinstance(msg, list) else [msg]

                for u in updates:
                    if not isinstance(u, dict):
                        continue

                    asset = str(u.get("asset_id") or u.get("assetId") or "")
                    if asset not in books:
                        continue

                    if "bids" in u:
                        books[asset]["bids"] = normalize_levels(u["bids"], "bids")
                    if "asks" in u:
                        books[asset]["asks"] = normalize_levels(u["asks"], "asks")

                if time.monotonic() - last_snapshot >= SNAPSHOT_INTERVAL_SECONDS:
                    last_snapshot = time.monotonic()

                    yb = books[yes_token]["bids"][:DEPTH_LEVELS]
                    ya = books[yes_token]["asks"][:DEPTH_LEVELS]
                    nb = books[no_token]["bids"][:DEPTH_LEVELS]
                    na = books[no_token]["asks"][:DEPTH_LEVELS]

                    row = [
                        now_ny_iso(),
                        best_price(yb), best_price(ya), depth_sum(yb), depth_sum(ya),
                        best_price(nb), best_price(na), depth_sum(nb), depth_sum(na),
                    ]
                    writer.writerow(row)
                    f.flush()

                    print(
                        f"{row[0]}  "
                        f"YES bid={row[1]} ask={row[2]}   "
                        f"NO bid={row[5]} ask={row[6]}"
                    )


# ======================
# RUN FOREVER (auto-advance slug every 15 minutes)
# ======================
async def run_forever(start_slug: str):
    prefix, ts = split_slug(start_slug)

    while True:
        slug = slug_for_timestamp(prefix, ts)
        end_ts = ts + 900  # collect until the end of this 15-min window

        try:
            await collect_one_event(slug, end_ts)
        except requests.HTTPError as e:
            # sometimes the next event appears a bit late; retry a few times
            print(f"[Gamma HTTPError for {slug}] {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            continue
        except Exception as e:
            print(f"[Error for {slug}] {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            continue

        # advance to next 15-min slug
        ts = next_15m_timestamp(ts)

        # if we finished early for any reason, wait until the next boundary
        # (keeps rollovers aligned)
        now = time.time()
        if now < ts:
            await asyncio.sleep(ts - now)


def main():
    asyncio.run(run_forever(EVENT_SLUG))


if __name__ == "__main__":
    main()
