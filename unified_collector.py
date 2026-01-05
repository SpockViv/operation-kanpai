#!/usr/bin/env python3
"""
Unified data collector combining:
- BTC price data (from Chainlink/Binance via Polymarket RTDS)
- Polymarket orderbook depth data (BTC up/down 15-minute events)

All data is collected into a single CSV file with aligned timestamps.
"""

import argparse
import asyncio
import csv
import json
import os
import time
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque

import requests
import websockets
from websockets.exceptions import ConnectionClosed

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    NY_TZ = None

# ======================
# CONFIGURATION
# ======================
RTDS_URL = "wss://ws-live-data.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEPTH_LEVELS = 10
SNAPSHOT_INTERVAL_SECONDS = 1.0

# ======================
# DATA MODELS
# ======================
@dataclass
class Bar1s:
    epoch_sec: int
    iso_time: str
    open: float
    high: float
    low: float
    close: float
    twap: float
    ticks: int


@dataclass
class PolymarketSnapshot:
    epoch_sec: int
    iso_time: str
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    yes_bid_depth: Optional[float]
    yes_ask_depth: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    no_bid_depth: Optional[float]
    no_ask_depth: Optional[float]
    event_slug: str


# ======================
# BTC PRICE COLLECTOR (from btc-price-collector.py)
# ======================
class OneSecondAggregator:
    """
    Builds 1-second OHLC + TWAP bars from tick prices.
    force_emit_up_to() emits every second even if no new ticks arrive (fills with last_price).
    """

    def __init__(self, tz_mode: str = "local"):
        self.tz_mode = tz_mode
        self.cur_sec: Optional[int] = None
        self.had_tick: bool = False
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.twap_numer_ms: float = 0.0
        self.twap_cursor_ms: Optional[int] = None
        self.twap_price: Optional[float] = None
        self.last_price: Optional[float] = None
        self.tick_count: int = 0

    def _sec_start_ms(self, sec: int) -> int:
        return sec * 1000

    def _iso(self, epoch_sec: int) -> str:
        if self.tz_mode == "utc":
            dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
            return dt.isoformat()
        return datetime.fromtimestamp(epoch_sec).strftime("%Y-%m-%d %H:%M:%S")

    def _start_new_second(self, sec: int):
        self.cur_sec = sec
        self.had_tick = False
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.twap_numer_ms = 0.0
        self.twap_cursor_ms = self._sec_start_ms(sec)
        self.twap_price = self.last_price
        self.tick_count = 0

    def _accumulate_twap_until(self, end_ms: int):
        if self.twap_cursor_ms is None or self.twap_price is None:
            return
        if end_ms <= self.twap_cursor_ms:
            return
        dt = end_ms - self.twap_cursor_ms
        self.twap_numer_ms += self.twap_price * dt
        self.twap_cursor_ms = end_ms

    def _finalize_current_bar(self) -> Optional[Bar1s]:
        if self.cur_sec is None:
            return None

        sec = self.cur_sec
        sec_end_ms = self._sec_start_ms(sec + 1)
        self._accumulate_twap_until(sec_end_ms)

        if not self.had_tick:
            if self.last_price is None:
                return None
            o = h = l = c = self.last_price
            twap = self.last_price
            ticks = 0
        else:
            o = float(self.open)
            h = float(self.high)
            l = float(self.low)
            c = float(self.close)
            twap = float(self.twap_numer_ms / 1000.0)
            ticks = self.tick_count

        return Bar1s(
            epoch_sec=sec,
            iso_time=self._iso(sec),
            open=o, high=h, low=l, close=c, twap=twap, ticks=ticks
        )

    def process_tick(self, ts_ms: int, price: float) -> List[Bar1s]:
        out: List[Bar1s] = []
        sec = ts_ms // 1000

        if self.cur_sec is None:
            self._start_new_second(sec)

        if sec < self.cur_sec:
            return out

        if sec > self.cur_sec:
            bar = self._finalize_current_bar()
            if bar:
                out.append(bar)

            missing_start = self.cur_sec + 1
            missing_end = sec - 1
            if missing_start <= missing_end:
                for s in range(missing_start, missing_end + 1):
                    self._start_new_second(s)
                    fill = self._finalize_current_bar()
                    if fill:
                        out.append(fill)

            self._start_new_second(sec)

        if self.twap_price is None:
            self.twap_price = price
        self._accumulate_twap_until(ts_ms)

        if not self.had_tick:
            self.open = price
            self.high = price
            self.low = price
            self.had_tick = True
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)

        self.close = price
        self.tick_count += 1
        self.twap_price = price
        self.last_price = price

        return out

    def force_emit_up_to(self, now_ms: int) -> List[Bar1s]:
        out: List[Bar1s] = []
        if self.cur_sec is None:
            return out

        now_sec = now_ms // 1000
        while self.cur_sec < now_sec:
            bar = self._finalize_current_bar()
            if bar:
                out.append(bar)
            self._start_new_second(self.cur_sec + 1)

        return out


def build_subscription(source: str, symbol: str) -> Tuple[dict, str]:
    source = source.lower()

    if source == "chainlink":
        sym = symbol.strip().lower()
        msg = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": sym}, separators=(",", ":")),
                }
            ],
        }
        return msg, sym

    if source == "binance":
        sym = symbol.strip().lower()
        msg = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": sym,
                }
            ],
        }
        return msg, sym

    raise ValueError("source must be 'chainlink' or 'binance'")


def extract_ticks_from_message(msg: dict) -> List[Tuple[int, float, Optional[str]]]:
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return []

    symbol = payload.get("symbol")

    if "data" in payload and isinstance(payload["data"], list):
        ticks = []
        for item in payload["data"]:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp")
            val = item.get("value")
            if isinstance(ts, (int, float)) and isinstance(val, (int, float)):
                ticks.append((int(ts), float(val), symbol))
        return ticks

    ts = payload.get("timestamp")
    val = payload.get("value")
    if isinstance(ts, (int, float)) and isinstance(val, (int, float)):
        return [(int(ts), float(val), symbol)]

    return []


# ======================
# POLYMARKET COLLECTOR (from polymarket_market_collector.py)
# ======================
def split_slug(slug: str):
    prefix, ts_str = slug.rsplit("-", 1)
    return prefix, int(ts_str)


def slug_for_timestamp(prefix: str, ts: int) -> str:
    return f"{prefix}-{ts}"


def next_15m_timestamp(ts: int) -> int:
    return ts + 900


def now_ny_iso() -> str:
    if NY_TZ is not None:
        return datetime.now(NY_TZ).isoformat()
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def fetch_event_by_slug(slug: str) -> dict:
    r = requests.get(f"{GAMMA_API}/events/slug/{slug}", timeout=15)
    r.raise_for_status()
    return r.json()


def parse_clob_token_ids(val):
    if val is None:
        return []
    if isinstance(val, list):
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
# UNIFIED CSV WRITER
# ======================
class UnifiedCsvWriter:
    """
    Writes unified CSV with columns from both BTC price and Polymarket data.
    Uses a buffer to align timestamps from both sources.
    """
    
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        
        # Buffers for aligning data
        self.btc_buffer: Dict[int, Bar1s] = {}
        self.pm_buffer: Dict[int, PolymarketSnapshot] = {}
        
        if is_new:
            self.w.writerow([
                "epoch_sec", "iso_time",
                "btc_open", "btc_high", "btc_low", "btc_close", "btc_twap", "btc_ticks",
                "yes_bid", "yes_ask", "yes_bid_depth", "yes_ask_depth",
                "no_bid", "no_ask", "no_bid_depth", "no_ask_depth",
                "event_slug"
            ])
            self.f.flush()

    def add_btc_bar(self, bar: Bar1s):
        """Add BTC price bar to buffer"""
        self.btc_buffer[bar.epoch_sec] = bar
        self._try_write_aligned(bar.epoch_sec)

    def add_polymarket_snapshot(self, snapshot: PolymarketSnapshot):
        """Add Polymarket snapshot to buffer"""
        self.pm_buffer[snapshot.epoch_sec] = snapshot
        self._try_write_aligned(snapshot.epoch_sec)

    def _try_write_aligned(self, epoch_sec: int):
        """Try to write a row if we have both BTC and PM data for this second"""
        btc = self.btc_buffer.get(epoch_sec)
        pm = self.pm_buffer.get(epoch_sec)
        
        if btc and pm:
            # We have both, write the row
            iso_time = btc.iso_time  # Use BTC time format
            self.w.writerow([
                epoch_sec, iso_time,
                btc.open, btc.high, btc.low, btc.close, btc.twap, btc.ticks,
                pm.yes_bid, pm.yes_ask, pm.yes_bid_depth, pm.yes_ask_depth,
                pm.no_bid, pm.no_ask, pm.no_bid_depth, pm.no_ask_depth,
                pm.event_slug
            ])
            self.f.flush()
            
            # Clean up old buffers (keep last 10 seconds)
            min_sec = epoch_sec - 10
            self.btc_buffer = {k: v for k, v in self.btc_buffer.items() if k >= min_sec}
            self.pm_buffer = {k: v for k, v in self.pm_buffer.items() if k >= min_sec}

    def flush(self):
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


# ======================
# BTC PRICE STREAMING TASK
# ======================
async def stream_btc_price(
    source: str,
    symbol: str,
    tz_mode: str,
    csv_writer: UnifiedCsvWriter,
    quiet: bool,
    stale_seconds: int,
    stop_event: asyncio.Event,
):
    """Stream BTC price data and add to unified CSV writer"""
    sub_msg, symbol_match = build_subscription(source, symbol)
    agg = OneSecondAggregator(tz_mode=tz_mode)
    backoff = 1
    last_tick_mono = time.monotonic()

    while not stop_event.is_set():
        try:
            async with websockets.connect(
                RTDS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=2000,
            ) as ws:
                await ws.send(json.dumps(sub_msg))
                backoff = 1
                last_tick_mono = time.monotonic()

                async def app_ping():
                    while not stop_event.is_set():
                        await asyncio.sleep(5)
                        try:
                            await ws.send("PING")
                        except Exception:
                            return

                async def heartbeat():
                    last_flush = time.monotonic()
                    while not stop_event.is_set():
                        await asyncio.sleep(1)
                        bars = agg.force_emit_up_to(int(time.time() * 1000))
                        for b in bars:
                            csv_writer.add_btc_bar(b)
                            if not quiet:
                                print(f"[BTC] {b.iso_time} O:{b.open:.2f} H:{b.high:.2f} L:{b.low:.2f} C:{b.close:.2f} TWAP:{b.twap:.2f} ticks:{b.ticks}")
                        if time.monotonic() - last_flush >= 2.0:
                            csv_writer.flush()
                            last_flush = time.monotonic()

                async def stale_watchdog():
                    nonlocal last_tick_mono
                    while not stop_event.is_set():
                        await asyncio.sleep(1)
                        if stop_event.is_set():
                            return
                        age = time.monotonic() - last_tick_mono
                        if age >= stale_seconds:
                            print(f"[stale] No BTC ticks for {int(age)}s. Forcing reconnect…")
                            with contextlib.suppress(Exception):
                                await ws.close()
                            return

                ping_task = asyncio.create_task(app_ping())
                hb_task = asyncio.create_task(heartbeat())
                stale_task = asyncio.create_task(stale_watchdog())

                try:
                    while not stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            continue
                        except ConnectionClosed:
                            raise

                        if isinstance(raw, str) and raw == "PONG":
                            continue

                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        msgs = msg if isinstance(msg, list) else [msg]
                        for one in msgs:
                            if not isinstance(one, dict):
                                continue

                            ticks = extract_ticks_from_message(one)
                            if not ticks:
                                continue

                            for ts_ms, price, sym in ticks:
                                if isinstance(sym, str) and sym.strip().lower() != symbol_match:
                                    continue

                                last_tick_mono = time.monotonic()
                                bars = agg.process_tick(ts_ms, price)
                                for b in bars:
                                    csv_writer.add_btc_bar(b)
                                    if not quiet:
                                        print(f"[BTC] {b.iso_time} O:{b.open:.2f} H:{b.high:.2f} L:{b.low:.2f} C:{b.close:.2f} TWAP:{b.twap:.2f} ticks:{b.ticks}")

                finally:
                    for t in (ping_task, hb_task, stale_task):
                        t.cancel()
                        with contextlib.suppress(Exception):
                            await t

        except Exception as e:
            if stop_event.is_set():
                return
            print(f"[BTC err] {type(e).__name__}: {e}")
            print(f"[BTC reconnect] in {backoff}s…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ======================
# POLYMARKET STREAMING TASK
# ======================
async def stream_polymarket(
    start_slug: str,
    csv_writer: UnifiedCsvWriter,
    quiet: bool,
    stop_event: asyncio.Event,
):
    """Stream Polymarket orderbook data and add to unified CSV writer"""
    prefix, ts = split_slug(start_slug)

    while not stop_event.is_set():
        slug = slug_for_timestamp(prefix, ts)
        end_ts = ts + 900

        try:
            event = fetch_event_by_slug(slug)
            yes_token, no_token = extract_yes_no_tokens(event)

            if not quiet:
                print(f"\n[PM] Event: {event.get('title')}")
                print(f"[PM] Slug: {slug}")
                print(f"[PM] YES token: {yes_token}, NO token: {no_token}")

            books = {
                yes_token: {"bids": [], "asks": []},
                no_token: {"bids": [], "asks": []},
            }

            async with websockets.connect(WS_URL) as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": [yes_token, no_token]
                }))

                last_snapshot = time.monotonic()

                while not stop_event.is_set():
                    if time.time() >= end_ts:
                        break

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        continue

                    if raw == "PONG":
                        continue

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

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

                        epoch_sec = int(time.time())
                        iso_time = now_ny_iso()

                        snapshot = PolymarketSnapshot(
                            epoch_sec=epoch_sec,
                            iso_time=iso_time,
                            yes_bid=best_price(yb),
                            yes_ask=best_price(ya),
                            yes_bid_depth=depth_sum(yb),
                            yes_ask_depth=depth_sum(ya),
                            no_bid=best_price(nb),
                            no_ask=best_price(na),
                            no_bid_depth=depth_sum(nb),
                            no_ask_depth=depth_sum(na),
                            event_slug=slug
                        )

                        csv_writer.add_polymarket_snapshot(snapshot)

                        if not quiet:
                            print(
                                f"[PM] {iso_time}  "
                                f"YES bid={snapshot.yes_bid} ask={snapshot.yes_ask}   "
                                f"NO bid={snapshot.no_bid} ask={snapshot.no_ask}"
                            )

        except requests.HTTPError as e:
            if stop_event.is_set():
                return
            print(f"[PM HTTPError for {slug}] {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            continue
        except Exception as e:
            if stop_event.is_set():
                return
            print(f"[PM Error for {slug}] {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            continue

        ts = next_15m_timestamp(ts)
        now = time.time()
        if now < ts:
            wait_time = ts - now
            # Check stop_event periodically during wait
            for _ in range(int(wait_time)):
                if stop_event.is_set():
                    return
                await asyncio.sleep(1)
            if not stop_event.is_set() and wait_time > int(wait_time):
                await asyncio.sleep(wait_time - int(wait_time))


# ======================
# MAIN
# ======================
async def main_async(
    source: str,
    symbol: str,
    tz_mode: str,
    csv_path: str,
    quiet: bool,
    stale_seconds: int,
    event_slug: str,
):
    """Run both collectors concurrently"""
    csv_writer = UnifiedCsvWriter(csv_path)
    print(f"[CSV] Writing unified data to: {csv_path}")

    stop_event = asyncio.Event()

    try:
        # Run both tasks concurrently
        btc_task = asyncio.create_task(
            stream_btc_price(source, symbol, tz_mode, csv_writer, quiet, stale_seconds, stop_event)
        )
        pm_task = asyncio.create_task(
            stream_polymarket(event_slug, csv_writer, quiet, stop_event)
        )

        # Wait for both tasks (they run until interrupted)
        await asyncio.gather(btc_task, pm_task)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping collectors...")
        stop_event.set()
        await asyncio.sleep(1)
    finally:
        csv_writer.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified collector: BTC price + Polymarket orderbook data → single CSV"
    )
    p.add_argument("--source", choices=["chainlink", "binance"], default="chainlink",
                   help="BTC price source (default: chainlink)")
    p.add_argument("--symbol", default="btc/usd",
                   help="For chainlink: btc/usd. For binance: btcusdt.")
    p.add_argument("--tz", choices=["local", "utc"], default="local",
                   help="Timezone for timestamps (default: local)")
    p.add_argument("--csv", default="",
                   help="CSV output path (auto-generated if omitted)")
    p.add_argument("--quiet", action="store_true",
                   help="Don't print data, only write CSV")
    p.add_argument("--stale-seconds", type=int, default=180,
                   help="Force BTC reconnect if no ticks for N seconds (default: 180)")
    p.add_argument("--event-slug", default="btc-updown-15m-1766941200",
                   help="Starting Polymarket event slug (default: btc-updown-15m-1766941200)")

    return p.parse_args()


def default_csv_path() -> str:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(data_dir / f"unified_data_{ts}.csv")


if __name__ == "__main__":
    args = parse_args()

    csv_path = args.csv.strip()
    if not csv_path:
        csv_path = default_csv_path()

    if args.source == "binance" and "/" in args.symbol:
        print("[note] Binance symbols are like btcusdt (no slash). Example: --symbol btcusdt")

    asyncio.run(
        main_async(
            source=args.source,
            symbol=args.symbol,
            tz_mode=args.tz,
            csv_path=csv_path,
            quiet=args.quiet,
            stale_seconds=args.stale_seconds,
            event_slug=args.event_slug,
        )
    )

