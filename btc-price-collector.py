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
from typing import Optional, List, Tuple

import websockets
from websockets.exceptions import ConnectionClosed

RTDS_URL = "wss://ws-live-data.polymarket.com"


# ----------------------------
# Data model
# ----------------------------
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


# ----------------------------
# 1-second Aggregator with forced emit (heartbeat)
# ----------------------------
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

        # If no ticks this second, fill using last_price (if available)
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

        # Ignore out-of-order ticks
        if sec < self.cur_sec:
            return out

        # If tick jumps forward, finalize current + fill gaps
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

        # TWAP integration up to tick time
        if self.twap_price is None:
            self.twap_price = price
        self._accumulate_twap_until(ts_ms)

        # Tick-based OHLC
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

        # After tick, piecewise-constant price for TWAP
        self.twap_price = price
        self.last_price = price

        return out

    def force_emit_up_to(self, now_ms: int) -> List[Bar1s]:
        out: List[Bar1s] = []
        if self.cur_sec is None:
            return out

        now_sec = now_ms // 1000

        # Emit bars until our internal second catches up to now_sec
        while self.cur_sec < now_sec:
            bar = self._finalize_current_bar()
            if bar:
                out.append(bar)
            self._start_new_second(self.cur_sec + 1)

        return out


# ----------------------------
# CSV writer (keep file open)
# ----------------------------
class CsvSink:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if is_new:
            self.w.writerow(["epoch_sec", "time", "open", "high", "low", "close", "twap", "ticks"])
            self.f.flush()

    def write_bar(self, bar: Bar1s):
        self.w.writerow([bar.epoch_sec, bar.iso_time, bar.open, bar.high, bar.low, bar.close, bar.twap, bar.ticks])

    def flush(self):
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


def print_bar(bar: Bar1s):
    print(
        f"{bar.iso_time}  "
        f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} "
        f"TWAP:{bar.twap:.2f} ticks:{bar.ticks}"
    )


# ----------------------------
# RTDS message handling
# ----------------------------
def build_subscription(source: str, symbol: str) -> Tuple[dict, str]:
    source = source.lower()

    if source == "chainlink":
        # IMPORTANT: compact JSON string (no spaces) in filters
        sym = symbol.strip().lower()  # btc/usd
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
        sym = symbol.strip().lower()  # btcusdt
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
    """
    Handles:
      - normal updates: payload {symbol, timestamp, value}
      - initial dump: payload {symbol, data:[{timestamp,value},...]}
    """
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return []

    symbol = payload.get("symbol")

    # initial dump format
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


# ----------------------------
# Defaults for "right click run"
# ----------------------------
def default_csv_path(source: str, symbol: str) -> str:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_").lower()
    return str(data_dir / f"{safe_symbol}_{source}_1s_{ts}.csv")


# ----------------------------
# Main streaming loop
# ----------------------------
async def stream_bars(
    source: str,
    symbol: str,
    tz_mode: str,
    csv_path: str,
    quiet: bool,
    heartbeat_print: bool,
    stale_seconds: int,
):
    """
    Fixes:
      1) Sends application-level "PING" (text) every 5 seconds (RTDS docs recommend this).
      2) Forces reconnect if no ticks for stale_seconds (prevents half-dead quiet connections).
    """
    sub_msg, symbol_match = build_subscription(source, symbol)
    agg = OneSecondAggregator(tz_mode=tz_mode)

    sink = CsvSink(csv_path)
    print(f"[csv] Writing to: {csv_path}")

    backoff = 1

    # Track last time we received a valid tick (monotonic clock = safe for elapsed time)
    last_tick_mono = time.monotonic()

    try:
        while True:
            try:
                print(f"[boot] Connecting to {RTDS_URL}")
                async with websockets.connect(
                    RTDS_URL,
                    # Keep websocket control-frame pings too (fine to keep)
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=2000,
                ) as ws:
                    print("[ok] Connected. Sending subscribe…")
                    await ws.send(json.dumps(sub_msg))
                    print(f"[ok] Subscribe sent: {sub_msg}")

                    backoff = 1

                    # Reset tick timer on a new connection
                    last_tick_mono = time.monotonic()

                    async def app_ping():
                        """
                        RTDS docs mention sending PING messages.
                        Many servers expect a literal "PING" text message and respond "PONG".
                        """
                        while True:
                            await asyncio.sleep(5)
                            try:
                                await ws.send("PING")
                            except Exception:
                                # If send fails, let outer loop reconnect
                                return

                    async def heartbeat():
                        """
                        Emit bars every second even without ticks.
                        Also flush periodically so CSV updates live.
                        """
                        last_flush = time.monotonic()
                        while True:
                            await asyncio.sleep(1)

                            bars = agg.force_emit_up_to(int(time.time() * 1000))
                            for b in bars:
                                sink.write_bar(b)
                                if not quiet:
                                    print_bar(b)

                            # periodic flush
                            if time.monotonic() - last_flush >= 2.0:
                                sink.flush()
                                last_flush = time.monotonic()

                    async def stale_watchdog():
                        """
                        If we haven't received ANY ticks for stale_seconds,
                        force a reconnect. This fixes "connected but no data forever".
                        """
                        nonlocal last_tick_mono
                        while True:
                            await asyncio.sleep(1)
                            age = time.monotonic() - last_tick_mono
                            if age >= stale_seconds:
                                print(f"[stale] No ticks for {int(age)}s (>= {stale_seconds}s). Forcing reconnect…")
                                with contextlib.suppress(Exception):
                                    await ws.close()
                                return

                    ping_task = asyncio.create_task(app_ping())
                    hb_task = asyncio.create_task(heartbeat())
                    stale_task = asyncio.create_task(stale_watchdog())

                    try:
                        while True:
                            # If the websocket is closed, ws.recv will raise.
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            except asyncio.TimeoutError:
                                # Still connected, heartbeat continues.
                                if heartbeat_print:
                                    print("[wait] No messages in 30s — still connected. (heartbeat continues)")
                                continue
                            except ConnectionClosed:
                                # Break and reconnect
                                raise

                            # RTDS may send "PONG" (string)
                            if isinstance(raw, str) and raw == "PONG":
                                continue

                            # Parse JSON messages
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue

                            # Sometimes servers send lists of messages; handle both
                            msgs = msg if isinstance(msg, list) else [msg]
                            for one in msgs:
                                if not isinstance(one, dict):
                                    continue

                                ticks = extract_ticks_from_message(one)
                                if not ticks:
                                    continue

                                for ts_ms, price, sym in ticks:
                                    # Filter symbol
                                    if isinstance(sym, str) and sym.strip().lower() != symbol_match:
                                        continue

                                    # We received a real tick: refresh tick timer
                                    last_tick_mono = time.monotonic()

                                    bars = agg.process_tick(ts_ms, price)
                                    for b in bars:
                                        sink.write_bar(b)
                                        if not quiet:
                                            print_bar(b)

                    finally:
                        for t in (ping_task, hb_task, stale_task):
                            t.cancel()
                            with contextlib.suppress(Exception):
                                await t

            except Exception as e:
                print(f"[err] {type(e).__name__}: {e}")
                print(f"[reconnect] in {backoff}s…")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    finally:
        sink.close()


def parse_args():
    p = argparse.ArgumentParser(description="Polymarket RTDS -> continuous 1-second OHLC+TWAP bars")
    p.add_argument("--source", choices=["chainlink", "binance"], default="chainlink")
    p.add_argument("--symbol", default="btc/usd",
                   help="For chainlink: btc/usd. For binance: btcusdt.")
    p.add_argument("--tz", choices=["local", "utc"], default="local")
    p.add_argument("--csv", default="", help="optional CSV output path (auto if omitted)")
    p.add_argument("--quiet", action="store_true", help="don’t print bars, only write CSV")
    p.add_argument("--heartbeat-print", action="store_true", help="debug message when socket is quiet")

    # Key new knob:
    p.add_argument("--stale-seconds", type=int, default=180,
                   help="force reconnect if no ticks arrive for this many seconds (default 180)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto CSV if you just right-click Run
    csv_path = args.csv.strip()
    if not csv_path:
        csv_path = default_csv_path(args.source, args.symbol)

    # Helper if switching sources
    if args.source == "binance" and "/" in args.symbol:
        print("[note] Binance symbols are like btcusdt (no slash). Example: --symbol btcusdt")

    asyncio.run(
        stream_bars(
            source=args.source,
            symbol=args.symbol,
            tz_mode=args.tz,
            csv_path=csv_path,
            quiet=args.quiet,
            heartbeat_print=args.heartbeat_print,
            stale_seconds=args.stale_seconds,
        )
    )
