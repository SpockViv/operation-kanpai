#!/usr/bin/env python3
"""
market_maker_v2.py – High-frequency spread-capture market maker for Polymarket
hourly BTC up/down binary markets.

Strategy  (MODEL-FREE)
--------
- Simultaneously quote BUY orders on both YES (UP) and NO (DOWN) tokens.
- Capture the spread: if both fill at combined cost < 1.00, guaranteed profit.
- Latency-aware adverse cancel: when Binance impulse over --impulse-ms exceeds
  --soft-move-usd, cancel ONLY the adverse side and hold for --soft-hold-ms.
  When it exceeds --hard-move-usd, enter full risk mode (cancel BOTH sides).
- Adapt the quoted spread based on realised BTC volatility (EMA of moves).
- Track inventory and skew quotes to reduce position risk after partial fills.
- Guard against stale data and auto-refresh long-resting orders.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import traceback
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Tuple
from zoneinfo import ZoneInfo

import certifi
import websockets

from logger_v2 import SQLiteEventLogger, epoch_ns, iso_utc_from_epoch_ns

# ── Fast JSON Required ───────────────────────────────────────────────────
import orjson as _orjson

def _json_loads(s):
    return json.loads(s)

def _json_dumps_str(d):
    return json.dumps(d, separators=(",", ":"), default=str)


try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
except Exception as e:
    raise SystemExit(
        "Missing py_clob_client. Activate venv and install py-clob-client."
    ) from e

try:
    import joblib as _joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
try:
    import numpy as _NP
    _HAS_NUMPY = True
except ImportError:
    _NP = None
    _HAS_NUMPY = False

# ── Constants ────────────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")
GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

BINANCE_WS_PRIMARY = "wss://stream.binance.com:9443/ws"
BINANCE_WS_FALLBACK = "wss://data-stream.binance.vision/ws"
BINANCE_SYMBOL = "btcusdt"

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ── Tiny helpers ─────────────────────────────────────────────────────────────


def now_ns() -> int:
    """Monotonic nanosecond clock – single call, no dict lookup overhead."""
    return time.monotonic_ns()


# ── Utility Helper Functions ────────────────────────────


def jprint(d: dict[str, Any]) -> None:
    print(_json_dumps_str(d), flush=True)


def load_env_file(path: str) -> None:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"env-file not found: {p}")
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        if not key or key in os.environ:
            continue
        vv = v.strip()
        if len(vv) >= 2 and vv[0] == vv[-1] and vv[0] in ("'", '"'):
            vv = vv[1:-1]
        os.environ[key] = vv


def parse_json_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        try:
            y = json.loads(x)
            if isinstance(y, list):
                return y
        except Exception:
            return []
    return []


def hour_label(dt_et: datetime) -> str:
    h = dt_et.hour
    sfx = "am" if h < 12 else "pm"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}{sfx}"


def hourly_slug(dt_et: datetime) -> str:
    d = dt_et.astimezone(ET).replace(minute=0, second=0, microsecond=0)
    return f"bitcoin-up-or-down-{d.strftime('%B').lower()}-{d.day}-{hour_label(d)}-et"


def http_json(url: str, timeout_sec: float) -> dict[str, Any]:
    req = urllib.request.Request(
        str(url),
        headers={
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": "poly-pair-mm/2.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_sec)) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def gamma_by_slug(slug: str, timeout_sec: float) -> dict[str, Any]:
    url = f"{GAMMA_BASE}/markets/slug/{slug}"
    try:
        return http_json(url, timeout_sec)
    except urllib.error.HTTPError as e:
        if int(getattr(e, "code", 0)) == 404:
            raise FileNotFoundError(slug) from e
        raise


def resolve_market_with_hour(
    walk_hours: int, timeout_sec: float
) -> tuple[str, dict[str, Any], datetime]:
    """Return (slug, market_json, hour_start_et) for the first market found walking forward."""
    base = datetime.now(ET).replace(minute=0, second=0, microsecond=0)
    for i in range(max(1, int(walk_hours)) + 1):
        cand = base + timedelta(hours=i)
        slug = hourly_slug(cand)
        try:
            mk = gamma_by_slug(slug, timeout_sec)
            return slug, mk, cand.replace(minute=0, second=0, microsecond=0)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No hourly market found in forward walk")


def extract_up_down_token_ids(market: dict[str, Any]) -> tuple[str, str]:
    tids = parse_json_list(market.get("clobTokenIds"))
    outs = [str(o).strip().lower() for o in parse_json_list(market.get("outcomes"))]
    if len(tids) < 2:
        raise RuntimeError("market missing clobTokenIds")
    if len(outs) != len(tids):
        return str(tids[0]), str(tids[1])
    up_id = None
    down_id = None
    for o, t in zip(outs, tids):
        if o in ("up", "yes"):
            up_id = str(t)
        elif o in ("down", "no"):
            down_id = str(t)
    if not up_id or not down_id:
        up_id, down_id = str(tids[0]), str(tids[1])
    return up_id, down_id


def best_from_levels(levels: Any, is_bid: bool) -> Optional[float]:
    best: Optional[float] = None
    for lv in levels or []:
        try:
            p = float(lv.get("price")) if isinstance(lv, dict) else float(lv[0])
        except Exception:
            continue
        if best is None:
            best = p
        elif is_bid and p > best:
            best = p
        elif (not is_bid) and p < best:
            best = p
    return best


def normalize_price(px: float, pmin: float, pmax: float, tick: float) -> float:
    x = max(float(pmin), min(float(pmax), float(px)))
    q = round(x / float(tick))
    out = q * float(tick)
    out = max(float(pmin), min(float(pmax), out))
    return float(f"{out:.6f}")


def summarize(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)

    def pct(p: float) -> float:
        i = int(round((p / 100.0) * (len(xs2) - 1)))
        i = max(0, min(len(xs2) - 1, i))
        return float(xs2[i])

    return {
        "n": len(xs2),
        "min": float(xs2[0]),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": float(xs2[-1]),
    }


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class BookTop:
    bid: Optional[float] = None
    ask: Optional[float] = None
    recv_ns: Optional[int] = None


@dataclass
class SideState:
    name: str  # "UP" or "DOWN"
    token_id: str
    order_id: Optional[str] = None
    px: Optional[float] = None
    cancel_in_flight: bool = False
    cancel_in_flight_since_ns: int = 0  # monotonic ns when cancel_in_flight was set
    post_in_flight: bool = False  # True while a PostReq is queued / being processed
    post_in_flight_since_ns: int = 0  # monotonic ns when post_in_flight was set
    post_ns: Optional[int] = None  # when the order was posted (monotonic ns)
    size_ordered: float = 0.0  # original order size (may differ from args.size after partials)
    last_seen_matched: float = 0.0  # matched qty as of last fill-watcher poll


@dataclass
class CancelReq:
    prio: int  # 0 = HI, 1 = LO
    seq: int
    t_enqueue_ns: int
    t_trigger_ns: int
    trigger_kind: str  # "BIN" | "FILL" | "REQUOTE" | "ROLLOVER" | "PARTIAL"
    side: str  # "UP" | "DOWN"
    order_id: str
    replace_px: Optional[float] = None
    replace_size: Optional[float] = None  # if set, replacement order uses this size instead of args.size
    step_id: Optional[int] = None
    prev_px: Optional[float] = None
    prev_post_ns: Optional[int] = None
    prev_size_ordered: float = 0.0
    prev_last_seen_matched: float = 0.0


@dataclass
class PostReq:
    prio: int  # 0 = HI, 1 = LO
    seq: int
    t_enqueue_ns: int
    trigger_ns: int
    trigger_kind: str  # "BIN" | "FILL" | "REQUOTE" | "INIT" | "RISK_CLEAR" | "FILL_REPOST" | "PARTIAL_REPOST"
    side: str  # "UP" | "DOWN"
    token_id: str
    px: float
    size: float
    prev_px: Optional[float] = None
    step_id: Optional[int] = None


# ── Volatility tracker ───────────────────────────────────────────────────────


class VolatilityTracker:
    """
    EMA of absolute BTC mid-to-mid moves (per Binance tick).
    Used to dynamically widen the quoted spread when the market is choppy.
    """

    __slots__ = ("_alpha", "_ema", "_last_mid", "min_vol", "max_vol")

    def __init__(
        self,
        half_life_ticks: int = 50,
        min_vol: float = 1.0,
        max_vol: float = 200.0,
    ):
        self._alpha = 2.0 / (half_life_ticks + 1)
        self._ema: Optional[float] = None
        self._last_mid: Optional[float] = None
        self.min_vol = min_vol
        self.max_vol = max_vol

    def update(self, mid: float) -> None:
        if self._last_mid is not None:
            abs_move = abs(mid - self._last_mid)
            if self._ema is None:
                self._ema = abs_move
            else:
                self._ema += self._alpha * (abs_move - self._ema)
        self._last_mid = mid

    @property
    def ema(self) -> float:
        if self._ema is None:
            return self.min_vol
        return max(self.min_vol, min(self.max_vol, self._ema))

    def spread_multiplier(self, base_move_usd: float) -> float:
        """
        Returns a multiplier >= 1.0 that widens the spread when vol is elevated.
        When EMA of moves equals *base_move_usd*, multiplier is 1.0.
        """
        if base_move_usd <= 0:
            return 1.0
        ratio = self.ema / base_move_usd
        return max(1.0, ratio)


# ── Client helpers ───────────────────────────────────────────────────────────


def build_client_from_env() -> Tuple[ClobClient, str]:
    key = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
    if not key:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY missing")
    host = os.environ.get(
        "POLYMARKET_CLOB_HOST", "https://clob.polymarket.com"
    ).strip()
    chain = int(float(os.environ.get("POLYMARKET_CHAIN_ID", "137")))
    funder = os.environ.get("POLYMARKET_PROXY", "").strip() or None
    st_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(st_raw.lstrip("$")) if st_raw else (1 if funder else 0)
    client = ClobClient(
        host,
        key=key,
        chain_id=chain,
        signature_type=signature_type,
        funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client, host


async def warmup_sign(
    loop: asyncio.AbstractEventLoop,
    sign_ex: ThreadPoolExecutor,
    client: ClobClient,
    token_id: str,
) -> None:
    """Pre-warm the signing path to eliminate first-call JIT / lazy-init penalty."""
    try:
        await loop.run_in_executor(
            sign_ex,
            client.create_order,
            OrderArgs(price=0.50, size=1.0, side=BUY, token_id=str(token_id)),
        )
    except Exception:
        pass


async def post_postonly(
    loop: asyncio.AbstractEventLoop,
    sign_ex: ThreadPoolExecutor,
    post_ex: ThreadPoolExecutor,
    client: ClobClient,
    *,
    token_id: str,
    px: float,
    size: float,
    live: bool,
) -> tuple[Optional[str], float, dict[str, Any]]:
    """Sign + post a BUY order with postOnly if supported.

    Raises RuntimeError if the exchange rejects (e.g. crosses book).
    """
    t0 = time.perf_counter()

    if not live:
        return None, 0.0, {"mode": "DRY_RUN"}

    signed = await loop.run_in_executor(
        sign_ex,
        client.create_order,
        OrderArgs(
            price=float(px),
            size=float(size),
            side=BUY,
            token_id=str(token_id),
        ),
    )

    def _post_call():
        try:
            return client.post_order(signed, OrderType.GTC, True)
        except TypeError:
            return client.post_order(signed, OrderType.GTC)

    resp = await loop.run_in_executor(post_ex, _post_call)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    oid = None
    out: dict[str, Any]
    if isinstance(resp, dict):
        oid = resp.get("orderID") or resp.get("orderId") or resp.get("id")
        out = dict(resp)
        # ── Detect exchange-level rejections returned as dicts ─────────
        # The exchange may reject a post-only order that would cross the
        # book and return the error in the response body without raising.
        if not oid:
            err_msg = str(
                resp.get("errorMsg")
                or resp.get("error")
                or resp.get("message")
                or ""
            ).lower()
            if err_msg:
                raise RuntimeError(
                    f"post_order rejected: {resp.get('errorMsg') or resp.get('error') or resp.get('message')}"
                )
    else:
        out = {"response": str(resp)}

    if not oid:
        raise RuntimeError(f"post_order returned no order ID: {out}")

    return str(oid), dt_ms, out


async def cancel_one(
    loop: asyncio.AbstractEventLoop,
    cancel_ex: ThreadPoolExecutor,
    client: ClobClient,
    order_id: str,
) -> float:
    t0 = time.perf_counter()
    await loop.run_in_executor(cancel_ex, client.cancel_orders, [str(order_id)])
    return (time.perf_counter() - t0) * 1000.0


async def get_order(
    loop: asyncio.AbstractEventLoop,
    read_ex: ThreadPoolExecutor,
    client: ClobClient,
    order_id: str,
) -> Optional[dict[str, Any]]:
    try:
        resp = await loop.run_in_executor(read_ex, client.get_order, str(order_id))
        return resp if isinstance(resp, dict) else None
    except Exception:
        return None


def size_matched(order: Optional[dict[str, Any]]) -> float:
    if not order:
        return 0.0
    sm = order.get("size_matched") or order.get("sizeMatched") or "0"
    try:
        return float(sm)
    except Exception:
        return 0.0


# ── Main market-making loop ──────────────────────────────────────────────────


async def run_one_market(
    args: argparse.Namespace,
    *,
    slug: str,
    up_id: str,
    down_id: str,
    hour_start_et: datetime,
) -> None:
    loop = asyncio.get_running_loop()

    # Separate thread pools – extra workers so both sides can sign/post/cancel
    # in parallel without head-of-line blocking.
    sign_ex = ThreadPoolExecutor(max_workers=2)
    post_ex = ThreadPoolExecutor(max_workers=2)
    cancel_ex = ThreadPoolExecutor(max_workers=2)
    read_ex = ThreadPoolExecutor(max_workers=1)

    client, host = build_client_from_env()
    jprint({
        "event": "market_resolved",
        "slug": slug,
        "up_token": up_id,
        "down_token": down_id,
        "host": host,
    })

    # ── Warmup signing path for both tokens ──────────────────────────────
    await asyncio.gather(
        warmup_sign(loop, sign_ex, client, up_id),
        warmup_sign(loop, sign_ex, client, down_id),
    )
    jprint({"event": "sign_warmup_done"})

    # ── SQLite logger for RL training data ────────────────────────────────
    run_id = uuid.uuid4().hex
    _debug_log_path = Path(__file__).resolve().with_name("debug-cda0d1.log")

    def _debug_log(hypothesis_id: str, location: str, message: str, data: Optional[dict[str, Any]] = None) -> None:
        try:
            payload = {
                "sessionId": "cda0d1",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": int(time.time() * 1000),
            }
            with _debug_log_path.open("a", encoding="utf-8") as f:
                f.write(_json_dumps_str(payload) + "\n")
        except Exception:
            pass
    args_json = _json_dumps_str({k: str(v) for k, v in vars(args).items()})
    logger = SQLiteEventLogger(
        str(args.db_path),
        run_id=run_id,
        max_queue=int(args.db_max_queue),
        flush_every_ms=int(args.db_flush_ms),
        log_binance_every_ms=int(args.db_log_binance_every_ms),
        log_poly_every_ms=int(args.db_log_poly_every_ms),
        args_json=args_json,
    )

    roll_et = hour_start_et + timedelta(hours=1)
    _script_dir = Path(__file__).resolve().parent

    # Snapshot of the latest model-driven side decision, included in step logs.
    _poly_last_decision: dict[str, Any] = {
        "model_enabled": False,
        "model_used": False,
        "gate_fail_no_quote_active": False,
        "balance_exception_active": False,
        "balance_only_side": "NONE",
        "gate_fail_reason": "",
        "pred_delta_up_bid": None,
        "pred_delta_down_bid": None,
        "pred_next_up_bid": None,
        "pred_next_down_bid": None,
        "side_state_decision_up": "keep",
        "side_state_decision_down": "keep",
        "model_requote_trigger_up": False,
        "model_requote_trigger_down": False,
        "model_requote_reason_up": "disabled",
        "model_requote_reason_down": "disabled",
        "best_bid_up_at_decision": None,
        "best_bid_down_at_decision": None,
    }

    logger.emit_market(
        slug=slug,
        up_token_id=up_id,
        down_token_id=down_id,
        hour_start_iso=hour_start_et.isoformat(),
        roll_iso=roll_et.isoformat(),
    )
    
    # Wait for the writer thread to flush the market row
    # First, ensure the writer thread is alive
    if not logger._thread.is_alive():
        jprint({
            "event": "db_writer_thread_dead",
            "slug": slug,
            "db_path": logger.db_path,
        })
        logger.close()
        raise RuntimeError("Database writer thread is not alive; check for FATAL_db_writer_init errors")
    
    # Wait for queue to drain (ensures emit_market was picked up).
    # Backward-compatible: older logger_v2.py may not have flush().
    if hasattr(logger, "flush"):
        flushed = logger.flush(timeout_ms=2000.0)  # type: ignore[attr-defined]
        if not flushed:
            jprint({
                "event": "db_flush_timeout",
                "slug": slug,
                "queue_size": logger._q.qsize(),
            })
    
    # Now poll for the market_id to appear in the database
    market_id = 0
    for _ in range(60):
        await asyncio.sleep(0.05)
        market_id = logger.resolve_market_id(slug)
        if market_id:
            break
    if not market_id:
        jprint({
            "event": "db_market_id_missing",
            "slug": slug,
            "thread_alive": logger._thread.is_alive(),
            "queue_size": logger._q.qsize(),
            "dropped_rows": logger.dropped_rows(),
        })
        logger.close()
        raise RuntimeError("Failed to resolve market_id; DB writer not flushing?")

    jprint({"event": "db_logger_ready", "run_id": run_id, "market_id": market_id})

    # ── Shared state ─────────────────────────────────────────────────────
    book_up = BookTop()
    book_dn = BookTop()
    poly_ready = asyncio.Event()

    bin_mid: Optional[float] = None
    bin_ref: Optional[float] = None
    bin_ready = asyncio.Event()
    
    # ── MODEL-FREE impulse horizon ─────────────────────────────────────────
    IMPULSE_MS = float(args.impulse_ms)
    IMPULSE_MEDIUM_MS = IMPULSE_MS * 10  # medium horizon for sustained moves

    # ── Adverse-selection guard (data-driven) ──────────────────────────────
    ASEL_TRIGGER_USD = 2.0  # default; can be overridden by CLI
    ASEL_BUCKETS_USD = (2.0, 5.0, 10.0)  # piecewise thresholds
    ASEL_EXTRA_TICKS = (1, 2, 3)  # extra ticks to pull back on adverse side
    # Binance price history: list of (time_ns, price) tuples, max 1000 entries
    bin_history: list[tuple[int, float]] = []
    MAX_BIN_HISTORY = 1000

    UP = SideState(name="UP", token_id=up_id)
    DN = SideState(name="DOWN", token_id=down_id)

    # Inventory: net units BOUGHT on each side (increases on fills)
    inventory_up: float = 0.0
    inventory_dn: float = 0.0

    # Cancelled-order audit: when ANY code-path cancels an order we haven't
    # fully fill-checked yet, save it here so fill_watcher can do one final
    # get_order and credit any unaccounted partial fills to inventory.
    #   order_id → (side_name, last_seen_matched_at_detach_time)
    pending_fill_audit: dict[str, tuple[str, float]] = {}
    # Cancel-restore snapshot: used to fully restore local order tracking if
    # cancel fails after we've detached local state.
    #   order_id → (px, post_ns, size_ordered, last_seen_matched)
    pending_cancel_snapshot: dict[str, tuple[Optional[float], Optional[int], float, float]] = {}

    # ── Post-only crossing backoff ─────────────────────────────────────
    # When a post-only order is rejected ("crosses book"), the exchange's
    # actual ask is AHEAD of our local WS-fed book.ask.  Rather than
    # immediately retrying at the same stale price, we back off by extra
    # ticks that increase on consecutive rejections and reset on success.
    _MAX_CROSS_BACKOFF = 3          # max extra ticks of safety margin
    _CROSS_BACKOFF_DECAY_NS = int(5e9)  # auto-decay backoff after 5s
    cross_backoff_up: int = 0       # extra ticks for UP side
    cross_backoff_dn: int = 0       # extra ticks for DOWN side
    cross_backoff_last_ns_up: int = 0  # last rejection timestamp (UP)
    cross_backoff_last_ns_dn: int = 0  # last rejection timestamp (DOWN)

    def detach_order_for_audit(S: SideState) -> Optional[str]:
        """Detach the active order from *S* and save for fill audit.
        Returns the old order_id (or None)."""
        oid = S.order_id
        if oid is not None:
            pending_fill_audit[oid] = (S.name, S.last_seen_matched)
            pending_cancel_snapshot[oid] = (
                S.px,
                S.post_ns,
                S.size_ordered,
                S.last_seen_matched,
            )
        S.order_id = None
        S.px = None
        S.post_ns = None
        S.size_ordered = 0.0
        S.last_seen_matched = 0.0
        S.post_in_flight = False
        return oid

    def clear_order_no_audit(S: SideState) -> None:
        """Clear order fields without auditing (fills already accounted for)."""
        S.order_id = None
        S.px = None
        S.post_ns = None
        S.size_ordered = 0.0
        S.last_seen_matched = 0.0
        S.post_in_flight = False

    # Volatility tracker
    vol_tracker = VolatilityTracker(
        half_life_ticks=int(args.vol_half_life),
        min_vol=1.0,
        max_vol=200.0,
    )

    # ── RL step counter ───────────────────────────────────────────────────
    step_counter = 0
    current_step_id: Optional[int] = None

    def _book_age_ms(bk: BookTop) -> Optional[float]:
        if bk.recv_ns is None:
            return None
        return (now_ns() - int(bk.recv_ns)) / 1e6

    def log_step(
        reason: str,
        *,
        action_id: int = 0,
        action: Optional[dict[str, Any]] = None,
        extra_state: Optional[dict[str, Any]] = None,
    ) -> int:
        """Record a decision snapshot and return the new step_id.

        SAFETY: this function must NEVER raise.  If the logger or JSON
        serialisation fails, we still increment and return the step_id so
        that order tracking continues to work correctly.
        """
        nonlocal step_counter, current_step_id
        step_counter += 1
        sid = step_counter
        current_step_id = sid

        try:
            up_mid = (0.5 * (book_up.bid + book_up.ask)) if (book_up.bid is not None and book_up.ask is not None) else None
            dn_mid = (0.5 * (book_dn.bid + book_dn.ask)) if (book_dn.bid is not None and book_dn.ask is not None) else None

            state: dict[str, Any] = {
                "time": {
                    "epoch_ns": epoch_ns(),
                    "mono_ns": now_ns(),
                    "secs_left": (roll_et - datetime.now(ET)).total_seconds(),
                    "roll_et": roll_et.isoformat(),
                },
                "binance": {
                    "mid": bin_mid,
                    "ref": bin_ref,
                    "move": (float(bin_mid) - float(bin_ref)) if (bin_mid is not None and bin_ref is not None) else None,
                    "vol_ema_usd": vol_tracker.ema,
                },
                "poly": {
                    "UP": {
                        "bid": book_up.bid, "ask": book_up.ask, "mid": up_mid,
                        "spread": (book_up.ask - book_up.bid) if (book_up.bid is not None and book_up.ask is not None) else None,
                        "age_ms": _book_age_ms(book_up),
                    },
                    "DOWN": {
                        "bid": book_dn.bid, "ask": book_dn.ask, "mid": dn_mid,
                        "spread": (book_dn.ask - book_dn.bid) if (book_dn.bid is not None and book_dn.ask is not None) else None,
                        "age_ms": _book_age_ms(book_dn),
                    },
                },
                "inventory": {
                    "up": inventory_up,
                    "down": inventory_dn,
                    "imbalance": inventory_up - inventory_dn,
                },
                "orders": {
                    "UP": {
                        "oid": UP.order_id, "px": UP.px,
                        "size_ordered": UP.size_ordered,
                        "last_seen_matched": UP.last_seen_matched,
                    },
                    "DOWN": {
                        "oid": DN.order_id, "px": DN.px,
                        "size_ordered": DN.size_ordered,
                        "last_seen_matched": DN.last_seen_matched,
                    },
                },
                "risk": {
                    "risk_active": risk_active,
                    "risk_dir": risk_dir,
                    "risk_until_ns": risk_until_ns,
                },
            }
            if extra_state:
                state["extra"] = extra_state

            action_dict = action if action is not None else {}

            logger.emit_step(
                market_id=market_id,
                step_id=sid,
                reason=reason,
                action_id=action_id,
                action_json=action_dict,
                state_json=state,
            )
        except Exception:
            pass  # emit_step already counts drops; never crash the trading loop

        return sid

    # ── Action menu (minimal 12-action grid for RL exploration) ────────
    ACTION_MENU_VERSION = 2
    ACTION_MID_OFFSET_TICKS = [0.0, 0.5, 1.0]
    ACTION_ASK_BUFFER_TICKS = [0.0, 0.5, 1.0, 2.0]

    ACTION_MENU: list[dict[str, float]] = []
    for _mo in ACTION_MID_OFFSET_TICKS:
        for _ab in ACTION_ASK_BUFFER_TICKS:
            ACTION_MENU.append({
                "mid_offset_ticks": _mo,
                "ask_buffer_ticks": _ab,
            })
    # ACTION_MENU[0] = {mid_offset_ticks: 0.0, ask_buffer_ticks: 0.0}  ... [11]

    def _float_close(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) < tol

    def action_id_from_knobs(mo: float, ab: float) -> int:
        """Reverse-lookup: current knobs → menu index, or -1 if off-menu."""
        for i, entry in enumerate(ACTION_MENU):
            if _float_close(entry["mid_offset_ticks"], mo) and _float_close(entry["ask_buffer_ticks"], ab):
                return i
        return -1

    # Effective overrides (start with CLI defaults)
    eff_mid_offset_ticks: float = float(args.mid_offset_ticks)
    eff_ask_buffer_ticks: float = float(args.ask_buffer_ticks)
    last_action_id: int = action_id_from_knobs(eff_mid_offset_ticks, eff_ask_buffer_ticks)
    last_action_params: dict[str, Any] = {
        "mid_offset_ticks": eff_mid_offset_ticks,
        "ask_buffer_ticks": eff_ask_buffer_ticks,
    }

    def apply_action(aid: int) -> None:
        """Set the effective quoting knobs from the action menu."""
        nonlocal eff_mid_offset_ticks, eff_ask_buffer_ticks
        nonlocal last_action_id, last_action_params
        if 0 <= aid < len(ACTION_MENU):
            entry = ACTION_MENU[aid]
            eff_mid_offset_ticks = entry["mid_offset_ticks"]
            eff_ask_buffer_ticks = entry["ask_buffer_ticks"]
            last_action_id = aid
            last_action_params = dict(entry)
        else:
            last_action_id = aid
            last_action_params = {
                "mid_offset_ticks": eff_mid_offset_ticks,
                "ask_buffer_ticks": eff_ask_buffer_ticks,
            }

    def choose_action_id() -> int:
        """Select an action id. For now, use the CLI override or default 0."""
        forced = int(getattr(args, "policy_action_id", -1))
        if 0 <= forced < len(ACTION_MENU):
            return forced
        # Default: map current CLI knobs to action id (or 0)
        aid = action_id_from_knobs(float(args.mid_offset_ticks), float(args.ask_buffer_ticks))
        return aid if aid >= 0 else 0

    # Priority queues
    cancel_q: asyncio.PriorityQueue[tuple[int, int, CancelReq]] = (
        asyncio.PriorityQueue()
    )
    post_q: asyncio.PriorityQueue[tuple[int, int, PostReq]] = (
        asyncio.PriorityQueue()
    )
    seq = 0

    # Metrics
    post_ms_list: list[float] = []
    cancel_ms_list: list[float] = []
    cancel_queue_ms_list: list[float] = []
    trigger_to_cancel_ms_list: list[float] = []
    trigger_to_replace_ms_list: list[float] = []
    reprice_effective_delta_ticks_list: list[float] = []
    post_inflight_age_ms_list: list[float] = []
    cancel_inflight_age_ms_list: list[float] = []
    fills_up = 0
    fills_dn = 0
    reprice_skipped_noop = 0
    reprice_skipped_hysteresis = 0
    post_skipped_noop_replace = 0
    requote_signal_suppressed = 0
    post_inflight_timeout_recovers = 0
    cancel_inflight_timeout_recovers = 0
    max_churn_requote_attempts = 0
    max_churn_bypassed_noop = 0
    max_churn_bypassed_hysteresis = 0

    shutdown = asyncio.Event()
    requote_event = asyncio.Event()  # signalled for immediate requote

    # Risk window
    risk_active = False
    risk_dir: Optional[str] = None
    risk_until_ns: Optional[int] = None
    last_risk_trigger_ns: Optional[int] = None
    last_fill_ns: int = 0  # timestamp of most recent fill (for post-fill cooldown)

    # MODEL-FREE soft holds: per-side pause after adverse-side cancel
    soft_hold_until_ns: dict[str, int] = {"UP": 0, "DOWN": 0}

    # Cancel backoff: suppress cancel retries after transient 425 errors
    _CANCEL_BACKOFF_NS = int(100e6)  # 100 ms
    cancel_backoff_until_ns: dict[str, int] = {"UP": 0, "DOWN": 0}
    requote_reentry_allowed_ns: dict[str, int] = {"UP": 0, "DOWN": 0}
    last_requote_cancel_ns: dict[str, int] = {"UP": 0, "DOWN": 0}
    last_requote_signal_ns: dict[str, int] = {}
    post_inflight_timeout_ns = int(float(args.post_inflight_timeout_ms) * 1e6)
    cancel_inflight_timeout_ns = int(float(args.cancel_inflight_timeout_ms) * 1e6)
    max_churn_enabled = bool(getattr(args, "requote_max_churn", False))

    def signal_requote(reason: str, *, min_gap_ms: float = 0.0) -> None:
        nonlocal requote_signal_suppressed
        gap_ns = 0 if max_churn_enabled else int(float(min_gap_ms) * 1e6)
        t_now = now_ns()
        last_ns = int(last_requote_signal_ns.get(reason, 0))
        if gap_ns > 0 and (t_now - last_ns) < gap_ns:
            requote_signal_suppressed += 1
            return
        last_requote_signal_ns[reason] = t_now
        requote_event.set()

    roll_et = hour_start_et + timedelta(hours=1)

    # Fill-quality model (optional): widen spread when P(good) is low
    _fill_quality_model = None
    _fill_quality_scaler = None
    _fill_quality_threshold = 0.3
    _fill_quality_scaled = False
    if getattr(args, "fill_quality_enable", False) and _HAS_JOBLIB:
        _model_path = args.fill_quality_model_path or str(_script_dir / "fill_quality_model.joblib")
        _threshold_path = args.fill_quality_threshold_path or str(_script_dir / "fill_quality_threshold.json")
        if Path(_model_path).is_file():
            try:
                _fill_quality_model = _joblib.load(_model_path)
                with open(_threshold_path, "r", encoding="utf-8") as _f:
                    _th = json.load(_f)
                _fill_quality_threshold = float(_th.get("threshold", 0.3))
                _fill_quality_scaled = bool(_th.get("scaled", False))
                if _fill_quality_scaled:
                    _scaler_path = args.fill_quality_scaler_path or str(_script_dir / "fill_quality_scaler.joblib")
                    if Path(_scaler_path).is_file():
                        _fill_quality_scaler = _joblib.load(_scaler_path)
                jprint({
                    "event": "fill_quality_model_loaded",
                    "threshold": _fill_quality_threshold,
                    "scaled": _fill_quality_scaled,
                })
            except Exception as e:
                jprint({"event": "fill_quality_model_load_failed", "error": str(e)})
        else:
            jprint({
                "event": "fill_quality_model_not_found",
                "path": _model_path,
            })

    # Poly move models (optional): drive quote state decisions (keep/cancel/re-enter/suppress).
    # Prefer a single bundle file when --poly-fair-weights-file is set.
    _poly_model_up = None
    _poly_model_dn = None
    _poly_scaler_up = None
    _poly_scaler_dn = None
    _poly_imputer_up = None
    _poly_imputer_dn = None
    _poly_report_up: dict[str, Any] = {}
    _poly_report_dn: dict[str, Any] = {}
    _poly_feature_cols_up: list[str] = []
    _poly_feature_cols_dn: list[str] = []
    _poly_fair_runtime_enabled = False
    if getattr(args, "poly_fair_enable", False) and _HAS_JOBLIB and _HAS_NUMPY:
        try:
            _weights_file = str(getattr(args, "poly_fair_weights_file", "") or "").strip()
            if _weights_file:
                _raw_bundle = Path(_weights_file).expanduser()
                _bundle_candidates: list[Path] = []
                if _raw_bundle.is_absolute():
                    _bundle_candidates = [_raw_bundle]
                else:
                    # Prefer current working directory (EC2 run convention), then script dir.
                    _bundle_candidates = [
                        (Path.cwd() / _raw_bundle),
                        (_script_dir / _raw_bundle),
                    ]
                _bundle_path = _bundle_candidates[0].resolve()
                for _cand in _bundle_candidates:
                    if _cand.is_file():
                        _bundle_path = _cand.resolve()
                        break
                jprint({
                    "event": "poly_fair_bundle_probe",
                    "cwd": str(Path.cwd()),
                    "weights_file_arg": _weights_file,
                    "resolved_weights_file": str(_bundle_path),
                    "exists": bool(_bundle_path.is_file()),
                })
                _bundle = _joblib.load(_bundle_path)
                for _k in ("up", "down", "meta"):
                    if _k not in _bundle:
                        raise KeyError(f"bundle missing key: {_k}")
                for _side_name in ("up", "down"):
                    _side = _bundle.get(_side_name, {})
                    for _k in ("model", "scaler", "imputer", "report"):
                        if _k not in _side:
                            raise KeyError(f"bundle side '{_side_name}' missing key: {_k}")

                _up = _bundle["up"]
                _dn = _bundle["down"]
                _poly_model_up = _up["model"]
                _poly_model_dn = _dn["model"]
                _poly_scaler_up = _up["scaler"]
                _poly_scaler_dn = _dn["scaler"]
                _poly_imputer_up = _up["imputer"]
                _poly_imputer_dn = _dn["imputer"]
                _poly_report_up = dict(_up.get("report", {}))
                _poly_report_dn = dict(_dn.get("report", {}))
                _poly_feature_cols_up = list(_up.get("feature_cols", [])) or list(_poly_report_up.get("feature_cols", [])) or [
                    "binance_move", "binance_vol_ema", "poly_down_bid", "poly_up_bid", "secs_left"
                ]
                _poly_feature_cols_dn = list(_dn.get("feature_cols", [])) or list(_poly_report_dn.get("feature_cols", [])) or [
                    "binance_move", "binance_vol_ema", "poly_down_bid", "poly_up_bid", "secs_left"
                ]
                _poly_fair_runtime_enabled = True
                jprint({
                    "event": "poly_fair_models_loaded",
                    "weights_file": str(_bundle_path),
                    "bundle_schema_version": str(_bundle.get("schema_version", "unknown")),
                    "feature_cols_up": _poly_feature_cols_up,
                    "feature_cols_dn": _poly_feature_cols_dn,
                })
                jprint({
                    "event": "poly_fair_model_lifecycle",
                    "active_weights_file": str(_bundle_path),
                    "retrain_entrypoint": "mm_analysis/train_poly_move_pipeline.py",
                    "workflow": {
                        "input_db": "mm_rl_log.sqlite",
                        "build_dataset": "next-tick-within-window labels",
                        "train_scope": "all_markets_no_holdout",
                        "bundle_output": "poly_move_bundle.joblib",
                        "deploy": "replace bundle atomically; keep rollback copy",
                    },
                })
            else:
                _model_dir = Path(args.poly_fair_model_dir).expanduser()
                if not _model_dir.is_absolute():
                    _model_dir = (_script_dir / _model_dir).resolve()
                up_target = "delta_up_bid"
                dn_target = "delta_down_bid"
                _poly_model_up = _joblib.load(_model_dir / f"poly_move_model_{up_target}.joblib")
                _poly_model_dn = _joblib.load(_model_dir / f"poly_move_model_{dn_target}.joblib")
                _poly_scaler_up = _joblib.load(_model_dir / f"poly_move_scaler_{up_target}.joblib")
                _poly_scaler_dn = _joblib.load(_model_dir / f"poly_move_scaler_{dn_target}.joblib")
                _poly_imputer_up = _joblib.load(_model_dir / f"poly_move_imputer_{up_target}.joblib")
                _poly_imputer_dn = _joblib.load(_model_dir / f"poly_move_imputer_{dn_target}.joblib")

                _up_rep_path = _model_dir / f"poly_move_model_report_{up_target}.json"
                _dn_rep_path = _model_dir / f"poly_move_model_report_{dn_target}.json"
                if _up_rep_path.is_file():
                    _poly_report_up = json.loads(_up_rep_path.read_text(encoding="utf-8"))
                if _dn_rep_path.is_file():
                    _poly_report_dn = json.loads(_dn_rep_path.read_text(encoding="utf-8"))
                _poly_feature_cols_up = list(_poly_report_up.get("feature_cols", [])) or [
                    "binance_move", "binance_vol_ema", "poly_down_bid", "poly_up_bid", "secs_left"
                ]
                _poly_feature_cols_dn = list(_poly_report_dn.get("feature_cols", [])) or [
                    "binance_move", "binance_vol_ema", "poly_down_bid", "poly_up_bid", "secs_left"
                ]
                _poly_fair_runtime_enabled = True
                jprint({
                    "event": "poly_fair_models_loaded",
                    "model_dir": str(_model_dir),
                    "feature_cols_up": _poly_feature_cols_up,
                    "feature_cols_dn": _poly_feature_cols_dn,
                })
                jprint({
                    "event": "poly_fair_model_lifecycle",
                    "active_model_dir": str(_model_dir),
                    "retrain_entrypoint": "mm_analysis/train_poly_move_pipeline.py",
                    "workflow": {
                        "input_db": "mm_rl_log.sqlite",
                        "build_dataset": "next-tick-within-window labels",
                        "train_targets": ["delta_up_bid", "delta_down_bid"],
                        "validate": "market-chronological train/val/test",
                        "deploy": "replace artifacts after validation; keep rollback copy",
                    },
                })
        except Exception as e:
            _poly_fair_runtime_enabled = False
            jprint({
                "event": "poly_fair_models_load_failed",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(limit=4),
            })
    elif getattr(args, "poly_fair_enable", False):
        # poly_fair was requested, but required deps are unavailable.
        jprint({
            "event": "poly_fair_models_load_skipped",
            "reason": "missing_dependency",
            "has_joblib": bool(_HAS_JOBLIB),
            "has_numpy": bool(_HAS_NUMPY),
        })

    # ── Pricing helpers ──────────────────────────────────────────────────

    def poly_ok() -> bool:
        return (
            book_up.bid is not None
            and book_up.ask is not None
            and book_dn.bid is not None
            and book_dn.ask is not None
        )

    def book_is_stale(max_age_ms: float) -> bool:
        """Return True if either Poly book is older than *max_age_ms*."""
        t = now_ns()
        for bk in (book_up, book_dn):
            if bk.recv_ns is None:
                return True
            if (t - bk.recv_ns) / 1e6 > max_age_ms:
                return True
        return False

    def inventory_skew_ticks() -> tuple[float, float]:
        """
        Per-side skew in ticks based on NET inventory imbalance.
        Only the OVERWEIGHT side gets pulled back; the wanted side
        has zero skew so it stays aggressive to rebalance.

        imbalance > 0  →  long UP  →  pull UP back, leave DOWN alone
        imbalance < 0  →  long DOWN →  pull DOWN back, leave UP alone
        imbalance == 0 →  balanced  →  no skew on either side
        """
        s = float(args.inventory_skew_ticks)
        imbalance = inventory_up - inventory_dn
        up_skew = max(0.0, imbalance) * s   # pull back UP when long UP
        dn_skew = max(0.0, -imbalance) * s  # pull back DOWN when long DOWN
        return up_skew, dn_skew

    def vol_spread_extra_ticks() -> float:
        """Extra offset ticks to add to each side when vol is elevated."""
        mult = vol_tracker.spread_multiplier(float(args.bin_move_usd))
        # mult >= 1.0; extra = (mult - 1) * scale, capped
        raw = (mult - 1.0) * float(args.vol_spread_scale)
        cap = float(args.vol_spread_max_ticks)
        return min(raw, cap) if cap > 0 else raw

    # Feature order must match train_fill_model.FEATURE_COLS for fill-quality model
    _FILL_QUALITY_FEATURE_COLS = [
        "abs_bin_move", "bin_move",
        "up_spread", "dn_spread", "spread_avg",
        "inv_imbalance", "abs_inv_imbalance",
        "risk_active_int",
        "mid_distance",
        "trigger_BIN_SOFT", "trigger_REQUOTE", "trigger_BIN",
        "trigger_FILL", "trigger_OTHER",
        "secs_left",
        "late_market",
        "high_move",
    ]

    def fill_quality_extra_ticks() -> float:
        """Extra ticks to widen spread when fill-quality P(good) is below threshold."""
        if _fill_quality_model is None:
            return 0.0
        t_ns = now_ns()
        bin_move = get_worst_impulse(t_ns)
        bin_move_val = float(bin_move) if bin_move is not None else 0.0
        abs_bin_move = abs(bin_move_val)
        up_spread = (book_up.ask - book_up.bid) if (book_up.ask is not None and book_up.bid is not None) else None
        dn_spread = (book_dn.ask - book_dn.bid) if (book_dn.ask is not None and book_dn.bid is not None) else None
        spread_avg = (up_spread + dn_spread) / 2 if up_spread is not None and dn_spread is not None else None
        imb = inventory_up - inventory_dn
        inv_imbalance = float(imb)
        abs_inv_imbalance = abs(inv_imbalance)
        risk_active_int = 1 if risk_active else 0
        up_mid = (book_up.bid + book_up.ask) / 2 if (book_up.bid is not None and book_up.ask is not None) else None
        mid_distance = abs(up_mid - 0.5) if up_mid is not None else None
        secs_left = (roll_et - datetime.now(ET)).total_seconds()
        late_market = 1 if secs_left <= 900 else 0
        high_move = 1 if abs_bin_move >= 15 else 0
        # At quote time no fill trigger: all trigger dummies 0
        trigger_BIN_SOFT = trigger_REQUOTE = trigger_BIN = trigger_FILL = trigger_OTHER = 0
        row_dict = {
            "abs_bin_move": abs_bin_move, "bin_move": bin_move_val,
            "up_spread": up_spread, "dn_spread": dn_spread, "spread_avg": spread_avg,
            "inv_imbalance": inv_imbalance, "abs_inv_imbalance": abs_inv_imbalance,
            "risk_active_int": risk_active_int,
            "mid_distance": mid_distance,
            "trigger_BIN_SOFT": trigger_BIN_SOFT, "trigger_REQUOTE": trigger_REQUOTE,
            "trigger_BIN": trigger_BIN, "trigger_FILL": trigger_FILL, "trigger_OTHER": trigger_OTHER,
            "secs_left": secs_left, "late_market": late_market, "high_move": high_move,
        }
        row = []
        for col in _FILL_QUALITY_FEATURE_COLS:
            v = row_dict.get(col)
            if v is None:
                return 0.0
            row.append(v)
        if not _HAS_NUMPY or _NP is None:
            return 0.0
        X = _NP.array([row], dtype=_NP.float64)
        if _fill_quality_scaler is not None:
            X = _fill_quality_scaler.transform(X)
        try:
            p_good = _fill_quality_model.predict_proba(X)[0, 1]
        except Exception:
            return 0.0
        if p_good >= _fill_quality_threshold:
            return 0.0
        scale = float(getattr(args, "fill_quality_spread_scale", 2.0))
        max_ticks = float(getattr(args, "fill_quality_max_ticks", 3.0))
        raw = (_fill_quality_threshold - p_good) * scale
        return min(max(raw, 0.0), max_ticks)

    # Polymarket minimum order notional ($1).
    _POLY_MIN_NOTIONAL = 1.0

    def _notional_ok(px: float, sz: float) -> bool:
        """True if px * sz >= Polymarket minimum ($1)."""
        return px * sz >= _POLY_MIN_NOTIONAL - 1e-9

    def _bin_at_lag(t_ns: int, lag_ns: int) -> Optional[float]:
        """Look up BTC mid at t_ns - lag_ns from bin_history."""
        target = t_ns - lag_ns
        for hist_t, hist_p in reversed(bin_history):
            if hist_t <= target:
                return hist_p
        return None

    def get_bin_move_usd(t_ns: int) -> Optional[float]:
        """Model-free impulse over IMPULSE_MS."""
        if bin_mid is None:
            return None
        b_lag = _bin_at_lag(t_ns, int(IMPULSE_MS * 1e6))
        if b_lag is None:
            return None
        return float(bin_mid) - b_lag

    def get_bin_move_usd_medium(t_ns: int) -> Optional[float]:
        """Model-free impulse over IMPULSE_MEDIUM_MS (sustained-move horizon)."""
        if bin_mid is None:
            return None
        b_lag = _bin_at_lag(t_ns, int(IMPULSE_MEDIUM_MS * 1e6))
        if b_lag is None:
            return None
        return float(bin_mid) - b_lag

    def get_worst_impulse(t_ns: int) -> Optional[float]:
        """Return the impulse with the largest absolute magnitude across
        short (IMPULSE_MS) and medium (IMPULSE_MEDIUM_MS) horizons.
        Sign is preserved so the caller can determine direction."""
        bm_s = get_bin_move_usd(t_ns)
        bm_m = get_bin_move_usd_medium(t_ns)
        if bm_s is None and bm_m is None:
            return None
        if bm_s is None:
            return bm_m
        if bm_m is None:
            return bm_s
        return bm_s if abs(bm_s) >= abs(bm_m) else bm_m

    def adverse_selection_extra_ticks(side_name: str, bin_move_usd: float) -> float:
        """Continuous proportional pullback starting at soft_move_usd.

        For each unit of soft_move_usd of adverse impulse, pull back
        0.5 ticks, capped at 3.0 ticks.  This replaces the old coarse
        bucket system so that moves between $0.10-$2 are also covered.
        """
        soft = float(args.soft_move_usd)
        if soft <= 0:
            return 0.0
        # Determine adverse magnitude for this side
        if side_name == "UP":
            adv = max(0.0, -float(bin_move_usd))
        else:
            adv = max(0.0, float(bin_move_usd))

        if adv < soft:
            return 0.0
        return min(3.0, (adv / soft) * 0.5)
    
    def _poly_feature_value(col: str, *, t_ns: int, bin_move: float, secs_left: float) -> Optional[float]:
        if col == "binance_move":
            return float(bin_move)
        if col == "binance_vol_ema":
            return float(vol_tracker.ema)
        if col == "poly_down_bid":
            return float(book_dn.bid) if book_dn.bid is not None else None
        if col == "poly_up_bid":
            return float(book_up.bid) if book_up.bid is not None else None
        if col == "secs_left":
            return float(secs_left)
        # Unknown feature columns are unsupported at runtime.
        return None

    def _poly_predict_delta(
        *,
        model: Any,
        feature_cols: list[str],
        imputer: Any,
        scaler: Any,
        t_ns: int,
        bin_move: float,
        secs_left: float,
    ) -> Optional[float]:
        if not (_HAS_NUMPY and _NP is not None):
            return None
        row: list[float] = []
        for col in feature_cols:
            val = _poly_feature_value(col, t_ns=t_ns, bin_move=bin_move, secs_left=secs_left)
            if val is None:
                return None
            row.append(float(val))
        X = _NP.array([row], dtype=_NP.float64)
        if imputer is not None:
            X = imputer.transform(X)
        if scaler is not None:
            X = scaler.transform(X)
        pred = model.predict(X)
        if pred is None or len(pred) < 1:
            return None
        return float(pred[0])

    def _balance_only_side() -> Optional[str]:
        imbalance = inventory_up - inventory_dn
        if imbalance > float(args.poly_fair_balance_inventory_threshold):
            return "DOWN"
        if imbalance < -float(args.poly_fair_balance_inventory_threshold):
            return "UP"
        return None

    def compute_fair_prices(t_ns: int) -> Optional[dict[str, Any]]:
        """
        Model-driven quote-state signal engine.
        Returns a decision payload for keep/cancel/re-enter/suppress decisions.
        """
        secs_left = max(0.0, (roll_et - datetime.now(ET)).total_seconds())
        out: dict[str, Any] = {
            "model_enabled": bool(args.poly_fair_enable),
            "model_used": False,
            "gate_fail_no_quote_active": False,
            "balance_exception_active": False,
            "balance_only_side": "NONE",
            "gate_fail_reason": "",
            "pred_delta_up_bid": None,
            "pred_delta_down_bid": None,
            "pred_next_up_bid": None,
            "pred_next_down_bid": None,
            "side_state_decision_up": "keep",
            "side_state_decision_down": "keep",
            "model_requote_trigger_up": False,
            "model_requote_trigger_down": False,
            "model_requote_reason_up": "disabled",
            "model_requote_reason_down": "disabled",
            "best_bid_up_at_decision": float(book_up.bid) if book_up.bid is not None else None,
            "best_bid_down_at_decision": float(book_dn.bid) if book_dn.bid is not None else None,
            "up_active": True,
            "dn_active": True,
        }
        if not args.poly_fair_enable:
            return out
        if not _poly_fair_runtime_enabled:
            out["gate_fail_no_quote_active"] = True
            out["gate_fail_reason"] = "model_not_loaded"
            out["up_active"] = False
            out["dn_active"] = False
            return out
        if not poly_ok():
            out["gate_fail_no_quote_active"] = True
            out["gate_fail_reason"] = "book_not_ready"
            out["up_active"] = False
            out["dn_active"] = False
            return out

        gate_fail_reason = ""
        if secs_left <= float(args.poly_fair_min_secs_left):
            gate_fail_reason = "late_market_gate"
        elif bool(args.poly_fair_disable_high_vol) and float(vol_tracker.ema) >= float(args.poly_fair_high_vol_ema_usd):
            gate_fail_reason = "high_vol_gate"
        elif book_is_stale(float(args.stale_book_ms)):
            gate_fail_reason = "stale_book_gate"
        elif risk_active:
            gate_fail_reason = "risk_active_gate"

        # Model side-classification should react to the immediate impulse,
        # not the longer "worst" horizon which can stay elevated from drift.
        bin_impulse = get_bin_move_usd(t_ns)
        bin_move = float(bin_impulse) if bin_impulse is not None else 0.0
        pred_up = _poly_predict_delta(
            model=_poly_model_up,
            feature_cols=_poly_feature_cols_up,
            imputer=_poly_imputer_up,
            scaler=_poly_scaler_up,
            t_ns=t_ns,
            bin_move=bin_move,
            secs_left=secs_left,
        )
        pred_dn = _poly_predict_delta(
            model=_poly_model_dn,
            feature_cols=_poly_feature_cols_dn,
            imputer=_poly_imputer_dn,
            scaler=_poly_scaler_dn,
            t_ns=t_ns,
            bin_move=bin_move,
            secs_left=secs_left,
        )
        out["pred_delta_up_bid"] = pred_up
        out["pred_delta_down_bid"] = pred_dn
        out["pred_next_up_bid"] = (
            float(book_up.bid) + float(pred_up) if (book_up.bid is not None and pred_up is not None) else None
        )
        out["pred_next_down_bid"] = (
            float(book_dn.bid) + float(pred_dn) if (book_dn.bid is not None and pred_dn is not None) else None
        )
        out["model_used"] = pred_up is not None and pred_dn is not None

        if gate_fail_reason:
            out["gate_fail_no_quote_active"] = True
            out["gate_fail_reason"] = gate_fail_reason
            out["up_active"] = False
            out["dn_active"] = False
            if bool(args.poly_fair_balance_only_on_gate_fail):
                bal_side = _balance_only_side()
                if bal_side is not None:
                    out["balance_exception_active"] = True
                    out["balance_only_side"] = bal_side
                    out["up_active"] = bal_side == "UP"
                    out["dn_active"] = bal_side == "DOWN"
            out["side_state_decision_up"] = "suppress" if not out["up_active"] else "keep"
            out["side_state_decision_down"] = "suppress" if not out["dn_active"] else "keep"
            out["model_requote_reason_up"] = "gated_out"
            out["model_requote_reason_down"] = "gated_out"
            # #region agent log
            _debug_log(
                "H1",
                "market_maker_v2.py:1390",
                "compute_fair_prices gate fail",
                {
                    "gate_fail_reason": gate_fail_reason,
                    "secs_left": secs_left,
                    "vol_ema": float(vol_tracker.ema),
                    "book_up_bid": book_up.bid,
                    "book_dn_bid": book_dn.bid,
                    "risk_active": bool(risk_active),
                    "up_active": bool(out["up_active"]),
                    "dn_active": bool(out["dn_active"]),
                },
            )
            # #endregion
            return out

        # Side classification from worst impulse sign.
        direction_thr = float(args.poly_fair_direction_threshold_usd)
        favorable: Optional[str] = None
        if bin_move > direction_thr:
            favorable = "UP"
        elif bin_move < -direction_thr:
            favorable = "DOWN"
        adverse = "DOWN" if favorable == "UP" else ("UP" if favorable == "DOWN" else None)
        min_requote = float(args.poly_fair_min_requote_delta)

        def _side_adverse_pred(name: str, pred: Optional[float]) -> bool:
            if pred is None:
                return False
            # For resting bids, a predicted downward move means our quote is stale/high-risk.
            return float(pred) <= -min_requote

        up_adverse_pred = _side_adverse_pred("UP", pred_up)
        dn_adverse_pred = _side_adverse_pred("DOWN", pred_dn)
        up_small = pred_up is None or abs(float(pred_up)) < min_requote
        dn_small = pred_dn is None or abs(float(pred_dn)) < min_requote

        up_active = True
        dn_active = True
        up_decision = "keep"
        dn_decision = "keep"
        up_reason = "sub_1c_no_churn" if up_small else "neutral"
        dn_reason = "sub_1c_no_churn" if dn_small else "neutral"

        if favorable == "UP":
            if bool(args.poly_fair_favored_keep_enable):
                up_reason = "favorable_keep"
            if bool(args.poly_fair_adverse_cancel_enable) and dn_adverse_pred:
                dn_active = False
                dn_decision = "suppress"
                dn_reason = "adverse_move_ge_1c"
            elif not dn_small and pred_dn is not None and pred_dn > 0:
                dn_decision = "reenter"
                dn_reason = "no_longer_adverse"
        elif favorable == "DOWN":
            if bool(args.poly_fair_favored_keep_enable):
                dn_reason = "favorable_keep"
            if bool(args.poly_fair_adverse_cancel_enable) and up_adverse_pred:
                up_active = False
                up_decision = "suppress"
                up_reason = "adverse_move_ge_1c"
            elif not up_small and pred_up is not None and pred_up > 0:
                up_decision = "reenter"
                up_reason = "no_longer_adverse"
        else:
            # No directional impulse: keep both sides if gates pass.
            up_reason = "non_directional_keep"
            dn_reason = "non_directional_keep"

        # Inventory-balance override for model suppress decisions:
        # if imbalance is large, force the balancing side active even when the
        # model would suppress it.
        bal_side = _balance_only_side()
        if bal_side == "UP" and not up_active:
            up_active = True
            up_decision = "keep"
            up_reason = "inventory_balance_override"
        elif bal_side == "DOWN" and not dn_active:
            dn_active = True
            dn_decision = "keep"
            dn_reason = "inventory_balance_override"

        out["up_active"] = up_active
        out["dn_active"] = dn_active
        out["side_state_decision_up"] = up_decision
        out["side_state_decision_down"] = dn_decision
        out["model_requote_reason_up"] = up_reason
        out["model_requote_reason_down"] = dn_reason
        out["model_requote_trigger_up"] = bool(out["model_used"] and up_active and not up_small)
        out["model_requote_trigger_down"] = bool(out["model_used"] and dn_active and not dn_small)
        if (not up_active) or (not dn_active):
            # #region agent log
            _debug_log(
                "H3",
                "market_maker_v2.py:1469",
                "compute_fair_prices side suppression",
                {
                    "bin_move": float(bin_move),
                    "direction_thr": float(direction_thr),
                    "favorable": favorable,
                    "pred_up": pred_up,
                    "pred_dn": pred_dn,
                    "up_active": bool(up_active),
                    "dn_active": bool(dn_active),
                    "up_reason": up_reason,
                    "dn_reason": dn_reason,
                    "balance_side": _balance_only_side(),
                    "inv_up": float(inventory_up),
                    "inv_dn": float(inventory_dn),
                },
            )
            # #endregion
        return out
    
    def compute_pair_px() -> Optional[tuple[float, float]]:
        nonlocal cross_backoff_up, cross_backoff_dn
        nonlocal cross_backoff_last_ns_up, cross_backoff_last_ns_dn
        """Compute desired BUY prices for UP and DOWN tokens.

        Anchor: best ask minus 1 tick (tightest maker price).
        Offsets pull back from there.
        Sum constraint: inventory-weighted — the OVERWEIGHT side absorbs
        more of the excess so the WANTED side stays near its ask.
        """
        if not poly_ok():
            return None
        tick = float(args.tick)
        pmin = float(args.price_min)

        # ── Maker ceiling: 1 tick below the ask (a BUY at the ask crosses) ─
        # Also include crossing-backoff ticks so that after a post-only
        # rejection we automatically pull back further on the next attempt.
        t_now_bo = now_ns()
        # Auto-decay backoff after 5 s of no rejections
        if cross_backoff_up > 0 and (t_now_bo - cross_backoff_last_ns_up) > _CROSS_BACKOFF_DECAY_NS:
            cross_backoff_up = 0
        if cross_backoff_dn > 0 and (t_now_bo - cross_backoff_last_ns_dn) > _CROSS_BACKOFF_DECAY_NS:
            cross_backoff_dn = 0
        up_ceil = float(book_up.ask) - tick - cross_backoff_up * tick  # type: ignore[arg-type]
        dn_ceil = float(book_dn.ask) - tick - cross_backoff_dn * tick  # type: ignore[arg-type]

        # ── Start from maker ceiling (model-free) ──────────────────────
        ask_buf  = float(eff_ask_buffer_ticks) * tick
        base_off = float(eff_mid_offset_ticks) * tick
        vol_extra = vol_spread_extra_ticks() * tick
        model_extra = fill_quality_extra_ticks() * tick
        up_skew_t, dn_skew_t = inventory_skew_ticks()

        # MODEL-FREE: anchor strictly off the current maker ceiling.
        px_up_base = up_ceil - ask_buf - base_off
        px_dn_base = dn_ceil - ask_buf - base_off

        # ── Adverse-selection guard: pull back adverse side by extra ticks ──
        # Use the worst (largest-magnitude) impulse across both short and
        # medium horizons so sustained moves are also captured.
        # Proportional pullback starts at soft_move_usd (no dead zone).
        t_now = now_ns()
        bin_move = get_worst_impulse(t_now)
        extra_up = 0.0
        extra_dn = 0.0
        if bin_move is not None:
            extra_up = adverse_selection_extra_ticks("UP", float(bin_move))
            extra_dn = adverse_selection_extra_ticks("DOWN", float(bin_move))

        px_up_base -= extra_up * tick
        px_dn_base -= extra_dn * tick
        
        px_up = px_up_base - vol_extra - model_extra - up_skew_t * tick
        px_dn = px_dn_base - vol_extra - model_extra - dn_skew_t * tick

        # ── Momentum bias: zero-sum skew toward the winning side ─────
        # When a side's mid is above the momentum threshold (e.g. 0.70),
        # we boost that side's bid AND pull the opposite side back by the
        # same amount.  This keeps px_up + px_dn constant so the sum
        # constraint doesn't claw back the bonus from the winning side.
        # Disabled by default (threshold=1.0, bonus=0.0).
        m_thresh = float(args.momentum_threshold)
        m_bonus = float(args.momentum_bonus_ticks) * tick
        if m_bonus > 0 and m_thresh < 1.0:
            up_mid = (float(book_up.bid) + float(book_up.ask)) / 2.0  # type: ignore[arg-type]
            dn_mid = (float(book_dn.bid) + float(book_dn.ask)) / 2.0  # type: ignore[arg-type]
            if up_mid >= m_thresh and dn_mid < m_thresh:
                # UP is winning → boost UP, pull back DN (sum unchanged)
                px_up += m_bonus
                px_dn -= m_bonus
            elif dn_mid >= m_thresh and up_mid < m_thresh:
                # DN is winning → boost DN, pull back UP (sum unchanged)
                px_dn += m_bonus
                px_up -= m_bonus

        # Clamp into [price_min, ceiling]
        px_up = max(pmin, min(px_up, up_ceil))
        px_dn = max(pmin, min(px_dn, dn_ceil))

        px_up = normalize_price(px_up, args.price_min, args.price_max, tick)
        px_dn = normalize_price(px_dn, args.price_min, args.price_max, tick)

        # ── Enforce sum constraint ─────────────────────────────────────
        # Weights use NET imbalance — only the overweight side absorbs
        # excess; the wanted side stays as aggressive as possible.
        target = float(args.sum_target)
        total = px_up + px_dn

        imbalance = inventory_up - inventory_dn
        net_up = max(0.0, imbalance)   # excess UP tokens
        net_dn = max(0.0, -imbalance)  # excess DOWN tokens
        total_net = net_up + net_dn    # = abs(imbalance)

        if total > target + 1e-9:
            # Over target — push prices down.
            # The overweight side absorbs MORE of the reduction.
            excess = total - target

            if total_net > 1e-9:
                up_share = net_up / total_net
                dn_share = net_dn / total_net
            else:
                # Balanced → proportional by price
                up_share = px_up / total if total > 1e-12 else 0.5
                dn_share = px_dn / total if total > 1e-12 else 0.5

            raw_up = px_up - excess * up_share
            raw_dn = px_dn - excess * dn_share

            # Clamp to price_min; if one side hits floor, shift
            # remaining excess to the other side.
            raw_up = max(pmin, raw_up)
            raw_dn = max(pmin, raw_dn)
            still_over = (raw_up + raw_dn) - target
            if still_over > 1e-9:
                # One side hit floor, give remainder to the other
                if raw_up > pmin + 1e-9:
                    raw_up = max(pmin, raw_up - still_over)
                elif raw_dn > pmin + 1e-9:
                    raw_dn = max(pmin, raw_dn - still_over)

            px_up = normalize_price(raw_up, args.price_min, args.price_max, tick)
            px_dn = normalize_price(raw_dn, args.price_min, args.price_max, tick)

            # Rounding may leave us 1 tick over – trim the overweight side
            if px_up + px_dn > target + 1e-9:
                if net_up > net_dn or (
                    abs(net_up - net_dn) < 1e-9 and px_up >= px_dn
                ):
                    px_up = normalize_price(
                        px_up - tick, args.price_min, args.price_max, tick
                    )
                else:
                    px_dn = normalize_price(
                        px_dn - tick, args.price_min, args.price_max, tick
                    )

        elif total < target - 1e-9:
            # Under target — boost prices up.
            # The WANTED side (less inventory) gets more boost.
            shortfall = target - total

            if total_net > 1e-9:
                # Inverse: wanted side gets all the boost
                up_boost = net_dn / total_net   # UP boost ∝ excess DOWN
                dn_boost = net_up / total_net   # DOWN boost ∝ excess UP
            else:
                up_boost = px_up / total if total > 1e-12 else 0.5
                dn_boost = px_dn / total if total > 1e-12 else 0.5

            raw_up = px_up + shortfall * up_boost
            raw_dn = px_dn + shortfall * dn_boost

            # Cap at ceiling (never cross the book).
            # Do NOT redistribute leftover — if the wanted side is already
            # at its ceiling and can't absorb the boost, the remainder is
            # simply lost.  Pushing it onto the overweight side would undo
            # the inventory skew (the exact bug we're preventing).
            raw_up = min(raw_up, up_ceil)
            raw_dn = min(raw_dn, dn_ceil)

            px_up = normalize_price(raw_up, args.price_min, args.price_max, tick)
            px_dn = normalize_price(raw_dn, args.price_min, args.price_max, tick)

            # Rounding may push us back over – final safety trim
            if px_up + px_dn > target + 1e-9:
                if px_up >= px_dn:
                    px_up = normalize_price(
                        px_up - tick, args.price_min, args.price_max, tick
                    )
                else:
                    px_dn = normalize_price(
                        px_dn - tick, args.price_min, args.price_max, tick
                    )

        return px_up, px_dn

    def risk_replace_px(side: str, base_up: float, base_dn: float) -> float:
        tick = float(args.tick)
        extra = float(args.risk_extra_ticks) * tick
        if side == "UP":
            return normalize_price(
                base_up - extra, args.price_min, args.price_max, tick
            )
        return normalize_price(
            base_dn - extra, args.price_min, args.price_max, tick
        )

    # ── Queue helpers ────────────────────────────────────────────────────

    def enqueue_cancel(
        *,
        prio: int,
        trigger_kind: str,
        side: str,
        oid: str,
        trigger_ns: int,
        replace_px: Optional[float],
        replace_size: Optional[float] = None,
        step_id: Optional[int] = None,
    ) -> None:
        nonlocal seq
        S = UP if side == "UP" else DN
        S.cancel_in_flight = True
        S.cancel_in_flight_since_ns = now_ns()
        seq += 1
        snap = pending_cancel_snapshot.get(str(oid))
        req = CancelReq(
            prio=prio,
            seq=seq,
            t_enqueue_ns=now_ns(),
            t_trigger_ns=int(trigger_ns),
            trigger_kind=trigger_kind,
            side=side,
            order_id=oid,
            replace_px=replace_px,
            replace_size=replace_size,
            step_id=step_id,
            prev_px=(snap[0] if snap is not None else None),
            prev_post_ns=(snap[1] if snap is not None else None),
            prev_size_ordered=(float(snap[2]) if snap is not None else 0.0),
            prev_last_seen_matched=(float(snap[3]) if snap is not None else 0.0),
        )
        cancel_q.put_nowait((req.prio, req.seq, req))
        logger.emit_order_event(
            market_id=market_id,
            step_id=step_id,
            kind="enqueue_cancel",
            side=side,
            order_id=oid,
            px=replace_px,
            size=replace_size,
            trigger_kind=trigger_kind,
            trigger_ns=int(trigger_ns),
            enqueue_ns=req.t_enqueue_ns,
            latency_ms=None,
            queue_ms=None,
            ok=None,
            payload={"replace_px": replace_px, "replace_size": replace_size},
        )

    def enqueue_post(
        *,
        prio: int,
        trigger_kind: str,
        side: str,
        token_id: str,
        px: float,
        trigger_ns: int,
        size: Optional[float] = None,
        prev_px: Optional[float] = None,
        step_id: Optional[int] = None,
    ) -> None:
        nonlocal seq
        S = UP if side == "UP" else DN
        # ── Duplicate guard: if a post is already queued / in-flight for this
        # side, skip to avoid orphan orders on the exchange.  The quote_manager
        # or cancel_worker will retry on the next cycle with fresh prices.
        if S.post_in_flight:
            # #region agent log
            _debug_log(
                "H6",
                "market_maker_v2.py:1796",
                "enqueue_post skipped duplicate post_in_flight",
                {
                    "side": side,
                    "trigger_kind": trigger_kind,
                    "px": float(px),
                    "has_order": bool(S.order_id),
                    "post_in_flight": bool(S.post_in_flight),
                    "cancel_in_flight": bool(S.cancel_in_flight),
                },
            )
            # #endregion
            jprint({
                "event": "enqueue_post_skipped_dup",
                "side": side,
                "trigger_kind": trigger_kind,
                "px": px,
            })
            return
        S.post_in_flight = True  # prevent duplicate enqueues until post_worker clears
        S.post_in_flight_since_ns = now_ns()
        seq += 1
        req = PostReq(
            prio=prio,
            seq=seq,
            t_enqueue_ns=now_ns(),
            trigger_ns=int(trigger_ns),
            trigger_kind=trigger_kind,
            side=side,
            token_id=token_id,
            px=float(px),
            size=float(size) if size is not None else float(args.size),
            prev_px=(float(prev_px) if prev_px is not None else None),
            step_id=step_id,
        )
        post_q.put_nowait((req.prio, req.seq, req))
        logger.emit_order_event(
            market_id=market_id,
            step_id=step_id,
            kind="enqueue_post",
            side=side,
            order_id=None,
            px=float(px),
            size=float(req.size),
            trigger_kind=trigger_kind,
            trigger_ns=int(trigger_ns),
            enqueue_ns=req.t_enqueue_ns,
            latency_ms=None,
            queue_ms=None,
            ok=None,
            payload={"token_id": token_id},
        )

    # ── Cancel worker (2 instances for parallelism) ──────────────────────

    async def cancel_worker() -> None:
        nonlocal inventory_up, inventory_dn, last_fill_ns
        while not shutdown.is_set():
            try:
                _, _, req = await asyncio.wait_for(cancel_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            t_start = now_ns()
            q_ms = (t_start - req.t_enqueue_ns) / 1e6
            cancel_queue_ms_list.append(q_ms)

            S = UP if req.side == "UP" else DN
            cancel_ok = False

            try:
                ms = await cancel_one(loop, cancel_ex, client, req.order_id)
                cancel_ms_list.append(ms)
                trigger_to_cancel_ms_list.append(
                    (now_ns() - req.t_trigger_ns) / 1e6
                )
                cancel_ok = True

                jprint({
                    "event": "cancel_done",
                    "side": req.side,
                    "oid": req.order_id,
                    "prio": "HI" if req.prio == 0 else "LO",
                    "trigger_kind": req.trigger_kind,
                    "cancel_ms": ms,
                    "queue_ms": cancel_queue_ms_list[-1],
                    "trigger_to_cancel_ms": trigger_to_cancel_ms_list[-1],
                })
                logger.emit_order_event(
                    market_id=market_id,
                    step_id=req.step_id,
                    kind="cancel_done",
                    side=req.side,
                    order_id=req.order_id,
                    px=req.replace_px,
                    size=req.replace_size,
                    trigger_kind=req.trigger_kind,
                    trigger_ns=req.t_trigger_ns,
                    enqueue_ns=req.t_enqueue_ns,
                    latency_ms=ms,
                    queue_ms=q_ms,
                    ok=True,
                    payload={
                        "trigger_to_cancel_ms": trigger_to_cancel_ms_list[-1]
                        if trigger_to_cancel_ms_list else None,
                    },
                )
            except Exception as e:
                _err_str = str(e)
                jprint({
                    "event": "cancel_error",
                    "side": req.side,
                    "oid": req.order_id,
                    "err": _err_str,
                })
                logger.emit_order_event(
                    market_id=market_id,
                    step_id=req.step_id,
                    kind="cancel_error",
                    side=req.side,
                    order_id=req.order_id,
                    px=req.replace_px,
                    size=req.replace_size,
                    trigger_kind=req.trigger_kind,
                    trigger_ns=req.t_trigger_ns,
                    enqueue_ns=req.t_enqueue_ns,
                    latency_ms=None,
                    queue_ms=q_ms,
                    ok=False,
                    payload={"err": _err_str},
                )

                # ── Transient 425 backoff: suppress cancel retries ────
                _is_transient = "425" in _err_str or "service not ready" in _err_str.lower()
                if _is_transient:
                    cancel_backoff_until_ns[req.side] = now_ns() + _CANCEL_BACKOFF_NS
                    jprint({
                        "event": "cancel_backoff_425",
                        "side": req.side,
                        "oid": req.order_id,
                        "backoff_ms": _CANCEL_BACKOFF_NS / 1e6,
                    })

            if not cancel_ok:
                # Cancel may not have gone through — old order could still be
                # live on the exchange.  Restore tracking so we don't orphan it
                # and let quote_manager retry on the next cycle.
                if S.order_id is None and not S.post_in_flight:
                    S.order_id = req.order_id
                    S.px = req.prev_px
                    S.post_ns = req.prev_post_ns
                    S.size_ordered = req.prev_size_ordered
                    S.last_seen_matched = req.prev_last_seen_matched
                    # Remove from audit — we're restoring, not abandoning
                    pending_fill_audit.pop(req.order_id, None)
                    pending_cancel_snapshot.pop(req.order_id, None)
                S.cancel_in_flight = False
                S.cancel_in_flight_since_ns = 0
                jprint({
                    "event": "cancel_restore",
                    "side": req.side,
                    "oid": req.order_id,
                    "restored_px": S.px,
                    "restored_post_ns": S.post_ns,
                    "restored_size_ordered": S.size_ordered,
                    "restored_last_matched": S.last_seen_matched,
                })
                continue  # do NOT post replacement

            # NOTE: keep cancel_in_flight = True throughout fill reconciliation
            # so that quote_manager cannot race us by enqueuing a duplicate post
            # while we await get_order().  We clear it AFTER the replacement is
            # enqueued (or requote_event is set).
            pending_cancel_snapshot.pop(req.order_id, None)

            # ── Post-cancel fill reconciliation ─────────────────────
            # The order may have been (partially) filled between
            # detach_order_for_audit and the cancel completing on-
            # exchange.  Reconcile now so inventory is correct BEFORE
            # we enqueue any replacement.
            cancel_fill_delta = 0.0
            audit_entry = pending_fill_audit.pop(req.order_id, None)
            if audit_entry is not None:
                audit_side, audit_prev_matched = audit_entry
                try:
                    od = await get_order(loop, read_ex, client, req.order_id)
                    final_matched = size_matched(od)
                except Exception:
                    final_matched = audit_prev_matched  # safe: assume no new fills
                cancel_fill_delta = final_matched - audit_prev_matched
                if cancel_fill_delta > 1e-9:
                    if audit_side == "UP":
                        inventory_up += cancel_fill_delta
                    else:
                        inventory_dn += cancel_fill_delta
                    last_fill_ns = now_ns()
                    recon_sid = log_step(
                        "cancel_fill_reconcile",
                        action_id=-1,
                        action={
                            "oid": req.order_id,
                            "side": audit_side,
                            "credited": cancel_fill_delta,
                        },
                    )
                    jprint({
                        "event": "cancel_fill_reconcile",
                        "side": audit_side,
                        "oid": req.order_id,
                        "prev_matched": audit_prev_matched,
                        "final_matched": final_matched,
                        "credited": cancel_fill_delta,
                        "inventory_up": inventory_up,
                        "inventory_dn": inventory_dn,
                    })
                    logger.emit_fill(
                        market_id=market_id,
                        step_id=recon_sid,
                        fill_kind="cancel_fill_reconcile",
                        side=audit_side,
                        order_id=str(req.order_id),
                        fill_qty=float(cancel_fill_delta),
                        total_matched=float(final_matched),
                        remaining=0.0,
                        px=req.prev_px,
                        inv_up=float(inventory_up),
                        inv_dn=float(inventory_dn),
                        payload={
                            "trigger_kind": req.trigger_kind,
                            "bin_mid": bin_mid,
                        },
                    )

            # ── Schedule replacement on post lane ───────────────────
            # Adjust replacement size for fills that happened during
            # the cancel window (relevant for PARTIAL replacements).
            repl_size = req.replace_size
            if cancel_fill_delta > 1e-9 and repl_size is not None:
                repl_size = max(0.0, repl_size - cancel_fill_delta)
            eff_size = repl_size if repl_size is not None else float(args.size)
            if req.replace_px is not None and eff_size > 0 and _notional_ok(float(req.replace_px), eff_size):
                enqueue_post(
                    prio=0,
                    trigger_kind=req.trigger_kind,
                    side=req.side,
                    token_id=S.token_id,
                    px=float(req.replace_px),
                    trigger_ns=req.t_trigger_ns,
                    size=repl_size,  # None → default (args.size); adjusted for partials
                    prev_px=req.prev_px,
                    step_id=req.step_id,
                )
                S.cancel_in_flight = False  # safe: post_in_flight now guards the side
                S.cancel_in_flight_since_ns = 0
            else:
                # No replace price, size consumed by fills, or notional
                # too low → signal requote so quote_manager re-posts.
                S.cancel_in_flight = False  # clear before requote so quote_manager can act
                S.cancel_in_flight_since_ns = 0
                requote_event.set()

    # ── Post worker (2 instances for parallelism) ────────────────────────

    async def post_worker() -> None:
        nonlocal cross_backoff_up, cross_backoff_dn
        nonlocal cross_backoff_last_ns_up, cross_backoff_last_ns_dn
        nonlocal post_skipped_noop_replace, max_churn_bypassed_noop
        while not shutdown.is_set():
            try:
                _, _, req = await asyncio.wait_for(post_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            S = UP if req.side == "UP" else DN

            # ── SOFT HOLD guard: never post during hold window ─────────────
            if now_ns() < soft_hold_until_ns[req.side]:
                S.post_in_flight = False
                jprint({
                    "event": "post_skipped_soft_hold",
                    "side": req.side,
                    "hold_until_ns": soft_hold_until_ns[req.side],
                    "trigger_kind": req.trigger_kind,
                    "px": req.px,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_skip_soft_hold", side=req.side, order_id=None,
                    px=req.px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None, queue_ms=None,
                    ok=False, payload={"hold_until_ns": soft_hold_until_ns[req.side]},
                )
                # Let quote_manager re-enqueue after hold expires
                continue

            # ── Duplicate guard: another post for this side already landed ──
            if S.order_id is not None:
                S.post_in_flight = False
                jprint({
                    "event": "post_skipped_dup",
                    "side": req.side,
                    "existing_oid": S.order_id,
                    "trigger_kind": req.trigger_kind,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_skip_dup", side=req.side, order_id=S.order_id,
                    px=req.px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None, queue_ms=None,
                    ok=False, payload={"existing_oid": S.order_id},
                )
                continue

            # ── Stale-data guard: skip post if book data is too old ─────────
            if book_is_stale(float(args.stale_book_ms)):
                S.post_in_flight = False
                jprint({
                    "event": "post_skipped_stale_book",
                    "side": req.side,
                    "trigger_kind": req.trigger_kind,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_skip_stale_book", side=req.side, order_id=None,
                    px=req.px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None, queue_ms=None,
                    ok=False, payload={},
                )
                continue

            # ── BIN-specific: skip if risk still active ─────────────────
            actual_px = req.px
            if req.trigger_kind == "BIN" and risk_active:
                S.post_in_flight = False
                jprint({
                    "event": "post_skipped_risk_active",
                    "side": req.side,
                    "trigger_kind": req.trigger_kind,
                    "original_px": req.px,
                    "risk_dir": risk_dir,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_skip_risk_active", side=req.side,
                    order_id=None, px=req.px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None,
                    queue_ms=None, ok=False,
                    payload={"risk_dir": risk_dir, "bin_mid": bin_mid},
                )
                requote_event.set()
                continue

            # ── Fresh-price re-check (ALL trigger kinds) ─────────────
            # Between cancel-enqueue and now the market may have moved.
            # Recompute and use the more conservative (lower BUY) of
            # original and fresh to avoid quoting into an adverse move
            # or crossing the book.
            q = compute_pair_px()
            if q is not None:
                fresh_up, fresh_dn = q
                fresh_px = fresh_up if req.side == "UP" else fresh_dn
                prev_actual = actual_px
                actual_px = min(actual_px, fresh_px)
                actual_px = normalize_price(
                    actual_px, args.price_min, args.price_max,
                    float(args.tick),
                )
                if abs(actual_px - prev_actual) > 1e-9:
                    jprint({
                        "event": "post_repriced_fresh",
                        "side": req.side,
                        "trigger_kind": req.trigger_kind,
                        "original_px": prev_actual,
                        "fresh_px": fresh_px,
                        "actual_px": actual_px,
                    })

            # ── JIT Binance impulse guard: skip posts into fresh moves ─
            # This is the last line of defence before sending the order.
            # Check BOTH short and medium horizons — if EITHER is adverse
            # and exceeds soft_move_usd, skip the post, extend the soft
            # hold, and wake quote_manager to retry later.
            bm_now = get_worst_impulse(now_ns())
            if bm_now is not None:
                soft_thr = float(args.soft_move_usd)
                adverse = False
                if req.side == "UP":
                    # BTC down (negative move) hurts UP bids
                    adverse = bm_now <= -soft_thr
                else:
                    # BTC up (positive move) hurts DOWN bids
                    adverse = bm_now >= soft_thr

                # #region agent log
                _debug_log("H1", "market_maker_v2.py:2191", "jit_guard", {"side":req.side,"bm_now":bm_now,"soft_thr":soft_thr,"adverse":adverse,"trigger_kind":req.trigger_kind,"actual_px":actual_px})
                # #endregion

                if adverse:
                    hold_ns = int(float(args.soft_hold_ms) * 1e6)
                    new_hold = now_ns() + hold_ns
                    if soft_hold_until_ns[req.side] < new_hold:
                        soft_hold_until_ns[req.side] = new_hold

                    S.post_in_flight = False
                    jprint({
                        "event": "post_skipped_bin_impulse",
                        "side": req.side,
                        "trigger_kind": req.trigger_kind,
                        "bm_now": bm_now,
                        "soft_move_usd": soft_thr,
                        "hold_until_ns": soft_hold_until_ns[req.side],
                    })
                    logger.emit_order_event(
                        market_id=market_id,
                        step_id=req.step_id,
                        kind="post_skip_bin_impulse",
                        side=req.side,
                        order_id=None,
                        px=actual_px,
                        size=req.size,
                        trigger_kind=req.trigger_kind,
                        trigger_ns=req.trigger_ns,
                        enqueue_ns=req.t_enqueue_ns,
                        latency_ms=None,
                        queue_ms=None,
                        ok=False,
                        payload={
                            "bm_now": bm_now,
                            "soft_move_usd": soft_thr,
                            "hold_until_ns": soft_hold_until_ns[req.side],
                        },
                    )
                    requote_event.set()
                    continue

            # ── Pre-flight guards: crossing & notional ─────────────────
            tick = float(args.tick)
            bo = cross_backoff_up if req.side == "UP" else cross_backoff_dn
            safety_ticks = 1 + bo  # 1 base + backoff

            # Use WS best ask only (no REST preflight fetch in hot path).
            book = book_up if req.side == "UP" else book_dn
            best_ask = float(book.ask) if book.ask is not None else None  # type: ignore[arg-type]

            if best_ask is not None and actual_px >= best_ask - safety_ticks * tick + 1e-9:
                # Would cross or sit too close to the ask → clamp
                actual_px = normalize_price(
                    best_ask - safety_ticks * tick, args.price_min, args.price_max, tick
                )

            if (
                req.trigger_kind == "REQUOTE"
                and req.prev_px is not None
                and abs(float(actual_px) - float(req.prev_px)) < tick - 1e-12
            ):
                if max_churn_enabled:
                    max_churn_bypassed_noop += 1
                else:
                    post_skipped_noop_replace += 1
                    requote_reentry_allowed_ns[req.side] = max(
                        int(requote_reentry_allowed_ns.get(req.side, 0)),
                        now_ns() + int(float(args.requote_min_dwell_ms) * 1e6),
                    )
                    S.post_in_flight = False
                    jprint({
                        "event": "post_skipped_noop_replace",
                        "side": req.side,
                        "trigger_kind": req.trigger_kind,
                        "prev_px": req.prev_px,
                        "req_px": req.px,
                        "actual_px": actual_px,
                        "dwell_ms": float(args.requote_min_dwell_ms),
                    })
                    signal_requote("post_noop_replace", min_gap_ms=float(args.requote_signal_min_gap_ms))
                    continue

            notional = actual_px * req.size
            if notional < _POLY_MIN_NOTIONAL - 1e-9:
                S.post_in_flight = False
                jprint({
                    "event": "post_skipped_low_notional",
                    "side": req.side,
                    "px": actual_px,
                    "size": req.size,
                    "notional": notional,
                    "trigger_kind": req.trigger_kind,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_skip_low_notional", side=req.side,
                    order_id=None, px=actual_px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None,
                    queue_ms=None, ok=False,
                    payload={"notional": notional},
                )
                continue

            try:
                oid, ms, resp = await post_postonly(
                    loop,
                    sign_ex,
                    post_ex,
                    client,
                    token_id=req.token_id,
                    px=actual_px,
                    size=req.size,
                    live=bool(args.live),
                )
            except Exception as e:
                S.post_in_flight = False
                err_lower = str(e).lower()
                is_crossing = "cross" in err_lower or "post only" in err_lower or "post_order rejected" in err_lower

                # ── Bump crossing backoff so next attempt uses wider margin ──
                if is_crossing:
                    t_reject = now_ns()
                    if req.side == "UP":
                        cross_backoff_up = min(cross_backoff_up + 1, _MAX_CROSS_BACKOFF)
                        cross_backoff_last_ns_up = t_reject
                    else:
                        cross_backoff_dn = min(cross_backoff_dn + 1, _MAX_CROSS_BACKOFF)
                        cross_backoff_last_ns_dn = t_reject

                jprint({
                    "event": "post_error",
                    "side": req.side,
                    "trigger_kind": req.trigger_kind,
                    "px": actual_px,
                    "err": str(e),
                    "is_crossing": is_crossing,
                    "cross_backoff": cross_backoff_up if req.side == "UP" else cross_backoff_dn,
                })
                logger.emit_order_event(
                    market_id=market_id, step_id=req.step_id,
                    kind="post_error", side=req.side, order_id=None,
                    px=actual_px, size=req.size,
                    trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                    enqueue_ns=req.t_enqueue_ns, latency_ms=None, queue_ms=None,
                    ok=False, payload={"err": str(e), "original_px": req.px,
                                       "is_crossing": is_crossing},
                )

                if is_crossing:
                    # Brief cooldown so Poly WS book data can catch up
                    # before the next attempt; avoids hammering the API.
                    await asyncio.sleep(0.15)

                # Wake quote_manager to repost with fresh (backed-off) prices
                requote_event.set()
                continue

            post_ms_list.append(ms)

            if oid is None:
                S.post_in_flight = False
                jprint({
                    "event": "post_dry_run",
                    "side": req.side,
                    "px": actual_px,
                    "size": req.size,
                    "trigger_kind": req.trigger_kind,
                })
                continue

            S.order_id = oid
            S.px = actual_px
            S.post_ns = now_ns()
            S.size_ordered = req.size
            S.last_seen_matched = 0.0
            S.post_in_flight = False

            # #region agent log
            _debug_log("H1", "market_maker_v2.py:2344", "post_sent", {"side":req.side,"px":actual_px,"trigger_kind":req.trigger_kind,"bm_at_post":get_bin_move_usd(now_ns()),"size":req.size})
            # #endregion

            # ── Successful post: only reset crossing backoff if we posted
            # at/near the ORIGINAL desired price (not a backed-off fallback).
            # If the actual_px was clamped down by safety_ticks > 1, the
            # underlying crossing condition may still exist — let the 5-second
            # time-based decay handle it instead.  This prevents the
            # cancel→post→reject→reset→cancel loop.
            _was_clamped_down = (
                req.px is not None
                and actual_px < float(req.px) - tick * 0.5
            )
            if not _was_clamped_down:
                if req.side == "UP":
                    cross_backoff_up = 0
                else:
                    cross_backoff_dn = 0

            if req.trigger_kind in (
                "BIN",
                "FILL",
                "REQUOTE",
                "FILL_REPOST",
                "RISK_CLEAR",
                "STALE",
                "PARTIAL",
                "PARTIAL_REPOST",
            ):
                trigger_to_replace_ms_list.append(
                    (now_ns() - req.trigger_ns) / 1e6
                )

            jprint({
                "event": "post",
                "side": req.side,
                "px": actual_px,
                "size": req.size,
                "post_ms": ms,
                "oid": oid,
                "prio": "HI" if req.prio == 0 else "LO",
                "trigger_kind": req.trigger_kind,
                "resp": resp,
                "trigger_to_replace_ms": (
                    trigger_to_replace_ms_list[-1]
                    if trigger_to_replace_ms_list
                    else None
                ),
            })
            logger.emit_order_event(
                market_id=market_id, step_id=req.step_id,
                kind="post_done", side=req.side, order_id=oid,
                px=actual_px, size=req.size,
                trigger_kind=req.trigger_kind, trigger_ns=req.trigger_ns,
                enqueue_ns=req.t_enqueue_ns, latency_ms=ms, queue_ms=None,
                ok=True, payload={"resp": resp, "original_px": req.px},
            )

    # ── Polymarket WebSocket (with reconnect) ────────────────────────────

    async def poly_ws() -> None:
        token_ids = [up_id, down_id]
        while not shutdown.is_set():
            try:
                async with websockets.connect(
                    POLY_WS,
                    ssl=SSL_CTX,
                    ping_interval=15,
                    ping_timeout=20,
                    max_queue=1,
                    compression=None,
                    close_timeout=3,
                    open_timeout=10,
                ) as ws:
                    await ws.send(
                        json.dumps({"type": "market", "assets_ids": token_ids})
                    )
                    jprint({"event": "poly_ws_subscribed", "slug": slug})
                    async for raw in ws:
                        if shutdown.is_set():
                            return
                        recv = now_ns()
                        msg = _json_loads(raw)
                        items = msg if isinstance(msg, list) else [msg]
                        tob_changed = False
                        for d in items:
                            if not isinstance(d, dict):
                                continue
                            if str(d.get("event_type", "")).lower() != "book":
                                continue
                            aid = str(d.get("asset_id", ""))
                            bids = d.get("bids") or d.get("buys") or []
                            asks = d.get("asks") or d.get("sells") or []
                            bb = best_from_levels(bids, True)
                            ba = best_from_levels(asks, False)
                            if bb is None or ba is None:
                                continue
                            if aid == up_id:
                                prev_bid, prev_ask = book_up.bid, book_up.ask
                                book_up.bid = float(bb)
                                book_up.ask = float(ba)
                                book_up.recv_ns = int(recv)
                                if book_up.bid != prev_bid or book_up.ask != prev_ask:
                                    tob_changed = True
                                logger.emit_tick(
                                    market_id=market_id,
                                    source="poly",
                                    symbol="UP",
                                    bid=float(bb),
                                    ask=float(ba),
                                    mid=0.5 * (float(bb) + float(ba)),
                                    payload={"asset_id": aid},
                                    sampler="poly",
                                )
                            elif aid == down_id:
                                prev_bid, prev_ask = book_dn.bid, book_dn.ask
                                book_dn.bid = float(bb)
                                book_dn.ask = float(ba)
                                book_dn.recv_ns = int(recv)
                                if book_dn.bid != prev_bid or book_dn.ask != prev_ask:
                                    tob_changed = True
                                logger.emit_tick(
                                    market_id=market_id,
                                    source="poly",
                                    symbol="DOWN",
                                    bid=float(bb),
                                    ask=float(ba),
                                    mid=0.5 * (float(bb) + float(ba)),
                                    payload={"asset_id": aid},
                                    sampler="poly",
                                )
                        if poly_ok():
                            poly_ready.set()
                            # Only wake quote_manager when top-of-book actually
                            # changed — deep-level updates don't affect our prices
                            # and would just churn queue position for nothing.
                            if tob_changed:
                                signal_requote("poly_tob", min_gap_ms=float(args.requote_signal_min_gap_ms))
            except Exception as e:
                jprint({"event": "poly_ws_error", "err": str(e)})
                if not shutdown.is_set():
                    await asyncio.sleep(1.0)

    # ── Binance WebSocket (with reconnect across both URLs) ──────────────

    async def bin_ws() -> None:
        nonlocal bin_mid, bin_ref
        nonlocal risk_active, risk_dir, risk_until_ns, last_risk_trigger_ns
        nonlocal bin_history
        urls = [
            f"{BINANCE_WS_PRIMARY}/{BINANCE_SYMBOL}@bookTicker",
            f"{BINANCE_WS_FALLBACK}/{BINANCE_SYMBOL}@bookTicker",
        ]
        while not shutdown.is_set():
            for url in urls:
                if shutdown.is_set():
                    return
                try:
                    async with websockets.connect(
                        url,
                        ssl=SSL_CTX,
                        ping_interval=15,
                        ping_timeout=20,
                        max_queue=1,
                        compression=None,
                        close_timeout=3,
                        open_timeout=10,
                    ) as ws:
                        jprint({"event": "bin_ws_connected", "url": url})
                        async for raw in ws:
                            if shutdown.is_set():
                                return
                            t = now_ns()
                            d = _json_loads(raw)
                            b = float(d["b"])
                            a = float(d["a"])
                            new_mid = 0.5 * (b + a)

                            logger.emit_tick(
                                market_id=market_id,
                                source="binance",
                                symbol=BINANCE_SYMBOL,
                                bid=b,
                                ask=a,
                                mid=new_mid,
                                payload={"raw_b": d.get("b"), "raw_a": d.get("a")},
                                sampler="binance",
                            )

                            # Update volatility tracker on every tick
                            vol_tracker.update(new_mid)

                            bin_mid = new_mid
                            
                            # ── Update Binance price history for impulse calc ──
                            t = now_ns()
                            bin_history.append((t, float(new_mid)))
                            # Keep only recent history (last MAX_BIN_HISTORY entries)
                            if len(bin_history) > MAX_BIN_HISTORY:
                                bin_history.pop(0)
                            
                            if not bin_ready.is_set():
                                bin_ready.set()

                            # ── MODEL-FREE: latency-aware adverse-side cancel ──
                            # Check BOTH short and medium impulse horizons.
                            # Trigger if EITHER exceeds the soft threshold.
                            bm = get_worst_impulse(t)
                            if (not risk_active) and bm is not None:
                                soft = float(args.soft_move_usd)
                                hold_ns = int(float(args.soft_hold_ms) * 1e6)

                                # BTC down → UP bids are adverse → cancel UP
                                if bm <= -soft:
                                    if soft_hold_until_ns["UP"] < now_ns() + hold_ns:
                                        soft_hold_until_ns["UP"] = now_ns() + hold_ns
                                    if (UP.order_id and not UP.cancel_in_flight
                                            and now_ns() >= cancel_backoff_until_ns["UP"]):
                                        UP.cancel_in_flight = True
                                        oid = detach_order_for_audit(UP)
                                        if oid:
                                            enqueue_cancel(
                                                prio=0,
                                                trigger_kind="BIN_SOFT",
                                                side="UP",
                                                oid=str(oid),
                                                trigger_ns=int(t),
                                                replace_px=None,
                                                step_id=None,
                                            )
                                        signal_requote("bin_soft_up", min_gap_ms=float(args.requote_signal_min_gap_ms))

                                # BTC up → DOWN bids are adverse → cancel DOWN
                                elif bm >= soft:
                                    if soft_hold_until_ns["DOWN"] < now_ns() + hold_ns:
                                        soft_hold_until_ns["DOWN"] = now_ns() + hold_ns
                                    if (DN.order_id and not DN.cancel_in_flight
                                            and now_ns() >= cancel_backoff_until_ns["DOWN"]):
                                        DN.cancel_in_flight = True
                                        oid = detach_order_for_audit(DN)
                                        if oid:
                                            enqueue_cancel(
                                                prio=0,
                                                trigger_kind="BIN_SOFT",
                                                side="DOWN",
                                                oid=str(oid),
                                                trigger_ns=int(t),
                                                replace_px=None,
                                                step_id=None,
                                            )
                                        signal_requote("bin_soft_down", min_gap_ms=float(args.requote_signal_min_gap_ms))

                            if bin_ref is None:
                                bin_ref = float(bin_mid)

                            # ── Risk-timer clear (checked every tick) ────
                            if (
                                risk_active
                                and risk_until_ns is not None
                                and t >= risk_until_ns
                            ):
                                risk_active = False
                                risk_dir = None
                                risk_until_ns = None
                                last_risk_trigger_ns = None
                                bin_ref = float(bin_mid)
                                jprint({
                                    "event": "risk_clear_timer",
                                    "t_ns": int(t),
                                    "bin_ref": bin_ref,
                                })
                                # Trigger immediate requote
                                requote_event.set()

                            if risk_active:
                                continue

                            # ── Adverse-selection requote trigger ────────
                            # If impulse (worst across both horizons) exceeds
                            # asel threshold, wake quote_manager so
                            # compute_pair_px adds extra ticks on the adverse side.
                            bm = get_worst_impulse(t)
                            if bm is not None:
                                # Use soft_move_usd as the trigger since the
                                # proportional pullback now starts there.
                                if abs(float(bm)) >= float(args.soft_move_usd):
                                    signal_requote("bin_impulse", min_gap_ms=float(args.requote_signal_min_gap_ms))

                            # ── Decay bin_ref toward current mid ─────────
                            decay = float(args.bin_ref_decay)
                            if 0.0 < decay < 1.0:
                                bin_ref = (
                                    bin_ref * (1.0 - decay) + bin_mid * decay
                                )

                            # ── HARD risk trigger: impulse-based ─────────
                            bm_hard = get_worst_impulse(t)
                            if bm_hard is None:
                                continue

                            hard = float(args.hard_move_usd)
                            if abs(bm_hard) < hard:
                                continue

                            risk_active = True
                            last_risk_trigger_ns = int(t)
                            risk_until_ns = int(
                                t + int(float(args.risk_hold_ms) * 1e6)
                            )
                            risk_dir = "up" if bm_hard > 0 else "down"

                            jprint({
                                "event": "risk_trigger",
                                "dir": risk_dir,
                                "move_usd": bm_hard,
                                "vol_ema": vol_tracker.ema,
                                "t_ns": int(t),
                            })

                            risk_sid = log_step(
                                "risk_trigger",
                                action_id=0,
                                action={"dir": risk_dir, "move_usd": bm_hard},
                            )

                            q = compute_pair_px()
                            if q is None:
                                # Reset ref so we don't re-trigger immediately
                                bin_ref = float(bin_mid)
                                continue
                            base_up, base_dn = q

                            # Cancel BOTH sides – not just adverse.
                            # Replace prices already include the adverse-
                            # selection guard from compute_pair_px().
                            for cancel_S, cancel_name, cancel_repl in [
                                (UP, "UP", risk_replace_px("UP", base_up, base_dn)),
                                (DN, "DOWN", risk_replace_px("DOWN", base_up, base_dn)),
                            ]:
                                if (cancel_S.order_id and not cancel_S.cancel_in_flight
                                        and now_ns() >= cancel_backoff_until_ns[cancel_name]):
                                    cancel_S.cancel_in_flight = True
                                    old_oid = detach_order_for_audit(cancel_S)
                                    enqueue_cancel(
                                        prio=0,
                                        trigger_kind="BIN",
                                        side=cancel_name,
                                        oid=str(old_oid),
                                        trigger_ns=int(t),
                                        replace_px=cancel_repl,
                                        step_id=risk_sid,
                                    )

                            # Anchor ref to current mid after triggering
                            bin_ref = float(bin_mid)

                except Exception as e:
                    jprint({
                        "event": "bin_ws_error",
                        "url": url,
                        "err": str(e),
                    })
                    if not shutdown.is_set():
                        await asyncio.sleep(1.0)

    # ── Risk-timer watchdog ──────────────────────────────────────────────
    # Independent timer that clears risk even if Binance WS goes quiet.

    async def risk_timer_watchdog() -> None:
        nonlocal risk_active, risk_dir, risk_until_ns
        nonlocal last_risk_trigger_ns, bin_ref
        while not shutdown.is_set():
            await asyncio.sleep(0.010)  # 10 ms check interval
            if not risk_active or risk_until_ns is None:
                continue
            if now_ns() >= risk_until_ns:
                risk_active = False
                risk_dir = None
                risk_until_ns = None
                last_risk_trigger_ns = None
                if bin_mid is not None:
                    bin_ref = float(bin_mid)
                jprint({"event": "risk_clear_watchdog", "bin_ref": bin_ref})
                requote_event.set()

    # ── Fill detection (handles partial + full fills) ────────────────────

    async def fill_watcher() -> None:
        nonlocal inventory_up, inventory_dn, fills_up, fills_dn, last_fill_ns
        if not bool(args.live):
            jprint({"event": "fill_watcher_disabled", "reason": "dry_run"})
            return
        while not shutdown.is_set():
            await asyncio.sleep(float(args.fill_poll_ms) / 1000.0)

            # ── 1. Audit cancelled orders for unaccounted partial fills ──
            # When any code-path cancels an order (BIN, REQUOTE,
            # ROLLOVER), it may not have polled get_order recently.  We do
            # one final check here so those fills aren't lost.
            audit_ids = list(pending_fill_audit.keys())
            for oid in audit_ids:
                side_name, prev_matched = pending_fill_audit[oid]
                od = await get_order(loop, read_ex, client, oid)
                final_matched = size_matched(od)
                new_fill = final_matched - prev_matched
                if new_fill > 1e-9:
                    if side_name == "UP":
                        inventory_up += new_fill
                    else:
                        inventory_dn += new_fill
                    audit_sid = log_step(
                        "audit_fill",
                        action_id=-1,
                        action={"oid": oid, "side": side_name, "credited": new_fill},
                    )
                    jprint({
                        "event": "audit_fill",
                        "side": side_name,
                        "oid": oid,
                        "prev_matched": prev_matched,
                        "final_matched": final_matched,
                        "credited": new_fill,
                    })
                    logger.emit_fill(
                        market_id=market_id,
                        step_id=audit_sid,
                        fill_kind="audit_fill",
                        side=side_name,
                        order_id=str(oid),
                        fill_qty=float(new_fill),
                        total_matched=float(final_matched),
                        remaining=0.0,
                        px=None,
                        inv_up=float(inventory_up),
                        inv_dn=float(inventory_dn),
                        payload={
                            "bin_mid": bin_mid,
                            "up_bid": book_up.bid, "up_ask": book_up.ask,
                            "dn_bid": book_dn.bid, "dn_ask": book_dn.ask,
                        },
                    )
                    # Inventory changed → signal requote so quotes adjust
                    requote_event.set()
                del pending_fill_audit[oid]

            # ── 2. Check active orders for incremental fills ─────────────
            for S, other, side_name, other_name in [
                (UP, DN, "UP", "DOWN"),
                (DN, UP, "DOWN", "UP"),
            ]:
                if not S.order_id:
                    continue

                od = await get_order(loop, read_ex, client, S.order_id)
                current_matched = size_matched(od)
                remaining = S.size_ordered - current_matched

                # Log every order poll for RL training data
                logger.emit_order_poll(
                    market_id=market_id,
                    step_id=current_step_id,
                    side=side_name,
                    order_id=str(S.order_id),
                    px=float(S.px) if S.px is not None else None,
                    size_ordered=float(S.size_ordered),
                    size_matched=float(current_matched),
                    remaining=float(remaining),
                    payload={
                        "raw": od,
                        "bin_mid": bin_mid,
                        "up_bid": book_up.bid, "up_ask": book_up.ask,
                        "dn_bid": book_dn.bid, "dn_ask": book_dn.ask,
                    },
                )

                new_fill = current_matched - S.last_seen_matched

                if new_fill < 1e-9:
                    continue  # no new fill since last check

                # ── Credit the new fill increment to inventory ───────────
                S.last_seen_matched = current_matched
                if side_name == "UP":
                    inventory_up += new_fill
                else:
                    inventory_dn += new_fill

                is_full = remaining < float(args.min_remaining_size)

                if is_full:
                    # ── FULL FILL (or remainder too small to bother) ─────
                    fill_sid = log_step(
                        "fill",
                        action_id=-1,
                        action={"oid": S.order_id, "side": side_name, "new_fill": new_fill},
                    )
                    jprint({
                        "event": "fill",
                        "side": side_name,
                        "oid": S.order_id,
                        "matched": current_matched,
                        "remaining": remaining,
                        "type": "full",
                        "inventory_up": inventory_up,
                        "inventory_dn": inventory_dn,
                    })
                    logger.emit_fill(
                        market_id=market_id,
                        step_id=fill_sid,
                        fill_kind="fill",
                        side=side_name,
                        order_id=str(S.order_id),
                        fill_qty=float(new_fill),
                        total_matched=float(current_matched),
                        remaining=float(remaining),
                        px=float(S.px) if S.px is not None else None,
                        inv_up=float(inventory_up),
                        inv_dn=float(inventory_dn),
                        payload={
                            "bin_mid": bin_mid,
                            "up_bid": book_up.bid, "up_ask": book_up.ask,
                            "dn_bid": book_dn.bid, "dn_ask": book_dn.ask,
                        },
                    )
                    if side_name == "UP":
                        fills_up += 1
                    else:
                        fills_dn += 1

                    # Fills already credited above – just clear state
                    clear_order_no_audit(S)
                    last_fill_ns = now_ns()

                    # Cancel the other side so it doesn't get picked off at a
                    # stale price.  DB analysis: temporal mismatch between
                    # paired fills is the #1 cause of pair cost > $1.
                    if args.cancel_other_on_fill:
                        if other.order_id and not other.cancel_in_flight:
                            other.cancel_in_flight = True
                            oid = detach_order_for_audit(other)
                            if oid:
                                enqueue_cancel(
                                    prio=0,
                                    trigger_kind="FILL",
                                    side=other_name,
                                    oid=str(oid),
                                    trigger_ns=now_ns(),
                                    replace_px=None,
                                    step_id=fill_sid,
                                )

                    # Do NOT immediately repost the filled side aggressively.
                    # Let quote_manager recompute both sides with inventory skew
                    # and adverse-selection guard applied.
                    # #region agent log
                    _debug_log(
                        "H5",
                        "market_maker_v2.py:2883",
                        "full fill observed",
                        {
                            "side": side_name,
                            "inventory_up": float(inventory_up),
                            "inventory_dn": float(inventory_dn),
                            "other_side_has_order": bool(other.order_id),
                            "book_up_bid": book_up.bid,
                            "book_dn_bid": book_dn.bid,
                        },
                    )
                    # #endregion
                    requote_event.set()

                else:
                    # ── PARTIAL FILL ─────────────────────────────────────
                    partial_sid = log_step(
                        "partial_fill",
                        action_id=-1,
                        action={"oid": S.order_id, "side": side_name, "new_fill": new_fill},
                    )
                    jprint({
                        "event": "partial_fill",
                        "side": side_name,
                        "oid": S.order_id,
                        "new_fill": new_fill,
                        "total_matched": current_matched,
                        "remaining": remaining,
                        "inventory_up": inventory_up,
                        "inventory_dn": inventory_dn,
                    })
                    logger.emit_fill(
                        market_id=market_id,
                        step_id=partial_sid,
                        fill_kind="partial_fill",
                        side=side_name,
                        order_id=str(S.order_id),
                        fill_qty=float(new_fill),
                        total_matched=float(current_matched),
                        remaining=float(remaining),
                        px=float(S.px) if S.px is not None else None,
                        inv_up=float(inventory_up),
                        inv_dn=float(inventory_dn),
                        payload={
                            "bin_mid": bin_mid,
                            "up_bid": book_up.bid, "up_ask": book_up.ask,
                            "dn_bid": book_dn.bid, "dn_ask": book_dn.ask,
                        },
                    )

                    if args.partial_fill_reprice:
                        # Cancel the remaining resting qty and re-post at an
                        # inventory-skew-adjusted price with reduced size.
                        # But SKIP if the new price is within 1 tick of the
                        # old price — no point destroying queue position for
                        # the same price.
                        _partial_old_px = float(S.px) if S.px is not None else None
                        tick = float(args.tick)

                        q = compute_pair_px()
                        if q is not None:
                            px_up, px_dn = q
                            new_px = px_up if side_name == "UP" else px_dn
                            _partial_delta = abs(float(new_px) - _partial_old_px) if _partial_old_px is not None else None

                            # #region agent log
                            _debug_log("H2", "market_maker_v2.py:2967", "partial_reprice", {"side":side_name,"old_px":_partial_old_px,"new_px":float(new_px),"delta_ticks":round(_partial_delta / tick, 2) if _partial_delta is not None else None,"remaining":float(remaining),"skipped":(_partial_delta is not None and _partial_delta < tick - 1e-12)})
                            # #endregion

                            if _partial_delta is not None and _partial_delta < tick - 1e-12:
                                # Price unchanged (< 1 tick) — skip cancel+replace,
                                # preserve queue position. Do not force requote;
                                # quote_manager will react to actual book moves.
                                pass
                            else:
                                # Price changed meaningfully — cancel+replace.
                                S.cancel_in_flight = True
                                old_oid = detach_order_for_audit(S)
                                enqueue_cancel(
                                    prio=0,
                                    trigger_kind="PARTIAL",
                                    side=side_name,
                                    oid=str(old_oid),
                                    trigger_ns=now_ns(),
                                    replace_px=float(new_px),
                                    replace_size=float(remaining),
                                    step_id=partial_sid,
                                )
                        else:
                            # Can't compute price – cancel without replace,
                            # quote_manager will re-post later.
                            S.cancel_in_flight = True
                            old_oid = detach_order_for_audit(S)
                            enqueue_cancel(
                                prio=0,
                                trigger_kind="PARTIAL",
                                side=side_name,
                                oid=str(old_oid),
                                trigger_ns=now_ns(),
                                replace_px=None,
                                step_id=partial_sid,
                            )
                    else:
                        # Leave the resting order alone and do not force an
                        # immediate requote; only actual book changes should
                        # trigger repricing/requoting.
                        pass

    # ── Quote manager ────────────────────────────────────────────────────

    async def quote_manager() -> None:
        nonlocal bin_ref
        nonlocal reprice_skipped_noop, reprice_skipped_hysteresis
        nonlocal post_inflight_timeout_recovers, cancel_inflight_timeout_recovers
        nonlocal max_churn_requote_attempts, max_churn_bypassed_noop, max_churn_bypassed_hysteresis
        await bin_ready.wait()
        await poly_ready.wait()

        if bin_mid is not None:
            bin_ref = float(bin_mid)

        # Select initial action
        apply_action(choose_action_id())

        jprint({
            "event": "quote_start",
            "bin_ref": bin_ref,
            "sum_target": args.sum_target,
            "mid_offset_ticks": eff_mid_offset_ticks,
            "ask_buffer_ticks": eff_ask_buffer_ticks,
            "post_inflight_timeout_ms": float(args.post_inflight_timeout_ms),
            "cancel_inflight_timeout_ms": float(args.cancel_inflight_timeout_ms),
            "requote_max_churn": bool(max_churn_enabled),
            "momentum_threshold": float(args.momentum_threshold),
            "momentum_bonus_ticks": float(args.momentum_bonus_ticks),
            "action_id": last_action_id,
            "vol_ema": vol_tracker.ema,
            "poly_fair": {
                "enable": bool(args.poly_fair_enable),
                "mode": str(getattr(args, "poly_fair_mode", "active")),
                "model_loaded": bool(_poly_fair_runtime_enabled),
                "model_dir": str(getattr(args, "poly_fair_model_dir", "")),
                "weights_file": str(getattr(args, "poly_fair_weights_file", "")),
                "min_secs_left": float(args.poly_fair_min_secs_left),
                "disable_high_vol": bool(args.poly_fair_disable_high_vol),
                "high_vol_ema_usd": float(args.poly_fair_high_vol_ema_usd),
                "min_requote_delta": float(args.poly_fair_min_requote_delta),
                "max_stale_ticks": float(args.poly_fair_max_stale_ticks),
                "balance_only_on_gate_fail": bool(args.poly_fair_balance_only_on_gate_fail),
            },
        })

        last_tick_ns = 0
        model_reentry_allowed_ns: dict[str, int] = {"UP": 0, "DOWN": 0}

        # Initial posts
        init_sid = log_step(
            "quote_manager_init",
            action_id=last_action_id,
            action={
                "size": float(args.size),
                "mode": "NORMAL",
                **last_action_params,
            },
        )
        if bool(args.poly_fair_enable):
            init_dec = compute_fair_prices(now_ns()) or {}
            _poly_last_decision.update(init_dec)
            init_up_active = bool(init_dec.get("up_active", False))
            init_dn_active = bool(init_dec.get("dn_active", False))
            init_px_up = (
                normalize_price(float(book_up.bid), args.price_min, args.price_max, float(args.tick))
                if init_up_active and book_up.bid is not None
                else None
            )
            init_px_dn = (
                normalize_price(float(book_dn.bid), args.price_min, args.price_max, float(args.tick))
                if init_dn_active and book_dn.bid is not None
                else None
            )
            if init_px_up is not None and _notional_ok(init_px_up, float(args.size)):
                enqueue_post(
                    prio=1,
                    trigger_kind="INIT",
                    side="UP",
                    token_id=UP.token_id,
                    px=init_px_up,
                    trigger_ns=now_ns(),
                    step_id=init_sid,
                )
            if init_px_dn is not None and _notional_ok(init_px_dn, float(args.size)):
                enqueue_post(
                    prio=1,
                    trigger_kind="INIT",
                    side="DOWN",
                    token_id=DN.token_id,
                    px=init_px_dn,
                    trigger_ns=now_ns(),
                    step_id=init_sid,
                )
        else:
            q = compute_pair_px()
            if q is not None:
                px_up, px_dn = q
                if _notional_ok(px_up, float(args.size)):
                    enqueue_post(
                        prio=1,
                        trigger_kind="INIT",
                        side="UP",
                        token_id=UP.token_id,
                        px=px_up,
                        trigger_ns=now_ns(),
                        step_id=init_sid,
                    )
                if _notional_ok(px_dn, float(args.size)):
                    enqueue_post(
                        prio=1,
                        trigger_kind="INIT",
                        side="DOWN",
                        token_id=DN.token_id,
                        px=px_dn,
                        trigger_ns=now_ns(),
                        step_id=init_sid,
                    )

        while not shutdown.is_set():
            # Wait for either the periodic interval OR an immediate requote
            # signal (from fill, risk-clear, cancel-without-replace, etc.)
            # We track whether this wakeup is signal-driven so we only
            # cancel existing orders when a real event occurred (Binance
            # move, Poly book move, fill).  Timer-only wakeups only
            # re-post empty sides to avoid destroying queue position.
            signal_driven = False
            try:
                await asyncio.wait_for(
                    requote_event.wait(),
                    timeout=float(args.quote_interval_ms) / 1000.0,
                )
                requote_event.clear()
                signal_driven = True
            except asyncio.TimeoutError:
                pass

            t = now_ns()
            if (t - last_tick_ns) < int(float(args.quote_min_gap_ms) * 1e6):
                continue

            # ── Post-fill cooldown: avoid requoting into adverse moves ───
            cooldown_ns = int(float(args.fill_cooldown_ms) * 1e6)
            if last_fill_ns > 0 and cooldown_ns > 0:
                remaining_ns = last_fill_ns + cooldown_ns - t
                if remaining_ns > 0:
                    await asyncio.sleep(remaining_ns / 1e9)
                    t = now_ns()  # refresh after sleep

            last_tick_ns = t

            # ── Refresh action for this cycle ─────────────────────────
            apply_action(choose_action_id())

            # Determine desired prices and side activity.
            skip_reason: Optional[str] = None
            side_active_up = True
            side_active_dn = True
            model_requote_trigger_up = False
            model_requote_trigger_dn = False
            model_decision = None

            poly_mode = str(getattr(args, "poly_fair_mode", "active")).strip().lower()
            if bool(args.poly_fair_enable):
                model_decision = compute_fair_prices(t) or {}
                _poly_last_decision.update(model_decision)
                if poly_mode == "shadow":
                    # Log-only mode: keep legacy quoting behavior unchanged.
                    q = compute_pair_px()
                    if q is None:
                        skip_reason = "no_prices_shadow"
                        px_up, px_dn = None, None
                    else:
                        px_up, px_dn = q
                    side_active_up = True
                    side_active_dn = True
                    model_requote_trigger_up = False
                    model_requote_trigger_dn = False
                elif poly_mode == "advisory":
                    # Advisory mode: gate suppression + churn blocking; no side suppression logic.
                    q = compute_pair_px()
                    if q is None:
                        skip_reason = "no_prices_advisory"
                        px_up, px_dn = None, None
                    else:
                        px_up, px_dn = q
                    side_active_up = True
                    side_active_dn = True
                    model_requote_trigger_up = bool(model_decision.get("model_requote_trigger_up", False))
                    model_requote_trigger_dn = bool(model_decision.get("model_requote_trigger_down", False))
                else:
                    side_active_up = bool(model_decision.get("up_active", False))
                    side_active_dn = bool(model_decision.get("dn_active", False))
                    model_requote_trigger_up = bool(model_decision.get("model_requote_trigger_up", False))
                    model_requote_trigger_dn = bool(model_decision.get("model_requote_trigger_down", False))
                    px_up = (
                        normalize_price(float(book_up.bid), args.price_min, args.price_max, float(args.tick))
                        if side_active_up and book_up.bid is not None
                        else None
                    )
                    px_dn = (
                        normalize_price(float(book_dn.bid), args.price_min, args.price_max, float(args.tick))
                        if side_active_dn and book_dn.bid is not None
                        else None
                    )
                # #region agent log
                _debug_log(
                    "H2",
                    "market_maker_v2.py:3175",
                    "quote_manager model decision applied",
                    {
                        "poly_mode": poly_mode,
                        "skip_reason_pre": skip_reason,
                        "book_up_bid": book_up.bid,
                        "book_dn_bid": book_dn.bid,
                        "side_active_up": bool(side_active_up),
                        "side_active_dn": bool(side_active_dn),
                        "px_up": px_up,
                        "px_dn": px_dn,
                        "model_used": bool(model_decision.get("model_used", False)),
                        "gate_fail_reason": model_decision.get("gate_fail_reason"),
                    },
                )
                # #endregion
                if bool(model_decision.get("gate_fail_no_quote_active", False)) and not bool(
                    model_decision.get("balance_exception_active", False)
                ):
                    skip_reason = str(model_decision.get("gate_fail_reason", "gated_out"))
                elif px_up is None and px_dn is None:
                    skip_reason = "no_active_best_bid"
            else:
                if risk_active:
                    skip_reason = "risk_active"
                elif book_is_stale(float(args.stale_book_ms)):
                    skip_reason = "stale_book"
                if skip_reason is not None:
                    log_step(
                        "quote_skip",
                        action_id=last_action_id,
                        action={"skip_reason": skip_reason, **last_action_params},
                    )
                    continue
                q = compute_pair_px()
                if q is None:
                    log_step(
                        "quote_skip",
                        action_id=last_action_id,
                        action={"skip_reason": "no_prices", **last_action_params},
                    )
                    continue
                px_up, px_dn = q

            # Model gating can request no-quote. Enforce by canceling live orders.
            if skip_reason is not None:
                for side_obj, name in [(UP, "UP"), (DN, "DOWN")]:
                    if side_obj.order_id and not side_obj.cancel_in_flight:
                        side_obj.cancel_in_flight = True
                        oid = detach_order_for_audit(side_obj)
                        enqueue_cancel(
                            prio=1,
                            trigger_kind="MODEL_SUPPRESS" if bool(args.poly_fair_enable) else "SKIP",
                            side=name,
                            oid=str(oid),
                            trigger_ns=now_ns(),
                            replace_px=None,
                            replace_size=None,
                            step_id=None,
                        )
                log_step(
                    "quote_skip",
                    action_id=last_action_id,
                    action={
                        "skip_reason": skip_reason,
                        "poly_decision": model_decision if bool(args.poly_fair_enable) else None,
                        **last_action_params,
                    },
                )
                continue

            # Notional check: each active side must meet Polymarket's $1 minimum.
            up_ok = bool(px_up is not None and side_active_up and _notional_ok(float(px_up), float(args.size)))
            dn_ok = bool(px_dn is not None and side_active_dn and _notional_ok(float(px_dn), float(args.size)))
            if bool(args.poly_fair_enable):
                # In model mode, inactive/notional-failing side is canceled; active side continues.
                if side_active_up and not up_ok:
                    side_active_up = False
                if side_active_dn and not dn_ok:
                    side_active_dn = False
                if (not up_ok and px_up is not None) or (not dn_ok and px_dn is not None):
                    # #region agent log
                    _debug_log(
                        "H4",
                        "market_maker_v2.py:3240",
                        "notional check deactivated side",
                        {
                            "px_up": px_up,
                            "px_dn": px_dn,
                            "size": float(args.size),
                            "up_ok": bool(up_ok),
                            "dn_ok": bool(dn_ok),
                            "side_active_up": bool(side_active_up),
                            "side_active_dn": bool(side_active_dn),
                        },
                    )
                    # #endregion
                if not side_active_up and not side_active_dn:
                    for side_obj, name in [(UP, "UP"), (DN, "DOWN")]:
                        if side_obj.order_id and not side_obj.cancel_in_flight:
                            side_obj.cancel_in_flight = True
                            oid = detach_order_for_audit(side_obj)
                            enqueue_cancel(
                                prio=1,
                                trigger_kind="MODEL_NOTIONAL",
                                side=name,
                                oid=str(oid),
                                trigger_ns=now_ns(),
                                replace_px=None,
                                replace_size=None,
                                step_id=None,
                            )
                    log_step(
                        "quote_skip",
                        action_id=last_action_id,
                        action={
                            "skip_reason": "model_notional_mismatch",
                            "desired_px_up": px_up,
                            "desired_px_dn": px_dn,
                            "up_ok": up_ok,
                            "dn_ok": dn_ok,
                            "poly_decision": model_decision,
                            **last_action_params,
                        },
                    )
                    continue
            elif not (up_ok and dn_ok):
                # Legacy one-sided notional rebalance behavior.
                imbalance = inventory_up - inventory_dn
                rebalance_up = up_ok and not dn_ok and imbalance < -1e-9
                rebalance_dn = dn_ok and not up_ok and imbalance > 1e-9

                if rebalance_up or rebalance_dn:
                    untradeable_side = DN if rebalance_up else UP
                    untradeable_name = "DOWN" if rebalance_up else "UP"
                    if untradeable_side.order_id and not untradeable_side.cancel_in_flight:
                        untradeable_side.cancel_in_flight = True
                        oid = detach_order_for_audit(untradeable_side)
                        enqueue_cancel(
                            prio=1,
                            trigger_kind="NOTIONAL_REBAL",
                            side=untradeable_name,
                            oid=str(oid),
                            trigger_ns=now_ns(),
                            replace_px=None,
                            replace_size=None,
                            step_id=None,
                        )
                else:
                    for side_obj, name in [(UP, "UP"), (DN, "DOWN")]:
                        if side_obj.order_id and not side_obj.cancel_in_flight:
                            side_obj.cancel_in_flight = True
                            oid = detach_order_for_audit(side_obj)
                            enqueue_cancel(
                                prio=1,
                                trigger_kind="NOTIONAL",
                                side=name,
                                oid=str(oid),
                                trigger_ns=now_ns(),
                                replace_px=None,
                                replace_size=None,
                                step_id=None,
                            )
                    log_step(
                        "quote_skip",
                        action_id=last_action_id,
                        action={
                            "skip_reason": "notional_mismatch",
                            "desired_px_up": px_up,
                            "desired_px_dn": px_dn,
                            "up_ok": up_ok,
                            "dn_ok": dn_ok,
                            "imbalance": imbalance,
                            **last_action_params,
                        },
                    )
                    continue

            tick = float(args.tick)

            # ── Decision-point step ───────────────────────────────────
            action_dict = {
                "desired_px_up": px_up,
                "desired_px_dn": px_dn,
                "size": float(args.size),
                "mode": "NORMAL",
                "skip_reason": skip_reason,
                "signal_driven": signal_driven,
                "bin_move_usd": get_bin_move_usd(now_ns()),
                "poly_model_enabled": bool(args.poly_fair_enable),
                "pred_delta_up_bid": (_poly_last_decision.get("pred_delta_up_bid") if bool(args.poly_fair_enable) else None),
                "pred_delta_down_bid": (_poly_last_decision.get("pred_delta_down_bid") if bool(args.poly_fair_enable) else None),
                "pred_next_up_bid": (_poly_last_decision.get("pred_next_up_bid") if bool(args.poly_fair_enable) else None),
                "pred_next_down_bid": (_poly_last_decision.get("pred_next_down_bid") if bool(args.poly_fair_enable) else None),
                "side_state_decision_up": (_poly_last_decision.get("side_state_decision_up") if bool(args.poly_fair_enable) else None),
                "side_state_decision_down": (_poly_last_decision.get("side_state_decision_down") if bool(args.poly_fair_enable) else None),
                "model_requote_trigger_up": (_poly_last_decision.get("model_requote_trigger_up") if bool(args.poly_fair_enable) else None),
                "model_requote_trigger_down": (_poly_last_decision.get("model_requote_trigger_down") if bool(args.poly_fair_enable) else None),
                "model_requote_reason_up": (_poly_last_decision.get("model_requote_reason_up") if bool(args.poly_fair_enable) else None),
                "model_requote_reason_down": (_poly_last_decision.get("model_requote_reason_down") if bool(args.poly_fair_enable) else None),
                "best_bid_up_at_decision": (_poly_last_decision.get("best_bid_up_at_decision") if bool(args.poly_fair_enable) else None),
                "best_bid_down_at_decision": (_poly_last_decision.get("best_bid_down_at_decision") if bool(args.poly_fair_enable) else None),
                **last_action_params,
            }

            sid = log_step(
                "quote_manager",
                action_id=last_action_id,
                action=action_dict,
            )

            for side, desired_px, name, side_active, model_requote_trigger in [
                (UP, px_up, "UP", side_active_up, model_requote_trigger_up),
                (DN, px_dn, "DOWN", side_active_dn, model_requote_trigger_dn),
            ]:
                # ── 425 cancel backoff: skip ALL cancel/reprice actions
                # for this side while we're in the transient-error cooldown.
                if now_ns() < cancel_backoff_until_ns[name]:
                    continue

                # MODEL-FREE soft hold: do not repost into the adverse impulse window
                if (not bool(args.poly_fair_enable)) and now_ns() < int(soft_hold_until_ns[name]):
                    # If an order is somehow still live, cancel it.
                    if side.order_id and not side.cancel_in_flight:
                        side.cancel_in_flight = True
                        oid = detach_order_for_audit(side)
                        if oid:
                            enqueue_cancel(
                                prio=0,
                                trigger_kind="SOFT_HOLD",
                                side=name,
                                oid=str(oid),
                                trigger_ns=now_ns(),
                                replace_px=None,
                                step_id=sid,
                            )
                    continue

                if side.cancel_in_flight:
                    _cif_age_ns = now_ns() - int(side.cancel_in_flight_since_ns or 0)
                    cancel_inflight_age_ms_list.append(_cif_age_ns / 1e6)
                    if _cif_age_ns > cancel_inflight_timeout_ns:
                        cancel_inflight_timeout_recovers += 1
                        side.cancel_in_flight = False
                        side.cancel_in_flight_since_ns = 0
                        jprint({
                            "event": "cancel_in_flight_timeout_recover",
                            "side": name,
                            "age_ms": round(_cif_age_ns / 1e6, 1),
                            "timeout_ms": round(cancel_inflight_timeout_ns / 1e6, 1),
                        })
                        signal_requote("cancel_timeout_recover")
                    else:
                        continue
                if side.post_in_flight:
                    _pif_age_ns = now_ns() - side.post_in_flight_since_ns
                    post_inflight_age_ms_list.append(_pif_age_ns / 1e6)
                    if _pif_age_ns > post_inflight_timeout_ns:
                        post_inflight_timeout_recovers += 1
                        side.post_in_flight = False
                        jprint({
                            "event": "post_in_flight_timeout_recover",
                            "side": name,
                            "age_ms": round(_pif_age_ns / 1e6, 1),
                            "timeout_ms": round(post_inflight_timeout_ns / 1e6, 1),
                        })
                        signal_requote("post_timeout_recover")
                    else:
                        continue

                # In model mode, inactive side is canceled/suppressed.
                if bool(args.poly_fair_enable) and (not side_active or desired_px is None):
                    # #region agent log
                    _debug_log(
                        "H5",
                        "market_maker_v2.py:3400",
                        "side suppressed in quote loop",
                        {
                            "side": name,
                            "side_active": bool(side_active),
                            "desired_px": desired_px,
                            "has_order": bool(side.order_id),
                            "cancel_in_flight": bool(side.cancel_in_flight),
                            "post_in_flight": bool(side.post_in_flight),
                            "model_requote_trigger": bool(model_requote_trigger),
                            "book_up_bid": book_up.bid,
                            "book_dn_bid": book_dn.bid,
                        },
                    )
                    # #endregion
                    cool_ns = int(float(args.poly_fair_reentry_cooldown_ms) * 1e6)
                    if cool_ns > 0:
                        model_reentry_allowed_ns[name] = max(model_reentry_allowed_ns[name], now_ns() + cool_ns)
                    if side.order_id and not side.cancel_in_flight:
                        side.cancel_in_flight = True
                        oid = detach_order_for_audit(side)
                        enqueue_cancel(
                            prio=1,
                            trigger_kind="MODEL_SUPPRESS",
                            side=name,
                            oid=str(oid),
                            trigger_ns=now_ns(),
                            replace_px=None,
                            step_id=sid,
                        )
                    continue

                # Skip posting if this side doesn't meet notional minimum
                # (handles one-sided rebalance mode gracefully).
                if desired_px is None or not _notional_ok(float(desired_px), float(args.size)):
                    continue
                if bool(args.poly_fair_enable) and now_ns() < int(model_reentry_allowed_ns.get(name, 0)):
                    continue
                if (not max_churn_enabled) and now_ns() < int(requote_reentry_allowed_ns.get(name, 0)):
                    continue

                if side.order_id is None:
                    # No resting order and no post queued → post
                    enqueue_post(
                        prio=1,
                        trigger_kind="REQUOTE",
                        side=name,
                        token_id=side.token_id,
                        px=float(desired_px),
                        trigger_ns=now_ns(),
                        step_id=sid,
                    )
                    continue

                # In model-active mode for active sides, keep strict top-of-book
                # maintenance (>=1 tick drift) even on timer wakeups.
                strict_model_best_bid_follow = bool(args.poly_fair_enable) and bool(side_active)

                # ── Only cancel existing resting orders when woken by a
                # real market signal (Binance move, Poly book change, fill,
                # risk-clear, etc.).  Timer-only wakeups must NOT cancel —
                # doing so destroys queue priority for no informational gain.
                # Exception: strict model best-bid follow for currently active side.
                if (not strict_model_best_bid_follow) and (not signal_driven):
                    continue

                # Check if order needs repricing (price moved >= 1 tick)
                needs_reprice = (
                    side.px is not None
                    and desired_px is not None
                    and (
                        abs(float(side.px) - float(desired_px)) > 1e-12
                        if max_churn_enabled
                        else abs(float(side.px) - float(desired_px)) >= tick - 1e-12
                    )
                )

                if needs_reprice:
                    if max_churn_enabled:
                        max_churn_requote_attempts += 1
                    t_requote = now_ns()
                    q_eff = compute_pair_px()
                    if q_eff is not None:
                        eff_side = q_eff[0] if name == "UP" else q_eff[1]
                        effective_desired_px = min(float(desired_px), float(eff_side))
                    else:
                        effective_desired_px = float(desired_px)
                    reprice_effective_delta = abs(float(side.px) - float(effective_desired_px)) if side.px is not None else None
                    if reprice_effective_delta is not None:
                        reprice_effective_delta_ticks_list.append(reprice_effective_delta / tick)
                    if reprice_effective_delta is not None and reprice_effective_delta < tick - 1e-12:
                        if max_churn_enabled:
                            max_churn_bypassed_noop += 1
                        else:
                            reprice_skipped_noop += 1
                            _debug_log(
                                "H3",
                                "market_maker_v2.py:3534",
                                "qm_reprice_skipped_noop_effective",
                                {
                                    "side": name,
                                    "old_px": float(side.px) if side.px is not None else None,
                                    "desired_px": float(desired_px) if desired_px is not None else None,
                                    "effective_desired_px": float(effective_desired_px),
                                    "effective_delta_ticks": (
                                        round(reprice_effective_delta / tick, 3)
                                        if reprice_effective_delta is not None else None
                                    ),
                                },
                            )
                            continue

                    dwell_ns = int(float(args.requote_min_dwell_ms) * 1e6)
                    within_dwell = (t_requote - int(last_requote_cancel_ns.get(name, 0))) < dwell_ns
                    if within_dwell and reprice_effective_delta is not None and reprice_effective_delta < (2.0 * tick - 1e-12):
                        if max_churn_enabled:
                            max_churn_bypassed_hysteresis += 1
                        else:
                            reprice_skipped_hysteresis += 1
                            _debug_log(
                                "H3",
                                "market_maker_v2.py:3534",
                                "qm_reprice_skipped_hysteresis",
                                {
                                    "side": name,
                                    "old_px": float(side.px) if side.px is not None else None,
                                    "desired_px": float(desired_px) if desired_px is not None else None,
                                    "effective_desired_px": float(effective_desired_px),
                                    "effective_delta_ticks": (
                                        round(reprice_effective_delta / tick, 3)
                                        if reprice_effective_delta is not None else None
                                    ),
                                    "dwell_ms": float(args.requote_min_dwell_ms),
                                },
                            )
                            continue

                    model_state_key = "side_state_decision_up" if name == "UP" else "side_state_decision_down"
                    model_active_key = "up_active" if name == "UP" else "dn_active"
                    model_state_before = _poly_last_decision.get(model_state_key)
                    model_state_after = model_state_before
                    reprice_blocked_by_model_state = False
                    reprice_reason = "legacy_signal_reprice"

                    # Model-first state control: right before repricing, refresh
                    # the latest model decision and let suppress/cancel preempt.
                    if strict_model_best_bid_follow:
                        latest_decision = compute_fair_prices(now_ns()) or {}
                        _poly_last_decision.update(latest_decision)
                        latest_active = bool(latest_decision.get(model_active_key, False))
                        model_state_after = latest_decision.get(model_state_key)
                        if not latest_active:
                            reprice_blocked_by_model_state = True
                            if side.order_id and not side.cancel_in_flight:
                                side.cancel_in_flight = True
                                oid = detach_order_for_audit(side)
                                enqueue_cancel(
                                    prio=1,
                                    trigger_kind="MODEL_SUPPRESS",
                                    side=name,
                                    oid=str(oid),
                                    trigger_ns=now_ns(),
                                    replace_px=None,
                                    step_id=sid,
                                )
                            _debug_log(
                                "H3",
                                "market_maker_v2.py:3534",
                                "reprice_blocked_by_model_state",
                                {
                                    "side": name,
                                    "old_px": float(side.px) if side.px is not None else None,
                                    "desired_px": float(desired_px) if desired_px is not None else None,
                                    "signal_driven": bool(signal_driven),
                                    "model_state_before_reprice": model_state_before,
                                    "model_state_after_reprice_check": model_state_after,
                                    "reprice_blocked_by_model_state": True,
                                    "reprice_reason": "model_state_suppress",
                                },
                            )
                            continue
                        reprice_reason = "model_best_bid_follow"
                    else:
                        stale_ticks = abs(float(side.px) - float(desired_px)) / tick  # type: ignore[arg-type]
                        if (
                            bool(args.poly_fair_enable)
                            and not bool(model_requote_trigger)
                            and stale_ticks < float(args.poly_fair_max_stale_ticks)
                        ):
                            # Model says sub-threshold move (< min delta) and quote is only
                            # mildly stale: avoid churn. If quote is materially stale,
                            # allow cancel/reprice regardless of model trigger.
                            continue
                    is_upward = float(desired_px) > float(side.px) + 1e-12  # type: ignore[arg-type]
                    if (not bool(args.poly_fair_enable)) and is_upward:
                        bm_worst = get_worst_impulse(now_ns())
                        if bm_worst is not None and abs(bm_worst) >= float(args.soft_move_usd):
                            _debug_log("H3", "market_maker_v2.py:3530", "qm_reprice_blocked_upward", {"side":name,"old_px":float(side.px) if side.px is not None else None,"desired_px":float(desired_px),"bm_worst":bm_worst})
                            continue

                    # #region agent log
                    _debug_log("H3", "market_maker_v2.py:3534", "qm_reprice", {"side":name,"old_px":float(side.px) if side.px is not None else None,"desired_px":float(desired_px),"delta_ticks":round(abs(float(side.px)-float(desired_px))/tick,2) if side.px is not None else None,"signal_driven":signal_driven,"bm":get_bin_move_usd(now_ns()),"is_upward":is_upward,"reprice_reason":reprice_reason,"model_state_before_reprice":model_state_before,"model_state_after_reprice_check":model_state_after,"reprice_blocked_by_model_state":reprice_blocked_by_model_state})
                    # #endregion
                    side.cancel_in_flight = True
                    oid = detach_order_for_audit(side)
                    last_requote_cancel_ns[name] = t_requote
                    enqueue_cancel(
                        prio=1,
                        trigger_kind="REQUOTE",
                        side=name,
                        oid=str(oid),
                        trigger_ns=now_ns(),
                        replace_px=float(desired_px),
                        step_id=sid,
                    )

    # ── Rollover ─────────────────────────────────────────────────────────

    async def roller() -> None:
        while not shutdown.is_set():
            await asyncio.sleep(0.25)
            if datetime.now(ET) >= roll_et:
                jprint({"event": "rollover", "at": roll_et.isoformat()})
                roll_sid = log_step(
                    "rollover",
                    action_id=-1,
                    action={"trigger": "hour_roll", "roll_et": roll_et.isoformat()},
                )
                # Cancel all outstanding orders before rolling
                for S, name in [(UP, "UP"), (DN, "DOWN")]:
                    if S.order_id and not S.cancel_in_flight:
                        S.cancel_in_flight = True
                        oid = detach_order_for_audit(S)
                        if oid:
                            enqueue_cancel(
                                prio=0,
                                trigger_kind="ROLLOVER",
                                side=name,
                                oid=str(oid),
                                trigger_ns=now_ns(),
                                replace_px=None,
                                step_id=roll_sid,
                            )
                # Give cancels time to process before tearing down
                await asyncio.sleep(0.5)
                shutdown.set()
                return

    # ── Periodic stats ───────────────────────────────────────────────────

    async def periodic() -> None:
        while not shutdown.is_set():
            await asyncio.sleep(float(args.summary_every_sec))
            # Show spread breakdown so user can see exactly what's widening
            up_skew_t, dn_skew_t = inventory_skew_ticks()
            vol_extra_t = vol_spread_extra_ticks()
            fill_quality_extra_t = fill_quality_extra_ticks()
            tick = float(args.tick)
            secs_to_roll = (roll_et - datetime.now(ET)).total_seconds()

            # Momentum bias (informational, mirrors compute_pair_px zero-sum skew)
            m_thresh = float(args.momentum_threshold)
            m_bonus_t = float(args.momentum_bonus_ticks)
            up_momentum_t = 0.0
            dn_momentum_t = 0.0
            if poly_ok() and m_bonus_t > 0 and m_thresh < 1.0:
                _um = (float(book_up.bid) + float(book_up.ask)) / 2.0  # type: ignore[arg-type]
                _dm = (float(book_dn.bid) + float(book_dn.ask)) / 2.0  # type: ignore[arg-type]
                if _um >= m_thresh and _dm < m_thresh:
                    up_momentum_t = m_bonus_t    # UP boosted
                    dn_momentum_t = -m_bonus_t   # DN pulled back
                elif _dm >= m_thresh and _um < m_thresh:
                    dn_momentum_t = m_bonus_t    # DN boosted
                    up_momentum_t = -m_bonus_t   # UP pulled back

            jprint({
                "event": "rolling",
                "post_ms": summarize(post_ms_list[-200:]),
                "cancel_ms": summarize(cancel_ms_list[-200:]),
                "cancel_queue_ms": summarize(cancel_queue_ms_list[-200:]),
                "trigger_to_cancel_ms": summarize(
                    trigger_to_cancel_ms_list[-200:]
                ),
                "trigger_to_replace_ms": summarize(
                    trigger_to_replace_ms_list[-200:]
                ),
                "reprice_effective_delta_ticks": summarize(
                    reprice_effective_delta_ticks_list[-200:]
                ),
                "post_inflight_age_ms": summarize(post_inflight_age_ms_list[-200:]),
                "cancel_inflight_age_ms": summarize(cancel_inflight_age_ms_list[-200:]),
                "risk_active": risk_active,
                "risk_dir": risk_dir,
                "cross_backoff": {"UP": cross_backoff_up, "DN": cross_backoff_dn},
                "spread_breakdown": {
                    "mid_offset_ticks": float(eff_mid_offset_ticks),
                    "ask_buffer_ticks": float(eff_ask_buffer_ticks),
                    "vol_extra_ticks": round(vol_extra_t, 3),
                    "fill_quality_extra_ticks": round(fill_quality_extra_t, 3),
                    "up_skew_ticks": round(up_skew_t, 3),
                    "dn_skew_ticks": round(dn_skew_t, 3),
                    "up_momentum_ticks": round(up_momentum_t, 3),
                    "dn_momentum_ticks": round(dn_momentum_t, 3),
                    "total_offset_up": round(
                        (eff_mid_offset_ticks + vol_extra_t + fill_quality_extra_t + up_skew_t - up_momentum_t) * tick, 4
                    ),
                    "total_offset_dn": round(
                        (eff_mid_offset_ticks + vol_extra_t + fill_quality_extra_t + dn_skew_t - dn_momentum_t) * tick, 4
                    ),
                },
                "vol_ema": vol_tracker.ema,
                "action_id": last_action_id,
                "secs_to_roll": round(secs_to_roll, 1),
                "UP_px": UP.px,
                "DN_px": DN.px,
                "inventory_up": inventory_up,
                "inventory_dn": inventory_dn,
                "fills_up": fills_up,
                "fills_dn": fills_dn,
                "UP_oid": UP.order_id is not None,
                "DN_oid": DN.order_id is not None,
                "reprice_skipped_noop": reprice_skipped_noop,
                "reprice_skipped_hysteresis": reprice_skipped_hysteresis,
                "post_skipped_noop_replace": post_skipped_noop_replace,
                "requote_signal_suppressed": requote_signal_suppressed,
                "post_inflight_timeout_recovers": post_inflight_timeout_recovers,
                "cancel_inflight_timeout_recovers": cancel_inflight_timeout_recovers,
                "max_churn_requote_attempts": max_churn_requote_attempts,
                "max_churn_bypassed_noop": max_churn_bypassed_noop,
                "max_churn_bypassed_hysteresis": max_churn_bypassed_hysteresis,
                "db_dropped": logger.dropped_rows(),
            })

    # ── Launch all tasks ─────────────────────────────────────────────────

    tasks = [
        asyncio.create_task(cancel_worker(), name="cancel_1"),
        asyncio.create_task(cancel_worker(), name="cancel_2"),
        asyncio.create_task(post_worker(), name="post_1"),
        asyncio.create_task(post_worker(), name="post_2"),
        asyncio.create_task(poly_ws(), name="poly_ws"),
        asyncio.create_task(bin_ws(), name="bin_ws"),
        asyncio.create_task(risk_timer_watchdog(), name="risk_watchdog"),
        asyncio.create_task(fill_watcher(), name="fill_watcher"),
        asyncio.create_task(quote_manager(), name="quote_manager"),
        asyncio.create_task(roller(), name="roller"),
        asyncio.create_task(periodic(), name="periodic"),
    ]

    try:
        await shutdown.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        sign_ex.shutdown(wait=False, cancel_futures=True)
        post_ex.shutdown(wait=False, cancel_futures=True)
        cancel_ex.shutdown(wait=False, cancel_futures=True)
        read_ex.shutdown(wait=False, cancel_futures=True)

        final_state = {
            "slug": slug,
            "post_ms": summarize(post_ms_list),
            "cancel_ms": summarize(cancel_ms_list),
            "cancel_queue_ms": summarize(cancel_queue_ms_list),
            "trigger_to_cancel_ms": summarize(trigger_to_cancel_ms_list),
            "trigger_to_replace_ms": summarize(trigger_to_replace_ms_list),
            "reprice_effective_delta_ticks": summarize(reprice_effective_delta_ticks_list),
            "post_inflight_age_ms": summarize(post_inflight_age_ms_list),
            "cancel_inflight_age_ms": summarize(cancel_inflight_age_ms_list),
            "inventory_up": inventory_up,
            "inventory_dn": inventory_dn,
            "fills_up": fills_up,
            "fills_dn": fills_dn,
            "reprice_skipped_noop": reprice_skipped_noop,
            "reprice_skipped_hysteresis": reprice_skipped_hysteresis,
            "post_skipped_noop_replace": post_skipped_noop_replace,
            "requote_signal_suppressed": requote_signal_suppressed,
            "post_inflight_timeout_recovers": post_inflight_timeout_recovers,
            "cancel_inflight_timeout_recovers": cancel_inflight_timeout_recovers,
            "max_churn_requote_attempts": max_churn_requote_attempts,
            "max_churn_bypassed_noop": max_churn_bypassed_noop,
            "max_churn_bypassed_hysteresis": max_churn_bypassed_hysteresis,
            "dropped_rows": logger.dropped_rows(),
        }

        jprint({"event": "summary", **final_state})

        # Log market_end step for RL data slicing
        log_step(
            "market_end",
            action_id=-1,
            action={
                "final_inventory_up": inventory_up,
                "final_inventory_dn": inventory_dn,
                "fills_up": fills_up,
                "fills_dn": fills_dn,
            },
        )

        logger.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HFT spread-capture market maker for Polymarket hourly BTC markets"
    )
    p.add_argument("--env-file", required=True)

    p.add_argument("--walk-hours-forward", type=int, default=6)
    p.add_argument("--gamma-timeout-sec", type=float, default=8.0)

    p.add_argument("--live", action="store_true")
    p.add_argument("--size", type=float, default=5.0)

    p.add_argument("--tick", type=float, default=0.01)
    p.add_argument("--price-min", type=float, default=0.01)
    p.add_argument("--price-max", type=float, default=0.99)

    p.add_argument("--sum-target", type=float, default=0.99,
                   help="Target combined BUY cost for UP + DOWN pair. "
                        "0.99 = $0.01 spread profit per completed pair.")
    p.add_argument("--mid-offset-ticks", type=float, default=0.0,
                   help="Ticks below the anchor (fair or ask ceiling) to quote. "
                        "Acts as a fixed edge buffer.")
    p.add_argument("--ask-buffer-ticks", type=float, default=1.0,
                   help="Ticks below best ask for the maker ceiling. "
                        "DB analysis: fills at ask-1 have +0.5c markout; at ask "
                        "they're 0. Default 1 tick ensures we never cross the ask.")

    p.add_argument("--bin-move-usd", type=float, default=2.0,
                   help="(Legacy) Binance move threshold ($).  Kept for vol_spread "
                        "multiplier reference; primary trigger is --hard-move-usd.")
    p.add_argument("--risk-extra-ticks", type=int, default=2,
                   help="Extra ticks to pull back on replacement after risk trigger.")
    p.add_argument("--risk-hold-ms", type=float, default=250.0)

    # ── MODEL-FREE latency-aware adverse cancel ────────────────────────
    p.add_argument("--impulse-ms", type=float, default=140.0,
                   help="Binance impulse horizon in ms (should match p90 cancel+post).")
    p.add_argument("--soft-move-usd", type=float, default=0.8,
                   help="Soft threshold: cancel ONLY the adverse side when "
                        "|dB_impulse| exceeds this.")
    p.add_argument("--soft-hold-ms", type=float, default=120.0,
                   help="After a soft adverse cancel, pause reposting that side "
                        "for this long (ms).")
    p.add_argument("--hard-move-usd", type=float, default=2.0,
                   help="Hard threshold: enter risk mode (cancel BOTH sides + hold).")

    # Adverse-selection guard
    p.add_argument(
        "--asel-trigger-usd",
        type=float,
        default=2.0,
        help="Minimum |ΔB(250ms)| in $ to activate adverse-selection guard. "
             "Pulls quotes back on the adverse side by 1-3 extra ticks based on "
             "ΔB magnitude. DB analysis: fills become toxic above ~$2.",
    )

    # Volatility-adaptive spread
    p.add_argument(
        "--vol-half-life",
        type=int,
        default=50,
        help="EMA half-life in Binance ticks for volatility tracker",
    )
    p.add_argument(
        "--vol-spread-scale",
        type=float,
        default=0.5,
        help="Extra offset ticks per unit of vol multiplier above 1.0. "
             "DB analysis: fills during volatile periods need wider spread. "
             "Set to 0 to disable (not recommended).",
    )
    p.add_argument(
        "--vol-spread-max-ticks",
        type=float,
        default=1.0,
        help="Hard cap on vol-adaptive spread widening in ticks (prevents blowout)",
    )

    # Momentum bias
    p.add_argument(
        "--momentum-threshold",
        type=float,
        default=1.0,
        help="Mid price above which the winning side gets a bonus bid "
             "(default: 1.0 = effectively OFF). Set to e.g. 0.70 to enable: "
             "when a side's mid >= this threshold, bid more aggressively on it.",
    )
    p.add_argument(
        "--momentum-bonus-ticks",
        type=float,
        default=0.0,
        help="Extra ticks to bid on the winning side (more aggressive). "
             "(default: 0.0 = OFF). Typical value when enabled: 0.3.",
    )

    # Inventory skew
    p.add_argument(
        "--inventory-skew-ticks",
        type=float,
        default=0.02,
        help="Ticks to skew per unit of inventory imbalance (positive = less "
             "aggressive on overweight side). DB analysis: max imbalance was ~36 "
             "units; at 0.02, that's 0.72 ticks of skew — meaningful but not "
             "punitive. Set higher (0.05-0.15) to be more aggressive on rebalancing.",
    )
    p.add_argument(
        "--cancel-other-on-fill",
        action="store_true",
        default=True,
        help="Cancel the other side's order when one side fills (default: True). "
             "DB analysis: temporal mismatch between paired fills causes pair cost > $1. "
             "Cancelling the other side and requoting at fresh prices prevents stale fills.",
    )
    p.add_argument(
        "--no-cancel-other-on-fill",
        dest="cancel_other_on_fill",
        action="store_false",
        help="Keep the other side's order alive after a fill (pure spread capture)",
    )

    # Partial fill behaviour
    p.add_argument(
        "--partial-fill-reprice",
        action="store_true",
        default=True,
        help="On partial fill: cancel remaining, reprice at inventory-skewed level with reduced size (default: True)",
    )
    p.add_argument(
        "--no-partial-fill-reprice",
        dest="partial_fill_reprice",
        action="store_false",
        help="On partial fill: leave resting order alone, only update inventory & signal requote for other side",
    )
    # Fill-quality model: widen spread when P(good) is low (e.g. near expiry, high move)
    p.add_argument(
        "--fill-quality-enable",
        action="store_true",
        default=False,
        help="Use saved fill-quality model to add extra spread ticks when "
             "conditions are unfavorable (low P(good)). Requires --fill-quality-model-path etc.",
    )
    p.add_argument(
        "--fill-quality-model-path",
        type=str,
        default="",
        help="Path to fill_quality_model.joblib. Default: <script_dir>/fill_quality_model.joblib",
    )
    p.add_argument(
        "--fill-quality-scaler-path",
        type=str,
        default="",
        help="Path to fill_quality_scaler.joblib (used if model expects scaled features).",
    )
    p.add_argument(
        "--fill-quality-threshold-path",
        type=str,
        default="",
        help="Path to fill_quality_threshold.json (threshold and scaled flag).",
    )
    p.add_argument(
        "--fill-quality-spread-scale",
        type=float,
        default=2.0,
        help="Extra ticks to add per unit (threshold - P(good)) when P(good) < threshold. "
             "E.g. scale=2, threshold=0.5, P=0.3 -> (0.5-0.3)*2 = 0.4 extra ticks.",
    )
    p.add_argument(
        "--fill-quality-max-ticks",
        type=float,
        default=3.0,
        help="Cap on fill-quality-driven spread widening in ticks (default: 3.0).",
    )
    # Poly fair state-transition model controls (Phase 1)
    p.add_argument(
        "--poly-fair-enable",
        action="store_true",
        default=False,
        help="Enable model-driven quote-state decisions (keep/cancel/re-enter/suppress).",
    )
    p.add_argument(
        "--poly-fair-mode",
        type=str,
        default="active",
        choices=["shadow", "advisory", "active"],
        help="Rollout mode: shadow logs only; advisory suppresses gated/churn; active controls side state.",
    )
    p.add_argument(
        "--poly-fair-model-dir",
        type=str,
        default="mm_analysis/out",
        help="Directory containing pre-trained poly_move_model/scaler/imputer artifacts.",
    )
    p.add_argument(
        "--poly-fair-weights-file",
        type=str,
        default="poly_move_bundle.joblib",
        help="Path to single bundle artifact (poly_move_bundle.joblib). Relative path resolves from CWD first, then script dir.",
    )
    p.add_argument(
        "--poly-fair-min-secs-left",
        type=float,
        default=900.0,
        help="If secs_left <= this, suppress quoting by default (unless balancing exception applies).",
    )
    p.add_argument(
        "--poly-fair-disable-high-vol",
        action="store_true",
        default=True,
        help="Suppress quoting in high-vol regimes by default.",
    )
    p.add_argument(
        "--no-poly-fair-disable-high-vol",
        dest="poly_fair_disable_high_vol",
        action="store_false",
        help="Allow model decisions even in high-vol regimes.",
    )
    p.add_argument(
        "--poly-fair-high-vol-ema-usd",
        type=float,
        default=20.0,
        help="EMA vol threshold (USD) considered high-vol for poly-fair gating.",
    )
    p.add_argument(
        "--poly-fair-direction-threshold-usd",
        type=float,
        default=0.8,
        help="Directional classification threshold (defaults to soft-move style threshold).",
    )
    p.add_argument(
        "--poly-fair-min-requote-delta",
        type=float,
        default=0.01,
        help="Minimum predicted move magnitude to trigger model-driven cancel/requote churn.",
    )
    p.add_argument(
        "--poly-fair-max-stale-ticks",
        type=float,
        default=2.0,
        help="Allow model churn-block only when quote staleness is below this many ticks; always reprice if more stale.",
    )
    p.add_argument(
        "--poly-fair-balance-only-on-gate-fail",
        action="store_true",
        default=True,
        help="If gates fail, allow one-sided quoting only for inventory balancing.",
    )
    p.add_argument(
        "--no-poly-fair-balance-only-on-gate-fail",
        dest="poly_fair_balance_only_on_gate_fail",
        action="store_false",
        help="Disable balancing exception on gate fail (strict no-quote).",
    )
    p.add_argument(
        "--poly-fair-balance-inventory-threshold",
        type=float,
        default=1.0,
        help="Inventory imbalance threshold (contracts) that activates one-sided balance exception.",
    )
    p.add_argument(
        "--poly-fair-favored-keep-enable",
        action="store_true",
        default=True,
        help="Keep favored side active by default when directional classification is available.",
    )
    p.add_argument(
        "--no-poly-fair-favored-keep-enable",
        dest="poly_fair_favored_keep_enable",
        action="store_false",
        help="Disable favorable-side keep bias.",
    )
    p.add_argument(
        "--poly-fair-adverse-cancel-enable",
        action="store_true",
        default=True,
        help="Allow adverse-side suppression when predicted adverse move >= min requote delta.",
    )
    p.add_argument(
        "--no-poly-fair-adverse-cancel-enable",
        dest="poly_fair_adverse_cancel_enable",
        action="store_false",
        help="Disable adverse-side model cancel behavior.",
    )
    p.add_argument(
        "--poly-fair-reentry-cooldown-ms",
        type=float,
        default=0.0,
        help="Optional cooldown before re-entering a previously suppressed side.",
    )
    p.add_argument(
        "--fill-cooldown-ms",
        type=float,
        default=100.0,
        help="Milliseconds to wait after a fill before requoting. "
             "Prevents immediate reposting into an adverse move that caused "
             "the fill. Set to 0 to disable. (default: 100ms)",
    )
    p.add_argument(
        "--min-remaining-size",
        type=float,
        default=0.5,
        help="Remaining order size below this is treated as a full fill (default: 0.5)",
    )

    # Stale data / order guards
    p.add_argument(
        "--preflight-book-fetch",
        action="store_true",
        default=False,
        help="Deprecated (no-op): REST preflight book fetch is disabled to avoid latency.",
    )
    p.add_argument(
        "--no-preflight-book-fetch",
        dest="preflight_book_fetch",
        action="store_false",
        help="Deprecated (no-op).",
    )
    p.add_argument(
        "--stale-book-ms",
        type=float,
        default=5000.0,
        help="Max age of Poly book data before suppressing quotes",
    )
    p.add_argument(
        "--order-max-age-ms",
        type=float,
        default=0.0,
        help="Deprecated (no-op): stale-order auto-refresh removed to preserve "
             "queue priority.  Orders are only cancelled on real market signals.",
    )

    # Reference price decay
    p.add_argument(
        "--bin-ref-decay",
        type=float,
        default=0.0,
        help="Per-tick decay of bin_ref toward current mid (0 = off, 1 = instant). "
             "Disabled by default: bin_ref decay masks trending moves.  Primary "
             "protection is now impulse-based (--hard-move-usd / --soft-move-usd).",
    )

    p.add_argument("--quote-interval-ms", type=float, default=15.0)
    p.add_argument("--quote-min-gap-ms", type=float, default=8.0)
    p.add_argument(
        "--post-inflight-timeout-ms",
        type=float,
        default=750.0,
        help="Timeout before quote_manager force-recovers a stuck post_in_flight side gate.",
    )
    p.add_argument(
        "--cancel-inflight-timeout-ms",
        type=float,
        default=750.0,
        help="Timeout before quote_manager force-recovers a stuck cancel_in_flight side gate.",
    )
    p.add_argument(
        "--requote-min-dwell-ms",
        type=float,
        default=120.0,
        help="Minimum dwell after a REQUOTE/no-op cycle before allowing another same-side REQUOTE cancel.",
    )
    p.add_argument(
        "--requote-signal-min-gap-ms",
        type=float,
        default=20.0,
        help="Debounce interval for noisy requote signals (Poly TOB and soft impulse wakes).",
    )
    p.add_argument(
        "--requote-max-churn",
        action="store_true",
        default=False,
        help="Enable maximum REQUOTE churn by bypassing no-op/hysteresis/debounce suppressors.",
    )
    p.add_argument("--fill-poll-ms", type=float, default=150.0)
    p.add_argument("--summary-every-sec", type=float, default=10.0)

    p.add_argument(
        "--max-runtime-sec",
        type=float,
        default=0.0,
        help="0 = forever across hours (rolls internally each hour)",
    )

    # ── RL data logging (SQLite) ─────────────────────────────────────────
    p.add_argument(
        "--db-path",
        type=str,
        default="mm_rl_log.sqlite",
        help="SQLite database path for RL training data",
    )
    p.add_argument(
        "--db-max-queue",
        type=int,
        default=200_000,
        help="Max pending rows before dropping (default: 200000)",
    )
    p.add_argument(
        "--db-flush-ms",
        type=int,
        default=250,
        help="Flush interval for the DB writer thread (default: 250ms)",
    )
    p.add_argument(
        "--db-log-binance-every-ms",
        type=int,
        default=25,
        help="Min interval between logged Binance ticks (default: 25ms)",
    )
    p.add_argument(
        "--db-log-poly-every-ms",
        type=int,
        default=25,
        help="Min interval between logged Poly ticks (default: 25ms)",
    )

    # ── RL policy action override ────────────────────────────────────────
    p.add_argument(
        "--policy-action-id",
        type=int,
        default=-1,
        help="Force a specific action menu index (-1 = use CLI knobs, default: -1)",
    )

    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)

    t0 = time.perf_counter()
    while True:
        slug, mk, hour_start_et = resolve_market_with_hour(
            args.walk_hours_forward, args.gamma_timeout_sec
        )
        up_id, down_id = extract_up_down_token_ids(mk)
        await run_one_market(
            args,
            slug=slug,
            up_id=up_id,
            down_id=down_id,
            hour_start_et=hour_start_et,
        )

        if float(args.max_runtime_sec) > 0 and (
            time.perf_counter() - t0
        ) >= float(args.max_runtime_sec):
            return

        # Brief pause before resolving next hour market
        await asyncio.sleep(0.5)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
