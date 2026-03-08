#!/usr/bin/env python3
"""
logger_v2.py – Append-only SQLite event logger for market_maker_v2.py

Provides:
    - epoch_ns(), iso_utc_from_epoch_ns()  – wall-clock helpers
    - SQLiteEventLogger                     – non-blocking, WAL-enabled logger
      with a background writer thread

Tables:
    runs           – metadata about each run (params, start/end, dropped rows)
    markets        – one row per hourly market
    steps          – decision snapshots  (state + action label)
    ticks          – sampled Binance & Poly top-of-book  (for markout)
    order_events   – enqueue / post / cancel outcomes  (for latency & success)
    order_polls    – raw order states from get_order  (for true fill details)
    fills          – incremental fills credited to inventory  (for trade outcomes)

Usage from market_maker_v2.py:

    from logger_v2 import SQLiteEventLogger, epoch_ns, iso_utc_from_epoch_ns

    logger = SQLiteEventLogger("mm_rl_log.sqlite", run_id=uuid.uuid4().hex)
    ...
    logger.close()
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

# ── Fast JSON (mirrors market_maker_v2 pattern) ──────────────────────────────
try:
    import orjson as _orjson

    def _json_dumps_str(d: Any) -> str:
        return _orjson.dumps(d, default=str).decode()
except ImportError:
    _orjson = None  # type: ignore[assignment]

    def _json_dumps_str(d: Any) -> str:  # type: ignore[misc]
        return json.dumps(d, separators=(",", ":"), default=str)


# ── Time helpers ─────────────────────────────────────────────────────────────


def now_ns() -> int:
    """Monotonic nanosecond clock (for in-process elapsed-time measurements)."""
    return time.monotonic_ns()


def epoch_ns() -> int:
    """Wall-clock epoch in nanoseconds (for cross-process alignment & markout)."""
    return time.time_ns()


def iso_utc_from_epoch_ns(t_ns: int) -> str:
    """UTC ISO-8601 timestamp from epoch nanoseconds."""
    return (
        datetime.utcfromtimestamp(t_ns / 1e9)
        .replace(tzinfo=ZoneInfo("UTC"))
        .isoformat()
    )


# ── Schema DDL ───────────────────────────────────────────────────────────────

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id           TEXT PRIMARY KEY,
  started_epoch_ns INTEGER,
  started_iso_utc  TEXT,
  args_json        TEXT,
  ended_epoch_ns   INTEGER,
  ended_iso_utc    TEXT,
  dropped_rows     INTEGER
);

CREATE TABLE IF NOT EXISTS markets (
  market_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT,
  slug             TEXT,
  up_token_id      TEXT,
  down_token_id    TEXT,
  hour_start_iso   TEXT,
  roll_iso         TEXT,
  created_epoch_ns INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      TEXT,
  market_id   INTEGER,
  step_id     INTEGER,
  epoch_ns    INTEGER,
  mono_ns     INTEGER,
  reason      TEXT,
  action_id   INTEGER,
  action_json TEXT,
  state_json  TEXT
);

CREATE TABLE IF NOT EXISTS ticks (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT,
  market_id    INTEGER,
  epoch_ns     INTEGER,
  mono_ns      INTEGER,
  source       TEXT,
  symbol       TEXT,
  bid          REAL,
  ask          REAL,
  mid          REAL,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS order_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT,
  market_id    INTEGER,
  step_id      INTEGER,
  epoch_ns     INTEGER,
  mono_ns      INTEGER,
  kind         TEXT,          -- enqueue_post, post_done, post_error, enqueue_cancel, cancel_done, cancel_error, ...
  side         TEXT,          -- UP/DOWN
  order_id     TEXT,
  px           REAL,
  size         REAL,
  trigger_kind TEXT,          -- BIN/FILL/REQUOTE/STALE/...
  trigger_ns   INTEGER,
  enqueue_ns   INTEGER,
  latency_ms   REAL,
  queue_ms     REAL,
  ok           INTEGER,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS order_polls (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT,
  market_id    INTEGER,
  step_id      INTEGER,
  epoch_ns     INTEGER,
  mono_ns      INTEGER,
  side         TEXT,
  order_id     TEXT,
  px           REAL,
  size_ordered REAL,
  size_matched REAL,
  remaining    REAL,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS fills (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT,
  market_id     INTEGER,
  step_id       INTEGER,
  epoch_ns      INTEGER,
  mono_ns       INTEGER,
  fill_kind     TEXT,     -- fill, partial_fill, audit_fill
  side          TEXT,
  order_id      TEXT,
  fill_qty      REAL,
  total_matched REAL,
  remaining     REAL,
  px            REAL,
  inv_up        REAL,
  inv_dn        REAL,
  payload_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_steps_market_step  ON steps(market_id, step_id);
CREATE INDEX IF NOT EXISTS idx_ticks_market_time  ON ticks(market_id, epoch_ns);
CREATE INDEX IF NOT EXISTS idx_fills_market_time  ON fills(market_id, epoch_ns);
"""


# ── SQLiteEventLogger ────────────────────────────────────────────────────────


class SQLiteEventLogger:
    """
    Non-blocking SQLite logger for RL training data.

    Hot-path calls ``emit_*()`` which push ``(sql, params)`` tuples onto a
    bounded ``queue.Queue``.  A daemon background thread drains the queue
    and batch-inserts rows into a WAL-mode database every *flush_every_ms*.

    Key properties
    ──────────────
    * WAL journal mode for concurrent reads while writing.
    * Bounded queue (default 200 000 items) – drops are *counted*, never
      silently lost.
    * Tick sampling: Binance and Poly updates are rate-limited to avoid
      overwhelming SQLite with high-frequency book updates.
    """

    def __init__(
        self,
        db_path: str,
        run_id: str,
        *,
        max_queue: int = 200_000,
        flush_every_ms: int = 250,
        log_binance_every_ms: int = 25,
        log_poly_every_ms: int = 25,
        args_json: Optional[str] = None,
    ):
        self.db_path = str(db_path)
        self.run_id = str(run_id)
        self.max_queue = int(max_queue)
        self.flush_every_ms = int(flush_every_ms)
        self.log_binance_every_ms = int(log_binance_every_ms)
        self.log_poly_every_ms = int(log_poly_every_ms)
        self.args_json = args_json

        self._q: "queue.Queue[tuple[str, tuple[Any, ...]]]" = queue.Queue(
            maxsize=self.max_queue
        )
        self._stop = threading.Event()
        self._dropped = 0
        self._dropped_last_warn_ns = 0

        # Sampling clocks (monotonic ns)
        self._last_bin_log_ns = 0
        # Poly is sampled per symbol (e.g. UP/DOWN) so one side cannot
        # suppress the other inside the same interval window.
        self._last_poly_log_ns_by_symbol: dict[str, int] = {}

        self._thread = threading.Thread(
            target=self._writer_loop, name="sqlite_logger", daemon=True
        )
        self._thread.start()

        self._emit_run_start()

    # ── Public properties ────────────────────────────────────────────────

    def dropped_rows(self) -> int:
        """Total number of rows dropped due to queue overflow."""
        return int(self._dropped)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush remaining rows and stop the writer thread.

        Records ``ended_epoch_ns``, ``ended_iso_utc``, and ``dropped_rows``
        into the ``runs`` row before shutting down.
        """
        self._emit_run_stop()
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._dropped > 0:
            print(
                _json_dumps_str({
                    "event": "logger_close_warning",
                    "dropped_rows": self._dropped,
                    "db_path": self.db_path,
                }),
                flush=True,
            )

    # ── Internal queue / writer ──────────────────────────────────────────

    def _put(self, sql: str, params: tuple[Any, ...]) -> None:
        try:
            self._q.put_nowait((sql, params))
        except queue.Full:
            self._dropped += 1
            # Rate-limit warnings to once per second
            t = time.monotonic_ns()
            if (t - self._dropped_last_warn_ns) > 1_000_000_000:
                self._dropped_last_warn_ns = t
                try:
                    print(
                        _json_dumps_str({
                            "event": "db_drop_warning",
                            "dropped_rows_total": self._dropped,
                            "max_queue": self.max_queue,
                            "suggest": (
                                "Increase sampling intervals: "
                                "--db-log-binance-every-ms 50 "
                                "--db-log-poly-every-ms 50"
                            ),
                        }),
                        flush=True,
                    )
                except Exception:
                    pass
        except Exception:
            # Never let the queue mechanism crash the caller
            self._dropped += 1

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA_DDL)
        conn.commit()

    def _flush_batch(
        self, conn: sqlite3.Connection, buf: list[tuple[str, tuple[Any, ...]]]
    ) -> None:
        """Write *buf* to the database.

        Strategy: try the whole batch in one transaction first (fast path).
        On failure, rollback and retry each row individually so one bad row
        can't take down the entire batch.
        """
        if not buf:
            return
        try:
            conn.execute("BEGIN;")
            for sql, params in buf:
                conn.execute(sql, params)
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass

            # Row-by-row fallback – save whatever we can
            for sql, params in buf:
                try:
                    conn.execute(sql, params)
                    conn.commit()
                except Exception:
                    self._dropped += 1
                    # Don't spam; the close() summary will report total drops.

    def _writer_loop(self) -> None:
        try:
            conn = self._connect()
            self._init_schema(conn)
        except Exception as exc:
            # If we can't even open the DB, print a loud warning and bail.
            # The main trading loop must NOT be affected.
            print(
                _json_dumps_str({
                    "event": "FATAL_db_writer_init",
                    "err": str(exc),
                    "db_path": self.db_path,
                }),
                flush=True,
            )
            return

        buf: list[tuple[str, tuple[Any, ...]]] = []
        last_flush = time.time()
        _reconnect_attempts = 0
        _MAX_RECONNECT = 5

        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=0.05)
                buf.append(item)
            except queue.Empty:
                pass

            now_t = time.time()
            if buf and (
                (now_t - last_flush) * 1000.0 >= self.flush_every_ms
                or len(buf) >= 5000
            ):
                try:
                    self._flush_batch(conn, buf)
                    _reconnect_attempts = 0
                except Exception as exc:
                    # Disk full / corruption – try to reconnect
                    _reconnect_attempts += 1
                    if _reconnect_attempts <= _MAX_RECONNECT:
                        print(
                            _json_dumps_str({
                                "event": "db_write_error",
                                "err": str(exc),
                                "attempt": _reconnect_attempts,
                            }),
                            flush=True,
                        )
                        try:
                            conn.close()
                        except Exception:
                            pass
                        try:
                            conn = self._connect()
                        except Exception:
                            pass
                    else:
                        # Give up reconnecting; drain queue to prevent memory
                        # blowup but count everything as dropped.
                        self._dropped += len(buf)
                buf.clear()
                last_flush = now_t

        # Final flush
        try:
            self._flush_batch(conn, buf)
        except Exception:
            self._dropped += len(buf)

        try:
            conn.close()
        except Exception:
            pass

    # ── Run lifecycle events ─────────────────────────────────────────────

    def _emit_run_start(self) -> None:
        t = epoch_ns()
        self._put(
            "INSERT OR REPLACE INTO runs"
            "(run_id, started_epoch_ns, started_iso_utc, args_json)"
            " VALUES (?,?,?,?)",
            (self.run_id, t, iso_utc_from_epoch_ns(t), self.args_json),
        )

    def _emit_run_stop(self) -> None:
        t = epoch_ns()
        self._put(
            "UPDATE runs SET ended_epoch_ns=?, ended_iso_utc=?, dropped_rows=?"
            " WHERE run_id=?",
            (t, iso_utc_from_epoch_ns(t), self._dropped, self.run_id),
        )

    # ── Market ───────────────────────────────────────────────────────────

    def emit_market(
        self,
        *,
        slug: str,
        up_token_id: str,
        down_token_id: str,
        hour_start_iso: str,
        roll_iso: str,
    ) -> None:
        """Insert a row into the ``markets`` table."""
        try:
            t_e = epoch_ns()
            self._put(
                "INSERT INTO markets"
                "(run_id, slug, up_token_id, down_token_id,"
                " hour_start_iso, roll_iso, created_epoch_ns)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    str(slug),
                    str(up_token_id),
                    str(down_token_id),
                    str(hour_start_iso),
                    str(roll_iso),
                    t_e,
                ),
            )
        except Exception:
            self._dropped += 1

    def resolve_market_id(self, slug: str) -> int:
        """Blocking SELECT to retrieve the ``market_id`` for a given slug.

        Called once per hour after ``emit_market()`` has flushed.
        Opens its own short-lived connection so it doesn't compete with the
        writer thread.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            row = conn.execute(
                "SELECT market_id FROM markets"
                " WHERE run_id=? AND slug=?"
                " ORDER BY market_id DESC LIMIT 1",
                (self.run_id, slug),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    # ── Steps (decision snapshots) ───────────────────────────────────────

    def emit_step(
        self,
        *,
        market_id: int,
        step_id: int,
        reason: str,
        action_id: int,
        action_json: Any,
        state_json: Any,
    ) -> None:
        """Log a decision snapshot.

        *action_json* and *state_json* accept dicts or pre-serialised strings.
        """
        try:
            t_e = epoch_ns()
            t_m = now_ns()
            self._put(
                "INSERT INTO steps"
                "(run_id, market_id, step_id, epoch_ns, mono_ns,"
                " reason, action_id, action_json, state_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    int(market_id),
                    int(step_id),
                    t_e,
                    t_m,
                    str(reason),
                    int(action_id),
                    _json_dumps_str(action_json)
                    if not isinstance(action_json, str)
                    else action_json,
                    _json_dumps_str(state_json)
                    if not isinstance(state_json, str)
                    else state_json,
                ),
            )
        except Exception:
            self._dropped += 1

    # ── Ticks (sampled market data) ──────────────────────────────────────

    def emit_tick(
        self,
        *,
        market_id: int,
        source: str,
        symbol: str,
        bid: Optional[float],
        ask: Optional[float],
        mid: Optional[float],
        payload: Any,
        sampler: str,
    ) -> None:
        """Log a tick, subject to per-source sampling.

        *sampler* must be ``"binance"`` or ``"poly"`` to select which
        rate-limit clock to check. Binance uses one source-wide clock.
        Poly uses per-symbol clocks (e.g. ``UP`` and ``DOWN``) so both
        sides can be logged independently at the configured cadence.
        If a tick arrives too soon for its clock, it is silently skipped
        (NOT counted as a drop).
        """
        try:
            t_e = epoch_ns()
            t_m = now_ns()

            # ── Sampling gate ────────────────────────────────────────────
            if sampler == "binance" and self.log_binance_every_ms > 0:
                if self._last_bin_log_ns and (t_m - self._last_bin_log_ns) < int(
                    self.log_binance_every_ms * 1e6
                ):
                    return
                self._last_bin_log_ns = t_m

            if sampler == "poly" and self.log_poly_every_ms > 0:
                sym_key = str(symbol).strip() if symbol is not None else ""
                if not sym_key:
                    sym_key = "__unknown__"
                last_poly_ns = self._last_poly_log_ns_by_symbol.get(sym_key, 0)
                if last_poly_ns and (t_m - last_poly_ns) < int(
                    self.log_poly_every_ms * 1e6
                ):
                    return
                self._last_poly_log_ns_by_symbol[sym_key] = t_m

            self._put(
                "INSERT INTO ticks"
                "(run_id, market_id, epoch_ns, mono_ns, source,"
                " symbol, bid, ask, mid, payload_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    int(market_id),
                    t_e,
                    t_m,
                    str(source),
                    str(symbol),
                    float(bid) if bid is not None else None,
                    float(ask) if ask is not None else None,
                    float(mid) if mid is not None else None,
                    _json_dumps_str(payload)
                    if not isinstance(payload, str)
                    else payload,
                ),
            )
        except Exception:
            self._dropped += 1

    # ── Order events (enqueue / post / cancel lifecycle) ─────────────────

    def emit_order_event(
        self,
        *,
        market_id: int,
        step_id: Optional[int],
        kind: str,
        side: str,
        order_id: Optional[str],
        px: Optional[float],
        size: Optional[float],
        trigger_kind: Optional[str],
        trigger_ns: Optional[int],
        enqueue_ns: Optional[int],
        latency_ms: Optional[float],
        queue_ms: Optional[float],
        ok: Optional[bool],
        payload: Any,
    ) -> None:
        """Log an order lifecycle event.

        *kind* examples: ``enqueue_post``, ``post_done``, ``post_error``,
        ``enqueue_cancel``, ``cancel_done``, ``cancel_error``.
        """
        try:
            t_e = epoch_ns()
            t_m = now_ns()
            self._put(
                "INSERT INTO order_events("
                "  run_id, market_id, step_id, epoch_ns, mono_ns,"
                "  kind, side, order_id, px, size,"
                "  trigger_kind, trigger_ns, enqueue_ns,"
                "  latency_ms, queue_ms, ok, payload_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    int(market_id),
                    int(step_id) if step_id is not None else None,
                    t_e,
                    t_m,
                    str(kind),
                    str(side),
                    str(order_id) if order_id else None,
                    float(px) if px is not None else None,
                    float(size) if size is not None else None,
                    str(trigger_kind) if trigger_kind else None,
                    int(trigger_ns) if trigger_ns is not None else None,
                    int(enqueue_ns) if enqueue_ns is not None else None,
                    float(latency_ms) if latency_ms is not None else None,
                    float(queue_ms) if queue_ms is not None else None,
                    1 if ok else (0 if ok is not None else None),
                    _json_dumps_str(payload)
                    if not isinstance(payload, str)
                    else payload,
                ),
            )
        except Exception:
            self._dropped += 1

    # ── Order polls (raw get_order snapshots) ────────────────────────────

    def emit_order_poll(
        self,
        *,
        market_id: int,
        step_id: Optional[int],
        side: str,
        order_id: str,
        px: Optional[float],
        size_ordered: float,
        size_matched: float,
        remaining: float,
        payload: Any,
    ) -> None:
        """Log a raw ``get_order`` snapshot for offline fill reconciliation."""
        try:
            t_e = epoch_ns()
            t_m = now_ns()
            self._put(
                "INSERT INTO order_polls("
                "  run_id, market_id, epoch_ns, mono_ns, step_id,"
                "  side, order_id, px, size_ordered, size_matched, remaining,"
                "  payload_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    int(market_id),
                    t_e,
                    t_m,
                    int(step_id) if step_id is not None else None,
                    str(side),
                    str(order_id),
                    float(px) if px is not None else None,
                    float(size_ordered),
                    float(size_matched),
                    float(remaining),
                    _json_dumps_str(payload)
                    if not isinstance(payload, str)
                    else payload,
                ),
            )
        except Exception:
            self._dropped += 1

    # ── Fills (incremental fills credited to inventory) ──────────────────

    def emit_fill(
        self,
        *,
        market_id: int,
        step_id: Optional[int],
        fill_kind: str,
        side: str,
        order_id: str,
        fill_qty: float,
        total_matched: float,
        remaining: float,
        px: Optional[float],
        inv_up: float,
        inv_dn: float,
        payload: Any,
    ) -> None:
        """Log an incremental fill event (the core *outcome* stream).

        *fill_kind* values: ``"fill"``, ``"partial_fill"``, ``"audit_fill"``.
        """
        try:
            t_e = epoch_ns()
            t_m = now_ns()
            self._put(
                "INSERT INTO fills("
                "  run_id, market_id, step_id, epoch_ns, mono_ns,"
                "  fill_kind, side, order_id,"
                "  fill_qty, total_matched, remaining, px, inv_up, inv_dn,"
                "  payload_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    int(market_id),
                    int(step_id) if step_id is not None else None,
                    t_e,
                    t_m,
                    str(fill_kind),
                    str(side),
                    str(order_id),
                    float(fill_qty),
                    float(total_matched),
                    float(remaining),
                    float(px) if px is not None else None,
                    float(inv_up),
                    float(inv_dn),
                    _json_dumps_str(payload)
                    if not isinstance(payload, str)
                    else payload,
                ),
            )
        except Exception:
            self._dropped += 1
