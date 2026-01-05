import argparse
import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

SLOT_SECONDS = 15 * 60  # 900 seconds

# ---------- Normal CDF ----------
SQRT2 = math.sqrt(2.0)
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

# ---------- Time helpers ----------
def iso_local(epoch_sec: int) -> str:
    return datetime.fromtimestamp(epoch_sec).strftime("%Y-%m-%d %H:%M:%S")

def slot_start_epoch(epoch_sec: int) -> int:
    return (epoch_sec // SLOT_SECONDS) * SLOT_SECONDS

def parse_local_start_time(s: str) -> int:
    """
    Parse local time like '2026-01-02 17:00:00' -> epoch seconds.
    Assumes your CSV times are in local timezone.
    """
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())

# ---------- Rolling volatility with irregular time steps ----------
@dataclass
class RollingIrregularBrownian:
    """
    We assume log-price follows (approximately) Brownian motion over short horizons:
      r_i = log(P_i/P_{i-1}) ~ N(mu*dt_i, sigma^2*dt_i)

    We estimate sigma from recent observations using dt-aware MLE:
      mu_hat = sum(r)/sum(dt)
      sigma_hat^2 = sum((r - mu_hat*dt)^2)/sum(dt)

    IMPORTANT: In this market-maker setting, we default to drift=0 for pricing
    (mu is too noisy in tiny windows and leads to crazy probabilities).
    """
    window_seconds: int
    min_samples: int

    # store tuples: (r, dt)
    obs: Deque[Tuple[float, int]] = None

    # running sums for fast rolling calc
    sum_dt: float = 0.0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    sum_rdt: float = 0.0
    sum_dt2: float = 0.0

    def __post_init__(self):
        self.obs = deque()

    def reset(self):
        self.obs.clear()
        self.sum_dt = self.sum_r = self.sum_r2 = self.sum_rdt = self.sum_dt2 = 0.0

    def push(self, r: float, dt: int):
        if dt <= 0:
            return

        self.obs.append((r, dt))
        self.sum_dt += dt
        self.sum_r += r
        self.sum_r2 += r * r
        self.sum_rdt += r * dt
        self.sum_dt2 += dt * dt

        # Keep only last ~window_seconds worth of dt
        while self.sum_dt > self.window_seconds and len(self.obs) > 1:
            old_r, old_dt = self.obs.popleft()
            self.sum_dt -= old_dt
            self.sum_r -= old_r
            self.sum_r2 -= old_r * old_r
            self.sum_rdt -= old_r * old_dt
            self.sum_dt2 -= old_dt * old_dt

    def estimate_mu_sigma(self) -> Tuple[Optional[float], Optional[float]]:
        if len(self.obs) < self.min_samples or self.sum_dt <= 0:
            return None, None

        # mu_hat = sum(r)/sum(dt)
        mu = self.sum_r / self.sum_dt

        # sigma^2 = sum((r - mu*dt)^2)/sum(dt)
        # expand:
        # sum(r^2) - 2*mu*sum(r*dt) + mu^2*sum(dt^2)
        num = self.sum_r2 - 2.0 * mu * self.sum_rdt + (mu * mu) * self.sum_dt2
        if num < 0:
            num = 0.0
        sigma2 = num / self.sum_dt
        sigma = math.sqrt(sigma2)
        return mu, sigma

def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(p.stem + "_fair.csv"))

def run_engine(
    input_csv: str,
    output_csv: str,
    vol_window_seconds: int,
    min_samples: int,
    price_field: str,
    reset_vol_each_slot: bool,
    strict_slot_start: bool,
    start_local: Optional[str],
    sigma_floor: float,
    drift_mode: str,
):
    if price_field not in {"close", "twap"}:
        raise ValueError("--price-field must be close or twap")

    # Drift handling:
    # - zero: mu = 0 (recommended)
    # - estimate: use mu_hat from window (can be unstable)
    if drift_mode not in {"zero", "estimate"}:
        raise ValueError("--drift must be 'zero' or 'estimate'")

    global_start_epoch = parse_local_start_time(start_local) if start_local else None

    vol = RollingIrregularBrownian(window_seconds=vol_window_seconds, min_samples=min_samples)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout)

        writer.writerow([
            "epoch_sec", "time_local",
            "slot_start_epoch", "slot_start_local",
            "elapsed_sec_in_slot", "remaining_sec_in_slot",
            "slot_start_price", "current_price",
            "log_return_so_far",
            "mu_per_sec", "sigma_per_sqrt_sec",
            "fair_up", "fair_down",
            "ticks"
        ])

        cur_slot_start: Optional[int] = None
        slot_S0: Optional[float] = None
        slot_active = False  # strict mode: only active after we hit exact boundary row

        # For dt-aware volatility we only update vol when we see a "real" update.
        last_real_epoch: Optional[int] = None
        last_real_price: Optional[float] = None

        for row in reader:
            try:
                epoch_sec = int(float(row["epoch_sec"]))
            except Exception:
                continue

            # Filter by global start (e.g. 17:00:00)
            if global_start_epoch is not None and epoch_sec < global_start_epoch:
                continue

            try:
                price = float(row[price_field])
            except Exception:
                continue

            try:
                ticks = int(float(row.get("ticks", "0")))
            except Exception:
                ticks = 0

            s_start = slot_start_epoch(epoch_sec)
            elapsed = epoch_sec - s_start
            remaining = SLOT_SECONDS - elapsed
            if remaining < 0:
                remaining = 0

            # Slot change
            if cur_slot_start is None or s_start != cur_slot_start:
                cur_slot_start = s_start
                slot_S0 = None
                slot_active = False

                last_real_epoch = None
                last_real_price = None

                if reset_vol_each_slot:
                    vol.reset()

            # STRICT SLOT START:
            # Only start the slot once we have a row exactly at boundary (elapsed==0).
            if strict_slot_start:
                if not slot_active:
                    if elapsed != 0:
                        # skip partial slot until boundary
                        continue
                    slot_active = True
                    slot_S0 = price

                    # initialize "real" obs state at boundary
                    if ticks > 0:
                        last_real_epoch = epoch_sec
                        last_real_price = price
                else:
                    if slot_S0 is None:
                        slot_S0 = price
            else:
                # non-strict: first row becomes slot start
                if slot_S0 is None:
                    slot_S0 = price

            # Update volatility only on "real" updates.
            # This avoids destroying sigma with tons of filled (ticks==0) repeats.
            if ticks > 0:
                if last_real_epoch is not None and last_real_price is not None and last_real_price > 0 and price > 0:
                    dt = epoch_sec - last_real_epoch
                    if dt > 0:
                        r = math.log(price / last_real_price)
                        vol.push(r, dt)
                last_real_epoch = epoch_sec
                last_real_price = price

            if slot_S0 is None or slot_S0 <= 0 or price <= 0:
                continue

            r0 = math.log(price / slot_S0)

            # If we're exactly at end-of-slot second, payoff is deterministic.
            if remaining == 0:
                fair_up = 1.0 if price >= slot_S0 else 0.0
                fair_down = 1.0 - fair_up
                writer.writerow([
                    epoch_sec, iso_local(epoch_sec),
                    cur_slot_start, iso_local(cur_slot_start),
                    elapsed, remaining,
                    slot_S0, price,
                    r0, "", "",
                    fair_up, fair_down,
                    ticks
                ])
                continue

            mu_hat, sigma_hat = vol.estimate_mu_sigma()

            # Drift: default to 0 to avoid the "mu * tau blows up" problem you saw.
            if drift_mode == "zero":
                mu = 0.0
            else:
                mu = 0.0 if mu_hat is None else float(mu_hat)

            # Volatility: enforce a floor so sigma never collapses to ~0.
            if sigma_hat is None:
                sigma = None
            else:
                sigma = max(float(sigma_hat), float(sigma_floor))

            # If not enough vol data, fall back to 0.5
            if sigma is None or sigma <= 0:
                fair_up = 0.5
                fair_down = 0.5
                writer.writerow([
                    epoch_sec, iso_local(epoch_sec),
                    cur_slot_start, iso_local(cur_slot_start),
                    elapsed, remaining,
                    slot_S0, price,
                    r0, "" if mu_hat is None else mu_hat, "" if sigma_hat is None else sigma_hat,
                    fair_up, fair_down,
                    ticks
                ])
                continue

            tau = float(remaining)

            # Probability that final price >= start price:
            # We need P( r0 + future_return >= 0 )
            # future_return ~ N(mu*tau, sigma^2*tau)
            # => P >= 0 = Phi( (r0 + mu*tau) / (sigma*sqrt(tau)) )
            z = (r0 + mu * tau) / (sigma * math.sqrt(tau))
            fair_up = norm_cdf(z)
            fair_up = max(0.0, min(1.0, fair_up))
            fair_down = 1.0 - fair_up

            writer.writerow([
                epoch_sec, iso_local(epoch_sec),
                cur_slot_start, iso_local(cur_slot_start),
                elapsed, remaining,
                slot_S0, price,
                r0, mu, sigma,
                fair_up, fair_down,
                ticks
            ])

    print(f"[ok] Wrote: {output_csv}")

def parse_args():
    p = argparse.ArgumentParser(description="15-min Up/Down fair price engine (fixed math inputs)")
    p.add_argument("--input", required=True, help="Path to BTC per-second CSV (epoch_sec,...).")
    p.add_argument("--output", default="", help="Output CSV (default: <input>_fair.csv)")
    p.add_argument("--price-field", choices=["close", "twap"], default="close")

    # Vol config
    p.add_argument("--vol-window", type=int, default=300, help="Rolling vol window in seconds (default 300s = 5min)")
    p.add_argument("--min-samples", type=int, default=20, help="Min real-update samples for vol estimate")
    p.add_argument("--sigma-floor", type=float, default=1e-4,
                   help="Floor on sigma per sqrt(second) to prevent collapse (default 1e-4)")

    # Slot behavior
    p.add_argument("--no-reset-vol", action="store_true", help="Do NOT reset vol each 15-min slot")
    p.add_argument("--strict-slot-start", action="store_true",
                   help="Require elapsed==0 row to start each slot (recommended)")
    p.add_argument("--start-local", default="", help="Ignore rows before this local time: 'YYYY-MM-DD HH:MM:SS'")

    # Drift behavior
    p.add_argument("--drift", choices=["zero", "estimate"], default="zero",
                   help="Use mu=0 (recommended) or estimate drift from rolling window")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    out = args.output.strip() or default_output_path(args.input)

    run_engine(
        input_csv=args.input,
        output_csv=out,
        vol_window_seconds=args.vol_window,
        min_samples=args.min_samples,
        price_field=args.price_field,
        reset_vol_each_slot=(not args.no_reset_vol),
        strict_slot_start=args.strict_slot_start,
        start_local=args.start_local.strip() or None,
        sigma_floor=args.sigma_floor,
        drift_mode=args.drift,
    )
