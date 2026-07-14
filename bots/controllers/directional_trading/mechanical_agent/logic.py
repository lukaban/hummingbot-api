from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_DECISIONS = {"long", "short", "wait", "no_trade"}
ALLOWED_FAMILIES = {
    "standard_pullback",
    "breakout_retest",
    "momentum_probe",
    "missed_no_trade",
}
DEFAULT_STRATEGY_VERSION = "liquidity_profile_v2"
V2_STRATEGY_VERSION = DEFAULT_STRATEGY_VERSION
SUPPORTED_STRATEGY_VERSIONS = {DEFAULT_STRATEGY_VERSION}
CANDIDATE_VALIDITY_SECONDS = 180


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not result.is_finite():
        raise ValueError(f"invalid {name}")
    return result


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    controller_id: str
    connector_name: str
    trading_pair: str
    trigger_type: str
    location_id: str
    triggered_at: float
    valid_until: float
    last_price: Decimal
    features: Mapping[str, str]
    recent_candles_5m: tuple[Mapping[str, str], ...]
    active_executors: tuple[Mapping[str, str], ...]
    strategy_version: str = DEFAULT_STRATEGY_VERSION
    origin_trigger_type: str | None = None
    origin_location_id: str | None = None
    invalidation_operator: str | None = None
    invalidation_level: Decimal | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["last_price"] = str(self.last_price)
        if self.invalidation_level is not None:
            payload["invalidation_level"] = str(self.invalidation_level)
        return payload


@dataclass(frozen=True)
class AgentTarget:
    price: Decimal
    allocation: Decimal


@dataclass(frozen=True)
class ValidatedExecutionPlan:
    event_id: str
    decision: str
    execution_family: str
    valid_until: float
    entry_low: Decimal
    entry_high: Decimal
    stop: Decimal
    targets: tuple[AgentTarget, ...]
    confidence: str
    reason: str


def parse_agent_result(
    payload: Mapping[str, Any],
    event: CandidateEvent,
    now: float,
    price: Decimal,
    minimum_rr: Decimal,
) -> ValidatedExecutionPlan | None:
    decision = str(payload.get("decision", ""))
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("invalid decision")
    if payload.get("event_id") != event.event_id:
        raise ValueError("event_id mismatch")
    valid_until = float(payload.get("valid_until", 0))
    if not math.isfinite(valid_until) or valid_until <= now or valid_until > event.valid_until:
        raise ValueError("analysis expired or exceeds event validity")
    if decision in {"wait", "no_trade"}:
        return None
    family = str(payload.get("execution_family", ""))
    if family not in ALLOWED_FAMILIES or family == "missed_no_trade":
        raise ValueError("invalid execution_family")
    entry_low = _decimal(payload.get("entry_low"), "entry_low")
    entry_high = _decimal(payload.get("entry_high"), "entry_high")
    stop = _decimal(payload.get("stop"), "stop")
    if entry_low > entry_high or not entry_low <= price <= entry_high:
        raise ValueError("current price outside entry range")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not 1 <= len(raw_targets) <= 3:
        raise ValueError("targets must contain 1 to 3 items")
    has_allocations = [target.get("allocation") is not None for target in raw_targets if isinstance(target, dict)]
    if len(has_allocations) != len(raw_targets) or any(has_allocations) != all(has_allocations):
        raise ValueError("allocations must be all present or all omitted")
    if all(has_allocations):
        allocations = [_decimal(target["allocation"], "allocation") for target in raw_targets]
    else:
        equal = Decimal(1) / Decimal(len(raw_targets))
        allocations = [equal] * len(raw_targets)
    if any(value <= 0 for value in allocations) or abs(sum(allocations) - Decimal(1)) > Decimal("0.000001"):
        raise ValueError("target allocations must sum to one")
    risk = price - stop if decision == "long" else stop - price
    if risk <= 0:
        raise ValueError("stop is on the wrong side")
    targets: list[AgentTarget] = []
    for raw_target, allocation in zip(raw_targets, allocations):
        target_price = _decimal(raw_target.get("price"), "target price")
        reward = target_price - price if decision == "long" else price - target_price
        if reward <= 0:
            raise ValueError("target direction is invalid")
        targets.append(AgentTarget(target_price, allocation))
    if decision == "long" and stop >= entry_low:
        raise ValueError("long stop must be below entry")
    if decision == "short" and stop <= entry_high:
        raise ValueError("short stop must be above entry")
    return ValidatedExecutionPlan(
        event_id=event.event_id,
        decision=decision,
        execution_family=family,
        valid_until=valid_until,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        targets=tuple(targets),
        confidence=str(payload.get("confidence", "unverified")),
        reason=str(payload.get("reason", "")),
    )


@dataclass(frozen=True)
class FeatureSnapshot:
    last_closed_5m_timestamp: int
    last_price: Decimal
    h4_close: Decimal
    h4_ema20: Decimal
    h1_close: Decimal
    h1_ema20: Decimal
    m15_close: Decimal
    previous_m15_close: Decimal
    m15_ema20: Decimal
    atr_5m: Decimal
    profile_vah: Decimal
    profile_val: Decimal
    profile_poc: Decimal
    profile_sample_bars: int
    confirmed_high: Decimal
    confirmed_low: Decimal
    recent_candles_5m: tuple[Mapping[str, str], ...]
    confirmed_high_is_pivot: bool = False
    confirmed_low_is_pivot: bool = False

    def as_agent_features(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in asdict(self).items()
            if key not in {
                "recent_candles_5m", "profile_sample_bars", "last_closed_5m_timestamp",
                "confirmed_high_is_pivot", "confirmed_low_is_pivot",
            }
        }


def _ema(values: Sequence[Decimal], period: int = 20) -> Decimal:
    if len(values) < period:
        raise ValueError(f"at least {period} values are required")
    multiplier = Decimal(2) / Decimal(period + 1)
    result = sum(values[:period]) / Decimal(period)
    for value in values[period:]:
        result = (value - result) * multiplier + result
    return result


def _atr(rows: Sequence[Mapping[str, Any]], period: int = 14) -> Decimal:
    if len(rows) < period + 1:
        raise ValueError("insufficient candles for ATR")
    ranges = []
    for previous, current in zip(rows, rows[1:]):
        high = _decimal(current["high"], "high")
        low = _decimal(current["low"], "low")
        previous_close = _decimal(previous["close"], "close")
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return sum(ranges[-period:]) / Decimal(period)


def _profile(rows: Sequence[Mapping[str, Any]], bins: int = 48) -> tuple[Decimal, Decimal, Decimal]:
    typical = [
        (_decimal(row["high"], "high") + _decimal(row["low"], "low") + _decimal(row["close"], "close")) / Decimal(3)
        for row in rows
    ]
    volumes = [_decimal(row["volume"], "volume") for row in rows]
    low, high = min(typical), max(typical)
    if high == low:
        return low, high, low
    width = (high - low) / Decimal(bins)
    bucket_volumes = [Decimal(0)] * bins
    for price, volume in zip(typical, volumes):
        index = min(int((price - low) / width), bins - 1)
        bucket_volumes[index] += volume
    poc = max(range(bins), key=bucket_volumes.__getitem__)
    left = right = poc
    accumulated = bucket_volumes[poc]
    target = sum(bucket_volumes) * Decimal("0.70")
    while accumulated < target and (left > 0 or right < bins - 1):
        left_volume = bucket_volumes[left - 1] if left > 0 else Decimal(-1)
        right_volume = bucket_volumes[right + 1] if right < bins - 1 else Decimal(-1)
        if right_volume > left_volume:
            right += 1
            accumulated += bucket_volumes[right]
        else:
            left -= 1
            accumulated += bucket_volumes[left]
    center = lambda index: low + (Decimal(index) + Decimal("0.5")) * width
    return center(left), center(right), center(poc)


def _resample_15m(rows: Sequence[Mapping[str, Any]]) -> list[Decimal]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        timestamp = int(row["timestamp"])
        groups.setdefault(timestamp // 900_000, []).append(row)
    closes = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda row: int(row["timestamp"]))
        timestamps = [int(row["timestamp"]) for row in ordered]
        aligned = timestamps and timestamps[0] % 900_000 == 0
        continuous = len(timestamps) == 3 and all(
            right - left == 300_000 for left, right in zip(timestamps, timestamps[1:])
        )
        if aligned and continuous and len(set(timestamps)) == 3:
            closes.append(_decimal(ordered[-1]["close"], "close"))
    return closes


def _confirmed_pivot(
    rows: Sequence[Mapping[str, Any]], key: str, *, high: bool, wing: int = 2
) -> tuple[Decimal, bool]:
    values = [_decimal(row[key], key) for row in rows]
    for index in range(len(values) - wing - 1, wing - 1, -1):
        neighbors = values[index - wing:index] + values[index + 1:index + wing + 1]
        if (high and values[index] > max(neighbors)) or (not high and values[index] < min(neighbors)):
            return values[index], True
    return (max(values) if high else min(values)), False


def build_feature_snapshot(
    candles_5m: Sequence[Mapping[str, Any]],
    candles_1h: Sequence[Mapping[str, Any]],
    candles_4h: Sequence[Mapping[str, Any]],
) -> FeatureSnapshot:
    if len(candles_5m) < 288:
        raise ValueError("at least 288 closed 5m candles are required")
    if len(candles_1h) < 20 or len(candles_4h) < 20:
        raise ValueError("at least 20 higher-timeframe candles are required")
    five = sorted(candles_5m, key=lambda row: int(row["timestamp"]))
    one = sorted(candles_1h, key=lambda row: int(row["timestamp"]))
    four = sorted(candles_4h, key=lambda row: int(row["timestamp"]))
    sample = five[-288:]
    closes_15m = _resample_15m(five)
    if len(closes_15m) < 20:
        raise ValueError("at least 20 complete 15m candles are required")
    val, vah, poc = _profile(sample)
    recent = five[-20:]
    structure_sample = five[-60:-1]
    confirmed_high, confirmed_high_is_pivot = _confirmed_pivot(structure_sample, "high", high=True)
    confirmed_low, confirmed_low_is_pivot = _confirmed_pivot(structure_sample, "low", high=False)
    return FeatureSnapshot(
        last_closed_5m_timestamp=int(five[-1]["timestamp"]),
        last_price=_decimal(five[-1]["close"], "close"),
        h4_close=_decimal(four[-1]["close"], "close"),
        h4_ema20=_ema([_decimal(row["close"], "close") for row in four]),
        h1_close=_decimal(one[-1]["close"], "close"),
        h1_ema20=_ema([_decimal(row["close"], "close") for row in one]),
        m15_close=closes_15m[-1],
        previous_m15_close=closes_15m[-2],
        m15_ema20=_ema(closes_15m),
        atr_5m=_atr(five),
        profile_vah=vah,
        profile_val=val,
        profile_poc=poc,
        profile_sample_bars=288,
        confirmed_high=confirmed_high,
        confirmed_low=confirmed_low,
        recent_candles_5m=tuple({key: str(value) for key, value in row.items()} for row in recent),
        confirmed_high_is_pivot=confirmed_high_is_pivot,
        confirmed_low_is_pivot=confirmed_low_is_pivot,
    )


def scan_candidate(
    features: FeatureSnapshot,
    controller_id: str,
    connector: str,
    pair: str,
    now: float,
    trigger_distance_atr: Decimal,
    *,
    active_executors: Sequence[Mapping[str, str]] = (),
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> CandidateEvent | None:
    if strategy_version not in SUPPORTED_STRATEGY_VERSIONS:
        raise ValueError("unsupported strategy_version")
    if trigger_distance_atr <= 0 or features.atr_5m <= 0:
        return None
    latest = features.recent_candles_5m[-1]
    latest_high = _decimal(latest["high"], "high")
    latest_low = _decimal(latest["low"], "low")
    latest_close = _decimal(latest["close"], "close")
    rejected_val = latest_low <= features.profile_val and latest_close > features.profile_val
    rejected_vah = latest_high >= features.profile_vah and latest_close < features.profile_vah
    if features.confirmed_high_is_pivot and latest_high > features.confirmed_high and latest_close < features.confirmed_high:
        trigger_type, location_id = "confirmed_high_sweep", "confirmed_high"
    elif features.confirmed_low_is_pivot and latest_low < features.confirmed_low and latest_close > features.confirmed_low:
        trigger_type, location_id = "confirmed_low_sweep", "confirmed_low"
    elif (
        features.previous_m15_close > features.profile_vah
        and features.profile_val <= features.m15_close <= features.profile_vah
    ) or (
        features.previous_m15_close < features.profile_val
        and features.profile_val <= features.m15_close <= features.profile_vah
    ):
        trigger_type, location_id = "m15_fast_reclaim", "profile_value"
    elif (
        features.profile_val <= features.previous_m15_close <= features.profile_vah
        and (features.m15_close > features.profile_vah or features.m15_close < features.profile_val)
    ):
        trigger_type, location_id = "m15_outside_value_close", "profile_value"
    elif rejected_val and not rejected_vah:
        trigger_type, location_id = "profile_val_rejection", "profile_val"
    elif rejected_vah and not rejected_val:
        trigger_type, location_id = "profile_vah_rejection", "profile_vah"
    else:
        return None
    invalidation_operator, invalidation_level = _invalidation_identity(
        features, trigger_type, location_id
    )
    if strategy_version == V2_STRATEGY_VERSION and invalidation_operator is None:
        return None
    raw = f"{strategy_version}|{connector}|{pair}|{trigger_type}|{location_id}|{features.last_closed_5m_timestamp}"
    event_id = hashlib.sha256(raw.encode()).hexdigest()
    return CandidateEvent(
        event_id=event_id,
        controller_id=controller_id,
        connector_name=connector,
        trading_pair=pair,
        trigger_type=trigger_type,
        location_id=location_id,
        triggered_at=now,
        valid_until=now + CANDIDATE_VALIDITY_SECONDS,
        last_price=features.last_price,
        features=features.as_agent_features(),
        recent_candles_5m=features.recent_candles_5m,
        active_executors=tuple(active_executors),
        strategy_version=strategy_version,
        invalidation_operator=invalidation_operator,
        invalidation_level=invalidation_level,
    )


def get_strategy(version: str):
    if version not in SUPPORTED_STRATEGY_VERSIONS:
        raise ValueError("unsupported strategy_version")
    return build_feature_snapshot, scan_candidate


def _invalidation_identity(
    features: FeatureSnapshot, trigger_type: str, location_id: str
) -> tuple[str | None, Decimal | None]:
    if trigger_type == "confirmed_high_sweep":
        return "close_below", features.confirmed_high
    if trigger_type == "confirmed_low_sweep":
        return "close_above", features.confirmed_low
    if trigger_type == "m15_outside_value_close":
        return (
            ("close_above", features.profile_vah)
            if features.m15_close > features.profile_vah
            else ("close_below", features.profile_val)
        )
    if trigger_type == "m15_fast_reclaim":
        return (
            ("close_below", features.profile_vah)
            if features.previous_m15_close > features.profile_vah
            else ("close_above", features.profile_val)
        )
    if trigger_type == "profile_val_rejection":
        return "close_above", features.profile_val
    if trigger_type == "profile_vah_rejection":
        return "close_below", features.profile_vah
    return None, None


def revalidate_armed_setup(
    armed: CandidateEvent,
    features: FeatureSnapshot,
    now: float,
    active_executors: Sequence[Mapping[str, str]] = (),
) -> CandidateEvent | None:
    armed_candle = int(_decimal(armed.recent_candles_5m[-1]["timestamp"], "timestamp"))
    if features.last_closed_5m_timestamp <= armed_candle:
        return None
    level = armed.invalidation_level
    operator = armed.invalidation_operator
    latest_close = _decimal(features.recent_candles_5m[-1]["close"], "close")
    if level is None or operator not in {"close_above", "close_below"}:
        return None
    if (operator == "close_above" and latest_close <= level) or (
        operator == "close_below" and latest_close >= level
    ):
        return None
    raw = (
        f"{V2_STRATEGY_VERSION}|{armed.connector_name}|{armed.trading_pair}|armed_revalidation|"
        f"{armed.trigger_type}|{armed.location_id}|{operator}|{features.last_closed_5m_timestamp}"
    )
    return CandidateEvent(
        event_id=hashlib.sha256(raw.encode()).hexdigest(),
        controller_id=armed.controller_id,
        connector_name=armed.connector_name,
        trading_pair=armed.trading_pair,
        trigger_type="armed_revalidation",
        location_id=armed.location_id,
        triggered_at=now,
        valid_until=now + CANDIDATE_VALIDITY_SECONDS,
        last_price=features.last_price,
        features=features.as_agent_features(),
        recent_candles_5m=features.recent_candles_5m,
        active_executors=tuple(active_executors),
        strategy_version=V2_STRATEGY_VERSION,
        origin_trigger_type=armed.trigger_type,
        origin_location_id=armed.location_id,
        invalidation_operator=operator,
        invalidation_level=level,
    )


def execution_amount(
    price: Decimal,
    stop: Decimal,
    allocation: Decimal,
    notional_cap_quote: Decimal,
    risk_amount_quote: Decimal | None,
) -> Decimal:
    capped = notional_cap_quote * allocation / price
    if risk_amount_quote is None:
        return capped
    return min(risk_amount_quote * allocation / abs(price - stop), capped)


class EventRegistry:
    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._in_flight: set[str] = set()
        self._finished_at: dict[str, float] = {}
        self._groups: set[str] = set()

    def start(self, event_id: str, now: float) -> bool:
        if event_id in self._groups or event_id in self._in_flight:
            return False
        if now - self._finished_at.get(event_id, float("-inf")) < self.cooldown_seconds:
            return False
        self._in_flight.add(event_id)
        return True

    def mark_finished(self, event_id: str, now: float) -> None:
        self._in_flight.discard(event_id)
        self._finished_at[event_id] = now

    def mark_group_created(self, event_id: str) -> None:
        self._in_flight.discard(event_id)
        self._groups.add(event_id)


class SetupJournal:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._events: list[dict[str, Any]] = []
        self._keys: set[tuple[str, str]] = set()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                setup_id = str(event["setup_instance_id"])
                event_type = str(event["event_type"])
                event_time = float(event["event_time"])
                if not setup_id or not event_type or not math.isfinite(event_time):
                    raise ValueError("invalid setup journal event")
                self._events.append(event)
                self._keys.add((setup_id, event_type))

    def record(self, setup_id: str, event_type: str, event_time: float, **changes: Any) -> bool:
        key = (str(setup_id), str(event_type))
        timestamp = float(event_time)
        if not key[0] or not key[1] or not math.isfinite(timestamp):
            raise ValueError("invalid setup journal event")
        if key in self._keys:
            return False
        event = {
            "setup_instance_id": key[0], "event_type": key[1],
            "event_time": timestamp, "changes": changes,
        }
        encoded = json.dumps(event, allow_nan=False, separators=(",", ":"), default=str)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(encoded + "\n")
                output.flush()
                os.fsync(output.fileno())
        self._events.append(event)
        self._keys.add(key)
        return True

    def has_event(self, setup_id: str, event_type: str) -> bool:
        return (setup_id, event_type) in self._keys

    def is_terminal(self, setup_id: str) -> bool:
        return self.has_event(setup_id, "cancelled") or self.has_event(setup_id, "closed")

    def execution_setup_ids(self) -> dict[str, str]:
        return {
            str(event["changes"]["execution_event_id"]): str(event["setup_instance_id"])
            for event in self._events
            if event["event_type"] == "executing" and event.get("changes", {}).get("execution_event_id")
        }

    def latest_plan(self) -> Mapping[str, Any] | None:
        for event in reversed(self._events):
            plan = event.get("changes", {}).get("plan")
            if event["event_type"] in {"confirmed", "executing"} and isinstance(plan, Mapping):
                return dict(plan)
        return None

    def restart_orphans(self) -> set[str]:
        last_type: dict[str, str] = {}
        for event in self._events:
            last_type[str(event["setup_instance_id"])] = str(event["event_type"])
        return {
            setup_id for setup_id, event_type in last_type.items()
            if event_type in {"armed", "confirmed"}
        }

    def summary(self) -> dict[str, Any]:
        setup_ids = {event["setup_instance_id"] for event in self._events if event["event_type"] == "armed"}
        types = {
            event_type: {event["setup_instance_id"] for event in self._events if event["event_type"] == event_type}
            for event_type in ("confirmed", "executing", "closed", "cancelled")
        }
        closed_pnl = [
            _decimal(event.get("changes", {}).get("pnl_quote", "0"), "pnl_quote")
            for event in self._events if event["event_type"] == "closed"
        ]
        return {
            "setups": len(setup_ids), "confirmed": len(types["confirmed"]),
            "executed": len(types["executing"]), "closed": len(types["closed"]),
            "wins": sum(value > 0 for value in closed_pnl),
            "losses": sum(value < 0 for value in closed_pnl),
            "cancelled": len(types["cancelled"]),
            "pnl_quote": str(sum(closed_pnl, Decimal(0))),
        }


def evaluate_execution_risk(
    executors: Sequence[Any], now: float, planned_risk_quote: Decimal,
    *, max_daily_loss_quote: Decimal | None, max_consecutive_losses: int | None,
    max_open_risk_quote: Decimal | None,
) -> tuple[str | None, dict[str, Any]]:
    try:
        groups: dict[str, list[Any]] = {}
        open_risk = Decimal(0)
        for executor in executors:
            level_id = str(getattr(getattr(executor, "config", None), "level_id", "") or "")
            if not level_id:
                continue
            groups.setdefault(level_id.split(":", 1)[0], []).append(executor)
            if bool(getattr(executor, "is_active", False)):
                config = executor.config
                open_risk += (
                    _decimal(config.entry_price, "entry_price")
                    * _decimal(config.amount, "amount")
                    * _decimal(config.triple_barrier_config.stop_loss, "stop_loss")
                )
        closed = []
        for group in groups.values():
            if group and all(bool(getattr(executor, "is_done", False)) for executor in group):
                timestamps = [float(executor.close_timestamp) for executor in group if executor.close_timestamp is not None]
                if timestamps:
                    pnl = sum((_decimal(executor.net_pnl_quote, "net_pnl_quote") for executor in group), Decimal(0))
                    closed.append((max(timestamps), pnl))
        closed.sort(reverse=True)
        day_start = math.floor(float(now) / 86400) * 86400
        daily_pnl = sum((pnl for timestamp, pnl in closed if timestamp >= day_start), Decimal(0))
        consecutive_losses = 0
        for _, pnl in closed:
            if pnl < 0:
                consecutive_losses += 1
            else:
                break
        status = {
            "gate": "pass", "daily_pnl_quote": str(daily_pnl),
            "consecutive_losses": consecutive_losses, "open_risk_quote": str(open_risk),
            "planned_risk_quote": str(planned_risk_quote),
        }
        reason = None
        if max_daily_loss_quote is not None and daily_pnl <= -max_daily_loss_quote:
            reason = "max daily loss reached"
        elif max_consecutive_losses is not None and consecutive_losses >= max_consecutive_losses:
            reason = "max consecutive losses reached"
        elif max_open_risk_quote is not None and open_risk + planned_risk_quote > max_open_risk_quote:
            reason = "max open risk reached"
        if reason:
            status["gate"] = "block"
        return reason, status
    except (AttributeError, TypeError, ValueError, InvalidOperation, OverflowError):
        return "risk data invalid", {"gate": "block"}


class AgentUnavailable(RuntimeError):
    pass


class AgentAnalysisAdapter:
    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 20,
        token: str | None = None,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token
        self._opener = opener
        self._sleep = sleep

    def analyze(self, event: CandidateEvent) -> dict[str, Any]:
        if not self.endpoint:
            raise AgentUnavailable("agent endpoint is disabled")
        body = json.dumps(event.to_payload(), sort_keys=True, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(f"{self.endpoint}/v1/analyze-candidate", data=body, headers=headers, method="POST")
        for attempt in range(2):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", 200)
                    if status < 200 or status >= 300:
                        raise AgentUnavailable(f"agent returned HTTP {status}")
                    payload = json.loads(response.read().decode())
                if not isinstance(payload, dict):
                    raise AgentUnavailable("agent returned invalid JSON object")
                return payload
            except HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == 1:
                    raise AgentUnavailable(f"agent HTTP error {error.code}") from error
            except (URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as error:
                if attempt == 1 or isinstance(error, json.JSONDecodeError):
                    raise AgentUnavailable("agent request failed") from error
            except Exception as error:
                raise AgentUnavailable("agent request failed") from error
            self._sleep(0.25)
        raise AgentUnavailable("agent request failed")
