from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, List, Literal

from pydantic import Field, model_validator

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig,
)
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction

from .logic import (
    AgentAnalysisAdapter,
    CandidateEvent,
    EventRegistry,
    SetupJournal,
    ValidatedExecutionPlan,
    evaluate_execution_risk,
    get_strategy,
    execution_amount,
    parse_agent_result,
    revalidate_armed_setup,
)


FAMILY_TIME_LIMIT_SECONDS = {
    "momentum_probe": 2700,
    "standard_pullback": 5400,
    "breakout_retest": 7200,
}


class MechanicalAgentControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "mechanical_agent"
    strategy_version: str = "liquidity_profile_v2"
    candles_connector: str | None = None
    candles_trading_pair: str | None = None
    agent_endpoint: str = ""
    agent_timeout_seconds: float = Field(default=20, gt=0)
    agent_cooldown_seconds: float = Field(
        default=300,
        gt=0,
        description="Controller cooldown before reanalyzing a candidate; separate from Hummingbot cooldown_time",
    )
    trigger_distance_atr: Decimal = Field(default=Decimal("0.25"), gt=0)
    minimum_rr: Decimal = Field(
        default=Decimal("1.5"), gt=0, description="Reference RR only; never an execution gate"
    )
    execution_mode: Literal["paper", "demo", "live"] = "paper"
    risk_amount_quote: Decimal | None = Field(default=None, gt=0)
    setup_journal_path: str = ""
    max_daily_loss_quote: Decimal | None = Field(default=None, gt=0)
    max_consecutive_losses: int | None = Field(default=None, gt=0)
    max_open_risk_quote: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution_connector(self):
        get_strategy(self.strategy_version)
        is_paper_connector = self.connector_name.endswith("_paper_trade")
        is_demo_connector = self.connector_name.endswith("_demo")
        if self.execution_mode == "paper" and not is_paper_connector:
            raise ValueError("paper mode requires a paper-trade connector")
        if self.execution_mode == "demo" and not is_demo_connector:
            raise ValueError("demo mode requires a demo connector")
        if self.execution_mode == "live" and (is_paper_connector or is_demo_connector):
            raise ValueError("live mode requires a live connector")
        return self


class MechanicalAgentController(DirectionalTradingControllerBase):
    def __init__(self, config: MechanicalAgentControllerConfig, *args, **kwargs):
        self.config = config
        self._build_features, self._scan_candidate = get_strategy(config.strategy_version)
        self._registry = EventRegistry(config.agent_cooldown_seconds)
        self._adapter = AgentAnalysisAdapter(
            endpoint=config.agent_endpoint,
            timeout_seconds=config.agent_timeout_seconds,
            token=self._agent_token(),
        )
        self._pending_plan: ValidatedExecutionPlan | None = None
        self._armed_event: CandidateEvent | None = None
        self._journal = SetupJournal(config.setup_journal_path)
        for setup_id in self._journal.restart_orphans():
            self._journal.record(setup_id, "cancelled", time.time(), reason="controller_restart")
        self._event_setup_ids = self._journal.execution_setup_ids()
        rate_config = config.model_copy(update={"connector_name": self.agent_connector})
        super().__init__(rate_config, *args, **kwargs)
        self.config = config
        self.processed_data["last_execution_plan"] = self._journal.latest_plan()

    @staticmethod
    def _agent_token() -> str | None:
        token_file = os.getenv("HUMMINGBOT_AGENT_TOKEN_FILE")
        return Path(token_file).read_text(encoding="utf-8").strip() if token_file else os.getenv("HUMMINGBOT_AGENT_TOKEN")

    @property
    def candles_connector(self) -> str:
        return self.config.candles_connector or self.config.connector_name

    @property
    def agent_connector(self) -> str:
        return self.config.connector_name.removesuffix("_paper_trade").removesuffix("_demo")

    @property
    def candles_trading_pair(self) -> str:
        return self.config.candles_trading_pair or self.config.trading_pair

    def get_candles_config(self) -> List[CandlesConfig]:
        return [
            CandlesConfig(connector=self.candles_connector, trading_pair=self.candles_trading_pair, interval="5m", max_records=300),
            CandlesConfig(connector=self.candles_connector, trading_pair=self.candles_trading_pair, interval="1h", max_records=120),
            CandlesConfig(connector=self.candles_connector, trading_pair=self.candles_trading_pair, interval="4h", max_records=120),
        ]

    def get_custom_info(self) -> dict[str, Any]:
        return {
            "scanner_state": self.processed_data.get("scanner_state", "starting"),
            "strategy_version": self.config.strategy_version,
            "last_event_id": self.processed_data.get("last_event_id"),
            "last_agent_decision": self.processed_data.get("last_agent_decision"),
            "last_rejection_reason": self.processed_data.get("last_rejection_reason"),
            "state_changed_at": self.processed_data.get("state_changed_at"),
            "last_event_at": self.processed_data.get("last_event_at"),
            "last_agent_decision_at": self.processed_data.get("last_agent_decision_at"),
            "last_execution_plan": self.processed_data.get("last_execution_plan"),
            "setup_stats": self._journal.summary(),
            "risk_status": self.processed_data.get("risk_status", {"gate": "not_requested"}),
            "execution_mode": self.config.execution_mode,
            "execution_connector": self.config.connector_name,
            "market_connector": self.agent_connector,
            "armed_invalidation_operator": self.processed_data.get("armed_invalidation_operator"),
            "armed_trigger_type": self.processed_data.get("armed_trigger_type"),
            "armed_location_id": self.processed_data.get("armed_location_id"),
            "armed_invalidation_level": self.processed_data.get("armed_invalidation_level"),
            "exchange_position_amount": self.processed_data.get("exchange_position_amount", "0"),
            "orphan_position": self.processed_data.get("orphan_position", False),
        }

    def _set_state(self, state: str, now: float, **changes: Any) -> None:
        if self.processed_data.get("scanner_state") != state:
            self.processed_data["state_changed_at"] = now
        self.processed_data.update(scanner_state=state, **changes)

    @staticmethod
    def _plan_payload(plan: ValidatedExecutionPlan, protection_status: str) -> dict[str, Any]:
        return {
            "event_id": plan.event_id, "direction": plan.decision,
            "execution_family": plan.execution_family,
            "entry_low": str(plan.entry_low), "entry_high": str(plan.entry_high),
            "stop": str(plan.stop),
            "targets": [str(target.price) for target in plan.targets],
            "valid_until": plan.valid_until, "protection_status": protection_status,
        }

    def _setup_id(self, event_id: str) -> str:
        return self._event_setup_ids.get(event_id, event_id)

    def _exchange_position_amount(self) -> Decimal:
        connectors = getattr(self.market_data_provider, "connectors", {})
        connector = connectors.get(self.config.connector_name) if connectors else None
        positions = getattr(connector, "account_positions", {}) if connector is not None else {}
        return sum(
            (
                abs(Decimal(str(position.amount)))
                for position in positions.values()
                if getattr(position, "trading_pair", None) == self.config.trading_pair
            ),
            Decimal(0),
        )

    def _sync_executor_lifecycle(self, now: float, exchange_position_amount: Decimal = Decimal(0)) -> None:
        groups: dict[str, list[Any]] = {}
        for executor in getattr(self, "executors_info", []):
            level_id = str(getattr(getattr(executor, "config", None), "level_id", "") or "")
            if level_id:
                groups.setdefault(level_id.split(":", 1)[0], []).append(executor)
        for execution_event_id, setup_id in self._journal.execution_setup_ids().items():
            group = groups.get(execution_event_id, [])
            if not group:
                continue
            if any(Decimal(str(getattr(executor, "filled_amount_quote", 0))) > 0 for executor in group):
                self._journal.record(setup_id, "filled", now, execution_event_id=execution_event_id)
            if all(bool(getattr(executor, "is_done", False)) for executor in group) and exchange_position_amount == 0:
                pnl = sum((Decimal(str(getattr(executor, "net_pnl_quote", 0))) for executor in group), Decimal(0))
                close_types = sorted({str(getattr(executor, "close_type", "unknown")) for executor in group})
                self._journal.record(
                    setup_id, "closed", now, execution_event_id=execution_event_id,
                    pnl_quote=str(pnl), close_types=close_types,
                )

    @staticmethod
    def _closed_records(frame) -> list[dict[str, Any]]:
        if frame is None or len(frame.index) < 2:
            return []
        records = frame.iloc[:-1].to_dict("records")
        for record in records:
            timestamp = int(record["timestamp"])
            record["timestamp"] = timestamp * 1000 if timestamp < 1_000_000_000_000 else timestamp
        return records

    async def update_processed_data(self):
        self.processed_data["signal"] = 0
        try:
            now = self.market_data_provider.time()
            exchange_position_amount = self._exchange_position_amount()
            self.processed_data["exchange_position_amount"] = str(exchange_position_amount)
            active_executor_exists = any(bool(getattr(executor, "is_active", False)) for executor in self.executors_info)
            orphan_position = exchange_position_amount > 0 and not active_executor_exists
            self.processed_data["orphan_position"] = orphan_position
            self._sync_executor_lifecycle(now, exchange_position_amount)
            if orphan_position:
                self.processed_data["risk_status"] = {
                    "gate": "blocked",
                    "reason": "exchange_position_without_active_executor",
                    "exchange_position_amount": str(exchange_position_amount),
                }
                self._set_state(
                    "orphaned_position", now,
                    last_rejection_reason="exchange_position_without_active_executor",
                )
                return
            _, risk_status = evaluate_execution_risk(
                self.executors_info, now, Decimal(0),
                max_daily_loss_quote=self.config.max_daily_loss_quote,
                max_consecutive_losses=self.config.max_consecutive_losses,
                max_open_risk_quote=self.config.max_open_risk_quote,
            )
            self.processed_data["risk_status"] = risk_status
            five_frame = self.market_data_provider.get_candles_df(
                self.candles_connector, self.candles_trading_pair, "5m", 300
            )
            one_frame = self.market_data_provider.get_candles_df(
                self.candles_connector, self.candles_trading_pair, "1h", 120
            )
            four_frame = self.market_data_provider.get_candles_df(
                self.candles_connector, self.candles_trading_pair, "4h", 120
            )
            features = self._build_features(
                self._closed_records(five_frame),
                self._closed_records(one_frame),
                self._closed_records(four_frame),
            )
            active_executors = tuple(
                {
                    "id": executor.id,
                    "side": str(executor.side),
                    "is_active": str(executor.is_active),
                }
                for executor in self.executors_info
            )
            self.processed_data["features"] = five_frame
            if self.config.strategy_version == "liquidity_profile_v2":
                if self._armed_event is None:
                    event = self._scan_candidate(
                        features, self.config.id, self.agent_connector,
                        self.config.trading_pair, now, self.config.trigger_distance_atr,
                        active_executors=active_executors,
                        strategy_version=self.config.strategy_version,
                    )
                    if event is not None and self._journal.is_terminal(event.event_id):
                        event = None
                    if event is not None:
                        self._armed_event = event
                        self._journal.record(
                            event.event_id, "armed", now,
                            trigger_type=event.trigger_type, location_id=event.location_id,
                            invalidation_operator=event.invalidation_operator,
                            invalidation_level=str(event.invalidation_level),
                        )
                        self._set_state(
                            "armed", now,
                            last_event_id=event.event_id,
                            last_event_at=now,
                            last_rejection_reason=None,
                            armed_invalidation_operator=event.invalidation_operator,
                            armed_trigger_type=event.trigger_type,
                            armed_location_id=event.location_id,
                            armed_invalidation_level=str(event.invalidation_level),
                        )
                    else:
                        self._set_state("idle", now)
                    return
                armed_candle = int(self._armed_event.recent_candles_5m[-1]["timestamp"])
                if features.last_closed_5m_timestamp <= armed_candle:
                    self._set_state("armed", now)
                    return
                armed_event = self._armed_event
                event = revalidate_armed_setup(
                    armed_event, features, now, active_executors
                )
                self._armed_event = None
                if event is None:
                    self._journal.record(armed_event.event_id, "cancelled", now, reason="invalidated")
                    self._set_state(
                        "invalidated", now,
                        last_rejection_reason="armed setup invalidated",
                    )
                    return
                self._event_setup_ids[event.event_id] = armed_event.event_id
            if event is None:
                self._set_state("idle", now)
                return
            if not self._registry.start(event.event_id, now):
                return
            setup_id = self._setup_id(event.event_id)
            self._set_state(
                "analyzing", now, last_event_id=event.event_id, last_event_at=now,
            )
            self.processed_data["last_rejection_reason"] = None
            try:
                payload = await asyncio.to_thread(self._adapter.analyze, event)
                price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
                )
                plan = parse_agent_result(
                    payload, event, self.market_data_provider.time(), Decimal(price), self.config.minimum_rr
                )
            except Exception as error:
                failed_at = self.market_data_provider.time()
                self._registry.mark_finished(event.event_id, failed_at)
                self._journal.record(setup_id, "cancelled", failed_at, reason=str(error))
                self._set_state(
                    "rejected", failed_at, last_rejection_reason=str(error), last_error_at=failed_at,
                )
                return
            if plan is None:
                decided_at = self.market_data_provider.time()
                self._registry.mark_finished(event.event_id, decided_at)
                self._journal.record(setup_id, "cancelled", decided_at, decision=payload.get("decision"))
                self._set_state(
                    "rejected", decided_at, last_agent_decision=payload.get("decision"),
                    last_agent_decision_at=decided_at,
                )
                return
            self._pending_plan = plan
            decided_at = self.market_data_provider.time()
            plan_payload = self._plan_payload(plan, "planned")
            self._journal.record(setup_id, "confirmed", decided_at, plan=plan_payload)
            self._set_state(
                "validated", decided_at,
                signal=1 if plan.decision == "long" else -1,
                last_agent_decision=plan.decision,
                last_agent_decision_at=decided_at,
                last_execution_plan=plan_payload,
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            failed_at = self.market_data_provider.time()
            self._set_state(
                "rejected", failed_at, last_rejection_reason=str(error), last_error_at=failed_at,
            )

    def determine_executor_actions(self) -> List[ExecutorAction]:
        plan = self._pending_plan
        if plan is None:
            return []
        now = self.market_data_provider.time()
        setup_id = self._setup_id(plan.event_id)
        if self._group_exists(plan.event_id):
            self._registry.mark_group_created(plan.event_id)
            self._pending_plan = None
            self._set_state("already_executed", now)
            return []
        if now >= plan.valid_until:
            self._registry.mark_finished(plan.event_id, now)
            self._pending_plan = None
            self._journal.record(setup_id, "cancelled", now, reason="analysis expired")
            self._set_state("rejected", now, last_rejection_reason="analysis expired")
            return []
        price = Decimal(self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        ))
        invalid_reason = self._permanent_plan_invalid_reason(plan, price)
        if invalid_reason is not None:
            self._registry.mark_finished(plan.event_id, now)
            self._pending_plan = None
            self._journal.record(setup_id, "cancelled", now, reason=invalid_reason)
            self._set_state("rejected", now, last_rejection_reason=invalid_reason)
            return []
        group_quote = self.config.total_amount_quote / Decimal(self.config.max_executors_per_side)
        planned_risk = min(
            self.config.risk_amount_quote or Decimal("Infinity"),
            group_quote * abs(price - plan.stop) / price,
        )
        risk_reason, risk_status = evaluate_execution_risk(
            self.executors_info, now, planned_risk,
            max_daily_loss_quote=self.config.max_daily_loss_quote,
            max_consecutive_losses=self.config.max_consecutive_losses,
            max_open_risk_quote=self.config.max_open_risk_quote,
        )
        self.processed_data["risk_status"] = risk_status
        if risk_reason:
            self._registry.mark_finished(plan.event_id, now)
            self._pending_plan = None
            self._journal.record(setup_id, "cancelled", now, reason=risk_reason)
            self._set_state("risk_blocked", now, last_rejection_reason=risk_reason)
            return []
        actions = self._create_actions_at_price(plan, price)
        if actions:
            self._registry.mark_group_created(plan.event_id)
            self._pending_plan = None
            plan_payload = self._plan_payload(plan, "configured")
            self._journal.record(
                setup_id, "executing", now,
                execution_event_id=plan.event_id, plan=plan_payload,
            )
            self._event_setup_ids[plan.event_id] = setup_id
            self._set_state("executing", now, last_execution_plan=plan_payload)
        return actions

    def _permanent_plan_invalid_reason(
        self, plan: ValidatedExecutionPlan, price: Decimal
    ) -> str | None:
        if not plan.entry_low <= price <= plan.entry_high:
            return "price left entry range"
        risk = price - plan.stop if plan.decision == "long" else plan.stop - price
        if risk <= 0:
            return "stop is invalid at current price"
        return None

    def _group_exists(self, event_id: str) -> bool:
        prefix = f"{event_id}:"
        return any(
            str(getattr(executor.config, "level_id", "") or "").startswith(prefix)
            for executor in self.executors_info
        )

    def _create_actions_at_price(
        self, plan: ValidatedExecutionPlan, price: Decimal
    ) -> List[CreateExecutorAction]:
        signal = 1 if plan.decision == "long" else -1
        active_same_side = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda executor: executor.is_active and (
                executor.side == TradeType.BUY if signal > 0 else executor.side == TradeType.SELL
            ),
        )
        if len(active_same_side) + len(plan.targets) > self.config.max_executors_per_side:
            return []
        if not self.can_create_executor(signal):
            return []
        side = TradeType.BUY if signal > 0 else TradeType.SELL
        group_quote = self.config.total_amount_quote / Decimal(self.config.max_executors_per_side)
        stop_loss = abs(price - plan.stop) / price
        configs = []
        for index, target in enumerate(plan.targets):
            take_profit = abs(target.price - price) / price
            amount = execution_amount(
                price,
                plan.stop,
                target.allocation,
                group_quote,
                self.config.risk_amount_quote,
            )
            barrier = TripleBarrierConfig(
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_limit=FAMILY_TIME_LIMIT_SECONDS[plan.execution_family],
                trailing_stop=self.config.trailing_stop,
                open_order_type=OrderType.MARKET,
                take_profit_order_type=OrderType.LIMIT,
                stop_loss_order_type=OrderType.MARKET,
                time_limit_order_type=OrderType.MARKET,
            )
            configs.append(PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=side,
                entry_price=price,
                amount=amount,
                triple_barrier_config=barrier,
                leverage=self.config.leverage,
                level_id=f"{plan.event_id}:{index}",
            ))
        return [CreateExecutorAction(controller_id=self.config.id, executor_config=config) for config in configs]
