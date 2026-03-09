"""Compute Market environment implementation."""

from __future__ import annotations

import os
import random
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from openenv.core.env_server import Environment

try:
    from ..models import (
        ActorProfile,
        ActorSignal,
        ComputeMarketAction,
        ComputeMarketObservation,
        ComputeMarketState,
        JobRecord,
        MarketEvent,
        MarketOffer,
        ReservationRecord,
    )
except ImportError:
    from models import (
        ActorProfile,
        ActorSignal,
        ComputeMarketAction,
        ComputeMarketObservation,
        ComputeMarketState,
        JobRecord,
        MarketEvent,
        MarketOffer,
        ReservationRecord,
    )


class ComputeMarketEnvironment(
    Environment[ComputeMarketAction, ComputeMarketObservation, ComputeMarketState]
):
    """Single-agent compute allocation market with scripted counterparties."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        total_gpus: int = 8,
        initial_budget: float = 150.0,
        max_ticks: int = 12,
        default_seed: int = 0,
    ) -> None:
        self.base_total_gpus = total_gpus
        self.base_initial_budget = initial_budget
        self.base_max_ticks = max_ticks
        self.total_gpus = total_gpus
        self.initial_budget = initial_budget
        self.max_ticks = max_ticks
        self.default_seed = default_seed
        self._scenario_variant = "baseline"
        self._price_bias = 0.0
        self._broker_enabled = True
        self._state = ComputeMarketState(
            episode_id=str(uuid4()),
            step_count=0,
            scenario_seed=default_seed,
            scenario_variant="baseline",
            max_ticks=max_ticks,
            total_gpus=total_gpus,
            budget_remaining=initial_budget,
        )
        self._rng = random.Random(default_seed)
        self._jobs: list[JobRecord] = []
        self._reservations: list[ReservationRecord] = []
        self._visible_offers: list[MarketOffer] = []
        self._hidden_actors: list[ActorProfile] = []
        self._actor_signals: list[ActorSignal] = []
        self._recent_events: list[MarketEvent] = []
        self._current_tick = 0
        self._market_price = 0.0
        self._free_gpus = total_gpus
        self._external_allocated_gpus = 0
        self._budget_remaining = initial_budget
        self._cumulative_reward = 0.0
        self._done = False
        self._initialize_episode(default_seed, episode_id=self._state.episode_id)

    def _initialize_episode(
        self,
        scenario_seed: int,
        episode_id: str | None = None,
        scenario_variant: str = "baseline",
    ) -> None:
        self._apply_scenario_variant(scenario_variant)
        self._rng = random.Random(scenario_seed)
        self._current_tick = 0
        self._done = False
        self._budget_remaining = float(self.initial_budget)
        self._cumulative_reward = 0.0
        self._jobs = self._build_jobs()
        self._reservations = []
        self._hidden_actors = self._build_actors()
        self._visible_offers = []
        self._actor_signals = []
        self._recent_events = [
            MarketEvent(
                tick=0,
                event_type="reset",
                message=f"Scenario '{self._scenario_variant}' initialized with scripted counterparties.",
            )
        ]
        self._refresh_market()
        self._state = self._snapshot_state(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_seed=scenario_seed,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> ComputeMarketObservation:
        scenario_seed = self.default_seed if seed is None else seed
        self._initialize_episode(
            scenario_seed,
            episode_id=episode_id,
            scenario_variant=kwargs.get("scenario_variant", "baseline"),
        )
        return self._build_observation(
            0.0,
            False,
            {"status": "ready", "scenario_variant": self._scenario_variant},
        )

    def step(self, action: ComputeMarketAction) -> ComputeMarketObservation:  # type: ignore[override]
        if self._done:
            return self._build_observation(
                0.0,
                True,
                {"error": "Episode already finished."},
            )

        self._state.step_count += 1
        reward = 0.0
        action_events: list[MarketEvent] = []
        error: str | None = None

        if action.action_type == "bid_for_capacity":
            reward, error, action_events = self._handle_bid(action)
        elif action.action_type == "accept_offer":
            reward, error, action_events = self._handle_accept_offer(action)
        elif action.action_type == "propose_swap":
            reward, error, action_events = self._handle_swap(action)
        elif action.action_type == "schedule_job":
            reward, error, action_events = self._handle_schedule(action)
        elif action.action_type == "delay_job":
            reward, error, action_events = self._handle_delay(action)
        elif action.action_type == "inspect_market":
            action_events = [
                self._event(
                    "inspect",
                    f"Market inspected: spot price ${self._market_price:.2f}, free GPUs {self._free_gpus}.",
                )
            ]
            reward -= 0.25
        elif action.action_type == "noop":
            action_events = [self._event("noop", "No action taken this tick.")]
        else:
            error = f"Unsupported action type: {action.action_type}"
            reward -= 2.0

        advance_reward, advance_events = self._advance_tick()
        total_reward = round(reward + advance_reward, 2)
        combined_events = action_events + advance_events
        self._recent_events = combined_events[-6:]
        self._cumulative_reward = round(self._cumulative_reward + total_reward, 2)
        self._state = self._snapshot_state(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            scenario_seed=self._state.scenario_seed,
        )
        metadata = {
            "events": [event.model_dump() for event in combined_events],
            "scenario_variant": self._scenario_variant,
        }
        if error:
            metadata["error"] = error
        return self._build_observation(total_reward, self._done, metadata)

    @property
    def state(self) -> ComputeMarketState:
        return self._state

    def _apply_scenario_variant(self, variant: str) -> None:
        allowed = {
            "baseline",
            "tight_capacity",
            "price_shock",
            "policy_shift",
            "job_mix",
        }
        self._scenario_variant = variant if variant in allowed else "baseline"
        self.total_gpus = self.base_total_gpus
        self.initial_budget = self.base_initial_budget
        self.max_ticks = self.base_max_ticks
        self._price_bias = 0.0
        self._broker_enabled = True

        if self._scenario_variant == "tight_capacity":
            self.total_gpus = max(4, self.base_total_gpus - 2)
        elif self._scenario_variant == "price_shock":
            self._price_bias = 2.25
        elif self._scenario_variant == "policy_shift":
            self._broker_enabled = False
        elif self._scenario_variant == "job_mix":
            self.max_ticks = self.base_max_ticks + 1

    def _build_jobs(self) -> list[JobRecord]:
        jitter = self._rng.randint(-4, 4)
        if self._scenario_variant == "job_mix":
            return [
                JobRecord(
                    job_id="job-a",
                    gpu_count=3,
                    total_duration=3,
                    remaining_duration=3,
                    deadline=6,
                    value=82 + jitter,
                    priority=2,
                ),
                JobRecord(
                    job_id="job-b",
                    gpu_count=2,
                    total_duration=1,
                    remaining_duration=1,
                    deadline=4,
                    value=36 + self._rng.randint(-2, 2),
                    priority=3,
                ),
                JobRecord(
                    job_id="job-c",
                    gpu_count=3,
                    total_duration=2,
                    remaining_duration=2,
                    deadline=8,
                    value=58 + self._rng.randint(-3, 3),
                    priority=2,
                    depends_on=["job-b"],
                ),
            ]

        return [
            JobRecord(
                job_id="job-a",
                gpu_count=4,
                total_duration=2,
                remaining_duration=2,
                deadline=4,
                value=100 + jitter,
                priority=3,
            ),
            JobRecord(
                job_id="job-b",
                gpu_count=2,
                total_duration=2,
                remaining_duration=2,
                deadline=7,
                value=46 + self._rng.randint(-3, 3),
                priority=2,
            ),
            JobRecord(
                job_id="job-c",
                gpu_count=1,
                total_duration=1,
                remaining_duration=1,
                deadline=8,
                value=24 + self._rng.randint(-2, 2),
                priority=1,
                depends_on=["job-a"],
            ),
        ]

    def _build_actors(self) -> list[ActorProfile]:
        actors = [
            ActorProfile(
                actor_id="urgent-tenant",
                policy_type="urgent_tenant",
                max_bid=round(7.0 + self._rng.uniform(0.5, 1.5), 2),
                preferred_gpu_count=4 + self._rng.randint(0, 2),
                visible_behavior="aggressive",
                swap_floor=round(6.0 + self._rng.uniform(0.2, 0.8), 2),
            ),
            ActorProfile(
                actor_id="budget-tenant",
                policy_type="cost_sensitive_tenant",
                max_bid=round(4.5 + self._rng.uniform(0.2, 1.0), 2),
                preferred_gpu_count=2 + self._rng.randint(0, 1),
                visible_behavior="steady",
                swap_floor=round(4.0 + self._rng.uniform(0.2, 0.8), 2),
            ),
            ActorProfile(
                actor_id="broker-1",
                policy_type="broker",
                max_bid=round(6.0 + self._rng.uniform(0.2, 1.2), 2),
                preferred_gpu_count=3 + self._rng.randint(0, 2),
                visible_behavior="opportunistic",
                swap_floor=round(5.0 + self._rng.uniform(0.2, 0.8), 2),
            ),
        ]

        if self._scenario_variant == "policy_shift":
            actors[0].max_bid = round(max(3.5, actors[0].max_bid - 2.0), 2)
            actors[0].visible_behavior = "steady"
            actors[1].max_bid = round(actors[1].max_bid + 1.4, 2)
            actors[1].preferred_gpu_count += 1
            actors[1].visible_behavior = "aggressive"

        if not self._broker_enabled:
            actors = [actor for actor in actors if actor.policy_type != "broker"]
        return actors

    def _handle_bid(self, action: ComputeMarketAction) -> tuple[float, str | None, list[MarketEvent]]:
        if action.gpu_count <= 0:
            return -2.0, "gpu_count must be positive.", []
        if action.price_per_gpu <= 0:
            return -2.0, "price_per_gpu must be positive.", []
        if action.gpu_count > self._free_gpus:
            return -2.0, f"Only {self._free_gpus} public GPUs are available this tick.", []
        if self._owned_gpus() + action.gpu_count > self.total_gpus:
            return -2.0, "Cluster capacity would be exceeded.", []
        if action.price_per_gpu < self._market_price:
            return -1.0, f"Bid ${action.price_per_gpu:.2f} is below current clearing price ${self._market_price:.2f}.", []
        total_cost = round(action.gpu_count * action.price_per_gpu * action.duration, 2)
        if total_cost > self._budget_remaining:
            return -2.0, "Insufficient budget for bid.", []

        self._budget_remaining = round(self._budget_remaining - total_cost, 2)
        self._reservations.append(
            ReservationRecord(
                reservation_id=f"res-{uuid4().hex[:8]}",
                source="spot-market",
                gpu_count=action.gpu_count,
                remaining_ticks=action.duration,
                price_per_gpu=action.price_per_gpu,
                acquired_at_tick=self._current_tick,
            )
        )
        return (
            -total_cost,
            None,
            [
                self._event(
                    "bid_won",
                    f"Won {action.gpu_count} GPU(s) for {action.duration} tick(s) at ${action.price_per_gpu:.2f}/GPU.",
                )
            ],
        )

    def _handle_accept_offer(self, action: ComputeMarketAction) -> tuple[float, str | None, list[MarketEvent]]:
        if not action.offer_id:
            return -2.0, "offer_id is required.", []
        offer = next((item for item in self._visible_offers if item.offer_id == action.offer_id), None)
        if offer is None:
            return -2.0, f"Offer {action.offer_id} is not available.", []
        if self._owned_gpus() + offer.gpu_count > self.total_gpus:
            return -2.0, "Cluster capacity would be exceeded.", []

        total_cost = round(offer.gpu_count * offer.price_per_gpu * offer.duration, 2)
        if total_cost > self._budget_remaining:
            return -2.0, "Insufficient budget for offer.", []

        self._budget_remaining = round(self._budget_remaining - total_cost, 2)
        self._reservations.append(
            ReservationRecord(
                reservation_id=f"res-{uuid4().hex[:8]}",
                source=offer.actor_id,
                gpu_count=offer.gpu_count,
                remaining_ticks=offer.duration,
                price_per_gpu=offer.price_per_gpu,
                acquired_at_tick=self._current_tick,
            )
        )
        self._visible_offers = [item for item in self._visible_offers if item.offer_id != offer.offer_id]
        return (
            -total_cost,
            None,
            [
                self._event(
                    "offer_accepted",
                    f"Accepted {offer.offer_type} offer from {offer.actor_id} for {offer.gpu_count} GPU(s).",
                )
            ],
        )

    def _handle_swap(self, action: ComputeMarketAction) -> tuple[float, str | None, list[MarketEvent]]:
        if not action.actor_id:
            return -2.0, "actor_id is required.", []
        if action.gpu_count <= 0 or action.price_per_gpu <= 0:
            return -2.0, "gpu_count and price_per_gpu must be positive.", []
        actor = next((item for item in self._hidden_actors if item.actor_id == action.actor_id), None)
        if actor is None:
            return -2.0, f"Unknown actor {action.actor_id}.", []
        if self._owned_gpus() + action.gpu_count > self.total_gpus:
            return -2.0, "Cluster capacity would be exceeded.", []
        total_cost = round(action.gpu_count * action.price_per_gpu * action.duration, 2)
        if total_cost > self._budget_remaining:
            return -2.0, "Insufficient budget for swap.", []
        if action.price_per_gpu < actor.swap_floor:
            return -1.0, f"{actor.actor_id} rejected the swap; offered price is below its floor.", []

        self._budget_remaining = round(self._budget_remaining - total_cost, 2)
        self._reservations.append(
            ReservationRecord(
                reservation_id=f"res-{uuid4().hex[:8]}",
                source=f"swap:{actor.actor_id}",
                gpu_count=action.gpu_count,
                remaining_ticks=action.duration,
                price_per_gpu=action.price_per_gpu,
                acquired_at_tick=self._current_tick,
            )
        )
        return (
            -total_cost,
            None,
            [
                self._event(
                    "swap_accepted",
                    f"{actor.actor_id} transferred {action.gpu_count} GPU(s) at ${action.price_per_gpu:.2f}/GPU.",
                )
            ],
        )

    def _handle_schedule(self, action: ComputeMarketAction) -> tuple[float, str | None, list[MarketEvent]]:
        if not action.job_id:
            return -2.0, "job_id is required.", []
        job = self._job(action.job_id)
        if job is None:
            return -2.0, f"Unknown job {action.job_id}.", []
        if job.status in {"completed", "missed"}:
            return -1.0, f"Job {action.job_id} is already terminal.", []
        if not self._deps_completed(job):
            return -1.0, f"Job {action.job_id} is blocked on dependencies {job.depends_on}.", []
        if self._idle_owned_gpus() < job.gpu_count:
            return -1.0, f"Need {job.gpu_count} idle owned GPU(s) to start {job.job_id}.", []

        job.status = "running"
        if job.started_at is None:
            job.started_at = self._current_tick
        return (
            1.0,
            None,
            [
                self._event(
                    "job_started",
                    f"Scheduled {job.job_id} using {job.gpu_count} GPU(s).",
                )
            ],
        )

    def _handle_delay(self, action: ComputeMarketAction) -> tuple[float, str | None, list[MarketEvent]]:
        if not action.job_id:
            return -2.0, "job_id is required.", []
        job = self._job(action.job_id)
        if job is None:
            return -2.0, f"Unknown job {action.job_id}.", []
        if job.status in {"completed", "missed"}:
            return -1.0, f"Job {job.job_id} is already terminal.", []
        job.status = "paused" if job.status == "running" else "pending"
        job.delay_count += 1
        return (
            -1.0,
            None,
            [
                self._event(
                    "job_delayed",
                    f"Delayed {job.job_id}; slack shrinks while the deadline stays fixed.",
                )
            ],
        )

    def _advance_tick(self) -> tuple[float, list[MarketEvent]]:
        tick_reward = 0.0
        events: list[MarketEvent] = []
        available_owned = self._owned_gpus()
        used_gpus = 0

        running_jobs = sorted(
            [job for job in self._jobs if job.status == "running"],
            key=lambda item: (-item.priority, item.job_id),
        )
        for job in running_jobs:
            if used_gpus + job.gpu_count <= available_owned:
                used_gpus += job.gpu_count
                job.remaining_duration -= 1
                events.append(
                    self._event(
                        "job_progress",
                        f"{job.job_id} progressed; {job.remaining_duration} tick(s) remaining.",
                    )
                )
                if job.remaining_duration == 0:
                    job.status = "completed"
                    job.completed_at = self._current_tick + 1
                    tick_reward += job.value
                    events.append(
                        self._event(
                            "job_completed",
                            f"{job.job_id} completed before deadline and earned ${job.value:.2f}.",
                        )
                    )
            else:
                job.status = "paused"
                tick_reward -= 3.0
                events.append(
                    self._event(
                        "job_paused",
                        f"{job.job_id} paused because owned capacity dropped below demand.",
                    )
                )

        idle_owned = max(0, available_owned - used_gpus)
        if idle_owned > 0:
            idle_penalty = round(0.5 * idle_owned, 2)
            tick_reward -= idle_penalty
            events.append(
                self._event(
                    "idle_penalty",
                    f"Paid ${idle_penalty:.2f} idle-hoarding penalty for {idle_owned} unused owned GPU(s).",
                )
            )

        for reservation in self._reservations:
            reservation.remaining_ticks = max(0, reservation.remaining_ticks - 1)
        expired = [item for item in self._reservations if item.remaining_ticks == 0]
        self._reservations = [item for item in self._reservations if item.remaining_ticks > 0]
        for reservation in expired:
            events.append(
                self._event(
                    "reservation_expired",
                    f"Reservation {reservation.reservation_id} from {reservation.source} expired.",
                )
            )

        next_tick = self._current_tick + 1
        for job in self._jobs:
            if job.status not in {"completed", "missed"} and next_tick > job.deadline:
                job.status = "missed"
                penalty = round(job.value * 0.6, 2)
                tick_reward -= penalty
                events.append(
                    self._event(
                        "deadline_missed",
                        f"{job.job_id} missed its deadline and incurred ${penalty:.2f} penalty.",
                    )
                )

        self._current_tick = next_tick
        self._done = self._current_tick >= self.max_ticks or all(
            job.status in {"completed", "missed"} for job in self._jobs
        ) or self._budget_remaining <= 0.0

        if not self._done:
            self._refresh_market()
            events.extend(self._market_events_for_tick())
        else:
            self._visible_offers = []
            self._actor_signals = []
            self._free_gpus = max(0, self.total_gpus - self._owned_gpus())

        return round(tick_reward, 2), events

    def _refresh_market(self) -> None:
        owned = self._owned_gpus()
        remaining_cluster = max(0, self.total_gpus - owned)
        base_price = 4.0 + 0.3 * self._current_tick + self._rng.uniform(0.0, 1.0) + self._price_bias
        actor_signals: list[ActorSignal] = []
        visible_offers: list[MarketOffer] = []
        external_demand = 0

        for actor in self._hidden_actors:
            if actor.policy_type == "urgent_tenant":
                gpu_demand = max(2, actor.preferred_gpu_count - (self._current_tick // 3))
                bid = round(actor.max_bid - 0.15 * self._current_tick, 2)
                pressure = "high" if gpu_demand >= 4 else "medium"
            elif actor.policy_type == "cost_sensitive_tenant":
                gpu_demand = max(1, actor.preferred_gpu_count - (self._current_tick // 4))
                bid = round(actor.max_bid - 0.1 * max(0, self._current_tick - 1), 2)
                pressure = "medium" if gpu_demand >= 2 else "low"
            else:
                gpu_demand = actor.preferred_gpu_count
                bid = round(actor.max_bid + 0.2 * self._current_tick, 2)
                pressure = "medium"
                if self._broker_enabled:
                    visible_offers.append(
                        MarketOffer(
                            offer_id=f"offer-{self._current_tick}-{actor.actor_id}",
                            actor_id=actor.actor_id,
                            gpu_count=min(
                                remaining_cluster or actor.preferred_gpu_count,
                                actor.preferred_gpu_count,
                            ),
                            price_per_gpu=round(bid + 0.6, 2),
                            duration=2,
                            expires_at_tick=self._current_tick + 1,
                            offer_type="broker",
                        )
                    )

            if actor.policy_type != "broker":
                external_demand += gpu_demand
            actor_signals.append(
                ActorSignal(
                    actor_id=actor.actor_id,
                    visible_behavior=actor.visible_behavior,
                    pressure_hint=pressure,
                    last_seen_bid=max(0.0, bid),
                )
            )

        self._external_allocated_gpus = min(remaining_cluster, external_demand)
        self._free_gpus = max(0, remaining_cluster - self._external_allocated_gpus)
        pressure_bump = 0.45 * (external_demand / max(1, self.total_gpus))
        self._market_price = round(base_price + pressure_bump, 2)
        self._visible_offers = [
            offer for offer in visible_offers if offer.gpu_count > 0 and offer.expires_at_tick >= self._current_tick
        ]
        self._actor_signals = actor_signals

    def _market_events_for_tick(self) -> list[MarketEvent]:
        messages = [
            self._event(
                "market_tick",
                f"Tick {self._current_tick}: spot price ${self._market_price:.2f}, public free GPUs {self._free_gpus}.",
            )
        ]
        for signal in self._actor_signals:
            messages.append(
                self._event(
                    "actor_signal",
                    f"{signal.actor_id} looks {signal.visible_behavior} with {signal.pressure_hint} pressure.",
                )
            )
        for offer in self._visible_offers:
            messages.append(
                self._event(
                    "offer_visible",
                    f"{offer.actor_id} posted {offer.gpu_count} GPU(s) at ${offer.price_per_gpu:.2f}/GPU for {offer.duration} tick(s).",
                )
            )
        return messages

    def _snapshot_state(
        self,
        episode_id: str,
        step_count: int,
        scenario_seed: int,
    ) -> ComputeMarketState:
        return ComputeMarketState(
            episode_id=episode_id,
            step_count=step_count,
            scenario_seed=scenario_seed,
            scenario_variant=self._scenario_variant,
            current_tick=self._current_tick,
            max_ticks=self.max_ticks,
            total_gpus=self.total_gpus,
            free_gpus=self._free_gpus,
            owned_gpus=self._owned_gpus(),
            idle_owned_gpus=self._idle_owned_gpus(),
            budget_remaining=self._budget_remaining,
            market_price=self._market_price,
            cumulative_reward=self._cumulative_reward,
            external_allocated_gpus=self._external_allocated_gpus,
            done=self._done,
            jobs=[job.model_copy(deep=True) for job in self._jobs],
            visible_offers=[offer.model_copy(deep=True) for offer in self._visible_offers],
            reservations=[reservation.model_copy(deep=True) for reservation in self._reservations],
            actor_signals=[signal.model_copy(deep=True) for signal in self._actor_signals],
            hidden_actors=[actor.model_copy(deep=True) for actor in self._hidden_actors],
            recent_events=[event.model_copy(deep=True) for event in self._recent_events],
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict,
    ) -> ComputeMarketObservation:
        return ComputeMarketObservation(
            scenario_variant=self._scenario_variant,
            current_tick=self._current_tick,
            max_ticks=self.max_ticks,
            total_gpus=self.total_gpus,
            free_gpus=self._free_gpus,
            owned_gpus=self._owned_gpus(),
            idle_owned_gpus=self._idle_owned_gpus(),
            budget_remaining=self._budget_remaining,
            market_price=self._market_price,
            jobs=[job.model_copy(deep=True) for job in self._jobs],
            visible_offers=[offer.model_copy(deep=True) for offer in self._visible_offers],
            recent_events=[event.model_copy(deep=True) for event in self._recent_events],
            actor_signals=[signal.model_copy(deep=True) for signal in self._actor_signals],
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def _deps_completed(self, job: JobRecord) -> bool:
        if not job.depends_on:
            return True
        completed = {item.job_id for item in self._jobs if item.status == "completed"}
        return all(dep in completed for dep in job.depends_on)

    def _job(self, job_id: str) -> JobRecord | None:
        return next((job for job in self._jobs if job.job_id == job_id), None)

    def _owned_gpus(self) -> int:
        return sum(reservation.gpu_count for reservation in self._reservations)

    def _idle_owned_gpus(self) -> int:
        running_gpu_demand = sum(job.gpu_count for job in self._jobs if job.status == "running")
        return max(0, self._owned_gpus() - running_gpu_demand)

    def _event(self, event_type: str, message: str) -> MarketEvent:
        return MarketEvent(tick=self._current_tick, event_type=event_type, message=message)


def create_environment_from_env() -> ComputeMarketEnvironment:
    """Factory used by the FastAPI app and tests."""

    return ComputeMarketEnvironment(
        total_gpus=int(os.getenv("COMPUTE_MARKET_TOTAL_GPUS", "8")),
        initial_budget=float(os.getenv("COMPUTE_MARKET_INITIAL_BUDGET", "150")),
        max_ticks=int(os.getenv("COMPUTE_MARKET_MAX_TICKS", "12")),
        default_seed=int(os.getenv("COMPUTE_MARKET_DEFAULT_SEED", "0")),
    )
