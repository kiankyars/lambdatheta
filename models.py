"""Data models for the Compute Market environment."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class JobRecord(BaseModel):
    """Represents an agent-owned job in the simulated compute market."""

    job_id: str
    gpu_count: int = Field(..., ge=1)
    total_duration: int = Field(..., ge=1)
    remaining_duration: int = Field(..., ge=0)
    deadline: int = Field(..., ge=1)
    value: float = Field(..., ge=0.0)
    priority: int = Field(default=1, ge=1)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "paused", "completed", "missed"] = "pending"
    delay_count: int = Field(default=0, ge=0)
    started_at: int | None = None
    completed_at: int | None = None


class MarketOffer(BaseModel):
    """Visible capacity offer from a scripted actor."""

    offer_id: str
    actor_id: str
    gpu_count: int = Field(..., ge=1)
    price_per_gpu: float = Field(..., ge=0.0)
    duration: int = Field(..., ge=1)
    expires_at_tick: int = Field(..., ge=0)
    offer_type: Literal["broker", "swap"] = "broker"


class ReservationRecord(BaseModel):
    """Capacity currently owned by the agent."""

    reservation_id: str
    source: str
    gpu_count: int = Field(..., ge=1)
    remaining_ticks: int = Field(..., ge=0)
    price_per_gpu: float = Field(..., ge=0.0)
    acquired_at_tick: int = Field(..., ge=0)


class MarketEvent(BaseModel):
    """Human-readable event surfaced in observations."""

    tick: int = Field(..., ge=0)
    event_type: str
    message: str


class ActorSignal(BaseModel):
    """Public signal about a scripted actor."""

    actor_id: str
    visible_behavior: Literal["aggressive", "steady", "opportunistic"]
    pressure_hint: Literal["low", "medium", "high"]
    last_seen_bid: float = Field(..., ge=0.0)


class ActorProfile(BaseModel):
    """Hidden actor configuration kept in the control-plane state."""

    actor_id: str
    policy_type: Literal["urgent_tenant", "cost_sensitive_tenant", "broker"]
    max_bid: float = Field(..., ge=0.0)
    preferred_gpu_count: int = Field(..., ge=1)
    visible_behavior: Literal["aggressive", "steady", "opportunistic"]
    swap_floor: float = Field(..., ge=0.0)


class ComputeMarketAction(Action):
    """Single-step action for the compute market."""

    action_type: Literal[
        "bid_for_capacity",
        "accept_offer",
        "propose_swap",
        "schedule_job",
        "delay_job",
        "inspect_market",
        "noop",
    ]
    job_id: str | None = Field(default=None)
    offer_id: str | None = Field(default=None)
    actor_id: str | None = Field(default=None)
    gpu_count: int = Field(default=0, ge=0)
    price_per_gpu: float = Field(default=0.0, ge=0.0)
    duration: int = Field(default=1, ge=1)


class ComputeMarketObservation(Observation):
    """Partial observation exposed to the trained agent."""

    scenario_variant: str = "baseline"
    current_tick: int = Field(default=0, ge=0)
    max_ticks: int = Field(default=0, ge=0)
    total_gpus: int = Field(default=0, ge=0)
    free_gpus: int = Field(default=0, ge=0)
    owned_gpus: int = Field(default=0, ge=0)
    idle_owned_gpus: int = Field(default=0, ge=0)
    budget_remaining: float = Field(default=0.0)
    market_price: float = Field(default=0.0, ge=0.0)
    jobs: list[JobRecord] = Field(default_factory=list)
    visible_offers: list[MarketOffer] = Field(default_factory=list)
    recent_events: list[MarketEvent] = Field(default_factory=list)
    actor_signals: list[ActorSignal] = Field(default_factory=list)


class ComputeMarketState(State):
    """Full control-plane state including hidden actor data."""

    scenario_seed: int = 0
    scenario_variant: str = "baseline"
    current_tick: int = 0
    max_ticks: int = 0
    total_gpus: int = 0
    free_gpus: int = 0
    owned_gpus: int = 0
    idle_owned_gpus: int = 0
    budget_remaining: float = 0.0
    market_price: float = 0.0
    cumulative_reward: float = 0.0
    external_allocated_gpus: int = 0
    done: bool = False
    jobs: list[JobRecord] = Field(default_factory=list)
    visible_offers: list[MarketOffer] = Field(default_factory=list)
    reservations: list[ReservationRecord] = Field(default_factory=list)
    actor_signals: list[ActorSignal] = Field(default_factory=list)
    hidden_actors: list[ActorProfile] = Field(default_factory=list)
    recent_events: list[MarketEvent] = Field(default_factory=list)
