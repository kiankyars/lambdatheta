"""Compute Market environment client."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
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


class ComputeMarketEnv(
    EnvClient[ComputeMarketAction, ComputeMarketObservation, ComputeMarketState]
):
    """Persistent client for the compute market environment."""

    def _step_payload(self, action: ComputeMarketAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ComputeMarketObservation]:
        obs_data = payload.get("observation", {})
        observation = ComputeMarketObservation(
            scenario_variant=obs_data.get("scenario_variant", "baseline"),
            current_tick=obs_data.get("current_tick", 0),
            max_ticks=obs_data.get("max_ticks", 0),
            total_gpus=obs_data.get("total_gpus", 0),
            free_gpus=obs_data.get("free_gpus", 0),
            owned_gpus=obs_data.get("owned_gpus", 0),
            idle_owned_gpus=obs_data.get("idle_owned_gpus", 0),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            market_price=obs_data.get("market_price", 0.0),
            jobs=[JobRecord(**item) for item in obs_data.get("jobs", [])],
            visible_offers=[MarketOffer(**item) for item in obs_data.get("visible_offers", [])],
            recent_events=[MarketEvent(**item) for item in obs_data.get("recent_events", [])],
            actor_signals=[ActorSignal(**item) for item in obs_data.get("actor_signals", [])],
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ComputeMarketState:
        return ComputeMarketState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            scenario_seed=payload.get("scenario_seed", 0),
            scenario_variant=payload.get("scenario_variant", "baseline"),
            current_tick=payload.get("current_tick", 0),
            max_ticks=payload.get("max_ticks", 0),
            total_gpus=payload.get("total_gpus", 0),
            free_gpus=payload.get("free_gpus", 0),
            owned_gpus=payload.get("owned_gpus", 0),
            idle_owned_gpus=payload.get("idle_owned_gpus", 0),
            budget_remaining=payload.get("budget_remaining", 0.0),
            market_price=payload.get("market_price", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            external_allocated_gpus=payload.get("external_allocated_gpus", 0),
            done=payload.get("done", False),
            jobs=[JobRecord(**item) for item in payload.get("jobs", [])],
            visible_offers=[MarketOffer(**item) for item in payload.get("visible_offers", [])],
            reservations=[ReservationRecord(**item) for item in payload.get("reservations", [])],
            actor_signals=[ActorSignal(**item) for item in payload.get("actor_signals", [])],
            hidden_actors=[ActorProfile(**item) for item in payload.get("hidden_actors", [])],
            recent_events=[MarketEvent(**item) for item in payload.get("recent_events", [])],
        )
