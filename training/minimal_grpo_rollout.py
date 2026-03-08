"""Minimal TRL/OpenEnv rollout helpers for Colab."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from compute_market_env import ComputeMarketAction, ComputeMarketEnv

ACTION_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = """You trade for scarce GPU capacity.
Choose exactly one JSON action per turn.
Valid action_type values are: bid_for_capacity, accept_offer, propose_swap, schedule_job, delay_job, inspect_market, noop.
Return strict JSON only."""


def observation_to_prompt(observation) -> str:
    jobs = [job.model_dump() for job in observation.jobs]
    offers = [offer.model_dump() for offer in observation.visible_offers]
    signals = [signal.model_dump() for signal in observation.actor_signals]
    return json.dumps(
        {
            "tick": observation.current_tick,
            "budget_remaining": observation.budget_remaining,
            "market_price": observation.market_price,
            "free_gpus": observation.free_gpus,
            "owned_gpus": observation.owned_gpus,
            "idle_owned_gpus": observation.idle_owned_gpus,
            "jobs": jobs,
            "visible_offers": offers,
            "actor_signals": signals,
        },
        indent=2,
    )


def parse_action(text: str) -> ComputeMarketAction:
    match = ACTION_RE.search(text)
    if not match:
        return ComputeMarketAction(action_type="inspect_market")
    try:
        payload = json.loads(match.group(0))
        return ComputeMarketAction(**payload)
    except Exception:
        return ComputeMarketAction(action_type="inspect_market")


def rollout_once(generate_completion, seed: int = 0, max_turns: int = 6) -> dict[str, Any]:
    env_url = os.environ.get("OPENENV_URL", "http://localhost:8000")
    rewards = []
    actions = []
    transcripts = []

    with ComputeMarketEnv(base_url=env_url) as env:
        result = env.reset(seed=seed)
        for _ in range(max_turns):
            if result.done:
                break
            prompt = observation_to_prompt(result.observation)
            completion = generate_completion(SYSTEM_PROMPT, prompt)
            action = parse_action(completion)
            result = env.step(action)
            actions.append(action.model_dump(exclude_none=True))
            rewards.append(float(result.reward or 0.0))
            transcripts.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "action": action.model_dump(exclude_none=True),
                    "reward": result.reward,
                }
            )

    return {
        "actions": actions,
        "rewards": rewards,
        "return": sum(rewards),
        "transcripts": transcripts,
    }
