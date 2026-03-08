"""GRPO training helpers for the Compute Market environment.

Designed to be imported from a Colab notebook after installing:
- the environment package from the Hugging Face Space or GitHub repo
- TRL / Unsloth / transformers runtime deps
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from compute_market_env import ComputeMarketAction, ComputeMarketEnv

ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """You are a compute allocator trading for scarce GPU capacity.
Return exactly one JSON object with a valid action.
Allowed action_type values: bid_for_capacity, accept_offer, propose_swap, schedule_job, delay_job, inspect_market, noop.
Only include fields needed by the chosen action.
Be conservative with budget and prioritize completing valuable jobs before their deadlines."""

DEFAULT_TASK_PROMPT = (
    "Maximize completed job value while minimizing spend, idle-hoarding penalties, "
    "and missed deadlines in the compute market."
)


@dataclass
class RolloutSummary:
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    episode_return: float
    valid_action_reward: float
    completion_bonus: float
    transcripts: list[dict[str, Any]]


def observation_to_prompt(observation: Any, task_prompt: str = DEFAULT_TASK_PROMPT) -> str:
    jobs = [job.model_dump() for job in observation.jobs]
    offers = [offer.model_dump() for offer in observation.visible_offers]
    signals = [signal.model_dump() for signal in observation.actor_signals]
    events = [event.model_dump() for event in observation.recent_events]
    return json.dumps(
        {
            "task": task_prompt,
            "scenario_variant": getattr(observation, "scenario_variant", "baseline"),
            "tick": observation.current_tick,
            "max_ticks": observation.max_ticks,
            "budget_remaining": observation.budget_remaining,
            "market_price": observation.market_price,
            "public_free_gpus": observation.free_gpus,
            "owned_gpus": observation.owned_gpus,
            "idle_owned_gpus": observation.idle_owned_gpus,
            "jobs": jobs,
            "visible_offers": offers,
            "actor_signals": signals,
            "recent_events": events,
        },
        indent=2,
    )


def parse_action(text: str) -> tuple[ComputeMarketAction, bool]:
    match = ACTION_JSON_RE.search(text)
    if not match:
        return ComputeMarketAction(action_type="inspect_market"), False
    try:
        payload = json.loads(match.group(0))
        return ComputeMarketAction(**payload), True
    except Exception:
        return ComputeMarketAction(action_type="inspect_market"), False


def _count_completed_jobs(observation: Any) -> int:
    return sum(1 for job in observation.jobs if job.status == "completed")


def rollout_once(
    trainer: Any,
    env: ComputeMarketEnv,
    tokenizer: Any,
    dataset_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = 6,
    seed: int | None = None,
) -> RolloutSummary:
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset(seed=seed)
    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    transcripts: list[dict[str, Any]] = []
    rewards: list[float] = []
    valid_action_reward = 0.0
    completed_before = 0

    for turn in range(max_turns):
        if result.done:
            break

        prompt_text = observation_to_prompt(result.observation, dataset_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        rollout_outputs = generate_rollout_completions(trainer, [rendered_prompt])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"],
            skip_special_tokens=True,
        )

        action, is_valid = parse_action(completion_text)
        result = env.step(action)
        reward = float(result.reward or 0.0)
        rewards.append(reward)
        valid_action_reward += 0.25 if is_valid else -1.0

        completed_after = _count_completed_jobs(result.observation)
        completion_gain = max(0, completed_after - completed_before)
        completed_before = completed_after

        transcripts.append(
            {
                "turn": turn,
                "prompt": prompt_text,
                "completion": completion_text,
                "action": action.model_dump(exclude_none=True),
                "is_valid_action": is_valid,
                "reward": reward,
                "completed_jobs": completed_after,
                "completion_gain": completion_gain,
            }
        )

    completion_bonus = float(completed_before)
    return RolloutSummary(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
        episode_return=sum(rewards),
        valid_action_reward=valid_action_reward,
        completion_bonus=completion_bonus,
        transcripts=transcripts,
    )


def rollout_func(
    prompts: list[str],
    trainer: Any | None = None,
    tokenizer: Any | None = None,
    env_url: str | None = None,
    max_turns: int = 6,
    seed_offset: int = 0,
) -> dict[str, Any]:
    if trainer is None:
        raise ValueError("trainer is required")
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    env_url = env_url or os.environ.get("OPENENV_URL", "http://localhost:8000")
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    episode_returns = []
    validity_rewards = []
    completion_bonuses = []
    transcripts = []

    with ComputeMarketEnv(base_url=env_url) as env:
        for idx, prompt_text in enumerate(prompts):
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                max_turns=max_turns,
                seed=seed_offset + idx,
            )
            episode_prompt_ids.append(episode.prompt_ids)
            episode_completion_ids.append(episode.completion_ids)
            episode_logprobs.append(episode.logprobs)
            episode_returns.append(episode.episode_return)
            validity_rewards.append(episode.valid_action_reward)
            completion_bonuses.append(episode.completion_bonus)
            transcripts.append(episode.transcripts)

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_reward": episode_returns,
        "valid_action_reward": validity_rewards,
        "completion_bonus": completion_bonuses,
        "transcripts": transcripts,
    }


def reward_env_return(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("env_reward") or []
    return [float(rewards[i]) if i < len(rewards) else 0.0 for i in range(len(completions))]


def reward_valid_action(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("valid_action_reward") or []
    return [float(rewards[i]) if i < len(rewards) else 0.0 for i in range(len(completions))]


def reward_job_completion(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("completion_bonus") or []
    return [float(rewards[i]) if i < len(rewards) else 0.0 for i in range(len(completions))]


def build_dataset(size: int = 128, prompt: str = DEFAULT_TASK_PROMPT):
    from datasets import Dataset

    return Dataset.from_dict({"prompt": [prompt] * size})


def build_grpo_config(
    output_dir: str = "outputs/compute-market-qwen3-4b",
    max_steps: int = 300,
    learning_rate: float = 5e-6,
    num_generations: int = 2,
    max_prompt_length: int = 1800,
    max_completion_length: int = 192,
    use_vllm: bool = True,
):
    from trl import GRPOConfig

    kwargs: dict[str, Any] = dict(
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=max_steps,
        report_to="none",
        output_dir=output_dir,
    )
    if use_vllm:
        kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.15,
        )
    return GRPOConfig(**kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    env_url: str,
    train_dataset: Any | None = None,
    output_dir: str = "outputs/compute-market-qwen3-4b",
    max_steps: int = 300,
    max_turns: int = 6,
):
    from trl import GRPOTrainer

    train_dataset = train_dataset or build_dataset()
    args = build_grpo_config(output_dir=output_dir, max_steps=max_steps)

    def bound_rollout_func(prompts: list[str], trainer: Any | None = None, **_: Any) -> dict[str, Any]:
        return rollout_func(
            prompts,
            trainer=trainer,
            tokenizer=tokenizer,
            env_url=env_url,
            max_turns=max_turns,
        )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_env_return,
            reward_valid_action,
            reward_job_completion,
        ],
        train_dataset=train_dataset,
        args=args,
        rollout_func=bound_rollout_func,
    )


def build_colab_setup_snippet(space_repo_id: str = "openenv-community/compute_market_env") -> str:
    return f"""# Colab install\n!pip install --upgrade uv\n!uv pip install unsloth vllm --torch-backend=auto\n!uv pip install --upgrade --no-cache-dir --no-deps unsloth unsloth_zoo\n!uv pip install transformers==4.56.2 'trl>=0.24.0' datasets openenv-core\n!pip install git+https://huggingface.co/spaces/{space_repo_id}\n\nimport os\nos.environ['OPENENV_URL'] = 'https://{space_repo_id.replace('/', '-').replace('_', '-')}.hf.space'\n"""


if __name__ == "__main__":
    print(build_colab_setup_snippet())
