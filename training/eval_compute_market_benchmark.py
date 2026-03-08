"""Tiny ID/OOD benchmark for Compute Market models.

Run this in Colab after installing the environment package from the HF Space and
cloning this repo for the helper code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compute_market_env import ComputeMarketEnv

from training.compute_market_grpo import (
    DEFAULT_TASK_PROMPT,
    SYSTEM_PROMPT,
    observation_to_prompt,
    parse_action,
)

BENCHMARK_SPLITS = {
    "id_baseline": {
        "tag": "ID",
        "scenario_variant": "baseline",
        "seeds": [11, 12, 13, 14, 15],
    },
    "ood_tight_capacity": {
        "tag": "OOD",
        "scenario_variant": "tight_capacity",
        "seeds": [101, 102, 103, 104, 105],
    },
    "ood_price_shock": {
        "tag": "OOD",
        "scenario_variant": "price_shock",
        "seeds": [201, 202, 203, 204, 205],
    },
    "ood_policy_shift": {
        "tag": "OOD",
        "scenario_variant": "policy_shift",
        "seeds": [301, 302, 303, 304, 305],
    },
    "ood_job_mix": {
        "tag": "OOD",
        "scenario_variant": "job_mix",
        "seeds": [401, 402, 403, 404, 405],
    },
}


@dataclass
class EpisodeStats:
    model_label: str
    split: str
    tag: str
    seed: int
    scenario_variant: str
    total_return: float
    completed_jobs: int
    missed_jobs: int
    invalid_actions: int
    budget_remaining: float
    turns: int


def load_model_and_tokenizer(
    model_ref: str,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    fast_inference: bool = True,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_ref,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_completion(
    model: Any,
    tokenizer: Any,
    observation: Any,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 160,
) -> str:
    prompt_text = observation_to_prompt(observation, task_prompt)
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
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def run_episode(
    env: ComputeMarketEnv,
    model: Any,
    tokenizer: Any,
    model_label: str,
    split: str,
    tag: str,
    seed: int,
    scenario_variant: str,
    max_turns: int = 6,
) -> EpisodeStats:
    result = env.reset(seed=seed, scenario_variant=scenario_variant)
    total_return = 0.0
    invalid_actions = 0
    turns = 0

    for _ in range(max_turns):
        if result.done:
            break
        completion = generate_completion(model, tokenizer, result.observation)
        action, is_valid = parse_action(completion)
        if not is_valid:
            invalid_actions += 1
        result = env.step(action)
        total_return += float(result.reward or 0.0)
        turns += 1

    jobs = result.observation.jobs
    completed_jobs = sum(1 for job in jobs if job.status == "completed")
    missed_jobs = sum(1 for job in jobs if job.status == "missed")
    return EpisodeStats(
        model_label=model_label,
        split=split,
        tag=tag,
        seed=seed,
        scenario_variant=scenario_variant,
        total_return=round(total_return, 2),
        completed_jobs=completed_jobs,
        missed_jobs=missed_jobs,
        invalid_actions=invalid_actions,
        budget_remaining=round(result.observation.budget_remaining, 2),
        turns=turns,
    )


def evaluate_model(
    model: Any,
    tokenizer: Any,
    model_label: str,
    env_url: str,
    benchmark_splits: dict[str, dict[str, Any]] = BENCHMARK_SPLITS,
    max_turns: int = 6,
) -> list[EpisodeStats]:
    stats: list[EpisodeStats] = []
    with ComputeMarketEnv(base_url=env_url) as env:
        for split_name, config in benchmark_splits.items():
            for seed in config["seeds"]:
                stats.append(
                    run_episode(
                        env=env,
                        model=model,
                        tokenizer=tokenizer,
                        model_label=model_label,
                        split=split_name,
                        tag=config["tag"],
                        seed=seed,
                        scenario_variant=config["scenario_variant"],
                        max_turns=max_turns,
                    )
                )
    return stats


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 2)


def summarize_results(stats: list[EpisodeStats]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EpisodeStats]] = {}
    for item in stats:
        grouped.setdefault((item.model_label, item.split), []).append(item)

    rows: list[dict[str, Any]] = []
    for (model_label, split), items in grouped.items():
        rows.append(
            {
                "model": model_label,
                "split": split,
                "tag": items[0].tag,
                "scenario_variant": items[0].scenario_variant,
                "episodes": len(items),
                "mean_return": _safe_mean([item.total_return for item in items]),
                "mean_completed_jobs": _safe_mean([item.completed_jobs for item in items]),
                "mean_missed_jobs": _safe_mean([item.missed_jobs for item in items]),
                "mean_budget_remaining": _safe_mean([item.budget_remaining for item in items]),
                "invalid_action_rate": _safe_mean(
                    [item.invalid_actions / max(1, item.turns) for item in items]
                ),
            }
        )

    for model_label in sorted({item.model_label for item in stats}):
        for tag_name in ["ID", "OOD"]:
            items = [item for item in stats if item.model_label == model_label and item.tag == tag_name]
            if not items:
                continue
            rows.append(
                {
                    "model": model_label,
                    "split": f"aggregate_{tag_name.lower()}",
                    "tag": tag_name,
                    "scenario_variant": "mixed",
                    "episodes": len(items),
                    "mean_return": _safe_mean([item.total_return for item in items]),
                    "mean_completed_jobs": _safe_mean([item.completed_jobs for item in items]),
                    "mean_missed_jobs": _safe_mean([item.missed_jobs for item in items]),
                    "mean_budget_remaining": _safe_mean([item.budget_remaining for item in items]),
                    "invalid_action_rate": _safe_mean(
                        [item.invalid_actions / max(1, item.turns) for item in items]
                    ),
                }
            )
    return rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "split",
        "tag",
        "mean_return",
        "mean_completed_jobs",
        "mean_missed_jobs",
        "mean_budget_remaining",
        "invalid_action_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def benchmark_two_models(
    base_model_ref: str,
    tuned_model_ref: str,
    env_url: str,
    max_turns: int = 6,
) -> tuple[list[EpisodeStats], list[dict[str, Any]], str]:
    all_stats: list[EpisodeStats] = []
    for label, model_ref in [
        ("Qwen3-4B-Base", base_model_ref),
        ("compute-market-qwen3-4b", tuned_model_ref),
    ]:
        model, tokenizer = load_model_and_tokenizer(model_ref)
        all_stats.extend(
            evaluate_model(
                model=model,
                tokenizer=tokenizer,
                model_label=label,
                env_url=env_url,
                max_turns=max_turns,
            )
        )
    rows = summarize_results(all_stats)
    return all_stats, rows, render_markdown_table(rows)


if __name__ == "__main__":
    import os

    env_url = os.environ.get(
        "OPENENV_URL",
        "https://openenv-community-compute-market-env.hf.space",
    )
    _, rows, table = benchmark_two_models(
        base_model_ref="Qwen/Qwen3-4B",
        tuned_model_ref="outputs/compute-market-qwen3-4b",
        env_url=env_url,
    )
    print(table)
