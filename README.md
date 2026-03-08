---
title: Compute Market Environment Server
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - multi-agent
  - compute-allocation
  - market-simulation
---

# Compute Market Environment

An OpenEnv environment for training a single allocator/trader in a scarce-GPU market with scripted background actors, hidden incentives, delayed rewards, and partial observability.

## What v1 implements

- One trained agent: the allocator/trader
- Scripted counterparties: urgent tenant, cost-sensitive tenant, broker
- Jobs with deadlines, value, dependencies, and delayed payoff
- Actions: `bid_for_capacity`, `accept_offer`, `propose_swap`, `schedule_job`, `delay_job`, `inspect_market`, `noop`
- Reward = completed job value minus compute spend, missed-deadline penalties, and idle-hoarding penalties
- Separate training helper for TRL/Colab in `training/minimal_grpo_rollout.py`

## Quick Start

```python
from compute_market_env import ComputeMarketAction, ComputeMarketEnv

with ComputeMarketEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=7)
    print(result.observation.market_price)
    print(result.observation.free_gpus)

    result = env.step(
        ComputeMarketAction(
            action_type="bid_for_capacity",
            gpu_count=4,
            price_per_gpu=6.5,
            duration=3,
        )
    )
    print(result.reward)
    print(result.observation.budget_remaining)
```

## Local Development

```bash
uv sync --extra dev
uv run pytest -q
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
openenv validate --verbose
```

## Docker

```bash
docker build -t compute-market-env:latest -f server/Dockerfile .
docker run -p 8000:8000 compute-market-env:latest
```

## Environment Loop

1. Agent observes market price, public free GPUs, visible offers, jobs, and public actor signals.
2. Agent takes one action.
3. The environment advances one tick.
4. Scripted actors update demand and offers.
5. Jobs progress, complete, pause, or miss deadlines.
6. The environment returns the next observation and realized reward for that tick.
