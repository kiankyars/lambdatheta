"""Tiny local smoke example for the compute market environment."""

from compute_market_env import ComputeMarketAction, ComputeMarketEnv


with ComputeMarketEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=5)
    print("reset", result.observation.market_price, result.observation.free_gpus)

    result = env.step(
        ComputeMarketAction(
            action_type="bid_for_capacity",
            gpu_count=min(4, max(1, result.observation.free_gpus)),
            price_per_gpu=max(6.0, result.observation.market_price + 0.5),
            duration=3,
        )
    )
    print("bid", result.reward)

    result = env.step(ComputeMarketAction(action_type="schedule_job", job_id="job-a"))
    print("schedule", result.reward)

    while not result.done:
        result = env.step(ComputeMarketAction(action_type="noop"))
        print("tick", result.observation.current_tick, result.reward)
