from compute_market_env import ComputeMarketAction
from compute_market_env.server.compute_market_environment import ComputeMarketEnvironment


def test_reset_is_deterministic_for_fixed_seed():
    env = ComputeMarketEnvironment(default_seed=1)
    obs1 = env.reset(seed=11)
    obs2 = env.reset(seed=11)

    assert obs1.market_price == obs2.market_price
    assert obs1.free_gpus == obs2.free_gpus
    assert [offer.price_per_gpu for offer in obs1.visible_offers] == [
        offer.price_per_gpu for offer in obs2.visible_offers
    ]


def test_initial_state_is_populated_before_explicit_reset():
    env = ComputeMarketEnvironment(default_seed=3)

    assert env.state.current_tick == 0
    assert env.state.market_price > 0
    assert len(env.state.jobs) > 0
    assert len(env.state.actor_signals) > 0


def test_step_before_explicit_reset_uses_initialized_state():
    env = ComputeMarketEnvironment(default_seed=3)

    result = env.step(ComputeMarketAction(action_type="noop"))

    assert result.current_tick == 1
    assert len(result.jobs) > 0
    assert result.market_price > 0


def test_bid_schedule_and_complete_job():
    env = ComputeMarketEnvironment(total_gpus=12, initial_budget=200.0, max_ticks=8, default_seed=0)
    env.reset(seed=3)

    result = env.step(
        ComputeMarketAction(
            action_type="bid_for_capacity",
            gpu_count=4,
            price_per_gpu=8.0,
            duration=3,
        )
    )
    assert result.reward < 0

    result = env.step(ComputeMarketAction(action_type="schedule_job", job_id="job-a"))
    assert any(job.status == "running" for job in result.jobs)

    result = env.step(ComputeMarketAction(action_type="noop"))
    job_a = next(job for job in result.jobs if job.job_id == "job-a")
    assert job_a.status == "completed"
    assert result.reward > 0


def test_invalid_action_returns_penalty_metadata():
    env = ComputeMarketEnvironment(default_seed=2)
    env.reset(seed=2)

    result = env.step(ComputeMarketAction(action_type="schedule_job", job_id="missing-job"))

    assert result.reward < 0
    assert "error" in result.metadata


def test_tight_capacity_variant_changes_capacity():
    env = ComputeMarketEnvironment(total_gpus=8, default_seed=0)
    result = env.reset(seed=7, scenario_variant="tight_capacity")

    assert result.scenario_variant == "tight_capacity"
    assert result.total_gpus == 6


def test_policy_shift_variant_removes_broker_offer():
    env = ComputeMarketEnvironment(default_seed=0)
    result = env.reset(seed=7, scenario_variant="policy_shift")

    assert result.scenario_variant == "policy_shift"
    assert all(offer.actor_id != "broker-1" for offer in result.visible_offers)
