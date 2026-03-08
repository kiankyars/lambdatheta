"""FastAPI app for the Compute Market environment."""

try:
    from openenv.core.env_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

try:
    from ..models import ComputeMarketAction, ComputeMarketObservation
    from .compute_market_environment import create_environment_from_env
except ImportError:
    from models import ComputeMarketAction, ComputeMarketObservation
    from server.compute_market_environment import create_environment_from_env


app = create_app(
    create_environment_from_env,
    ComputeMarketAction,
    ComputeMarketObservation,
    env_name="compute_market_env",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
