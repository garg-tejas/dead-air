"""FastAPI application for the Dead Air Environment."""

try:
    from openenv.core import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from ..models import DispatchAction, DispatchObservation
    from .dispatcher_environment import DispatcherEnvironment
except (ModuleNotFoundError, ImportError):
    try:
        from models import DispatchAction, DispatchObservation
        from server.dispatcher_environment import DispatcherEnvironment
    except (ModuleNotFoundError, ImportError):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models import DispatchAction, DispatchObservation
        from server.dispatcher_environment import DispatcherEnvironment


app = create_app(
    DispatcherEnvironment,
    DispatchAction,
    DispatchObservation,
    env_name="dead_air",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
