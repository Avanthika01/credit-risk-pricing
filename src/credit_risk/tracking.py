"""Optional MLflow logging. The pipeline runs identically when MLflow is absent."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on environment
    import mlflow

    HAVE_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    HAVE_MLFLOW = False


EXPERIMENT = "credit-risk-pd"


@contextmanager
def run(name: str, enabled: bool = True, tracking_dir: Path | None = None):
    """Yield a logger object; a no-op if MLflow is missing, disabled or fails to start.

    Tracking is auxiliary: an MLflow problem logs a warning instead of failing a run.
    Uses a local SQLite backend (MLflow 3 no longer accepts the plain file store).
    """
    if not (enabled and HAVE_MLFLOW):
        if enabled:
            log.info("MLflow not installed; skipping experiment tracking.")
        yield _NoOp()
        return
    try:
        if tracking_dir is not None:
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}")
            if mlflow.get_experiment_by_name(EXPERIMENT) is None:
                mlflow.create_experiment(EXPERIMENT, artifact_location=(tracking_dir / "artifacts").resolve().as_uri())
        mlflow.set_experiment(EXPERIMENT)
        active = mlflow.start_run(run_name=name)
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow tracking unavailable (%s); continuing without it.", exc)
        yield _NoOp()
        return
    with active:
        yield _MLflow()


class _NoOp:
    def params(self, d: dict) -> None: ...
    def metrics(self, d: dict, prefix: str = "") -> None: ...
    def artifact(self, path: Path) -> None: ...


class _MLflow:
    def params(self, d: dict) -> None:
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d: dict, prefix: str = "") -> None:
        mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in d.items() if isinstance(v, (int, float))})

    def artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))
