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


@contextmanager
def run(name: str, enabled: bool = True, tracking_dir: Path | None = None):
    """Yield a logger object; a no-op if MLflow is missing or disabled."""
    if not (enabled and HAVE_MLFLOW):
        if enabled:
            log.info("MLflow not installed; skipping experiment tracking.")
        yield _NoOp()
        return
    if tracking_dir is not None:
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("credit-risk-pd")
    with mlflow.start_run(run_name=name):
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
