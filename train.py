"""Train, evaluate and report the PD models.

Real data (after scripts/download_data.py):
    python train.py
SYNTHETIC data (tests/CI smoke run; outputs go to artifacts/synthetic/, never reports/):
    python train.py --synthetic 40000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from credit_risk.config import Config  # noqa: E402
from credit_risk.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=int, metavar="N_ROWS", default=None,
                    help="Use N rows of SYNTHETIC data instead of the real Lending Club file.")
    ap.add_argument("--pd-cutoff", type=float, default=None, help="Approve if PD <= cutoff (default 0.20).")
    ap.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    ap.add_argument("--refresh", action="store_true", help="Rebuild the parquet cache from the raw CSV.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    if args.pd_cutoff is not None:
        cfg = dataclasses.replace(cfg, decision=dataclasses.replace(cfg.decision, pd_cutoff=args.pd_cutoff))

    m = run(cfg, synthetic_rows=args.synthetic, use_mlflow=not args.no_mlflow, refresh=args.refresh)
    summary = {"data_source": m["data_source"], "champion": m["champion"],
               "test_calibrated": m["test_calibrated"], "score_psi": m["psi"]["score"], "lgd": m["lgd"]["lgd"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
