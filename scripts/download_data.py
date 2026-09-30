"""Download the Lending Club accepted-loans file (2007-2018Q4) from Kaggle.

Auth: any method supported by the kaggle package (>=2.0), e.g. ~/.kaggle/kaggle.json,
~/.kaggle/access_token, the KAGGLE_API_TOKEN env var, or `kaggle auth login`.

The file name is discovered from the dataset listing rather than hard-coded, and
the compressed .csv.gz is preferred (pandas reads it directly; smaller download).

Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import KAGGLE_DATASET, Paths  # noqa: E402


def pick_accepted_file(names: list[str]) -> str:
    accepted = [n for n in names if "accepted" in n.lower() and ".csv" in n.lower()]
    if not accepted:
        raise RuntimeError(f"No accepted-loans file found in {KAGGLE_DATASET}: {names}")
    gz = [n for n in accepted if n.lower().endswith(".csv.gz")]
    return (gz or accepted)[0]


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported late: authenticates on import

    api = KaggleApi()
    api.authenticate()
    raw_dir = Paths().raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    listing = api.dataset_list_files(KAGGLE_DATASET, page_size=100)
    names = [f.name for f in listing.files]
    print(f"Files in {KAGGLE_DATASET}: {names}")
    target = pick_accepted_file(names)
    print(f"Downloading {target} -> {raw_dir}")
    api.dataset_download_file(KAGGLE_DATASET, target, path=str(raw_dir), quiet=False)

    for z in raw_dir.glob("*.zip"):  # Kaggle may deliver single files zipped
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        z.unlink()

    files = [p for p in raw_dir.rglob("*") if p.is_file()]
    for p in files:
        print(f"  {p.relative_to(raw_dir)}  {p.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
