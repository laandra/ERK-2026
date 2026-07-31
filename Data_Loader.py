from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import os
import pandas as pd


def _find_workspace_root(start: Optional[Path] = None) -> Path:
    """Resolve repository root by walking upwards until core files are found."""
    here = start or Path(__file__).resolve().parent
    core_markers = {"Environment.py", "Basic_Functions.py", "Input data"}

    for current in [here, *here.parents]:
        if all((current / marker).exists() for marker in core_markers):
            return current
    return Path.cwd()


WORKSPACE_ROOT = _find_workspace_root()
IN_COLAB = os.path.exists("/content")
INPUT_DATA_DIR = WORKSPACE_ROOT / "Input data"
AUSGRID_DIR = INPUT_DATA_DIR / "Ausgrid"


def ausgrid_household_path(household_id: int) -> Path:
    return AUSGRID_DIR / f"Ausgrid {int(household_id)}.csv"


def load_household_data(household_id: int, *, dataset: str = "ausgrid") -> pd.DataFrame:
    """Load one household profile and normalize index/columns for the env."""
    dataset_key = str(dataset).lower()

    if dataset_key == "ausgrid":
        csv_path = ausgrid_household_path(household_id)
    elif dataset_key in {"greek", "greeksmarthome"}:
        csv_path = INPUT_DATA_DIR / "GreekSmartHome.csv"
    else:
        raise ValueError(f"Unsupported dataset={dataset!r}. Use 'ausgrid' or 'greek'.")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {csv_path}. Expected data under {INPUT_DATA_DIR}."
        )

    df = pd.read_csv(csv_path)
    if "Timestamp_UTC" not in df.columns:
        raise ValueError(f"Missing 'Timestamp_UTC' column in {csv_path}.")

    df.index = pd.to_datetime(df["Timestamp_UTC"], format="ISO8601")
    return df.drop(columns=["Timestamp_UTC"])


def load_multiple_households(household_ids: Iterable[int], *, dataset: str = "ausgrid") -> dict:
    """Load multiple household datasets keyed by household id."""
    result = {}
    for hid in household_ids:
        result[int(hid)] = load_household_data(int(hid), dataset=dataset)
    return result


def available_ausgrid_households(limit: Optional[int] = None) -> List[int]:
    """List discovered Ausgrid household ids from filenames."""
    if not AUSGRID_DIR.exists():
        return []

    ids = []
    for csv_path in AUSGRID_DIR.glob("Ausgrid *.csv"):
        suffix = csv_path.stem.replace("Ausgrid ", "", 1)
        if suffix.isdigit():
            ids.append(int(suffix))

    ids.sort()
    if limit is not None:
        return ids[: int(limit)]
    return ids
