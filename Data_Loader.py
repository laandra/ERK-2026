from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

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
INPUT_DATA_DIR = WORKSPACE_ROOT / "Input data"


def _single_file_datasets() -> dict[str, Path]:
    """One-profile datasets: the CSVs sitting directly under Input data."""
    return {path.stem: path for path in sorted(INPUT_DATA_DIR.glob("*.csv"))}


def _dataset_dir_map() -> dict[str, Path]:
    dataset_dirs = {
        path.name.lower(): path
        for path in INPUT_DATA_DIR.iterdir()
        if path.is_dir()
    }
    dataset_dirs.update(
        {name.lower(): path for name, path in _single_file_datasets().items()}
    )
    dataset_dirs["greek"] = INPUT_DATA_DIR / "GreekSmartHome.csv"
    return dataset_dirs


def available_input_datasets() -> List[str]:
    """Dataset names: the folders under Input data, plus the single-file ones."""
    dataset_names = sorted(path.name for path in INPUT_DATA_DIR.iterdir() if path.is_dir())
    dataset_names.extend(_single_file_datasets())
    return dataset_names


def _resolve_dataset_source(dataset: str) -> Path:
    dataset_map = _dataset_dir_map()
    dataset_key = str(dataset).strip().lower()

    try:
        return dataset_map[dataset_key]
    except KeyError as exc:
        available = ", ".join(available_input_datasets())
        raise ValueError(
            f"Unsupported dataset={dataset!r}. Available datasets: {available}."
        ) from exc


def _finalize_timeseries(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if "Timestamp_UTC" not in df.columns:
        raise ValueError(f"Missing 'Timestamp_UTC' column in {csv_path}.")

    df = df.copy()
    df.index = pd.to_datetime(df["Timestamp_UTC"], format="ISO8601")
    return df.drop(columns=["Timestamp_UTC"])


def _household_csv_path(dataset_dir: Path, household_id: int) -> Path:
    csv_matches = sorted(dataset_dir.glob(f"* {int(household_id)}.csv"))
    if not csv_matches:
        raise FileNotFoundError(
            f"Household id {household_id} not found in dataset {dataset_dir.name!r}."
        )
    return csv_matches[0]


def household_column_names(household_id: int, *, dataset: str = "Ausgrid") -> List[str]:
    """Return the original column names for one household dataset file."""
    dataset_source = _resolve_dataset_source(dataset)

    if not dataset_source.is_dir():
        csv_path = dataset_source
    elif dataset_source.name.lower() == "smp":
        raise ValueError(
            "Dataset 'SMP' is price-only. Use load_smp_data(country_id=...) instead."
        )
    else:
        csv_path = _household_csv_path(dataset_source, household_id)

    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def load_smp_data(country_id: str) -> pd.DataFrame:
    """Load SMP market prices for a given country file under Input data/SMP."""
    smp_dir = INPUT_DATA_DIR / "SMP"
    csv_path = smp_dir / f"{str(country_id).strip()}.csv"

    if not csv_path.exists():
        available = sorted(path.stem for path in smp_dir.glob("*.csv")) if smp_dir.exists() else []
        raise FileNotFoundError(
            f"SMP country {country_id!r} not found under {smp_dir}. "
            f"Available countries: {', '.join(available)}."
        )

    df = pd.read_csv(csv_path)
    return _finalize_timeseries(df, csv_path)


def load_household_data(
    household_id: int,
    *,
    dataset: str = "Ausgrid",
    print_column_names: bool = False,
) -> pd.DataFrame:
    """Load one household profile."""
    dataset_source = _resolve_dataset_source(dataset)

    if dataset_source.is_dir():
        if dataset_source.name.lower() == "smp":
            raise ValueError(
                "Dataset 'SMP' is price-only. Use load_smp_data(country_id=...) instead."
            )
        csv_path = _household_csv_path(dataset_source, household_id)
    else:
        csv_path = dataset_source

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {csv_path}. Expected data under {INPUT_DATA_DIR}."
        )

    df = pd.read_csv(csv_path)
    if print_column_names:
        print(df.columns.tolist())
    return _finalize_timeseries(df, csv_path)


def load_multiple_households(household_ids: Iterable[int], *, dataset: str = "Ausgrid") -> dict:
    """Load multiple household datasets keyed by household id."""
    return {int(hid): load_household_data(int(hid), dataset=dataset) for hid in household_ids}
