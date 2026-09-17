"""Cluster household load shapes, one ranked CSV per cluster count.

    python Clustering/cluster_households.py --dataset Ausgrid --k-max 30

Writes `Clustering/<dataset>/user_ids_sorted_by_cluster_{k}.csv` for k = 1..k-max,
each row a household with the cluster it landed in, its distance to that cluster's
centroid, and its rank within the cluster by that distance.

`hems_study.study_units` reads the k=30 file for Ausgrid and takes the rank-1
household of each cluster as the study roster -- one household per consumption
shape present in the population, rather than a transcribed list of ids.

The method is unchanged from the original Ausgrid sweep: each household becomes a
72-dimensional mean weekly profile (weekday / Saturday / Sunday x 24 hours),
normalised by its own daily maximum so that shape rather than magnitude drives the
clustering, then KMeans at a fixed seed.

What is new is that the input is the per-household CSVs under `Input data/`, read
through `Data_Loader`, rather than a single combined long-format file. That file
(`Ausgrid_2010_2013_Orange_Combined.csv`) is not in the repository, so the original
script could not be re-run; going through `Data_Loader` also means any dataset laid
out the same way -- the Fluvius groups, say -- can be clustered by the same method.

The committed `Clustering/Ausgrid/` CSVs predate this script and were fit on that
combined file. They are authoritative for the published study: re-running here may
permute cluster labels, and should not be used to overwrite them.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Data_Loader as dl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# The schemas differ between datasets: Ausgrid calls it `Energy_Consumption`,
# Fluvius `Consumption_Volume_kWh`. Tried in order; the first one present wins.
CONSUMPTION_COLUMNS = (
    "Energy_Consumption",
    "Consumption_Volume_kWh",
    "Consumption",
)


def consumption_column(df: pd.DataFrame) -> str:
    """The load column of one household frame, whatever this dataset calls it."""
    for name in CONSUMPTION_COLUMNS:
        if name in df.columns:
            return name
    raise KeyError(
        f"No consumption column found. Tried {list(CONSUMPTION_COLUMNS)}; the file "
        f"has {df.columns.tolist()}. Add the name to CONSUMPTION_COLUMNS."
    )


def household_ids(dataset: str) -> list[int]:
    """The ids present in a dataset directory, ascending.

    Files are named `<Dataset> <id>.csv`, which is the same convention
    `Data_Loader._household_csv_path` globs for.
    """
    source = dl._resolve_dataset_source(dataset)
    if not source.is_dir():
        raise ValueError(
            f"dataset={dataset!r} is a single file, not a population of households. "
            f"Clustering needs a directory of per-household CSVs."
        )
    ids = []
    for path in source.glob("* *.csv"):
        match = re.search(r"\s(\d+)\.csv$", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def load_population(dataset: str, n_users: int | None) -> pd.DataFrame:
    """Wide frame of consumption: timestamps down, one `user_<id>` column across."""
    ids = household_ids(dataset)
    if n_users is not None:
        ids = ids[:n_users]
        if len(ids) < n_users:
            raise ValueError(
                f"Requested {n_users} households, dataset {dataset!r} has {len(ids)}."
            )
    print(f"Loading {len(ids)} households from {dataset} ...")

    columns = {}
    for ident in ids:
        df = dl.load_household_data(ident, dataset=dataset)
        series = pd.to_numeric(df[consumption_column(df)], errors="coerce")
        # tz-naive throughout: the stamps carry an offset (Ausgrid) or a `Z`
        # suffix (Fluvius), and a mix of the two will not align on concat.
        series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
        columns[f"user_{ident}"] = series.astype(np.float32)

    data = pd.DataFrame(columns).sort_index().fillna(0.0)
    print(f"Data shape: {data.shape}")
    return data


def reshape2threedays(df_in: pd.DataFrame) -> pd.DataFrame:
    """Hourly series -> one mean weekly profile per household, as a row.

    Columns are (day type, hour), where day type is 0 for any weekday, 5 for
    Saturday and 6 for Sunday -- so 3 x 24 = 72 features per household.
    """
    df = df_in.copy()
    df.columns.name = "ts_id"
    df.index.name = "timestamp"
    df = df.unstack().reset_index()

    dayofweek = df.timestamp.dt.dayofweek
    days = dayofweek.where(dayofweek >= 5, 0).rename("dayofweek")

    return (
        pd.pivot_table(df, values=0, index=df.ts_id,
                       columns=[days, df.timestamp.rename("hour").dt.hour])
        .round(3)
    )


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each profile by its own daily maximum, so shape drives the distance."""
    # fillna covers profiles that are all zeros, which would divide by zero.
    return df.apply(lambda s: s / df.max(axis=1)).fillna(0)


def cluster_sweep(features: pd.DataFrame, k_max: int, out_dir: str) -> None:
    X = features.to_numpy(dtype=float)
    print(f"Feature matrix: {X.shape}")
    os.makedirs(out_dir, exist_ok=True)

    for k in range(1, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labels = model.labels_
        dist = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)

        ranking = pd.DataFrame({
            "user_id": features.index,
            "cluster": labels,
            "dist_to_centroid": dist,
        }).sort_values(["cluster", "dist_to_centroid"]).reset_index(drop=True)
        ranking["rank_in_cluster"] = ranking.groupby("cluster").cumcount() + 1

        out_path = os.path.join(out_dir, f"user_ids_sorted_by_cluster_{k}.csv")
        ranking.to_csv(out_path, index=False)
        print(f"  k={k:2d} -> {os.path.basename(out_path)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="Ausgrid",
                        help="a directory under `Input data/` (default: Ausgrid)")
    parser.add_argument("--k-max", type=int, default=30,
                        help="sweep cluster counts 1..k-max (default: 30)")
    parser.add_argument("--n-users", type=int, default=None,
                        help="use only the first N household ids (default: all)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: Clustering/<dataset>/)")
    args = parser.parse_args(argv)

    out_dir = args.out or os.path.join(HERE, args.dataset)
    data = load_population(args.dataset, args.n_users)

    print("Computing mean weekly profiles ...")
    features = normalize_df(reshape2threedays(data.resample("1h").mean()))

    cluster_sweep(features, args.k_max, out_dir)
    print(f"Done. {args.k_max} files in {out_dir}")


if __name__ == "__main__":
    main()
