"""Screen the ported forecaster against the incumbents, resumably.

    python run_hbd_benchmark.py [id ...]

Every fit is cached -- per AR column, so an interrupted run resumes mid-channel
rather than restarting a ~150 s fit. Rerun until it prints ALL DONE; nothing
already computed is recomputed. That matters on any machine that will not let
one process run for the ~10 minutes a household of AR fitting takes.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hems_study as hs                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "Input data", "Ausgrid")
OUT = os.path.join(HERE, "results_local", "hbd_screen")
KINDS = ["persistence", "median14", "hbd_baseline", "hbd", "hbd_median14"]


def main(ids):
    os.makedirs(OUT, exist_ok=True)
    done = []
    for ident in ids:
        path = os.path.join(OUT, f"{ident}.csv")
        if not os.path.exists(path):
            hs.forecast_benchmark(DATA, dataset_ids=[ident], kinds=KINDS,
                                  out_path=path)
            print(f"  finished {ident}", flush=True)
        done.append(path)

    if len(done) == len(ids):
        frame = pd.concat([pd.read_csv(p) for p in done], ignore_index=True)
        frame.to_csv(os.path.join(OUT, "all.csv"), index=False)
        print(frame.pivot_table(index="kind", columns="channel",
                                values="skill_vs_naive", aggfunc="median")
                   .reindex(KINDS).round(3).to_string())
        print("ALL DONE")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else [str(i) for i in hs.study_units().index.tolist()])
