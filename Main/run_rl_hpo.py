"""Hyperparameter optimisation of the learned controllers -- the driver.

    python3 run_rl_hpo.py all --jobs 12                  # the whole pipeline
    python3 run_rl_hpo.py search --studies AU:bc SI:bc --jobs 12
    python3 run_rl_hpo.py refine --studies AU:bc SI:bc --jobs 12
    python3 run_rl_hpo.py search --studies AU:dqn SI:dqn AU:bc_dqn SI:bc_dqn
    python3 run_rl_hpo.py refine --studies AU:dqn SI:dqn AU:bc_dqn SI:bc_dqn
    python3 run_rl_hpo.py confirm --jobs 12              # every study with a winner
    python3 run_rl_hpo.py status                         # progress, safe any time
    python3 run_rl_hpo.py report                         # winners in TUNED form

What is searched, on which households and against which objective is
`rl_hpo`'s docstring. The pipeline, in the order `all` runs it:

    1  search  bc            the clone's optimiser (cheap: ~1 s a fit)
    2  refine  bc            top configurations x 3 seeds -> winner.json
    3  search  dqn, bc_dqn   the DQN cold, and the fine-tune of the bc winner
    4  refine  dqn, bc_dqn
    5  confirm all           winner vs the panel's settings, test year, 30 units

INTERRUPT AT ANY TIME (Ctrl-C, `kill <pid>`, a closed laptop) and rerun the
SAME command: finished households come back from disk, trials that were open
are re-queued with their parameters, finished stages are skipped in seconds.
`--trials N` is a target, not an increment: rerunning with a larger N extends
a study. `--hours H` stops opening new trials after H hours (open ones
finish), so a night can be budgeted; the next run continues where it stopped.

Nothing here promotes a winner. `report` prints the winners as TUNED entries;
pasting them into `run_rl_benchmark.TUNED` and re-running the panel is the
deliberate, reviewable step it has always been.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rl_hpo as hpo                                             # noqa: E402

STAGE_1 = [("AU", "bc"), ("SI", "bc")]
STAGE_2 = [("AU", "dqn"), ("SI", "dqn"), ("AU", "bc_dqn"), ("SI", "bc_dqn")]


def _names(studies, tag=""):
    return [f"{t}_{m}" + (f"_{tag}" if tag else "") for t, m in studies]


def status(root=None):
    import pandas as pd
    df = hpo.status_frame(root)
    lock = hpo.driver_running(root)
    print(f"driver: {'running, pid ' + str(lock['pid']) + ' since ' + lock['started'] if lock else 'not running'}")
    if df.empty:
        print("no studies yet")
        return df
    with pd.option_context("display.width", 200):
        print(df.round(2).to_string(index=False))
    if not lock and (df.get("running", 0) > 0).any():
        print("RUNNING trials with no driver were interrupted; the next "
              "search re-queues them.")
    return df


def report(root=None):
    import pandas as pd
    status(root)
    for name in hpo.list_studies(root):
        w = hpo.winner(name, root)
        if w:
            print(f"\n{name}: winner {w['label']} -- validation "
                  f"{w['mean_saving_val_a']:.2f}/a vs incumbent "
                  f"{w['incumbent_saving_val_a']:.2f}/a, "
                  f"Δ {w['delta_vs_incumbent']:+.2f} (p {w['p_vs_incumbent']:.3f}, "
                  f"{w['n_better']}/{len(w['households'])} households)")
    cs = hpo.confirm_summary(root=root)
    if not cs.empty:
        print("\nTEST YEAR, 30 study households, winner vs the panel's settings "
              "(gain = lower bill + wear per year, seed-averaged):")
        with pd.option_context("display.width", 200):
            print(cs.drop(columns=["d"]).round(3).to_string(index=False))
    print("\nwinners as TUNED entries (only fields that differ from the "
          "TrainConfig / BCOptions defaults; paste by hand, then re-run the panel):")
    print(hpo.tuned_entries(root) or "    (none yet)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["search", "refine", "confirm", "all",
                                      "status", "report"])
    ap.add_argument("--studies", nargs="*", default=None,
                    help="<AU|SI>:<bc|dqn|bc_dqn> ... (default: all six)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--trials", type=int, default=None,
                    help="target finished trials per study (default per method)")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop opening new search trials after this long")
    ap.add_argument("--households", type=int, default=None,
                    help="tuning households per study (default per method)")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"DQN budget per run (default {hpo.STEPS}, the panel's)")
    ap.add_argument("--tag", default="", help="suffix for a second study of a pair")
    ap.add_argument("--clone", default="hpo", choices=["hpo", "current"],
                    help="bc_dqn: fine-tune the IL winner or the panel's clone")
    ap.add_argument("--top", type=int, default=hpo.REFINE_TOP)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="refine: extra seeds (default 1 2); confirm: seeds "
                         "(default 0 1 2)")
    ap.add_argument("--root", default=None,
                    help="results root (default results_local/rl_hpo); use a "
                         "scratch directory for smoke tests")
    ap.add_argument("--models", default=None,
                    help="network root (default rl_models/hpo)")
    args = ap.parse_args(argv)

    if args.phase == "status":
        status(args.root)
        return
    if args.phase == "report":
        report(args.root)
        return

    studies = ([hpo.parse_study(s) for s in args.studies] if args.studies
               else STAGE_1 + STAGE_2)
    common = dict(n_jobs=args.jobs, root=args.root, models=args.models)
    search_kw = dict(common, trials=args.trials, hours=args.hours,
                     n_households=args.households, tag=args.tag,
                     steps=args.steps, clone_source=args.clone)
    refine_kw = dict(common, top=args.top,
                     seeds=tuple(args.seeds) if args.seeds is not None
                     else hpo.REFINE_SEEDS)
    confirm_kw = dict(common, seeds=tuple(args.seeds) if args.seeds is not None
                      else hpo.CONFIRM_SEEDS)
    t0 = time.time()
    try:
        if args.phase == "search":
            hpo.search(studies, **search_kw)
        elif args.phase == "refine":
            hpo.refine(_names(studies, args.tag), **refine_kw)
        elif args.phase == "confirm":
            names = (_names(studies, args.tag) if args.studies
                     else hpo.list_studies(args.root))
            hpo.confirm(names, **confirm_kw)
        elif args.phase == "all":
            # One `--hours` budget across both search stages: what stage 1
            # spends, stage 2 does not get.
            deadline = time.time() + args.hours * 3600 if args.hours else None

            def _left():
                return None if deadline is None else max((deadline - time.time()) / 3600, 0.0)

            s1 = [s for s in studies if s[1] == "bc"]
            s2 = [s for s in studies if s[1] != "bc"]
            if s1:
                print("== stage 1: search bc ==", flush=True)
                hpo.search(s1, **dict(search_kw, hours=_left()))
                print("== stage 2: refine bc ==", flush=True)
                hpo.refine(_names(s1, args.tag), **refine_kw)
            if s2:
                print("== stage 3: search dqn / bc_dqn ==", flush=True)
                hpo.search(s2, **dict(search_kw, hours=_left()))
                print("== stage 4: refine dqn / bc_dqn ==", flush=True)
                hpo.refine(_names(s2, args.tag), **refine_kw)
            print("== stage 5: confirm on the study households (test year) ==",
                  flush=True)
            hpo.confirm(_names(studies, args.tag),
                        **dict(common, seeds=hpo.CONFIRM_SEEDS))
            report(args.root)
    except KeyboardInterrupt:
        print(f"stopped after {(time.time() - t0) / 60:.1f} min; rerun the same "
              f"command to resume", flush=True)
        sys.exit(130)
    print(f"ALL DONE ({args.phase}, {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
