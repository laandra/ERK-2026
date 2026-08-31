#!/usr/bin/env python3
"""
Batch runner for LocalModel.ipynb across 30 representative users.

Usage (from the Andraz/ directory):
    python run_batch.py              # full run (all 30 users, 100 basic + 410 extended episodes)
    python run_batch.py --test       # quick test: 1 user, 2 episodes only
    python run_batch.py --user 138   # run only user 138 with full episodes

Outputs go to "Local model resoults/" (existing empty folder):
  - user_{id}_basic_price_comparison.pdf / .png
  - user_{id}_extended_price_comparison.pdf / .png
  - summary_results.csv  (one row per user, columns for both modes)
"""

import os
import sys
import csv
import copy
import warnings

# ── Must set backend BEFORE any other matplotlib import ─────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message=".*non-interactive.*", category=UserWarning)

# ── Working directory: always run from Andraz/ ───────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import json
import nbformat
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Local model resoults")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_results.csv")
NOTEBOOK_PATH = os.path.join(SCRIPT_DIR, "LocalModel.ipynb")
CLUSTER_CSV = os.path.join(SCRIPT_DIR, "clustering_results", "user_ids_sorted_by_cluster_30.csv")

CSV_FIELDNAMES = [
    "user_id", "cluster_id",
    "no_battery_cost", "milp_cost",
    "basic_dqn_cost", "basic_dqn_reward",
    "basic_dqn_improvement_pct", "basic_milp_improvement_pct",
    "basic_dqn_gap_vs_milp_pct", "basic_dqn_fraction_theoretical_pct",
    "extended_dqn_cost", "extended_dqn_reward",
    "extended_dqn_improvement_pct",
    "extended_dqn_gap_vs_milp_pct", "extended_dqn_fraction_theoretical_pct",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_30_user_ids():
    """Return [(user_id_int, cluster_id), ...] for rank=1 users in each cluster."""
    df = pd.read_csv(CLUSTER_CSV)
    top = df[df["rank_in_cluster"] == 1].sort_values("cluster")
    return [(int(row["user_id"].split("_")[1]), int(row["cluster"]))
            for _, row in top.iterrows()]


def load_notebook_cells():
    """Return dict {cell_index: source_string} for code cells only."""
    with open(NOTEBOOK_PATH, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    return {i: "".join(cell.source)
            for i, cell in enumerate(nb.cells)
            if cell.cell_type == "code"}


def run_cell(code, ns, label=""):
    """exec code in namespace ns; propagate exceptions with a label."""
    try:
        exec(compile(code, f"<cell {label}>", "exec"), ns)
    except Exception as exc:
        print(f"  [ERROR] cell {label}: {type(exc).__name__}: {exc}")
        raise


def compute_metrics(ns):
    """Pull scalar metrics from the evaluation namespace."""
    placiloDQN = ns["placiloDQN"]
    nagradaDQN = ns["nagradaDQN"]
    no_battery_cost = ns["agent_brez_baterije"].Price[-1]
    milp_cost = float(ns["df_milp"]["Cum_Cost"].iloc[-1])
    dqn_cost = float(placiloDQN[-1])
    dqn_reward = float(nagradaDQN[-1])

    denom = no_battery_cost - milp_cost
    improvement_dqn = ((no_battery_cost - dqn_cost) / no_battery_cost) * 100
    improvement_milp = ((no_battery_cost - milp_cost) / no_battery_cost) * 100
    gap_dqn_vs_milp = ((dqn_cost - milp_cost) / milp_cost) * 100 if milp_cost else 0.0
    fraction_theoretical = ((no_battery_cost - dqn_cost) / denom) * 100 if denom else 0.0

    return {
        "no_battery_cost": no_battery_cost,
        "milp_cost": milp_cost,
        "dqn_cost": dqn_cost,
        "dqn_reward": dqn_reward,
        "dqn_improvement_pct": improvement_dqn,
        "milp_improvement_pct": improvement_milp,
        "dqn_gap_vs_milp_pct": gap_dqn_vs_milp,
        "dqn_fraction_theoretical_pct": fraction_theoretical,
    }


def save_plot(ns, uid, mode):
    """Build and save the price-comparison plot for one user/mode."""
    placiloDQN = list(ns["placiloDQN"])
    no_batt_price = list(ns["agent_brez_baterije"].Price)
    milp_cum = ns["df_milp"]["Cum_Cost"].to_numpy().tolist()
    x_dates = ns["nakljucni_agent"].Date
    plotMultiY = ns["plotMultiY"]

    # Trim to same length
    n = min(len(placiloDQN), len(no_batt_price), len(milp_cum)) - 1
    Y = [placiloDQN[:n], no_batt_price[:n], milp_cum[:n]]
    X = list(x_dates)[:n]

    mode_label = "Basic Training" if mode == "basic" else "Extended Convergence"
    title = f"User {uid} – {mode_label} – Electricity Cost [DQN vs MILP]"

    plt.figure(figsize=(12, 6))
    plotMultiY(
        X, Y,
        "Time", ["Cost [EUR]"],
        ["Optimized DQN", "No Battery", "MILP (Global Optimum)"],
        title,
        save_pdf=False, save=False, show_title=True,
    )

    fig = plt.gcf()
    base = f"user_{uid}_{mode}_price_comparison"
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}.png"), bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"    Saved {base}.pdf / .png")


def append_csv_row(row):
    file_exists = os.path.isfile(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Per-user setup code ──────────────────────────────────────────────────────

BASIC_EPISODES = 100      # default for full run
EXTENDED_ITERS = 40      # default for full run (40 x 10 = 400 + 10 initial = 410 total episodes)

BASIC_TRAIN_CODE = """
train_agent, train_env = Learning_DQN(reset=True, ponovitev=BASIC_EPISODES)
"""

EVAL_CODE = """
train_agent.epsilon = 0.0
nagradaDQN, placiloDQN, baterijaDQN, NagradaKapaciteta, NagradaSprememba, NagradaPlacilo = (
    Learning_DQN(ucenje=False, agent=train_agent)
)
"""

MILP_CODE = """
df_milp = run_milp_benchmark(env, use_discrete_actions=True)
"""


def run_user(uid, cluster_id, cells):
    print(f"\n{'='*60}")
    print(f"USER {uid}  (cluster {cluster_id})")
    print(f"{'='*60}")

    # ── Fresh namespace per user ──────────────────────────────────────────────
    ns = {"__builtins__": __builtins__}

    # ── Setup cells ──────────────────────────────────────────────────────────
    print("  [setup] imports...")
    run_cell(cells[3], ns, "3:imports")

    print("  [setup] data loading...")
    data_code = cells[6].replace(
        '"../Input data/Ausgrid/Ausgrid 123.csv"',
        f'"../Input data/Ausgrid/Ausgrid {uid}.csv"',
    )
    run_cell(data_code, ns, "6:data")

    print("  [setup] preprocessing...")
    run_cell(cells[7], ns, "7:split")
    run_cell(cells[9], ns, "9:params")
    run_cell(cells[10], ns, "10:normalize")
    run_cell(cells[12], ns, "12:build_env")

    print("  [setup] DQN class & Learning_DQN...")
    run_cell(cells[16], ns, "16:DQN_class")
    run_cell(cells[17], ns, "17:Learning_DQN")

    # ── Baselines (once per user) ─────────────────────────────────────────────
    print("  [baselines] test env + MILP function...")
    run_cell(cells[31], ns, "31:test_env+milp_fn")

    print("  [baselines] SARSA agents + random DQN eval...")
    run_cell(cells[33], ns, "33:sarsa_envs")
    run_cell(cells[34], ns, "34:sarsa_agents")

    print("  [baselines] no-battery agent...")
    run_cell(cells[36], ns, "36:no_battery")

    print("  [baselines] MILP benchmark (this may take several minutes)...")
    run_cell(MILP_CODE, ns, "milp")

    # Capture baseline scalars before DQN training overwrites anything
    no_battery_cost = float(ns["agent_brez_baterije"].Price[-1])
    milp_cost = float(ns["df_milp"]["Cum_Cost"].iloc[-1])
    print(f"    No-battery cost: {no_battery_cost:.2f} EUR | MILP cost: {milp_cost:.2f} EUR")

    row = {"user_id": uid, "cluster_id": cluster_id}

    # ── BASIC mode ───────────────────────────────────────────────────────────
    print(f"  [BASIC] training ({BASIC_EPISODES} episodes)...")
    ns["BASIC_EPISODES"] = BASIC_EPISODES
    run_cell(BASIC_TRAIN_CODE, ns, "basic:train")

    print("  [BASIC] evaluation...")
    run_cell(EVAL_CODE, ns, "basic:eval")

    m = compute_metrics(ns)
    print(f"    DQN cost: {m['dqn_cost']:.2f} EUR  "
          f"(improvement: {m['dqn_improvement_pct']:.1f}%  "
          f"fraction of theoretical: {m['dqn_fraction_theoretical_pct']:.1f}%)")

    save_plot(ns, uid, "basic")

    row.update({
        "no_battery_cost": no_battery_cost,
        "milp_cost": milp_cost,
        "basic_dqn_cost": m["dqn_cost"],
        "basic_dqn_reward": m["dqn_reward"],
        "basic_dqn_improvement_pct": m["dqn_improvement_pct"],
        "basic_milp_improvement_pct": m["milp_improvement_pct"],
        "basic_dqn_gap_vs_milp_pct": m["dqn_gap_vs_milp_pct"],
        "basic_dqn_fraction_theoretical_pct": m["dqn_fraction_theoretical_pct"],
    })

    # ── EXTENDED convergence mode ─────────────────────────────────────────────
    print(f"  [EXTENDED] convergence training ({EXTENDED_ITERS} x 10 episodes with eval)...")
    ns["EXTENDED_ITERS"] = EXTENDED_ITERS
    # Patch the convergence loop iteration count without modifying the notebook
    extended_code = cells[23].replace("for i in range(40):", "for i in range(EXTENDED_ITERS):")
    run_cell(extended_code, ns, "23:convergence")

    print("  [EXTENDED] final evaluation with best weights...")
    run_cell(EVAL_CODE, ns, "extended:eval")

    m = compute_metrics(ns)
    print(f"    DQN cost: {m['dqn_cost']:.2f} EUR  "
          f"(improvement: {m['dqn_improvement_pct']:.1f}%  "
          f"fraction of theoretical: {m['dqn_fraction_theoretical_pct']:.1f}%)")

    save_plot(ns, uid, "extended")

    row.update({
        "extended_dqn_cost": m["dqn_cost"],
        "extended_dqn_reward": m["dqn_reward"],
        "extended_dqn_improvement_pct": m["dqn_improvement_pct"],
        "extended_dqn_gap_vs_milp_pct": m["dqn_gap_vs_milp_pct"],
        "extended_dqn_fraction_theoretical_pct": m["dqn_fraction_theoretical_pct"],
    })

    append_csv_row(row)
    print(f"  [done] user {uid} written to CSV.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BASIC_EPISODES, EXTENDED_ITERS

    # ── CLI args ─────────────────────────────────────────────────────────────
    test_mode = "--test" in sys.argv
    single_user = None
    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        single_user = int(sys.argv[idx + 1])

    if test_mode:
        BASIC_EPISODES = 2
        EXTENDED_ITERS = 2
        print("[TEST MODE] Using 2 episodes for basic, 2 iterations for extended.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove stale summary so we start fresh (comment out to resume a partial run)
    if os.path.isfile(SUMMARY_CSV) and not single_user:
        os.remove(SUMMARY_CSV)

    all_users = get_30_user_ids()
    if single_user is not None:
        users = [(uid, cid) for uid, cid in all_users if uid == single_user]
        if not users:
            print(f"User {single_user} not in the 30-user list: {[u for u, _ in all_users]}")
            return
    elif test_mode:
        users = all_users[:1]   # just one user for a quick smoke test
    else:
        users = all_users

    print(f"Running LocalModel for {len(users)} users: {[u for u, _ in users]}")

    cells = load_notebook_cells()

    for i, (uid, cluster_id) in enumerate(users, 1):
        print(f"\n[{i}/{len(users)}]", end="")
        try:
            run_user(uid, cluster_id, cells)
        except Exception as exc:
            print(f"\n  !! FAILED for user {uid}: {exc}  — continuing with next user")

    print(f"\n{'='*60}")
    print(f"Batch complete. Results in: {OUTPUT_DIR}")
    if os.path.isfile(SUMMARY_CSV):
        df = pd.read_csv(SUMMARY_CSV)
        print(df.to_string(index=False))

 
if __name__ == "__main__":
    main()
