"""Invariant checks for the hyperparameter search (rl_hpo.py, run_rl_hpo.py)
and the clone-option hooks it added to rl_control / run_rl_benchmark.

Plain script like `test_rl_control.py`: one PASS/FAIL per check, non-zero exit
on failure. Each check is a way the search could silently be wrong: tuning a
configuration that is not the one later trained, moving a result that was
already on disk, touching the test year, or losing or redoing work across an
interruption.

    cd Main && python3 test_rl_hpo.py        (~1-2 min; one real clone, one
                                              short DQN, the rest in a temp dir)
"""

import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.simplefilter("ignore")

import rl_control as rl                                          # noqa: E402
import rl_hpo as hpo                                             # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


TMP = tempfile.mkdtemp(prefix="rl_hpo_test_")
ROOT, MODELS = os.path.join(TMP, "results"), os.path.join(TMP, "models")

# --------------------------------------------------------------------------
# 1. The clone-option hooks leave every existing result where it was
# --------------------------------------------------------------------------
cfg = rb.make_config(hpo.STEPS, 0)
spec = {t: rb.variant_specs(t)[hpo.VARIANT] for t in hpo.TARIFFS}
defaults = {rb.CLONE_PREFIX + k: v for k, v in rl.BCOptions().config().items()
            if v is not None}
for t in hpo.TARIFFS:
    for m in hpo.METHODS:
        tuned = dict(rb.TUNED.get((t, m), {}))
        a = rb.run_digest(cfg, spec[t], t, m)
        b = rb.run_digest(cfg, spec[t], t, m, overrides=dict(tuned, **defaults))
        check(f"{t} {m}: spelling out the default clone options keeps the digest",
              a == b)
        c = rb.run_digest(cfg, spec[t], t, m,
                          overrides=dict(tuned, clone_batch=1024))
        if m == "dqn":
            check(f"{t} dqn: a clone option cannot move a pure-DQN digest", a == c)
        else:
            check(f"{t} {m}: a changed clone option changes the digest", a != c)

check("_method_config's default output is unchanged (run_rl_global calls it)",
      rb._method_config(cfg, "bc") == {k: getattr(cfg, k) for k in rb._BC_FIELDS})
check("effective_config ignores clone keys, effective_clone applies them",
      rb.effective_config(cfg, "AU", "bc", {"clone_batch": 64}).batch == cfg.batch
      and rb.effective_clone("AU", "bc", {"clone_batch": 64}).batch == 64)
try:
    rb.effective_config(cfg, "AU", "bc", {"no_such_field": 1})
    check("an unknown override still raises", False)
except AttributeError:
    check("an unknown override still raises", True)

# The panel's stored clone digest for household 138 is reproduced.
panel = os.path.join(rb.OUT, "AU", f"{hpo.VARIANT}__bc", "138.json")
if os.path.exists(panel):
    with open(panel, encoding="utf-8") as fh:
        stored = json.load(fh)
    check("the panel's clone for Ausgrid 138 is still current under the hooks",
          stored["digest"] == rb.run_digest(cfg, spec["AU"], "AU", "bc"))

# --------------------------------------------------------------------------
# 2. The search space and its incumbent
# --------------------------------------------------------------------------
for t in hpo.TARIFFS:
    for m in hpo.METHODS:
        inc = hpo.incumbent_params(t, m)
        bad = hpo.in_space(m, inc)
        check(f"{t} {m}: the current settings are a point of the space",
              not bad, f"outside: {bad}")
        if m != "bc_dqn":
            over = hpo.trial_overrides(dict(rb.TUNED.get((t, m), {})), {}, inc)
            check(f"{t} {m}: trial 0 trains exactly the panel's configuration",
                  rb.run_digest(cfg, spec[t], t, m, overrides=over)
                  == rb.run_digest(cfg, spec[t], t, m))
names = {m: [n for n, _, _ in hpo.SPACES[m]] for m in hpo.METHODS}
check("bc_dqn does not search the width (the warm start fixes it)",
      "hidden" not in names["bc_dqn"])
check("every searched name is a TrainConfig field or a clone option",
      all(hasattr(rl.TrainConfig(), n) or
          (n.startswith(rb.CLONE_PREFIX) and
           hasattr(rl.BCOptions(), n[len(rb.CLONE_PREFIX):]))
          for m in hpo.METHODS for n in names[m]))

hh = hpo.tuning_households(16)
units = set(hpo.study_units())
check("tuning households never include a study household",
      not (set(hh) & units), f"{sorted(set(hh) & units)}")
check("tuning households are deterministic", hh == hpo.tuning_households(16))
import pandas as pd                                              # noqa: E402
import hems_study as hs                                          # noqa: E402
cl = pd.read_csv(hs.CLUSTERING_CSV)
cl["ident"] = cl["user_id"].str.removeprefix("user_").astype(int)
clusters = cl.set_index("ident").loc[hh[:8], "cluster"]
check("the first 8 tuning households are 8 different shape types",
      clusters.nunique() == 8, f"{clusters.nunique()} types")

# --------------------------------------------------------------------------
# 3. Scheduler, interruption and resume -- with a fake evaluator
# --------------------------------------------------------------------------
CALLS = []


def fake_job(job):
    """Writes what run_one + the baseline would write, from a smooth function
    of the configuration, and records that it was called."""
    rows = []
    for ev in job["evals"]:
        CALLS.append(hpo.eval_key(ev))
        t, m, i = ev["tariff"], ev["method"], int(ev["ident"])
        over = ev["overrides"] or {}
        lr = over.get("lr", 1e-3)
        saving = 100.0 + (i % 7) - 50.0 * (np.log10(lr) + 2.5) ** 2 + ev["seed"] * 0.1
        base = 1000.0
        val_net = base - saving * hpo.N_VAL_DAYS / 365.0
        res = {"digest": rb.run_digest(rb.make_config(ev["steps"], ev["seed"]),
                                       spec[t], t, m, overrides=ev["overrides"]),
               "val_cost_net_of_wear": val_net, "val_cost_closed": val_net,
               "val_efc": 10.0, "train_converged": True}
        if ev.get("score_test"):
            res.update({"cost_eur_total": 900.0 - saving, "cost_eur_closed": 900.0 - saving,
                        "fixed_eur": 0.0, "efc": 150.0})
        if not ev.get("read_only"):
            hpo._write_json(hpo._result_path(ev["out_root"], t, m, i), res)
        hpo._write_json(hpo.baseline_path(t, i, ev["root"]),
                        {"digest": hpo._baseline_digest(t), "val_idle_cost": base})
        rows.append(hpo._row(ev, res, base))
    return rows


H4 = [hh[0], hh[1], hh[2], hh[3]]
kw = dict(n_jobs=0, root=ROOT, models=MODELS, job_fn=fake_job,
          households=H4, verbose=False)
src = hpo.search([("AU", "bc")], trials=5, **kw)[0]
tf = hpo.trials_frame("AU_bc", ROOT)
done = tf[tf.state.isin(["COMPLETE", "PRUNED"])]
check("search reaches its trial target", len(done) == 5, f"{len(done)}")
t0 = tf.sort_values("number").iloc[0]
check("trial 0 is the incumbent, at the current settings",
      bool(t0.incumbent) and abs(t0.p_lr - hpo.current_value("AU", "bc", "lr")) < 1e-15)

# Simulate a driver killed mid-trial: a trial asked and never told.
st = hpo.load_study("AU_bc", ROOT)
orphan = st.ask()
orphan_params = hpo.suggest(orphan, "bc")
n_calls = len(CALLS)
src = hpo.search([("AU", "bc")], trials=7, **kw)[0]
tf = hpo.trials_frame("AU_bc", ROOT)
check("an interrupted (RUNNING) trial is re-queued on restart", src.requeued == 1)
req = tf[tf.requeued_from == orphan.number]
check("the re-queued trial keeps its parameters",
      len(req) == 1 and abs(req.iloc[0].p_lr - orphan_params["lr"]) < 1e-15)
check("the orphan stays in the journal, closed as FAIL",
      tf.set_index("number").loc[orphan.number, "state"] == "FAIL")
done = tf[tf.state.isin(["COMPLETE", "PRUNED"])]
check("a larger --trials extends the study", len(done) == 7, f"{len(done)}")
check("finished households were not recomputed on resume",
      len(set(CALLS[:n_calls]) & set(CALLS[n_calls:])) == 0)

n_calls = len(CALLS)
hpo.search([("AU", "bc")], trials=7, **kw)
check("rerunning a finished study computes nothing", len(CALLS) == n_calls)

# Pruning path (rungs), with the dqn defaults on 4 households: rungs (3, 4).
src = hpo.search([("AU", "dqn")], trials=12, **kw)[0]
tf = hpo.trials_frame("AU_dqn", ROOT)
check("dqn study uses rungs and finishes its target",
      src.rungs == [3, 4] and tf.state.isin(["COMPLETE", "PRUNED"]).sum() == 12)
inc = tf[tf.incumbent]
check("the incumbent is never pruned", (inc.state == "COMPLETE").all())

# A second driver is refused while the first is alive.
lockp = os.path.join(ROOT, "driver.lock")
hpo._write_json(lockp, {"pid": os.getppid(), "started": "test"})
try:
    hpo.search([("AU", "bc")], trials=7, **kw)
    check("a second driver is refused while one is alive", False)
except RuntimeError:
    check("a second driver is refused while one is alive", True)
os.remove(lockp)

# Refine: incumbent + top configurations x seeds, winner.json.
out = hpo.refine(["AU_bc"], n_jobs=0, top=2, seeds=(1,), root=ROOT,
                 models=MODELS, job_fn=fake_job, verbose=False)
w = hpo.winner("AU_bc", ROOT)
check("refine writes a winner with the incumbent among its candidates",
      w is not None and w["incumbent_tag"] in w["candidates"]
      and len(w["candidates"]) == 3)
check("the winner is the best seed-averaged candidate",
      w["mean_saving_val_a"] >= max(c["mean"] for c in w["candidates"].values()) - 1e-9)
check("refine evaluated every candidate under seeds 0 and 1",
      all(c["n_seeds"] == 2 for c in w["candidates"].values()))

# A changed study definition is refused, not silently extended.
try:
    hpo.search([("AU", "bc")], trials=7, **dict(kw, households=H4[:3]))
    check("a study reopened under other households is refused", False)
except RuntimeError:
    check("a study reopened under other households is refused", True)

# Confirm: winner vs current on (two) study units, test-year rows.
if w["tag"] != w["incumbent_tag"]:
    hpo.confirm(["AU_bc"], n_jobs=0, seeds=(0,), units=sorted(units)[:2],
                root=ROOT, models=MODELS, job_fn=fake_job, verbose=False)
    cf_ = hpo.confirm_frame("AU_bc", ROOT)
    check("confirm scores winner and current on the test year",
          set(cf_.label) == {"winner", "current"} and cf_["test_net"].notna().all())
    evs = hpo._confirm_evals("AU_bc", (0, 1), sorted(units)[:2], ROOT, MODELS)
    cur0 = [e for e in evs if e["label"] == "current" and e["seed"] == 0]
    cur1 = [e for e in evs if e["label"] == "current" and e["seed"] == 1]
    check("the panel's own clone is reused READ-ONLY for the current settings",
          all(e["out_root"] == rb.OUT and e.get("read_only") for e in cur0)
          and all(e["out_root"] != rb.OUT and not e.get("read_only") for e in cur1))
else:
    print("SKIP  confirm (the fake search kept the incumbent)")

# --------------------------------------------------------------------------
# 4. One REAL clone and one REAL short DQN through the worker entry point
# --------------------------------------------------------------------------
REAL = os.path.join(TMP, "real")
ev = {"study": "t", "trial": 0, "tag": "current", "label": "current",
      "tariff": "AU", "method": "bc", "ident": 138, "seed": 0,
      "overrides": None, "steps": hpo.STEPS, "score_test": False,
      "out_root": os.path.join(REAL, "runs"), "models_root": os.path.join(REAL, "m"),
      "root": REAL}
rows = hpo.run_job({"tariff": "AU", "ident": 138, "evals": [ev]})
r = rows[0]
check("a real clone trains through the worker", r.get("error") is None, r.get("error"))
if os.path.exists(panel) and r.get("error") is None:
    check("the real clone reproduces the panel's validation bill bit for bit",
          r["val_bill"] == stored["val_cost_closed"],
          f"{r['val_bill']!r} vs {stored['val_cost_closed']!r}")
res = json.load(open(hpo._result_path(ev["out_root"], "AU", "bc", 138)))
check("a search run never writes a test-year number",
      "cost_eur_closed" not in res and "cost_eur_total" not in res
      and "cost_eur_closed" not in r)
check("the idle baseline is above the clone's validation bill (it saves)",
      r["val_idle"] > r["val_net"] and r["saving_val_a"] > 0,
      f"saving {r['saving_val_a']:.1f}/a")

ev2 = dict(ev, method="dqn", steps=20_000, overrides={"gamma": 0.99, "n_step": 2,
                                                       "eval_every": 10_000})
rows = hpo.run_job({"tariff": "AU", "ident": 138, "evals": [ev2]})
check("a real (short) DQN trains through the worker",
      rows[0].get("error") is None, rows[0].get("error"))

ev3 = dict(ev, overrides={"clone_batch": 256, "clone_class_power": 0.5,
                          "clone_weight_decay": 1e-5, "clone_label_smoothing": 0.05,
                          "clone_patience": 5, "lr": 3e-3, "hidden": 64},
           out_root=os.path.join(REAL, "runs2"))
rows = hpo.run_job({"tariff": "AU", "ident": 138, "evals": [ev3]})
check("a clone under non-default options trains and differs",
      rows[0].get("error") is None and rows[0]["val_bill"] != r["val_bill"],
      rows[0].get("error"))

shutil.rmtree(TMP, ignore_errors=True)
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL PASS")
