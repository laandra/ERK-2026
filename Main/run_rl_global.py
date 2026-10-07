"""Global and per-type learned controllers, against the per-household ones.

    python3 run_rl_global.py --phase cache --jobs 12          # teachers + features
    python3 run_rl_global.py --phase train --schemes global global_typed --jobs 3
    python3 run_rl_global.py --phase train --schemes type --jobs 9
    python3 run_rl_global.py --phase eval --jobs 12
    python3 run_rl_global.py --phase report

`run_rl_benchmark.py` trains one network per study household on that household
alone. This driver trains the same learners (`rl_global`, a pooled port of
`rl_control`) on POOLS of households and scores them on the same 30 study units,
the same test year, through the same `run_policy` + arm `settle` -- so every
number here is paired, household by household, with the local result already on
disk under `results_local/rl_screen/<tariff>/fc_h24__<method>/`.

The pools come from the study's own clustering, not from a new list: the k=30
file (`Clustering/Ausgrid/user_ids_sorted_by_cluster_30.csv`) assigns all 300
Ausgrid households to the 30 load-shape clusters whose rank-1 members ARE the
study units (`hs.study_units`). A cluster is therefore a consumer TYPE with the
study household as its most typical member, and 1-25 more households of the same
shape behind it.

    global         all 300 households, one network, no type information
    global_typed   all 300 households, one network, cluster one-hot in the input
    type           one network per cluster, trained on its members only

Stage 2, run once stage 1 says which direction pays:

    global_ft      the global_typed network FINE-TUNED on one type's members:
                   localisation by data on top of localisation by input, the
                   global weights as the prior (same scaler, same one-hot)
    subtype        a type of >= SUB_MIN members split again (KMeans on the
                   clustering's own 72-d weekly profile, TRAINING YEARS ONLY),
                   one network for the sub-type holding the study unit

Same three-way split as the local screen, applied to EVERY household in a pool:
TRAIN = days 0-730 minus every 5th week, VALIDATION = those weeks, TEST = days
730-1095 of the study units only, read once at the end. The hyperparameters are
the local ones (`run_rl_benchmark.TUNED`), not re-tuned for pooling -- a stated
limitation, and the conservative one: anything a pooled learner gains here it
gains under settings chosen for the local learner.

Resumable everywhere: the per-household cache, every model and every scored
result carry a digest; a rerun recomputes nothing that is current and
recomputes everything that is not. Outputs:

    rl_models/global/cache/<tariff>/<id>.npz       features, teacher labels
    rl_models/global/<tariff>/<scheme>/<group>/<method>.pt      (gitignored)
    results_local/rl_global/models/<tariff>/<scheme>/<group>/<method>.json
    results_local/rl_global/<tariff>/<scheme>__<method>/<id>.json   test year
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import rl_global as rg                                           # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results_local", "rl_global")
MODEL_OUT = os.path.join(OUT, "models")
MODELS = os.path.join(HERE, "rl_models", "global")
CACHE = os.path.join(MODELS, "cache")
CLUSTER_CSV = os.path.join(ROOT, "Clustering", "Ausgrid",
                           "user_ids_sorted_by_cluster_30.csv")

# One observation contract for every scheme: the local screen's baseline
# variant (median14 forecast on both channels, 24 h of lookahead, SI contract
# features on SI). The question here is the DATA a learner is given, so the
# features are held fixed at the variant every local comparison is made on.
VARIANT = "fc_h24"
H = rb.H
SCHEMES = ("global", "global_typed", "type")            # stage 1
STAGE2_SCHEMES = ("global_ft", "subtype")                 # stage 2
# Stage 3: types the CONTROLLER cares about. The k=30 clusters are load-SHAPE
# types (max-normalised weekly profiles): measured over the 299 pooled
# households they hold 0 % of the variance in annual PV and 14 % of the
# variance in annual load -- the two quantities that decide what a battery
# should do. `ctrl_type` re-types the population on exactly those (training
# years only), `ctrl_ft` fine-tunes the global network on each such type.
CTRL_SCHEMES = ("ctrl_type", "ctrl_ft", "rand_type")
# `rand_type` is the control for `type`: each study unit pooled with as many
# households as its shape type has members -- but drawn at RANDOM from the
# population. If shape-type pools do no better than random pools of the same
# size, the shape types carry nothing the controller can use, and the per-type
# deficit against the global model is the cost of the smaller pool alone.
CTRL_K = 6                                   # ~50 households a type
CTRL_TYPES = os.path.join(OUT, "control_types.csv")
METHODS = ("bc", "dqn", "bc_dqn")
# Stage 2 sub-clusters only a type large enough to split into pools that are
# still pools: >= SUB_MIN members, about SUB_SIZE households per sub-type.
SUB_MIN, SUB_SIZE = 8, 5
PROFILES = os.path.join(MODELS, "profiles_train.csv")

# Reinforcement budget as a function of pool size. The local learner gets
# 500k steps for one household (~17 passes over its 590 training days); a pool
# of n has n times the distinct days, but the network is the same size and the
# Python-bound loop is the cost. sqrt(n) grows the budget with the pool without
# letting the 300-household pool cost an afternoon per model; the cap keeps a
# global run near an hour. Early stopping still ends any run that has settled.
LOCAL_STEPS = 500_000
TYPE_STEP_CAP = 2_500_000
GLOBAL_STEPS = 3_000_000
N_EVALS = 25           # validation evaluations per run, whatever its budget


def dqn_budget(n_members: int, scheme: str) -> tuple[int, int]:
    """(total_steps, eval_every) for a pool of `n_members`."""
    if scheme in ("global", "global_typed"):
        steps = GLOBAL_STEPS
    else:
        steps = int(np.clip(LOCAL_STEPS * np.sqrt(n_members), LOCAL_STEPS,
                            TYPE_STEP_CAP))
        steps = int(round(steps / 10_000) * 10_000)
    eval_every = max(20_000, int(round(steps / N_EVALS / 2_000) * 2_000))
    return steps, eval_every


# ---------------------------------------------------------------------------
# The population and its types
# ---------------------------------------------------------------------------
def population():
    """Every clustered household: index = Ausgrid id, columns cluster / rank.

    Read from the same committed CSV `hs.study_units` reads, so the study units
    are, by construction, the rank-1 member of each cluster here.
    """
    import pandas as pd
    df = pd.read_csv(CLUSTER_CSV)
    df["ident"] = df["user_id"].str.removeprefix("user_").astype(int)
    return df.set_index("ident")[["cluster", "rank_in_cluster", "dist_to_centroid"]]


def study_units() -> list[int]:
    return [int(i) for i in hs.study_units(CLUSTER_CSV).index]


def groups_for(scheme: str, pop=None, pool_cap: int | None = None,
               clusters=None) -> dict:
    """{group_name: {"members", "val_members", "n_types", "type_of"}}.

    `pool_cap` (smoke tests only) keeps at most that many households per
    cluster, closest to the centroid first; it lands in every digest through the
    member list, so a capped run can never be served as a full one.
    """
    pop = population() if pop is None else pop
    if pool_cap:
        pop = pop[pop["rank_in_cluster"] <= pool_cap]
    units = study_units()
    if clusters is not None:
        pop = pop[pop["cluster"].isin(clusters)]
        units = [u for u in units if u in pop.index]
    if scheme in ("global", "global_typed"):
        typed = scheme == "global_typed"
        return {"all": {
            "members": sorted(int(i) for i in pop.index),
            # The study units' validation weeks: one household per type, so the
            # checkpoint is chosen across every type rather than by whichever
            # clusters happen to be large.
            "val_members": sorted(units),
            "n_types": int(population()["cluster"].nunique()) if typed else 0,
            "type_of": ({int(i): int(c) for i, c in pop["cluster"].items()}
                        if typed else {}),
        }}
    if scheme in ("type", "global_ft"):
        ft = scheme == "global_ft"
        n_types = int(population()["cluster"].nunique())
        out = {}
        for c, g in pop.groupby("cluster"):
            members = sorted(int(i) for i in g.index)
            out[f"c{int(c):02d}"] = {
                "members": members, "val_members": members,
                "n_types": n_types if ft else 0,
                "type_of": {i: int(c) for i in members} if ft else {},
                # The model a fine-tune starts from, by (scheme, group); the
                # method is the fine-tune's own.
                **({"parent": ["global_typed", "all"]} if ft else {})}
        return out
    if scheme == "rand_type":
        full = population()
        rng = np.random.default_rng(42)
        out = {}
        for c, gq in full.groupby("cluster"):
            unit = next(u for u in units if u in gq.index)
            others = [int(i) for i in full.index if i != unit]
            pick = rng.choice(others, size=len(gq) - 1, replace=False)
            members = sorted([unit] + [int(i) for i in pick])
            out[f"r{int(c):02d}"] = {"members": members, "val_members": members,
                                     "n_types": 0, "type_of": {}}
        return out
    if scheme in CTRL_SCHEMES:
        ct = control_types()
        if pool_cap or clusters is not None:
            ct = ct[ct.index.isin(pop.index)]
        out = {}
        for k, gq in ct.groupby("ctrl_type"):
            members = sorted(int(i) for i in gq.index)
            out[f"k{int(k)}"] = {
                "members": members, "val_members": members, "n_types": 0,
                "type_of": {},
                **({"parent": ["global", "all"]} if scheme == "ctrl_ft" else {})}
        return out
    if scheme == "subtype":
        out = {}
        for c, (members, sub_of) in subclusters(pop).items():
            unit = next(u for u in units if u in members)
            mine = sorted(i for i in members if sub_of[i] == sub_of[unit])
            out[f"c{c:02d}s"] = {"members": mine, "val_members": mine,
                                 "n_types": 0, "type_of": {}}
        return out
    raise ValueError(f"unknown scheme {scheme!r}")


def group_of_unit(scheme: str, ident: int, pop=None) -> str:
    if scheme in ("global", "global_typed"):
        return "all"
    if scheme == "rand_type":
        pop = population() if pop is None else pop
        return f"r{int(pop.loc[ident, 'cluster']):02d}"
    if scheme in CTRL_SCHEMES:
        return f"k{int(control_types().loc[ident, 'ctrl_type'])}"
    pop = population() if pop is None else pop
    tail = "s" if scheme == "subtype" else ""
    return f"c{int(pop.loc[ident, 'cluster']):02d}{tail}"


def household_scale(tariff: str = "AU"):
    """Annual load and PV per household over the TRAINING span, read off the
    cached base features (columns 6 / 7: this interval's consumption and PV,
    kWh) -- two years summed, halved. Identical on both tariffs: the meter
    does not know the tariff."""
    import pandas as pd
    rows = []
    for i in cached_members(tariff, population().index):
        with np.load(cache_path(tariff, i)) as z:
            b = z["base"]
        rows.append({"ident": int(i), "load_kwh_a": float(b[:, 6].sum()) / 2,
                     "pv_kwh_a": float(b[:, 7].sum()) / 2})
    d = pd.DataFrame(rows).set_index("ident")
    d["pv_ratio"] = d["pv_kwh_a"] / d["load_kwh_a"]
    return d


def control_types(k: int = CTRL_K):
    """Households typed by what the dispatch depends on, not by load shape.

    KMeans (the clustering's own seed) on the standardised log annual load and
    PV-to-load ratio, both over the training years only -- how much the house
    draws, and how much of it the roof can cover. Cached to CTRL_TYPES.
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    if os.path.exists(CTRL_TYPES):
        d = pd.read_csv(CTRL_TYPES, index_col="ident")
        if d["k"].iloc[0] == k:
            return d
    d = household_scale()
    X = np.column_stack([np.log(d["load_kwh_a"]), d["pv_ratio"]])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    d["ctrl_type"] = KMeans(n_clusters=k, random_state=42,
                            n_init="auto").fit(X).labels_
    d["k"] = k
    d["shape_cluster"] = population().loc[d.index, "cluster"]
    os.makedirs(os.path.dirname(CTRL_TYPES), exist_ok=True)
    d.to_csv(CTRL_TYPES)
    return d


def training_profiles(idents) -> "pd.DataFrame":
    """The clustering's 72-d normalised weekly profile, from the TRAINING
    YEARS only (days 0-730), one row per household. Cached to PROFILES.

    `Clustering/cluster_households.py`'s own `reshape2threedays` +
    `normalize_df`, so a sub-type is a finer cut of the same notion of shape
    the types were cut on. Only the training years, unlike the committed k=30
    file (which saw every year): a split this study makes for itself must not
    read the scored year.
    """
    import importlib.util
    import pandas as pd
    have = (pd.read_csv(PROFILES, index_col=0) if os.path.exists(PROFILES)
            else pd.DataFrame())
    need = [i for i in idents if f"user_{i}" not in have.index]
    if need:
        spec = importlib.util.spec_from_file_location(
            "cluster_households",
            os.path.join(ROOT, "Clustering", "cluster_households.py"))
        ch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ch)
        cols = {}
        for i in need:
            frames = hs.load_study_frames(os.path.join(rb.DATA, f"Ausgrid {i}.csv"),
                                          H=H, delta_t=rb.DELTA_T,
                                          n_train=rb.N_TRAIN, n_sim=rb.N_SIM)
            ser = frames["df_all_kwh"]["Energy_Consumption"].iloc[: rb.N_TRAIN * H]
            ser.index = pd.to_datetime(ser.index, utc=True).tz_localize(None)
            cols[f"user_{i}"] = ser.astype(np.float32)
        new = ch.normalize_df(ch.reshape2threedays(
            pd.DataFrame(cols).resample("1h").mean()))
        new.columns = [f"{a}_{b}" for a, b in new.columns]
        have = pd.concat([have, new]) if len(have) else new
        os.makedirs(os.path.dirname(PROFILES), exist_ok=True)
        have.to_csv(PROFILES)
    return have.loc[[f"user_{i}" for i in idents]]


def subclusters(pop=None) -> dict:
    """{cluster: (members, {ident: sub_label})} for every type >= SUB_MIN.

    KMeans at the clustering's fixed seed, k = round(n / SUB_SIZE). Only
    households with a current cache in BOTH tariffs are split, so the AU and
    SI sub-types are the same households and stay comparable.
    """
    from sklearn.cluster import KMeans
    pop = population() if pop is None else pop
    out = {}
    for c, g in pop.groupby("cluster"):
        members = sorted(int(i) for i in g.index)
        members = [i for i in members
                   if i in set(cached_members("AU", members))
                   and i in set(cached_members("SI", members))]
        if len(members) < SUB_MIN:
            continue
        X = training_profiles(members).to_numpy(dtype=float)
        k = max(2, int(round(len(members) / SUB_SIZE)))
        lab = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X).labels_
        out[int(c)] = (members, {i: int(l) for i, l in zip(members, lab)})
    return out


# ---------------------------------------------------------------------------
# Phase A: per-household cache (teacher labels and base features)
# ---------------------------------------------------------------------------
def _digest(*dicts) -> str:
    return hashlib.sha256(
        json.dumps(dicts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cache_digest(tariff: str) -> str:
    spec = rb.variant_specs(tariff)[VARIANT]
    return _digest(spec.config(), {"tariff": tariff, "teach_span": rb.TEACH_SPAN,
                                   "bc_algo": rl.BC_ALGO_VERSION,
                                   "n_train": rb.N_TRAIN,
                                   "soc_init": rb.SOC_INIT_USABLE})


def cache_path(tariff: str, ident: int) -> str:
    return os.path.join(CACHE, tariff, f"{ident}.npz")


def build_cache(ident: int, tariff: str, prep=None) -> str:
    """Base features over the training span, and the teacher walk's labels.

    The teacher is `run_rl_benchmark.teacher_setpoints` -- the SAME cache the
    local screen reads, so the 30 study units reuse their solves and the other
    270 households get theirs solved identically. The walk is
    `rl.teacher_actions` over `TEACH_SPAN`, the local clone's own walk; what is
    stored is what `rl.train_bc` would assemble from it (dynamic features and
    labels per step), so a pooled clone fits exactly the rows a local clone of
    each member would.
    """
    path = cache_path(tariff, ident)
    dg = cache_digest(tariff)
    if os.path.exists(path):
        try:
            with np.load(path) as z:
                if str(z["digest"]) == dg:
                    return path
        except Exception:
            pass
    prep = prep or rb.prepare_household(ident, tariff)
    spec = rb.variant_specs(tariff)[VARIANT]
    fb = rl.FeatureBuilder(spec)
    tr = prep["train"]
    base = fb.build_static(tr["sig"], load_fc=prep["fc_train"][0],
                           pv_fc=prep["fc_train"][1])
    setpoints = rb.teacher_setpoints(prep)
    a, b = rb.TEACH_SPAN
    respect_peak = tariff == "SI"
    soc_trace, peaks, labels = rl.teacher_actions(
        tr["sig"], tr["settle"], tr["env"], setpoints, a * H, b * H,
        rb.SOC_INIT_USABLE, respect_peak)
    dyn = np.stack([fb.dynamic(tr["sig"], idx, soc_trace[j], peaks[j])
                    for j, idx in enumerate(range(a * H, b * H))])
    baseline = rl.no_battery_cost_trace(tr["sig"], tr["settle"], tr["env"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, base=base.astype(np.float32), dyn=dyn.astype(np.float32),
             labels=labels.astype(np.int8), baseline=baseline,
             teach_start=a * H, digest=dg)
    os.replace(tmp, path)
    return path


def _cache_worker(ident, tariff):
    warnings.simplefilter("ignore")
    t0 = time.time()
    try:
        build_cache(ident, tariff)
        return (tariff, ident, time.time() - t0, None)
    except Exception as exc:
        return (tariff, ident, time.time() - t0, repr(exc))


def run_cache(tariffs, idents, n_jobs):
    from joblib import Parallel, delayed
    todo = []
    for t in tariffs:
        dg = cache_digest(t)
        for i in idents:
            p = cache_path(t, i)
            if os.path.exists(p):
                try:
                    with np.load(p) as z:
                        if str(z["digest"]) == dg:
                            continue
                except Exception:
                    pass
            todo.append((i, t))
    # SI first and the slow solves spread: the SI teacher MILP is ~15x the AU one.
    todo.sort(key=lambda x: x[1] != "SI")
    print(f"cache: {len(todo)} household-tariff(s) to build, "
          f"{len(tariffs) * len(idents) - len(todo)} current", flush=True)
    if not todo:
        return []
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_cache_worker)(i, t) for i, t in todo)
    failed = [r for r in res if r[3]]
    print(f"cache built in {(time.time() - t0) / 60:.1f} min; "
          f"{len(failed)} failed", flush=True)
    for t, i, _, exc in failed:
        print(f"  FAILED {t} Ausgrid {i}: {exc}")
    return failed


def cached_members(tariff: str, idents) -> list[int]:
    """The subset of `idents` with a current cache -- a household whose data
    could not be built drops out of every pool, visibly (printed by the caller),
    rather than failing every model it belongs to."""
    dg = cache_digest(tariff)
    ok = []
    for i in idents:
        p = cache_path(tariff, i)
        if os.path.exists(p):
            with np.load(p) as z:
                if str(z["digest"]) == dg:
                    ok.append(int(i))
    return ok


# ---------------------------------------------------------------------------
# Phase B: training
# ---------------------------------------------------------------------------
# Fine-tunes that may never come back worse than their parent on validation.
GUARDED_SCHEMES = ("ctrl_ft",)


def train_config(tariff: str, method: str, scheme: str, n_members: int,
                 seed: int = 0) -> rl.TrainConfig:
    cfg = rb.effective_config(rb.make_config(LOCAL_STEPS, seed), tariff, method)
    if method != "bc":
        cfg.total_steps, cfg.eval_every = dqn_budget(n_members, scheme)
    return cfg


def model_digest(tariff, scheme, group, method, g, cfg, bc_digest=None,
                 parent_digest=None) -> str:
    spec = rb.variant_specs(tariff)[VARIANT]
    version = {"bc": (rl.BC_ALGO_VERSION,),
               "dqn": (rl.ALGO_VERSION,),
               "bc_dqn": (rl.BC_ALGO_VERSION, rl.ALGO_VERSION)}[method]
    method_cfg = (rg.BCConfigPool().config() | rb._method_config(cfg, "bc")
                  if method == "bc" else cfg.config())
    return _digest(spec.config(), method_cfg,
                   {"tariff": tariff, "scheme": scheme, "group": group,
                    "method": method, "algo": version,
                    "global_algo": rg.GLOBAL_ALGO_VERSION,
                    "members": g["members"], "val_members": g["val_members"],
                    "n_types": g["n_types"], "type_of": g["type_of"],
                    "train_days": rb.TRAIN_BLOCKS, "val_days": rb.VAL_BLOCKS,
                    "teach_span": rb.TEACH_SPAN, "cache": cache_digest(tariff),
                    "bc_prior": bc_digest, "parent": parent_digest,
                    **({"guarded": True} if scheme in GUARDED_SCHEMES else {})})


def seed_tag(seed: int) -> str:
    """Suffix that keeps a seed replicate beside, not on top of, seed 0.

    A global scheme is ONE network for all 30 study units, so the paired test
    across households cannot see training-seed noise -- the 30 local networks
    average theirs out, a global one does not. Replicates (`--seed 1`, `2`)
    are what show whether a global result is a property of pooling or of one
    lucky run. Seed 0 keeps the bare names, so its files never moved.
    """
    return f"_s{int(seed)}" if seed else ""


def _paths(tariff, scheme, group, method):
    return (os.path.join(MODELS, tariff, scheme, group, f"{method}.pt"),
            os.path.join(MODEL_OUT, tariff, scheme, group, f"{method}.json"))


def model_current(tariff, scheme, group, method, digest) -> dict | None:
    pt, js = _paths(tariff, scheme, group, method)
    if os.path.exists(js) and os.path.exists(pt):
        with open(js, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("digest") == digest:
            return meta
    return None


def _train_rows():
    return np.concatenate([np.arange(a * H, b * H) for a, b in rb.TRAIN_BLOCKS])


def _val_mask(n_rows: int, teach_start: int) -> np.ndarray:
    held = set(rl.block_days(rb.VAL_BLOCKS).tolist())
    days = (teach_start + np.arange(n_rows)) // H
    return np.isin(days, list(held))


class _Pool:
    """A group's members, loaded lazily: cache arrays always, env bundles only
    when reinforcement needs them (17 MB a household, ~5 GB for the global
    pool -- the reason global jobs run on fewer workers)."""

    def __init__(self, tariff, scheme, group, g, verbose=True):
        self.tariff, self.scheme, self.group, self.g = tariff, scheme, group, g
        self.verbose = verbose
        self.spec = rb.variant_specs(tariff)[VARIANT]
        self.fb = rg.TypedFeatureBuilder(self.spec, n_types=g["n_types"])
        self.cache = {}
        for i in g["members"]:
            with np.load(cache_path(tariff, i)) as z:
                self.cache[i] = {k: z[k] for k in ("base", "dyn", "labels",
                                                   "baseline", "teach_start")}
        if g.get("parent"):
            # A fine-tune reads the world through its parent's scaler: the
            # weights it starts from were fitted to THOSE inputs, and a
            # refitted scaler would hand them different numbers for the same
            # state.
            _, pfb, _ = rg.load_model(_paths(tariff, *g["parent"], "bc")[0])
            self.fb.norm_mean, self.fb.norm_std = pfb.norm_mean, pfb.norm_std
        else:
            rows = _train_rows()
            self.fb.fit_norm_pooled(self.cache[i]["base"][rows]
                                    for i in g["members"])
        self.members = None

    def bound(self, ident):
        return self.fb.for_type(self.g["type_of"].get(ident) if self.g["n_types"]
                                else None)

    def bc_data(self):
        """(X_tr, y_tr, X_va, y_va) over every member's teacher walk."""
        Xtr, ytr, Xva, yva = [], [], [], []
        for i in self.g["members"]:
            c = self.cache[i]
            s0 = int(c["teach_start"])
            n = len(c["labels"])
            fb = self.bound(i)
            X = np.hstack([fb.normalize_base(c["base"][s0:s0 + n]), c["dyn"]])
            mask = _val_mask(n, s0)
            y = c["labels"].astype(np.int64)
            Xtr.append(X[~mask]); ytr.append(y[~mask])
            Xva.append(X[mask]); yva.append(y[mask])
        return (np.concatenate(Xtr), np.concatenate(ytr),
                np.concatenate(Xva), np.concatenate(yva))

    def rl_members(self):
        if self.members is not None:
            return self.members
        t0 = time.time()
        out = {}
        for n, i in enumerate(self.g["members"], 1):
            prep = rb.prepare_household(i, self.tariff)
            tr = prep["train"]
            fb = self.bound(i)
            c = self.cache[i]
            out[i] = rg.Member(ident=str(i), sig=tr["sig"], settle=tr["settle"],
                               env=tr["env"],
                               static_norm=fb.normalized_rows(c["base"]),
                               baseline=c["baseline"])
            # Reinforcement never returns to the clone's arrays; at 300
            # households they are a gigabyte held for nothing.
            for k in ("base", "dyn", "labels"):
                c.pop(k, None)
            del prep
            if self.verbose and (n % 25 == 0 or n == len(self.g["members"])):
                print(f"    prepared {n}/{len(self.g['members'])} households "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        self.members = out
        return out


def train_group(tariff, scheme, group, g, methods, seed=0, verbose=True):
    """Train `methods` (in order bc, dqn, bc_dqn) for one pool. Resumable."""
    warnings.simplefilter("ignore")
    rb._calendar(tariff)
    respect_peak = tariff == "SI"
    pool = None
    done = {}
    out = []
    order = [m for m in METHODS if m in methods]
    if "bc_dqn" in order and "bc" not in order:
        order = ["bc"] + order          # the prior is part of the result
    for method in order:
        cfg = train_config(tariff, method, scheme, len(g["members"]), seed)
        bc_dg = done.get("bc", {}).get("digest") if method == "bc_dqn" else None
        if method == "bc_dqn" and bc_dg is None:
            raise RuntimeError("bc_dqn needs its clone trained first")
        parent_meta = None
        if g.get("parent"):
            pjs = _paths(tariff, *g["parent"], method)[1]
            if not os.path.exists(pjs):
                out.append((tariff, scheme, group, method, 0.0,
                            f"parent {g['parent']} {method} not trained"))
                continue
            with open(pjs, encoding="utf-8") as fh:
                parent_meta = json.load(fh)
        dg = model_digest(tariff, scheme, group, method, g, cfg, bc_dg,
                          parent_meta["digest"] if parent_meta else None)
        meta = model_current(tariff, scheme, group, method, dg)
        if meta is not None:
            done[method] = meta
            out.append((tariff, scheme, group, method, "cached", None))
            continue
        t0 = time.time()
        try:
            if pool is None:
                pool = _Pool(tariff, scheme, group, g, verbose)
            if verbose:
                print(f"[{tariff} {scheme}/{group}] {method}: "
                      f"{len(g['members'])} members", flush=True)
            hist_bc = hist_dqn = None
            parent_net = (rg.load_model(_paths(tariff, *g["parent"], method)[0])[0]
                          if parent_meta else None)
            if method == "bc":
                Xtr, ytr, Xva, yva = pool.bc_data()
                # A fine-tuned clone steps at a third of the rate: it starts
                # at a 300-household optimum and has tens of thousands of
                # rows to move it with, not millions.
                net, hist_bc = rg.train_bc_pool(Xtr, ytr, Xva, yva, cfg,
                                                verbose=verbose,
                                                init_net=parent_net,
                                                lr_scale=0.3 if parent_net else 1.0)
                del Xtr, ytr, Xva, yva
            else:
                mem = pool.rl_members()
                members = [mem[i] for i in g["members"]]
                val_members = [mem[i] for i in g["val_members"]]
                init = bc_net = None
                if method == "bc_dqn":
                    bc_net, _, _ = rg.load_model(_paths(tariff, scheme, group, "bc")[0])
                    init = bc_net
                if parent_net is not None:
                    # Fine-tune: start from the global network of the SAME
                    # method (rl's warm-start protection applies -- eps 0.2,
                    # half rate); a bc_dqn fine-tune is still held to its
                    # clone, now the type-localised one trained just before.
                    init = parent_net
                net, hist_dqn = rg.train_dqn_pool(
                    members, val_members, pool.fb, cfg, respect_peak,
                    rb.TRAIN_BLOCKS, rb.VAL_BLOCKS, init_net=init, bc_net=bc_net,
                    soc_target=rb.SOC_INIT_USABLE, verbose=verbose,
                    val_wear=rb.val_wear_fn(tariff),
                    # The fallback, for the control-relevant fine-tune only:
                    # `global_ft` predates it and is kept as trained.
                    init_is_candidate=scheme in GUARDED_SCHEMES)
            pt, js = _paths(tariff, scheme, group, method)
            rg.save_model(pt, net, pool.fb, cfg, {"bc": hist_bc, "dqn": hist_dqn},
                          extra={"tariff": tariff, "scheme": scheme,
                                 "group": group, "method": method})
            hist = hist_dqn or hist_bc
            meta = {"digest": dg, "tariff": tariff, "scheme": scheme,
                    "group": group, "method": method,
                    "n_members": len(g["members"]), "members": g["members"],
                    "val_members": g["val_members"], "n_types": g["n_types"],
                    "parent": g.get("parent"),
                    "parent_digest": parent_meta["digest"] if parent_meta else None,
                    "converged": bool(hist.get("converged", False)),
                    "steps_run": hist.get("steps_run"),
                    "best_val_cost_closed": hist.get("best_val_cost_closed"),
                    "bc_val_agreement": (hist_bc or {}).get("final_val_agreement"),
                    "train_runtime_s": hist.get("runtime_s"),
                    "wall_s": time.time() - t0, "train_config": cfg.config(),
                    "model_path": os.path.relpath(pt, HERE)}
            os.makedirs(os.path.dirname(js), exist_ok=True)
            with open(js, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)
            done[method] = meta
            out.append((tariff, scheme, group, method, time.time() - t0, None))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out.append((tariff, scheme, group, method, time.time() - t0, repr(exc)))
            if method == "bc":
                break                    # bc_dqn cannot run without it
    return out


def run_train(tariffs, schemes, methods, n_jobs, seed=0, pool_cap=None,
              clusters=None, verbose=False):
    from joblib import Parallel, delayed
    jobs = []
    for t in tariffs:
        for s in schemes:
            for group, g in groups_for(s, pool_cap=pool_cap, clusters=clusters).items():
                group = group + seed_tag(seed)
                ok = cached_members(t, g["members"])
                missing = sorted(set(g["members"]) - set(ok))
                if missing:
                    print(f"  {t} {s}/{group}: {len(missing)} member(s) without "
                          f"a cache, dropped: {missing}")
                g = dict(g, members=ok,
                         val_members=[i for i in g["val_members"] if i in ok])
                if not g["members"] or not g["val_members"]:
                    continue
                jobs.append((t, s, group, g))
    # Biggest pools first: they set the wall clock.
    jobs.sort(key=lambda j: -len(j[3]["members"]))

    if any(s in ("global", "global_typed") for s in schemes):
        # Two stages for the global pools so no bc_dqn waits on a clone being
        # trained in another worker: every clone first, then reinforcement.
        stages = [["bc"], [m for m in methods if m != "bc"]]
    else:
        stages = [list(methods)]    # one worker runs bc -> dqn -> bc_dqn in order
    t0 = time.time()
    results = []
    for stage in stages:
        if not stage:
            continue
        print(f"train: {len(jobs)} pool(s) x {stage} on {n_jobs} worker(s)",
              flush=True)
        res = Parallel(n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=10)(
            delayed(train_group)(t, s, grp, g, stage, seed, verbose)
            for t, s, grp, g in jobs)
        results += [r for group in res for r in group]
    failed = [r for r in results if r[5]]
    print(f"train done in {(time.time() - t0) / 60:.1f} min: "
          f"{sum(1 for r in results if r[4] == 'cached')} cached, "
          f"{sum(1 for r in results if r[4] != 'cached' and not r[5])} trained, "
          f"{len(failed)} failed", flush=True)
    for r in failed:
        print(f"  FAILED {r[0]} {r[1]}/{r[2]} {r[3]}: {r[5]}")
    return failed


# ---------------------------------------------------------------------------
# Phase C: the test year, study units only
# ---------------------------------------------------------------------------
def _local_result(tariff, method, ident) -> dict:
    p = os.path.join(rb.OUT, tariff, f"{VARIANT}__{method}", f"{ident}.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        r = json.load(fh)
    return {"local_cost_eur_closed": r.get("cost_eur_closed"),
            "local_fixed_eur": r.get("fixed_eur"),
            "local_cost_eur_total": r.get("cost_eur_total"),
            "local_efc": r.get("efc"), "local_digest": r.get("digest")}


def eval_unit(tariff, ident, schemes, methods, seed=0):
    """Score every available (scheme, method) model on one study unit's test
    year through `rbc.run_policy` -- the local screen's own scoring call."""
    warnings.simplefilter("ignore")
    pop = population()
    prep = None
    out = []
    for scheme in schemes:
        group = group_of_unit(scheme, ident, pop) + seed_tag(seed)
        for method in methods:
            pt, js = _paths(tariff, scheme, group, method)
            if not os.path.exists(js):
                out.append((tariff, scheme, method, ident, "no model"))
                continue
            with open(js, encoding="utf-8") as fh:
                meta = json.load(fh)
            if ident not in meta["members"]:
                out.append((tariff, scheme, method, ident, "unit not in pool"))
                continue
            res_path = os.path.join(OUT, tariff,
                                    f"{scheme}__{method}{seed_tag(seed).replace('_', '__')}",
                                    f"{ident}.json")
            if os.path.exists(res_path):
                with open(res_path, encoding="utf-8") as fh:
                    if json.load(fh).get("model_digest") == meta["digest"]:
                        out.append((tariff, scheme, method, ident, None))
                        continue
            try:
                if prep is None:
                    prep = rb.prepare_household(ident, tariff)
                net, fb, _ = rg.load_model(pt)
                fb = fb.for_type(int(pop.loc[ident, "cluster"]) if fb.n_types
                                 else None)
                spec = fb.spec
                sim = prep["sim"]
                policy = rl.LearnedPolicy(
                    net, fb, tariff == "SI",
                    load_fc=prep["fc_sim"][0], pv_fc=prep["fc_sim"][1],
                    name=f"{method}_{scheme}",
                    label=f"{method.upper()} {scheme}", causal=spec.causal)
                t0 = time.time()
                o = rbc.run_policy(sim["env"], policy, n_steps=rb.N_SIM * H,
                                   settle=sim["settle"],
                                   soc_init_kwh=rb.SOC_INIT_USABLE,
                                   rates=sim["rates"])
                result = {
                    "dataset": f"Ausgrid {ident}", "ident": str(ident),
                    "tariff": tariff, "scheme": scheme, "group": group,
                    "method": method, "variant": VARIANT, "seed": int(seed),
                    "model_digest": meta["digest"],
                    "n_members": meta["n_members"], "n_types": meta["n_types"],
                    "cluster": int(pop.loc[ident, "cluster"]),
                    "train_converged": meta["converged"],
                    "train_steps": meta["steps_run"],
                    "bc_val_agreement": meta["bc_val_agreement"],
                    "cost_eur_closed": o["Cost_EUR_Closed"],
                    "cost_eur": o["Cost_EUR"], "fixed_eur": o["Fixed_EUR"],
                    "cost_eur_total": o["Cost_EUR_Total"],
                    "efc": o["Equivalent_Full_Cycles"],
                    "import_kwh": o["Import_kWh"], "export_kwh": o["Export_kWh"],
                    "peak_import_kw": o["Peak_Import_kW"],
                    "agreed_power_iters": o["Agreed_Power_Iters"],
                    "agreed_power_converged": o["Agreed_Power_Converged"],
                    "eval_s": time.time() - t0,
                }
                result.update(rb._comparators(ident, tariff))
                result.update(_local_result(tariff, method, ident))
                os.makedirs(os.path.dirname(res_path), exist_ok=True)
                with open(res_path, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=1)
                out.append((tariff, scheme, method, ident, None))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                out.append((tariff, scheme, method, ident, repr(exc)))
    return out


def run_eval(tariffs, schemes, methods, n_jobs, units=None, seed=0):
    from joblib import Parallel, delayed
    units = units or study_units()
    jobs = [(t, i) for t in tariffs for i in units]
    jobs.sort(key=lambda j: j[0] != "SI")       # SI converges a contract: slower
    print(f"eval: {len(jobs)} unit-tariff(s) x {len(schemes)} scheme(s) x "
          f"{len(methods)} method(s) on {n_jobs} worker(s)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=5)(
        delayed(eval_unit)(t, i, schemes, methods, seed) for t, i in jobs)
    flat = [r for g in res for r in g]
    bad = [r for r in flat if r[4] and r[4] not in ("no model", "unit not in pool")]
    nomodel = [r for r in flat if r[4] == "no model"]
    print(f"eval done in {(time.time() - t0) / 60:.1f} min: "
          f"{len(flat) - len(bad) - len(nomodel)} scored, {len(nomodel)} without "
          f"a model, {len(bad)} failed", flush=True)
    for r in bad:
        print(f"  FAILED {r}")
    return bad


# ---------------------------------------------------------------------------
# Phase C2: every model on each study unit's OWN validation weeks
# ---------------------------------------------------------------------------
# Training picked each pooled model's weights on validation summed over its
# POOL -- the 30 study units for a global model, every member for a type model.
# Nothing ever asked the question a deployment would: on THIS household's own
# held-out weeks, is the localized model better than the global one it
# replaces, or than the household's own local model? This answers it, for every
# model on disk, without touching the test year: same greedy rollout, same
# weeks, same SOC and peak seeding as the validation every model trained on.
UNIT_VAL = os.path.join(OUT, "unit_validation.csv")
SEEDS = (0, 1, 2)


def _unit_candidates(tariff, ident, pop):
    """(scheme, method, seed, model_path, digest, kind) for every model that
    could drive `ident` on `tariff`. `kind` is "local" or "pooled"."""
    out = []
    for method in METHODS:
        js = os.path.join(rb.OUT, tariff, f"{VARIANT}__{method}", f"{ident}.json")
        pt = os.path.join(rb.MODELS, tariff, f"{VARIANT}__{method}", f"{ident}.pt")
        if os.path.exists(js) and os.path.exists(pt):
            with open(js, encoding="utf-8") as fh:
                out.append(("local", method, 0, pt, json.load(fh)["digest"], "local"))
    for scheme in SCHEMES + STAGE2_SCHEMES + CTRL_SCHEMES:
        for seed in SEEDS:
            group = group_of_unit(scheme, ident, pop) + seed_tag(seed)
            for method in METHODS:
                pt, js = _paths(tariff, scheme, group, method)
                if not (os.path.exists(pt) and os.path.exists(js)):
                    continue
                with open(js, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if ident in meta["members"]:
                    out.append((scheme, method, seed, pt, meta["digest"], "pooled"))
    return out


def _unit_val_worker(tariff, ident, have):
    warnings.simplefilter("ignore")
    pop = population()
    # A model digest is shared by every household a pooled model drives (and
    # by every local run of one configuration), so the key is the pair.
    todo = [c for c in _unit_candidates(tariff, ident, pop)
            if f"{tariff}|{ident}|{c[4]}" not in have]
    if not todo:
        return []
    prep = rb.prepare_household(ident, tariff)
    tr = prep["train"]
    respect_peak = tariff == "SI"
    n_val_days = sum(b - a for a, b in rb.VAL_BLOCKS)
    wear = rb.val_wear_fn(tariff)
    rows = []
    for scheme, method, seed, pt, digest, kind in todo:
        if kind == "local":
            net, fb, _, _ = rl.load_model(pt)
        else:
            net, fb, _ = rg.load_model(pt)
            fb = fb.for_type(int(pop.loc[ident, "cluster"]) if fb.n_types else None)
        static = fb.normalize(fb.build_static(tr["sig"], load_fc=prep["fc_train"][0],
                                              pv_fc=prep["fc_train"][1]))
        v = rl.validation_rollout(net, fb, static, tr["sig"], tr["settle"],
                                  tr["env"], rb.VAL_BLOCKS, rb.SOC_INIT_USABLE,
                                  respect_peak)
        rows.append({"tariff": tariff, "ident": int(ident), "scheme": scheme,
                     "method": method, "seed": int(seed), "digest": digest,
                     "val_cost_closed": v["cost_eur_closed"], "val_efc": v["efc"],
                     "val_net": v["cost_eur_closed"] + wear(v["efc"], n_val_days)})
    return rows


def unit_validation(tariffs=("AU", "SI"), n_jobs=12, units=None):
    """Validation cost of every model on every study unit's own held-out weeks.

    Cached in UNIT_VAL by model digest: a retrained model is re-rolled, a
    current one is read back. Returns the frame.
    """
    import pandas as pd
    from joblib import Parallel, delayed
    units = units or study_units()
    old = pd.read_csv(UNIT_VAL) if os.path.exists(UNIT_VAL) else pd.DataFrame()
    key = lambda d: d["tariff"] + "|" + d["ident"].astype(str) + "|" + d["digest"]
    have = set(key(old)) if len(old) else set()
    jobs = [(t, i) for t in tariffs for i in units]
    t0 = time.time()
    res = Parallel(n_jobs=min(n_jobs, len(jobs)), backend="loky")(
        delayed(_unit_val_worker)(t, i, have) for t, i in jobs)
    new = pd.DataFrame([r for g in res for r in g])
    df = pd.concat([old, new], ignore_index=True) if len(new) else old
    # Keep only rows whose model is still the one on disk.
    pop = population()
    current = {f"{t}|{i}|{c[4]}" for t in tariffs for i in units
               for c in _unit_candidates(t, i, pop)}
    df = df[key(df).isin(current)]
    df = df[~key(df).duplicated(keep="last")]
    df.to_csv(UNIT_VAL, index=False)
    print(f"unit validation: {len(new)} new rollouts in "
          f"{(time.time() - t0) / 60:.1f} min, {len(df)} on file", flush=True)
    return df


# ---------------------------------------------------------------------------
# Phase C3: the noise floor and the type input
# ---------------------------------------------------------------------------
# A pooled clone and a local clone are each ONE training run per household, so
# their per-household difference carries two runs' worth of seed noise. The
# local screen has one seed; these re-run its clone under others -- in their own
# tree, never over the published local results -- so a scheme's gap to local
# can be read against the gap between two local clones that differ only in seed.
LOCAL_SEEDS_OUT = os.path.join(OUT, "local_seeds")
LOCAL_SEEDS_MODELS = os.path.join(MODELS, "local_seeds")


def _local_seed_worker(tariff, ident, seeds, methods):
    warnings.simplefilter("ignore")
    prep = rb.prepare_household(ident, tariff)
    out = []
    for seed in seeds:
        for method in methods:
            cfg = rb.make_config(LOCAL_STEPS, seed)
            r = rb.run_one(prep, VARIANT, method, cfg, verbose=False,
                           out_root=os.path.join(LOCAL_SEEDS_OUT, f"s{seed}"),
                           models_root=os.path.join(LOCAL_SEEDS_MODELS, f"s{seed}"))
            out.append({"tariff": tariff, "ident": int(ident), "method": method,
                        "seed": seed, "cost_eur_total": r["cost_eur_total"],
                        "cost_eur_closed": r["cost_eur_closed"], "efc": r["efc"]})
    return out


def local_seed_runs(seeds=(1, 2), methods=("bc",), tariffs=("AU", "SI"),
                    n_jobs=12):
    """The local clone under other seeds, scored on the test year. Resumable
    through `run_one`'s own digest cache. Returns one long frame, seed 0 (the
    published local result) included."""
    import pandas as pd
    from joblib import Parallel, delayed
    jobs = [(t, i) for t in tariffs for i in study_units()]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_local_seed_worker)(t, i, seeds, methods) for t, i in jobs)
    df = pd.DataFrame([r for g in res for r in g])
    base = []
    for t in tariffs:
        for i in study_units():
            for m in methods:
                p = os.path.join(rb.OUT, t, f"{VARIANT}__{m}", f"{i}.json")
                with open(p, encoding="utf-8") as fh:
                    r = json.load(fh)
                base.append({"tariff": t, "ident": int(i), "method": m, "seed": 0,
                             "cost_eur_total": r["cost_eur_total"],
                             "cost_eur_closed": r["cost_eur_closed"],
                             "efc": r["efc"]})
    return pd.concat([pd.DataFrame(base), df], ignore_index=True)


def _wrong_type_worker(tariff, ident, methods, shift):
    """The typed global model driven with the household's TRUE type and with a
    WRONG one (cluster + shift, mod 30), test year. If the two score alike,
    the network is not using its type input."""
    warnings.simplefilter("ignore")
    pop = population()
    prep = rb.prepare_household(ident, tariff)
    sim = prep["sim"]
    true_k = int(pop.loc[ident, "cluster"])
    out = []
    for method in methods:
        net, fb, _ = rg.load_model(_paths(tariff, "global_typed", "all", method)[0])
        for label, k in (("true", true_k), ("wrong", (true_k + shift) % fb.n_types)):
            policy = rl.LearnedPolicy(net, fb.for_type(k), tariff == "SI",
                                      load_fc=prep["fc_sim"][0],
                                      pv_fc=prep["fc_sim"][1])
            o = rbc.run_policy(sim["env"], policy, n_steps=rb.N_SIM * H,
                               settle=sim["settle"],
                               soc_init_kwh=rb.SOC_INIT_USABLE, rates=sim["rates"])
            out.append({"tariff": tariff, "ident": int(ident), "method": method,
                        "type_input": label, "cost_eur_total": o["Cost_EUR_Total"],
                        "efc": o["Equivalent_Full_Cycles"]})
    return out


def wrong_type_test(tariffs=("AU", "SI"), methods=METHODS, shift=15, n_jobs=12):
    """Seed-0 typed global model, true vs wrong type, every study unit.
    Cached to OUT/wrong_type.csv (keyed on nothing but the inputs: rerun after
    retraining global_typed)."""
    import pandas as pd
    from joblib import Parallel, delayed
    path = os.path.join(OUT, "wrong_type.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    jobs = [(t, i) for t in tariffs for i in study_units()]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_wrong_type_worker)(t, i, methods, shift) for t, i in jobs)
    df = pd.DataFrame([r for g in res for r in g])
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Joining onto the study frame
# ---------------------------------------------------------------------------
def merge_pooled(df, schemes, methods=METHODS, seeds=SEEDS, verbose=True):
    """Add `cost_/efc_/fixed_/total_<method>_<scheme>` to the study's wide frame.

    `hs.merge_learned_controllers` does the joining -- the pooled results are on
    disk in the very layout it reads (`<tariff>/<scheme>__<method>/<id>.json`),
    so the bill, the wear rate and the SI standing charge are put on the study's
    axis by the same code that puts the local learners there. A scheme trained
    under several seeds is joined once per seed and AVERAGED per household: one
    global network drives all 30 units, so a single seed's luck would otherwise
    be read as a property of the scheme. The cost is linear in each line and
    every run lives the same 12 y, so the mean of the bills is the bill of the
    mean. `out.attrs["pooled_seeds"]` records how many seeds each column holds.
    """
    out = df
    n_seeds = {}
    for scheme in schemes:
        for method in methods:
            got = []
            for seed in seeds:
                m = method + seed_tag(seed).replace("_", "__")
                out = hs.merge_learned_controllers(
                    out, controllers=[(m, scheme)], screen_dir=OUT, verbose=False)
                if f"cost_{m}_{scheme}" in out.columns:
                    got.append(f"{m}_{scheme}")
            if not got:
                continue
            name = f"{method}_{scheme}"
            for pre in ("cost", "efc", "fixed", "total"):
                out[f"{pre}_{name}"] = out[[f"{pre}_{c}" for c in got]].mean(axis=1)
            out = out.drop(columns=[f"{pre}_{c}" for c in got if c != name
                                    for pre in ("cost", "efc", "fixed", "total")])
            n_seeds[name] = len(got)
    out.attrs["pooled_seeds"] = n_seeds
    if verbose:
        print("pooled controllers joined (seeds averaged):",
              ", ".join(f"{k} x{v}" for k, v in n_seeds.items()))
    return out


# ---------------------------------------------------------------------------
# Phase D: report -- every scheme against the local learner, paired
# ---------------------------------------------------------------------------
def load_results():
    import pandas as pd
    rows = []
    for tariff in ("AU", "SI"):
        base = os.path.join(OUT, tariff)
        if not os.path.isdir(base):
            continue
        for key in sorted(os.listdir(base)):
            d = os.path.join(base, key)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if f.endswith(".json"):
                    with open(os.path.join(d, f), encoding="utf-8") as fh:
                        rows.append(json.load(fh))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].fillna(0).astype(int) if "seed" in df else 0
    df["saving"] = df["ref_cost_no_battery"] - df["cost_eur_closed"]
    df["d_total_vs_local"] = df["cost_eur_total"] - df["local_cost_eur_total"]
    return df


def report(df=None, axis: str = "cost_eur_total"):
    """Per (tariff, method, scheme): mean saving and the paired difference to
    the LOCAL learner on `axis` (default: the invoice incl. the contract-
    dependent standing charge, which on SI moves with the agent's own peaks).
    Negative = the pooled model is cheaper. Wilcoxon signed-rank, the study's
    own `hs.paired_comparison`, with Holm across the whole table."""
    import pandas as pd
    df = load_results() if df is None else df
    if df is None:
        print("no global results on disk")
        return None
    rows = []
    for (t, m, s, sd), g in df.groupby(["tariff", "method", "scheme", "seed"]):
        g = g.dropna(subset=[axis, f"local_{axis}"])
        pc = hs.paired_comparison(g, axis, f"local_{axis}")
        rows.append({"tariff": t, "method": m, "scheme": s, "seed": sd, "n": len(g),
                     "saving_closed": g["saving"].mean(),
                     "local_saving_closed": (g["ref_cost_no_battery"]
                                             - g["local_cost_eur_closed"]).mean(),
                     "mean_diff": (g[axis] - g[f"local_{axis}"]).mean(),
                     "median_diff": pc["median_diff"],
                     "pooled_cheaper": pc["n_b_greater"],
                     "local_cheaper": pc["n_a_greater"],
                     "p": pc.get("p_value", float("nan")),
                     "efc": g["efc"].mean(), "local_efc": g["local_efc"].mean(),
                     "converged": g["train_converged"].mean()})
    out = pd.DataFrame(rows)
    out["p_holm"] = rb.holm(out["p"].tolist())
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"\nPOOLED vs LOCAL on {axis} (EUR/yr, negative diff = pooled cheaper)")
        print(out.round(3).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["cache", "train", "eval", "report", "all"],
                    default="all")
    ap.add_argument("--tariffs", nargs="*", default=["AU", "SI"])
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES),
                    help=f"any of {SCHEMES + STAGE2_SCHEMES + CTRL_SCHEMES}")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-cap", type=int, default=None,
                    help="smoke tests: at most this many households per cluster")
    ap.add_argument("--clusters", nargs="*", type=int, default=None,
                    help="restrict to these clusters (smoke tests)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    pop = population()
    if args.pool_cap:
        pop = pop[pop["rank_in_cluster"] <= args.pool_cap]
    if args.clusters is not None:
        pop = pop[pop["cluster"].isin(args.clusters)]
    if args.phase in ("cache", "all"):
        run_cache(args.tariffs, sorted(int(i) for i in pop.index), args.jobs)
    if args.phase in ("train", "all"):
        run_train(args.tariffs, args.schemes, args.methods, args.jobs, args.seed,
                  args.pool_cap, args.clusters, args.verbose)
    if args.phase in ("eval", "all"):
        units = [u for u in study_units() if u in pop.index]
        run_eval(args.tariffs, args.schemes, args.methods, args.jobs, units,
                 args.seed)
    if args.phase in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
