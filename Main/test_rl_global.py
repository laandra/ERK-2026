"""Invariant checks for the global / per-type learners (rl_global.py,
run_rl_global.py).

Plain script like `test_rl_control.py`: one PASS/FAIL per check, non-zero exit
on failure. Every check is a way a pooled learner could differ from the local
one for a reason OTHER than the data it was given -- which is the only
difference the comparison is allowed to measure.

    cd Main && python3 test_rl_global.py        (~1 min, 120-day slice)
"""

import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.simplefilter("ignore")

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import rl_global as rg                                           # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402
import run_rl_global as rgg                                      # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Fixture: 120 days of two households, SI settlement (the stateful one)
# --------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Input data", "Ausgrid")
N_DAYS, H = 120, 48
N_STEPS = N_DAYS * H
rb._calendar("SI")


def _bundle(ident):
    frames = hs.load_study_frames(os.path.join(DATA, f"Ausgrid {ident}.csv"))
    df_kwh = frames["df_all_kwh"].iloc[:N_STEPS + H]
    env = hs.align_envelope(
        hs.build_study_env(df_kwh, battery_cap=10.0, soc_min_pct=0.10,
                           soc_max_pct=0.80, p_max=1.5, eff=0.95, delta_t=0.5,
                           H=48), 1.5, 0.95, 0.5)
    df_kw = df_kwh.copy()
    df_kw[["Energy_Generation", "Energy_Consumption"]] /= 0.5
    rates = hs.build_rate_vectors("SI", env, df_kw.index, df_kw["SMP"].values, 30)
    settle = hs.build_settlement("SI", env, rates, 0.5)
    sig = rbc.build_signals(env, n_steps=N_STEPS, rates=rates)
    fc = rl.median14_forecast(df_kwh["Energy_Consumption"].values, H)[:N_STEPS]
    pv = rl.median14_forecast(df_kwh["Energy_Generation"].values, H)[:N_STEPS]
    return {"env": env, "settle": settle, "sig": sig, "fc": (fc, pv)}


B = {1: _bundle(1), 138: _bundle(138)}
spec = rl.FeatureSpec(horizon=48, peak_features=True)
TRAIN = [(14, 50), (57, 120)]
VAL = [(50, 57)]

# --------------------------------------------------------------------------
# 1. The typed observation is the local observation plus the type, and the
#    cached-base path builds exactly what the policy will see at test time
# --------------------------------------------------------------------------
base = {i: rl.FeatureBuilder(spec).build_static(b["sig"], *b["fc"])
        for i, b in B.items()}
fb = rg.TypedFeatureBuilder(spec, n_types=3)
fb.fit_norm_pooled(base[i] for i in B)
cat = np.concatenate([base[1], base[138]]).astype(np.float64)  # f32 moments drift ~1e-4
check("pooled scaler equals fit_norm on the concatenation",
      np.allclose(fb.norm_mean[:cat.shape[1]], cat.mean(axis=0), atol=1e-6)
      and np.allclose(fb.norm_std[:cat.shape[1]], cat.std(axis=0), rtol=1e-5))
check("type columns are exempt from normalisation",
      np.all(fb.norm_mean[-3:] == 0) and np.all(fb.norm_std[-3:] == 1))
fb2 = fb.for_type(2)
via_policy = fb2.normalize(fb2.build_static(B[1]["sig"], *B[1]["fc"]))
via_cache = fb2.normalize_base(base[1])
check("normalize(build_static) == normalize_base(cached base)  [typed]",
      np.allclose(via_policy, via_cache, atol=1e-5))
check("the bound type is a one-hot in the last columns",
      np.all(via_cache[:, -3:] == np.array([0, 0, 1], dtype=np.float32)))
rows = fb2.normalized_rows(base[1])
probe = [0, 17, 4000, N_STEPS - 1]
check("TypedRows view reads row-for-row like the materialised matrix",
      rows.shape == via_cache.shape
      and all(np.array_equal(rows[i], via_cache[i]) for i in probe))
fb_plain = rg.TypedFeatureBuilder(spec, n_types=0)
fb_plain.fit_norm(base[1])
local_fb = rl.FeatureBuilder(spec)
local_fb.fit_norm(base[1])
check("an untyped builder is the local builder exactly",
      np.array_equal(fb_plain.normalize_base(base[1]),
                     local_fb.normalize(base[1]))
      and fb_plain.dim == local_fb.dim)
try:
    fb.build_static(B[1]["sig"], *B[1]["fc"])
    check("an unbound typed builder refuses to build", False)
except RuntimeError:
    check("an unbound typed builder refuses to build", True)

# --------------------------------------------------------------------------
# 2. A one-household pool IS the local learner, bit for bit
# --------------------------------------------------------------------------
cfg = rl.TrainConfig(total_steps=4_000, learn_start=500, eval_every=1_000,
                     target_sync=500, buffer=5_000, seed=3, n_step=4)
b1 = B[1]
lf = rl.FeatureBuilder(spec)
lf.fit_norm(base[1])
sn = lf.normalize(base[1])
net_l, hist_l = rl.train_dqn(b1["sig"], b1["settle"], b1["env"], lf, sn, cfg,
                             respect_peak=True, train_days=TRAIN, val_days=VAL,
                             soc_target=4.0, verbose=False)
m1 = rg.Member("1", b1["sig"], b1["settle"], b1["env"], sn)
net_p, hist_p = rg.train_dqn_pool([m1], [m1], lf, cfg, True, TRAIN, VAL,
                                  soc_target=4.0, verbose=False)
same_w = all(torch.equal(a, b) for a, b in
             zip(net_l.state_dict().values(), net_p.state_dict().values()))
check("one-member pool reproduces rl.train_dqn (weights)", same_w)
check("one-member pool reproduces rl.train_dqn (validation trace)",
      hist_l["val_cost_closed"] == hist_p["val_cost_closed"],
      f"{hist_l['val_cost_closed'][-1]:.4f} vs {hist_p['val_cost_closed'][-1]:.4f}")

# Warm start + BC guide through the pool path too.
net_bl, _ = rl.train_dqn(b1["sig"], b1["settle"], b1["env"], lf, sn, cfg,
                         respect_peak=True, train_days=TRAIN, val_days=VAL,
                         init_net=net_l, bc_net=net_l, soc_target=4.0,
                         verbose=False)
net_bp, _ = rg.train_dqn_pool([m1], [m1], lf, cfg, True, TRAIN, VAL,
                              init_net=net_l, bc_net=net_l, soc_target=4.0,
                              verbose=False)
check("one-member pool reproduces the warm-started, BC-guided fine-tune",
      all(torch.equal(a, b) for a, b in
          zip(net_bl.state_dict().values(), net_bp.state_dict().values())))

# A two-household typed pool runs, draws both households, never trains on
# the validation week.
fbt = rg.TypedFeatureBuilder(spec, n_types=2)
fbt.fit_norm_pooled(base[i] for i in B)
mem = [rg.Member(str(i), B[i]["sig"], B[i]["settle"], B[i]["env"],
                 fbt.for_type(k).normalized_rows(base[i]))
       for k, i in enumerate(B)]
_, hist2 = rg.train_dqn_pool(mem, mem, fbt, cfg, True, TRAIN, VAL,
                             soc_target=4.0, verbose=False)
eps_days = [d + j for d in hist2["episode_start_day"] for j in range(7)]
check("a pooled run draws every member",
      set(hist2["episode_member"]) == {0, 1})
check("no pooled episode touches a validation day",
      not any(50 <= d < 57 for d in eps_days))
try:
    rg.train_dqn_pool(mem, mem, fbt, cfg, True, [(14, 54), (57, 120)], VAL,
                      soc_target=4.0, verbose=False)
    check("an overlapping split is refused", False)
except ValueError:
    check("an overlapping split is refused", True)

# --------------------------------------------------------------------------
# 3. Pooled cloning learns an expressible teacher
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.normal(size=(40_000, 8)).astype(np.float32)
y = (X[:, 0] > 0).astype(np.int64) + 2 * (X[:, 1] > 0.5).astype(np.int64)
net_c, hc = rg.train_bc_pool(X[:30_000], y[:30_000], X[30_000:], y[30_000:],
                             rl.TrainConfig(seed=0),
                             rg.BCConfigPool(batch=512, max_epochs=30, patience=3),
                             verbose=False)
check("pooled BC learns an expressible teacher",
      hc["final_val_agreement"] > 0.95, f"{hc['final_val_agreement']:.3f}")

# Save / load round trip, typed.
import tempfile
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.pt")
    net_t = rl.QNet(fbt.dim, 32)
    rg.save_model(p, net_t, fbt, rl.TrainConfig(hidden=32), {"x": 1})
    net_r, fb_r, _ = rg.load_model(p)
    o = torch.from_numpy(np.asarray(mem[1].static_norm[100])[None, :]
                         .repeat(1, 0))
    o = torch.cat([o, torch.zeros((1, fbt.n_dynamic))], dim=1)
    check("typed save/load round trip (weights, scaler, type width)",
          torch.equal(net_t(o), net_r(o)) and fb_r.n_types == 2
          and np.array_equal(fb_r.norm_std, fbt.norm_std))

# --------------------------------------------------------------------------
# 4. Pools come from the study's clustering, and the test year is untouched
# --------------------------------------------------------------------------
pop = rgg.population()
units = rgg.study_units()
types = rgg.groups_for("type", pop)
check("300 households, 30 types, partitioned",
      len(pop) == 300 and len(types) == 30
      and sum(len(g["members"]) for g in types.values()) == 300
      and len({i for g in types.values() for i in g["members"]}) == 300)
check("every study unit is a member of its own type pool",
      all(u in types[rgg.group_of_unit("type", u, pop)]["members"] for u in units))
check("every study unit is its type's rank-1 household",
      all(pop.loc[u, "rank_in_cluster"] == 1 for u in units))
glob = rgg.groups_for("global_typed", pop)["all"]
check("global pools hold every household; typed carries the cluster",
      len(glob["members"]) == 300 and glob["n_types"] == 30
      and all(glob["type_of"][u] == pop.loc[u, "cluster"] for u in units))
capped = rgg.groups_for("global", pop, pool_cap=2)["all"]
check("a capped pool changes the digest (members are in it)",
      rgg.model_digest("SI", "global", "all", "bc",
                       capped, rl.TrainConfig()) !=
      rgg.model_digest("SI", "global", "all", "bc",
                       rgg.groups_for("global", pop)["all"], rl.TrainConfig()))
check("the teacher and the cached features stop at the training years",
      rb.TEACH_SPAN[1] <= rb.N_TRAIN
      and max(b for _, b in rb.TRAIN_BLOCKS + rb.VAL_BLOCKS) <= rb.N_TRAIN)
mask = rgg._val_mask(rb.N_TRAIN * H, 0)
check("BC holdout mask is exactly the validation weeks",
      mask.sum() == sum(b - a for a, b in rb.VAL_BLOCKS) * H
      and not mask[rgg._train_rows()].any())
s_g, e_g = rgg.dqn_budget(300, "global")
s_1, _ = rgg.dqn_budget(1, "type")
check("a one-household type gets the local budget; global gets more",
      s_1 == rgg.LOCAL_STEPS and s_g > s_1 and s_g // e_g == rgg.N_EVALS)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL PASS")
