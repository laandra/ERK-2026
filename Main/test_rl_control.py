"""Invariant checks for the learned controllers (rl_control.py).

Like `test_hems_study.py`: plain script, one PASS/FAIL per check, non-zero
exit on failure. Every check here is a way a learned controller can be
quietly wrong in this study -- an acausal feature, an infeasible action, an
agent scored through different accounting than the rules, a training run that
is not reproducible.

    cd Main && python3 test_rl_control.py        (~1 min, 120-day slice)
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.simplefilter("ignore")

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Fixture: 120 days of one household, both tariffs' settlements
# --------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Input data", "Ausgrid")
N_DAYS, H = 120, 48
N_STEPS = N_DAYS * H

hs._si_cas.nastavi_koledar(drzava="AU", podrocje="NSW",
                           visja_sezona_meseci={5, 6, 7, 8}, casovni_pas="naive")
hs.TariffCalculator.HOLIDAY_COUNTRY = "AU"
hs.TariffCalculator.HOLIDAY_SUBDIV = "NSW"
hs.TariffCalculator.LOCAL_TZ = None

frames = hs.load_study_frames(os.path.join(DATA, "Ausgrid 1.csv"))
df_kwh = frames["df_all_kwh"].iloc[:N_STEPS + H]
env = hs.align_envelope(
    hs.build_study_env(df_kwh, battery_cap=10.0, soc_min_pct=0.10,
                       soc_max_pct=0.80, p_max=1.5, eff=0.95, delta_t=0.5, H=48),
    1.5, 0.95, 0.5)
df_kw = df_kwh.copy()
df_kw[["Energy_Generation", "Energy_Consumption"]] /= 0.5

bundles = {}
for tariff in ("SI", "AU"):
    rates = hs.build_rate_vectors(tariff, env, df_kw.index, df_kw["SMP"].values, 30)
    settle = hs.build_settlement(tariff, env, rates, 0.5)
    sig = rbc.build_signals(env, n_steps=N_STEPS, rates=rates)
    bundles[tariff] = (rates, settle, sig)

sig_si = bundles["SI"][2]
CAP = sig_si.capacity_kwh


# --------------------------------------------------------------------------
# 1. Causality of the forecast and the lookahead features
# --------------------------------------------------------------------------
vals = np.asarray(df_kwh["Energy_Consumption"].values[:N_STEPS], dtype=float)
fc_a = rl.median14_forecast(vals, H)
poisoned = vals.copy()
poisoned[60 * H:] += 1000.0
fc_b = rl.median14_forecast(poisoned, H)
check("median14 is causal (future poison never reaches the past)",
      np.allclose(fc_a[:60 * H], fc_b[:60 * H]))

t_probe, horizon = 40 * H, 48
arr = np.asarray(sig_si.import_rate, dtype=float).copy()
bins_full = rl._lookahead_bins(arr, horizon, N_STEPS, mean=True)
arr_poison = arr.copy()
arr_poison[t_probe + 1 + horizon:] = 1e9
bins_poison = rl._lookahead_bins(arr_poison, horizon, N_STEPS, mean=True)
check("lookahead bins read at most `horizon` steps ahead",
      np.allclose(bins_full[t_probe], bins_poison[t_probe]))

spec_truth = rl.FeatureSpec(horizon=48, load_channel="truth",
                            pv_channel="truth", peak_features=True)
fb_t = rl.FeatureBuilder(spec_truth)
static_t = fb_t.build_static(sig_si)
check("truth-channel spec is flagged non-causal",
      not spec_truth.causal and rl.FeatureSpec(horizon=48).causal)


# --------------------------------------------------------------------------
# 2. Actions are feasible, and grid charging respects the peak room
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)
ok_bounds, ok_room = True, True
for _ in range(500):
    idx = int(rng.integers(0, N_STEPS))
    soc = float(rng.uniform(0, CAP))
    lo = -min(env.max_discharge_kwh, soc * sig_si.eta_dis)
    hi = min(env.max_charge_kwh / sig_si.eta_ch, (CAP - soc) / sig_si.eta_ch)
    peak_state = {b: float(rng.uniform(0, 3)) for b in (1, 2, 3, 4, 5)}
    for a in range(rl.N_ACTIONS):
        p = rl.action_setpoint(a, sig_si, idx, lo, hi, peak_state, True)
        if not (lo - 1e-9 <= p <= hi + 1e-9):
            ok_bounds = False
    room = rbc._grid_charge_room(sig_si, idx, hi, peak_state, True)
    p_any = rl.action_setpoint(rl.A_CHARGE_ANY, sig_si, idx, lo, hi, peak_state, True)
    if p_any > room + 1e-9:
        ok_room = False
check("every semantic action lands inside [lo, hi]", ok_bounds)
check("charge_any never exceeds the peak-aware grid room", ok_room)


# --------------------------------------------------------------------------
# 3. A learned policy is scored exactly like a rule
# --------------------------------------------------------------------------
spec = rl.FeatureSpec(horizon=48, peak_features=True)
fb = rl.FeatureBuilder(spec)
fc_con = rl.median14_forecast(df_kwh["Energy_Consumption"].values[:N_STEPS], H)
fc_gen = rl.median14_forecast(df_kwh["Energy_Generation"].values[:N_STEPS], H)
static = fb.build_static(sig_si, load_fc=fc_con, pv_fc=fc_gen)
fb.fit_norm(static)

import torch                                                     # noqa: E402
torch.manual_seed(0)
net = rl.QNet(fb.dim)

policy = rl.LearnedPolicy(net, fb, respect_peak=True,
                          load_fc=fc_con, pv_fc=fc_gen)
_, settle_si, _ = bundles["SI"]
out = rbc.run_policy(env, policy, n_steps=N_STEPS, settle=settle_si,
                     soc_init_kwh=4.0, rates=bundles["SI"][0])
ref = rbc.run_policy(env, rbc.make_policy("self_consumption"),
                     n_steps=N_STEPS, settle=settle_si, soc_init_kwh=4.0,
                     rates=bundles["SI"][0])
# Machine epsilon, not a real drift: the runner's own rules report ~5e-17
# from the same clamp arithmetic, so the learned policy is held to the same
# standard rather than to exact zero.
check("run_policy executes a LearnedPolicy with zero SOC drift",
      out["SOC_Drift_kWh"] <= 1e-12, f"drift {out['SOC_Drift_kWh']:.1e}")
check("learned-policy result carries the same keys as a rule's",
      set(out) == set(ref))

out2 = rbc.run_policy(env, policy, n_steps=N_STEPS, settle=settle_si,
                      soc_init_kwh=4.0, rates=bundles["SI"][0])
check("evaluation is deterministic (same net, same cost, twice)",
      abs(out["Cost_EUR_Closed"] - out2["Cost_EUR_Closed"]) < 1e-9,
      f"{out['Cost_EUR_Closed']:.4f}")


# --------------------------------------------------------------------------
# 4. Behaviour cloning learns a teacher it can express
# --------------------------------------------------------------------------
# The teacher is the fixed-schedule rule -- fully expressible by the semantic
# actions and readable from the calendar features alone, so a cloner that
# cannot reach high agreement on it is broken, not unlucky.
teacher = rbc.make_policy("fixed_schedule", charge_hours=(11.0, 15.0),
                          discharge_hours=(15.0, 21.0), respect_peak=True)
t_out = rbc.run_policy(env, teacher, n_steps=N_STEPS, settle=settle_si,
                       soc_init_kwh=4.0, rates=bundles["SI"][0],
                       keep_traces=True)
cfg = rl.TrainConfig(seed=0, hidden=64)
static_norm = fb.normalize(static)
net_bc, hist_bc = rl.train_bc(sig_si, settle_si, env, fb, static_norm,
                              t_out["_setpoints"], start=0, stop=N_STEPS,
                              soc_init=4.0, respect_peak=True, cfg=cfg,
                              verbose=False)
check("BC reaches >= 0.75 held-out agreement on an expressible teacher",
      hist_bc["final_val_agreement"] >= 0.75,
      f"agreement {hist_bc['final_val_agreement']:.3f}")


# --------------------------------------------------------------------------
# 5. DQN training: runs, tracks convergence, best weights are the kept ones
# --------------------------------------------------------------------------
cfg_q = rl.TrainConfig(total_steps=8_000, eval_every=2_000, learn_start=500,
                       episode_days=3, hidden=64, patience=100, seed=0,
                       wear_eur_per_efc=0.4167)
net_q, hist_q = rl.train_dqn(sig_si, settle_si, env, fb, static_norm, cfg_q,
                             respect_peak=True, train_days=(0, 90),
                             val_days=(90, 120), soc_target=3.5, verbose=False)
check("DQN history carries the convergence trace",
      len(hist_q["val_step"]) >= 3 and "converged" in hist_q)
re_val = rl.greedy_rollout(net_q, fb, static_norm, sig_si, settle_si, env,
                           90 * H, 120 * H, 3.5, True)
check("kept weights reproduce the recorded best validation cost",
      abs(re_val["cost_eur_closed"] - hist_q["best_val_cost_closed"]) < 1e-6,
      f"{re_val['cost_eur_closed']:.4f} vs {hist_q['best_val_cost_closed']:.4f}")

# The validation score early stopping reads must be the one the results are
# reported on: bill plus the lifetime wear. A constant charge shifts every
# evaluation alike, so the kept weights cannot move -- only the score can.
_, hist_vw = rl.train_dqn(sig_si, settle_si, env, fb, static_norm, cfg_q,
                          respect_peak=True, train_days=(0, 90),
                          val_days=(90, 120), soc_target=3.5, verbose=False,
                          val_wear=lambda efc, days: 100.0)
check("val_wear is added to every validation score early stopping reads",
      abs(hist_vw["best_val_cost_closed"] - hist_q["best_val_cost_closed"] - 100.0) < 1e-6,
      f"{hist_vw['best_val_cost_closed']:.2f} vs {hist_q['best_val_cost_closed']:.2f} + 100")

net_q2, hist_q2 = rl.train_dqn(sig_si, settle_si, env, fb, static_norm, cfg_q,
                               respect_peak=True, train_days=(0, 90),
                               val_days=(90, 120), soc_target=3.5, verbose=False)
check("training is reproducible under one seed",
      abs(hist_q["best_val_cost_closed"] - hist_q2["best_val_cost_closed"]) < 1e-6)


# --------------------------------------------------------------------------
# 5b. Warm start survives fine-tuning, and n-step returns are what they claim
# --------------------------------------------------------------------------
# The regression this guards: a clone handed to train_dqn used to be destroyed
# inside the first evaluations, because its weights are cross-entropy logits
# and the first TD updates rescale the net to Q-magnitudes. A warm-started run
# must therefore START near the clone's own greedy cost, not near a cold net's.
bc_cost = rl.greedy_rollout(net_bc, fb, static_norm, sig_si, settle_si, env,
                            90 * H, 120 * H, 3.5, True)["cost_eur_closed"]
cfg_w = rl.TrainConfig(total_steps=8_000, eval_every=2_000, learn_start=500,
                       episode_days=3, hidden=64, patience=100, seed=0,
                       wear_eur_per_efc=0.4167)
net_w, hist_w = rl.train_dqn(sig_si, settle_si, env, fb, static_norm, cfg_w,
                             respect_peak=True, train_days=(0, 90),
                             val_days=(90, 120), init_net=net_bc, bc_net=net_bc,
                             soc_target=3.5, verbose=False)
first = hist_w["val_cost_closed"][0]
cold_first = hist_q["val_cost_closed"][0]
check("a warm-started fine-tune starts from the clone, not from scratch",
      abs(first - bc_cost) < abs(first - cold_first),
      f"clone {bc_cost:.1f}, warm start {first:.1f}, cold start {cold_first:.1f}")

# The BC prior must be a COPY: train_dqn is handed the same object twice, and
# if it trains the object it also consults, the guide drifts with the policy.
before = net_bc.adv.weight.detach().clone()
rl.train_dqn(sig_si, settle_si, env, fb, static_norm,
             rl.TrainConfig(total_steps=3_000, eval_every=2_000, learn_start=200,
                            episode_days=2, hidden=64, patience=100, seed=0),
             respect_peak=True, train_days=(0, 90), val_days=(90, 120),
             init_net=net_bc, bc_net=net_bc, soc_target=3.5, verbose=False)
check("training never mutates the caller's network in place",
      torch.equal(before, net_bc.adv.weight.detach()))

# n-step returns: sum_{i<k} gamma^i r_i, bootstrapped at gamma^k, TRUNCATED at
# the first exploratory continuation. Replicates train_dqn's `_flush` so the
# arithmetic is pinned independently of a training run finishing.
def _nstep(rewards, explored, gamma=0.9, n=3):
    buf = rl._Replay(16, 3)
    pending = []
    for t, (r, e) in enumerate(zip(rewards, explored)):
        obs = np.full(3, float(t), dtype=np.float32)
        nxt = np.full(3, float(t + 1), dtype=np.float32)
        pending.append((obs, t % rl.N_ACTIONS, r, e, nxt))
        if len(pending) >= n:
            ret, disc, k = 0.0, 1.0, 0
            for j, (_, _, rr, ex, _) in enumerate(pending):
                if j > 0 and ex:
                    break
                ret += disc * rr
                disc *= gamma
                k = j + 1
            buf.push(pending[0][0], pending[0][1], ret, pending[k - 1][4],
                     0.0, disc)
            pending.pop(0)
    return buf

b = _nstep([1.0, 2.0, 4.0, 8.0], [False] * 4)
want = 1.0 + 0.9 * 2.0 + 0.81 * 4.0
check("n-step return sums gamma-discounted rewards over its own window",
      abs(b.rew[0] - want) < 1e-5 and abs(b.disc[0] - 0.9 ** 3) < 1e-6,
      f"{b.rew[0]:.3f} vs {want:.3f}, disc {b.disc[0]:.4f}")

# Second step exploratory -> the window truncates to one transition, and the
# bootstrap must come from the state that FIRST step landed in.
b = _nstep([1.0, 2.0, 4.0, 8.0], [False, True, False, False])
check("an exploratory continuation truncates the n-step window",
      abs(b.rew[0] - 1.0) < 1e-6 and abs(b.disc[0] - 0.9) < 1e-6
      and abs(float(b.nxt[0][0]) - 1.0) < 1e-6,
      f"return {b.rew[0]:.3f}, disc {b.disc[0]:.3f}, bootstrap s{b.nxt[0][0]:.0f}")

# The FIRST action being exploratory is fine -- it is the action being scored,
# not a continuation -- so the window is not truncated by it.
b = _nstep([1.0, 2.0, 4.0, 8.0], [True, False, False, False])
check("an exploratory first action does not truncate its own window",
      abs(b.rew[0] - want) < 1e-5, f"{b.rew[0]:.3f} vs {want:.3f}")


# --------------------------------------------------------------------------
# 5c. A mid-window start inherits the ratchet peak it is really standing in
# --------------------------------------------------------------------------
# Zero-seeding a mid-month episode tells the agent every peak ahead of it is
# new, when in the scored year most are sunk. The environment already answers
# this question for its own episodes; these check we ask it and agree.
opens = int(np.flatnonzero(np.diff(env.reset_window_ids[:N_STEPS]) != 0)[0]) + 1
mid = opens + 10 * H
check("a window-opening index seeds a zero peak state",
      all(v == 0.0 for v in rl.seed_peak_state(env, opens).values()),
      f"idx {opens}")
seeded = rl.seed_peak_state(env, mid)
check("a mid-window index inherits a nonzero peak state",
      any(v > 0.0 for v in seeded.values()),
      f"idx {mid}: " + ", ".join(f"B{b}={v:.2f}" for b, v in seeded.items() if v))
check("the seed is the environment's own, not a second opinion",
      seeded == {int(b): float(v)
                 for b, v in env.compute_seed_peak_kw(mid).items()})


# --------------------------------------------------------------------------
# 5d. Train / validation / test are disjoint, and training respects it
# --------------------------------------------------------------------------
# The first split trained on year 2 only and its knobs were tuned on scored-
# year numbers. These pin the replacement: three disjoint parts, validation in
# every season, no episode or clone fit reaching a held-out day.
import run_rl_benchmark as rb                                    # noqa: E402

tr = set(rl.block_days(rb.TRAIN_BLOCKS).tolist())
va = set(rl.block_days(rb.VAL_BLOCKS).tolist())
check("train and validation days are disjoint and cover the training period",
      not (tr & va) and tr | va == set(range(rb.N_TRAIN)),
      f"{len(tr)} train / {len(va)} validation days")
check("neither train nor validation reaches the test year",
      max(tr | va) < rb.N_TRAIN and rb.TEACH_SPAN[1] <= rb.N_TRAIN)
months = {(int(d) // 30) % 12 for d in va}
check("validation weeks fall in every season of the year",
      len({m // 3 for m in months}) == 4, f"{len(months)} month bins")

try:
    rl.train_dqn(sig_si, settle_si, env, fb, static_norm,
                 rl.TrainConfig(total_steps=100, eval_every=100, learn_start=50,
                                episode_days=3, hidden=64, seed=0),
                 respect_peak=True, train_days=[(0, 60)],
                 val_days=[(50, 55)], soc_target=3.5, verbose=False)
    overlap_raised = False
except ValueError:
    overlap_raised = True
check("a training range that contains a validation week is refused",
      overlap_raised)

_, hist_h = rl.train_bc(sig_si, settle_si, env, fb, static_norm,
                        t_out["_setpoints"], start=0, stop=N_STEPS,
                        soc_init=4.0, respect_peak=True, cfg=cfg,
                        verbose=False, holdout_days=[(20, 27), (90, 97)])
n_held = int(round(sum(hist_h["label_counts"])))
check("BC fits only outside its held-out weeks",
      n_held == (N_DAYS - 14) * H, f"{n_held} fitted steps")


# --------------------------------------------------------------------------
# 6. Save / load round trip
# --------------------------------------------------------------------------
import tempfile                                                  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "m.pt")
    rl.save_model(path, net_q, fb, spec, cfg_q, hist_q)
    net_l, fb_l, spec_l, _ = rl.load_model(path)
    pol_l = rl.LearnedPolicy(net_l, fb_l, True, load_fc=fc_con, pv_fc=fc_gen)
    out_l = rbc.run_policy(env, pol_l, n_steps=N_STEPS, settle=settle_si,
                           soc_init_kwh=4.0, rates=bundles["SI"][0])
    pol_o = rl.LearnedPolicy(net_q, fb, True, load_fc=fc_con, pv_fc=fc_gen)
    out_o = rbc.run_policy(env, pol_o, n_steps=N_STEPS, settle=settle_si,
                           soc_init_kwh=4.0, rates=bundles["SI"][0])
    check("a reloaded model reproduces its own evaluation to the cent",
          abs(out_l["Cost_EUR_Closed"] - out_o["Cost_EUR_Closed"]) < 1e-9)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
