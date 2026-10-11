"""Invariant checks for the reward functions (rl_reward.py).

Like `test_rl_control.py`: plain script, one PASS/FAIL per check, non-zero
exit on failure. Every check is a way a reward can be quietly wrong in this
study -- a "new" reward that silently changes the old one's results, a
contract charge that is not the standing charge the invoice bills, a feature
that reads the month's future, a validation measure that cannot see the thing
the reward was changed to pay for.

    cd Main && python3 test_rl_reward.py        (~1 min, 180-day slice)
"""

import json
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
import rl_reward as rr                                           # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402
import run_rl_contract as rc                                     # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Fixture: 180 days of one household (July-December: two high-season months,
# then low season), on AU and on each SI contract type
# --------------------------------------------------------------------------
N_DAYS, H = 180, 48
N_STEPS = N_DAYS * H
rb._calendar("SI")
frames = hs.load_study_frames(os.path.join(rb.DATA, "Ausgrid 1.csv"))
df_kwh = frames["df_all_kwh"].iloc[:N_STEPS + H]
df_kw = df_kwh.copy()
df_kw[["Energy_Generation", "Energy_Consumption"]] /= 0.5

au = rb.build_bundle("AU", df_kwh, df_kw, N_STEPS)
monthly = rb.build_bundle("SI", df_kwh, df_kw, N_STEPS)
annual = rb.build_bundle("SI", df_kwh, df_kw, N_STEPS,
                         contract=rr.contract_env_kwargs("annual"))
signed = rr.operator_contract(monthly["sig"], 0, N_STEPS)
fixed = rb.build_bundle("SI", df_kwh, df_kw, N_STEPS,
                        contract=rr.contract_env_kwargs("fixed", signed))
by_price = {"monthly": monthly}
for _c in ("active", "single"):
    by_price[_c] = rb.build_bundle("SI", df_kwh, df_kw, N_STEPS,
                                   contract=rr.contract_env_kwargs(_c))
sig = monthly["sig"]
CAP = sig.capacity_kwh
naive = rr.naive_import_kw(sig)
fc = (rl.median14_forecast(df_kwh["Energy_Consumption"].values[:N_STEPS], H),
      rl.median14_forecast(df_kwh["Energy_Generation"].values[:N_STEPS], H))
TRAIN_DAYS, VAL_DAYS = [(14, 84), (91, 180)], [(84, 91)]


def features(spec, bundle):
    fb = rl.FeatureBuilder(spec)
    static = fb.build_static(bundle["sig"], load_fc=fc[0], pv_fc=fc[1])
    fb.fit_norm(static)
    return fb, fb.normalize(static)


def same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                 b.state_dict().values()))


# --------------------------------------------------------------------------
# 1. One reward per bill
# --------------------------------------------------------------------------
check("AU is paid for energy, SI fixed for energy + excess power, the two "
      "moving contracts for the contract charge",
      rr.reward_class("AU") is rr.AUEnergyReward
      and rr.reward_class("SI", "fixed") is rr.SIExcessPowerReward
      and rr.reward_class("SI", "monthly") is rr.SIMonthlyContractReward
      and rr.reward_class("SI", "annual") is rr.SIAnnualContractReward)
check("every (tariff, contract) has its own reward function",
      len(set(rr.REWARDS.values())) == len(rr.REWARDS) == 6
      and len({c.name for c in rr.REWARDS.values()}) == 6)
check("the study's contract is in both families: its price list among the "
      "price lists, its agreed-power rule among the rules",
      rr.family("monthly") == (rr.CONTRACT_TYPES, rr.PRICE_TYPES)
      and rr.family("active") == (rr.PRICE_TYPES,)
      and rr.family("fixed") == (rr.CONTRACT_TYPES,)
      and set(rr.ALL_TYPES) == set(rr.CONTRACT_TYPES) | set(rr.PRICE_TYPES)
      and all(t in rc.SPECIALIST and t in rc.AGREED_KW_SCALE
              and t in rc.PRICE_SCALE and t in rr.CONTRACT_WINDOW
              for t in rr.ALL_TYPES))
check("the study's own contract is the reference contract type",
      rr.REFERENCE_CONTRACT == "monthly"
      and monthly["env"].agreed_power_n_months_window == 1
      and annual["env"].agreed_power_n_months_window == 12)


# --------------------------------------------------------------------------
# 2. The legacy rewards ARE the loop that trained every stored result
# --------------------------------------------------------------------------
cfg = rl.TrainConfig(total_steps=4000, eval_every=2000, learn_start=500,
                     buffer=5000, seed=3)
for name, bundle, cls, peak in (("AU", au, rr.AUEnergyReward, False),
                                ("SI", monthly, rr.SIExcessPowerReward, True)):
    spec = rl.FeatureSpec(horizon=48, peak_features=peak)
    fb, sn = features(spec, bundle)
    common = dict(respect_peak=peak, train_days=TRAIN_DAYS, val_days=VAL_DAYS,
                  verbose=False)
    net_a, hist_a = rl.train_dqn(bundle["sig"], bundle["settle"], bundle["env"],
                                 fb, sn, cfg, **common)
    reward = cls(bundle, fb, sn, val_blocks=VAL_DAYS, soc_init=0.5 * CAP,
                 respect_peak=peak, load_fc=fc[0], pv_fc=fc[1])
    net_b, hist_b = rl.train_dqn(bundle["sig"], bundle["settle"], bundle["env"],
                                 fb, sn, cfg, reward_fn=reward, **common)
    check(f"{name}: {cls.__name__} trains bit for bit as reward_fn=None does",
          same_weights(net_a, net_b)
          and hist_a["val_cost_closed"] == hist_b["val_cost_closed"]
          and hist_a["episode_return_eur"] == hist_b["episode_return_eur"])
    check(f"{name}: {cls.__name__} adds no contract state to an episode",
          reward.start_episode(0, 7 * H) is None)

plain = rl.FeatureSpec(horizon=48, peak_features=True)
check("a spec without contract features digests as it always did",
      "contract_features" not in plain.config()
      and rl.FeatureSpec(horizon=48, peak_features=True,
                         contract_features=True).config()["contract_features"])
stored = os.path.join(rb.OUT, "SI", "fc_h24__bc", "138.json")
if os.path.exists(stored):
    with open(stored, encoding="utf-8") as fh:
        blob = json.load(fh)
    check("the panel's stored SI clone still carries the current digest",
          blob["digest"] == rb.run_digest(rb.make_config(500_000, 0),
                                          rb.variant_specs("SI")["fc_h24"],
                                          "SI", "bc"))
try:
    rl.FeatureSpec(horizon=48, contract_features=True)
    check("contract features without peak features are refused", False)
except ValueError:
    check("contract features without peak features are refused", True)


# --------------------------------------------------------------------------
# 3. The contract charge is the standing charge the invoice bills
# --------------------------------------------------------------------------
cal = rr.calendar_for(sig, 1)
env = monthly["env"]
shaved = np.minimum(naive, np.quantile(naive, 0.97))     # a clipped profile


def month_fixed(m):
    """Standing charge the environment bills in month index `m`, EUR."""
    zero = {b: 0.0 for b in (1, 2, 3, 4, 5)}
    return sum(monthly["settle"](env, int(i), 0.0, zero)[3]
               for i in np.flatnonzero(cal.month == m))


# October -> November: both low season, so the same blocks are billed and the
# pool's cost must equal the next month's standing charge to the cent.
m_oct = 3
env.clear_achieved_power()
fixed_naive = month_fixed(m_oct + 1)
env.set_achieved_power_kw(shaved)
fixed_shaved = month_fixed(m_oct + 1)
env.clear_achieved_power()
want = fixed_shaved - fixed_naive
got = cal.pool_cost(shaved, m_oct) - cal.pool_cost(naive, m_oct)
check("a pool's cost moves exactly as the next month's standing charge does",
      want < -0.01 and abs(got - want) < 1e-6,
      f"environment {want:.4f}, contract charge {got:.4f} EUR")

a, b = 100 * H, 114 * H
tracker = rr.ContractTracker(cal, shaved, exclude=(a, b))
total = sum(tracker.push(i, shaved[i]) for i in range(a, b))
want = sum(cal.pool_cost(shaved, m) - cal.pool_cost(shaved, m, exclude=(a, b))
           for m in sorted(set(cal.month[a:b].tolist())))
check("'marginal' pricing telescopes to the pool with the episode minus "
      "without", abs(total - want) < 1e-9, f"{total:.4f} EUR")

tracker = rr.ContractTracker(cal, np.zeros(N_STEPS), span=(a, b))
total = sum(tracker.push_priced(i, naive[i]) for i in range(a, b))
want = 0.0
for m in sorted(set(cal.month[a:b].tolist())):
    rows = np.arange(a, b)[cal.month[a:b] == m]
    lists = {}
    for blk in rr.MEASURED_BLOCKS:
        v = np.sort(np.concatenate([np.zeros(cal.k),
                                    naive[rows[cal.blocks[rows] == blk]]]))
        lists[blk] = v[-cal.k:].tolist()
    want += cal.share(a, b, m) * cal.cost(lists, m)
check("'episode' pricing bills the episode its time-share of the contract "
      "its own peaks set", abs(total - want) < 1e-9, f"{total:.4f} EUR")
check("an episode's time-shares add up to its length in billing periods",
      abs(sum(cal.share(a, b, m) for m in range(cal.n_months))
          - (b - a) / (H * 30.4375)) < 1e-9)

trace = cal.ratchet_trace(naive)
for m in (1, 3):
    rows = np.flatnonzero(cal.month == m)
    check(f"'ratchet' pricing over whole month {m} sums to its pool's cost",
          abs(trace[rows].sum() - cal.pool_cost(naive, m)) < 1e-9)
check("a draw the contract does not read (block 5) is never charged",
      float(np.abs(trace[cal.blocks == 5]).sum()) == 0.0
      and float(np.abs(cal.leave_one_out(naive)[cal.blocks == 5]).sum()) == 0.0)


# --------------------------------------------------------------------------
# 4. The contract state the agent reads is causal
# --------------------------------------------------------------------------
probe = 70 * H + 20
poisoned = naive.copy()
poisoned[probe:] += 50.0
check("the contract threshold never reads the month's future",
      rr.ContractTracker(cal, naive).threshold_at(probe)
      == rr.ContractTracker(cal, poisoned).threshold_at(probe))

spec_c = rl.FeatureSpec(horizon=48, peak_features=True, contract_features=True)
fb_c, sn_c = features(spec_c, monthly)


class Recording(rl.LearnedPolicy):
    """A learned policy that writes down the threshold it was shown."""

    def setpoint(self, sig_, idx, soc, lo, hi, peak_state):
        self.seen.append(self._tracker.threshold_at(idx))
        return super().setpoint(sig_, idx, soc, lo, hi, peak_state)


torch.manual_seed(0)
pol = Recording(rl.QNet(fb_c.dim), fb_c, True, load_fc=fc[0], pv_fc=fc[1],
                tracker_factory=rr.tracker_factory(1))
pol.seen = []
out = rbc._run_policy_once(env, pol, n_steps=N_STEPS, signals=sig,
                           keep_traces=True, settle=monthly["settle"],
                           soc_init_kwh=0.5 * CAP)
metered = np.maximum(out["_net_trace"], 0.0) / sig.hours
idxs = np.random.default_rng(1).integers(0, N_STEPS, 300)
check("in the field the threshold is built from the agent's own meter",
      all(pol.seen[i] == rr.ContractTracker(cal, metered).threshold_at(int(i))
          for i in idxs))
check("a new month opens with nothing in the pool",
      pol.seen[int(cal.first_row[2])] == 0.0)

cal12 = rr.calendar_for(annual["sig"], 12)
boot = annual["env"].agreed_power_bootstrap_kw
first_b2 = int(np.flatnonzero(cal12.blocks == 2)[0])
check("a twelve-month pool opens on the meter history the household "
      "walked in with",
      rr.ContractTracker(cal12, naive).threshold_at(first_b2)
      == sorted(boot[2])[-cal12.k:][0] > 0.0)
check("a twelve-month pool prices a kW at twelve months of standing charge",
      all(abs(cal12.rate[0][blk]
              - sum(cal.rate_by_calendar_month.get((blk, mo), 0.0)
                    for mo in range(1, 13))
              - (sum(cal.rate_by_calendar_month.get((5, mo), 0.0)
                     for mo in range(1, 13)) if blk == 4 else 0.0)) < 1e-9
          for blk in rr.MEASURED_BLOCKS))


# --------------------------------------------------------------------------
# 5. Contract types are different bills
# --------------------------------------------------------------------------
before = fixed["env"].agreed_power_schedule
fixed["env"].set_achieved_power_kw(shaved)
check("a signed contract is not moved by any dispatch",
      fixed["env"].agreed_power_schedule == before)
vals = [signed[blk] for blk in sorted(signed)]
check("the signed contract is the operator's: 0.1 kW steps, rising by block",
      all(abs(v * 10 - round(v * 10)) < 1e-9 for v in vals)
      and vals == sorted(vals) and len(vals) == 5)


def idle_fixed(bundle):
    bundle["env"].clear_achieved_power()
    return rbc.run_policy(bundle["env"], rbc._Idle(), n_steps=N_STEPS,
                          settle=bundle["settle"], soc_init_kwh=0.5 * CAP,
                          rates=bundle["rates"])["Fixed_EUR"]


fb_f, sn_f = features(plain, fixed)
static_f = fb_f.build_static(fixed["sig"], load_fc=fc[0], pv_fc=fc[1])
col = fb_f.agreed_kw_column
check("the agreed-power column is where FeatureBuilder says it is",
      np.allclose(static_f[:, col], fixed["sig"].agreed_kw[:N_STEPS])
      and rl.FeatureBuilder(rl.FeatureSpec(horizon=48)).agreed_kw_column is None)
# The disease the floor exists for: a signed line is (nearly) one number for
# the whole of training, so its own spread scales a 0.6 kW change in the line
# -- an ordinary year-to-year move -- into an input no network was trained on.
spread = float(fb_f.norm_std[col])
check("a signed contract's line has (almost) no spread to be scaled by",
      spread < 0.25, f"std {spread:.3f} kW")
check("...so the signed contract takes the line out, the annual one floors "
      "its scale, and the monthly one -- the panel's -- keeps its own spread",
      set(rc.AGREED_KW_SCALE) == set(rr.ALL_TYPES)
      and rc.AGREED_KW_SCALE["monthly"] is None
      and 1.8 / rc.AGREED_KW_SCALE["fixed"] == 0.0
      and 0.0 < rc.AGREED_KW_SCALE["annual"] < float("inf"))
cfg0 = rb.make_config(500_000, 0)
check("the floor is part of a result's identity where it applies",
      len({rc.run_digest(c, rc.SPECIALIST[c], "bc", cfg0)
           for c in rr.ALL_TYPES}) == len(rr.ALL_TYPES))

bills = [idle_fixed(x) for x in (fixed, monthly, annual)]
check("the three contract types bill one household three standing charges",
      len({round(v, 2) for v in bills}) == 3,
      "fixed %.2f, monthly %.2f, annual %.2f EUR" % tuple(bills))
try:
    rr.SIMonthlyContractReward(monthly, fb_c, sn_c, val_blocks=VAL_DAYS,
                               soc_init=0.5 * CAP, respect_peak=True,
                               options=rr.ContractOptions(pricing="bogus"))
    check("an unknown pricing is refused", False)
except ValueError:
    check("an unknown pricing is refused", True)
try:
    rr.SIAnnualContractReward(annual, fb_c, sn_c, val_blocks=VAL_DAYS,
                              soc_init=0.5 * CAP, respect_peak=True,
                              load_fc=fc[0], pv_fc=fc[1],
                              options=rr.ContractOptions(pricing="episode"))
    check("an episode is not allowed to stand in for a twelve-month pool", False)
except ValueError:
    check("an episode is not allowed to stand in for a twelve-month pool", True)


# --------------------------------------------------------------------------
# 5b. Price lists are different bills too, and each is billed as it is shown
# --------------------------------------------------------------------------
study_opts = monthly["env"].pricing_options
check("the study's own environment is exactly as it was: its price list, "
      "and no option the panel's runs never carried",
      study_opts["paket_id"] == "GENI_SAMO_DINAMICNI"
      and "meritve_15min" not in study_opts)
check("another price list is installed with the 15-minute tariff structure "
      "the rate vectors are built with -- and keeps the study's tariff year",
      all(by_price[c]["env"].pricing_options
          == dict(study_opts, paket_id=rr.PRICE_PAKET[c], meritve_15min=True)
          for c in ("active", "single")))
energy = {}
for c, b in by_price.items():
    b["env"].clear_achieved_power()
    out_c = rbc.run_policy(b["env"], rbc._Idle(), n_steps=N_STEPS,
                           settle=b["settle"], soc_init_kwh=0.5 * CAP,
                           rates=b["rates"])
    s_c = b["sig"]
    net_c = s_c.consumption - s_c.generation
    from_rates = float(np.sum(np.maximum(net_c, 0.0) * s_c.import_rate)
                       - np.sum(np.maximum(-net_c, 0.0) * s_c.export_credit))
    energy[c] = out_c["Energy_EUR"]
    check(f"{c}: the rates a controller decides against are the rates it is "
          f"billed at", abs(out_c["Energy_EUR"] - from_rates) < 1e-6,
          f"{out_c['Energy_EUR']:.2f} EUR")
check("the three price lists bill one household three energy lines",
      len({round(v, 2) for v in energy.values()}) == 3,
      "dynamic %.2f, active %.2f, single %.2f EUR" % (
          energy["monthly"], energy["active"], energy["single"]))
check("the single-tariff list has one export price; the dynamic one moves",
      float(np.ptp(by_price["single"]["sig"].export_credit)) < 1e-9
      and float(np.ptp(monthly["sig"].export_credit)) > 0.01)
fb_p, _ = features(plain, by_price["single"])
static_p = fb_p.build_static(by_price["single"]["sig"], load_fc=fc[0], pv_fc=fc[1])
check("the price columns are where FeatureBuilder says they are",
      np.allclose(static_p[:, fb_p.price_columns[0]],
                  by_price["single"]["sig"].import_rate)
      and np.allclose(static_p[:, fb_p.price_columns[1]],
                      by_price["single"]["sig"].export_credit)
      and len(fb_p.price_columns) == 6
      and rl.FeatureBuilder(rl.FeatureSpec(
          horizon=48, price_features=False)).price_columns == [])
check("a price that never moves in training is scaled by a stated floor",
      float(fb_p.norm_std[fb_p.price_columns[1]]) < 1e-4 < rc.PRICE_SCALE["single"]
      and rc.PRICE_SCALE["monthly"] is None)
check("the price lists share the study's agreed-power rule and its reward's "
      "arithmetic", all(by_price[c]["env"].agreed_power_n_months_window == 1
                        and issubclass(rr.reward_class("SI", c),
                                       rr.SIMonthlyContractReward)
                        for c in rr.PRICE_TYPES))


# --------------------------------------------------------------------------
# 6. Validation sees the contract, and the contract it leaves is the agent's
# --------------------------------------------------------------------------
roster = {p.name: p for p in hs.rule_roster("SI")}


def meter():
    return rr.SIMonthlyContractReward(monthly, None, None, val_blocks=VAL_DAYS,
                                      soc_init=0.5 * CAP, respect_peak=True)


v_idle = meter().settle_contract(rbc._Idle(), passes=3)
v_shave = meter().settle_contract(roster["self_consumption_peak_shaving"],
                                  passes=3)
check("the held-out measure charges an unshaved week more contract than a "
      "shaved one", v_shave["contract_eur"] < v_idle["contract_eur"],
      f"{v_shave['contract_eur']:.2f} vs {v_idle['contract_eur']:.2f} EUR")

flat = meter()
level = float(np.quantile(naive, 0.9))
out_flat = {"_import_kw": np.minimum(naive, level),
            "_cost_trace": np.zeros(N_STEPS), "_soc_trace": np.full(N_STEPS, 1.0),
            "_setpoints": np.zeros(N_STEPS), "Final_SOC_kWh": 1.0}
out_high = dict(out_flat, _import_kw=np.minimum(naive, 2.0 * level))
check("...and a lower LEVEL less than a higher one, even when no week "
      "stands out", flat.held_out(out_flat)["contract_eur"]
      < flat.held_out(out_high)["contract_eur"])

reward = rr.SIMonthlyContractReward(monthly, fb_c, sn_c, val_blocks=VAL_DAYS,
                                    soc_init=0.5 * CAP, respect_peak=True,
                                    load_fc=fc[0], pv_fc=fc[1])
torch.manual_seed(0)
net = rl.QNet(fb_c.dim)
agreed_before = reward.sig.agreed_kw.copy()
val, sig_after, _ = reward.validate(net)
check("validation re-agrees the contract from the agent's own draw",
      env.agreed_power_is_endogenous
      and not np.array_equal(agreed_before, sig_after.agreed_kw)
      and np.array_equal(env.achieved_power_kw[:N_STEPS], reward.profile))
check("...and scores bill + contract on the held-out week",
      abs(val["cost_eur_closed"] - val["cost_eur"] - val["contract_eur"]) < 50.0
      and val["contract_eur"] > 0.0)
fresh = rr.SIMonthlyContractReward(monthly, fb_c, sn_c, val_blocks=VAL_DAYS,
                                   soc_init=0.5 * CAP, respect_peak=True,
                                   load_fc=fc[0], pv_fc=fc[1])
check("a new reward starts from the no-battery contract, whatever ran before",
      not env.agreed_power_is_endogenous
      and np.array_equal(fresh.sig.agreed_kw, agreed_before))


# --------------------------------------------------------------------------
# 7. Training under the contract charge
# --------------------------------------------------------------------------
def contract_run(pricing=None, weight=1.0):
    r = rr.SIMonthlyContractReward(
        monthly, fb_c, sn_c, val_blocks=VAL_DAYS, soc_init=0.5 * CAP,
        respect_peak=True, load_fc=fc[0], pv_fc=fc[1],
        options=rr.ContractOptions(pricing=pricing, weight=weight))
    return rl.train_dqn(r.sig, monthly["settle"], env, fb_c, r.static_norm, cfg,
                        respect_peak=True, train_days=TRAIN_DAYS,
                        val_days=VAL_DAYS, verbose=False, reward_fn=r)


net_1, hist_1 = contract_run()
net_2, hist_2 = contract_run()
check("a contract-charge run is reproducible from its seed",
      same_weights(net_1, net_2)
      and hist_1["val_cost_closed"] == hist_2["val_cost_closed"])
net_3, hist_3 = contract_run(weight=0.0)
check("the contract charge reaches the update (weight 0 trains another net)",
      not same_weights(net_1, net_3))
for pricing in ("ratchet", "marginal"):
    net_p, hist_p = contract_run(pricing=pricing)
    check(f"'{pricing}' pricing trains and validates",
          len(hist_p["val_cost_closed"]) == 2
          and all(np.isfinite(hist_p["val_cost_closed"])))

setpoints = np.zeros(N_STEPS)
setpoints[::7] = 0.3
r_bc = rr.SIMonthlyContractReward(monthly, fb_c, sn_c, val_blocks=VAL_DAYS,
                                  soc_init=0.5 * CAP, respect_peak=True,
                                  load_fc=fc[0], pv_fc=fc[1])
bc_cfg = rl.TrainConfig(seed=0, hidden=32)
net_bc, hist_bc = rl.train_bc(
    r_bc.sig, monthly["settle"], env, fb_c, r_bc.static_norm, setpoints,
    start=0, stop=N_STEPS, soc_init=0.5 * CAP, respect_peak=True, cfg=bc_cfg,
    verbose=False, holdout_days=VAL_DAYS,
    opts=rl.BCOptions(max_epochs=3), reward_fn=r_bc)
taus = r_bc.teacher_thresholds(setpoints, 0, N_STEPS,
                               rl.teacher_actions(
                                   r_bc.sig, monthly["settle"], env, setpoints,
                                   0, N_STEPS, 0.5 * CAP, True)[0])
check("the cloner is shown the teacher's contract threshold, built causally",
      net_bc.adv.out_features == rl.N_ACTIONS and len(taus) == N_STEPS
      and taus[int(cal.first_row[2])] == 0.0 and taus.max() > 0.0)
env.clear_achieved_power()


# --------------------------------------------------------------------------
# 8. The references are the study's own (when both are on disk)
# --------------------------------------------------------------------------
ref_path = os.path.join(rc.OUT, "monthly", "refs", "138.json")
arm = rb._comparators(138, "SI")
if os.path.exists(ref_path) and arm:
    with open(ref_path, encoding="utf-8") as fh:
        ctrl = json.load(fh)["controllers"]
    worst = max(abs(ctrl[name]["cost_eur_closed"] - arm[f"ref_cost_{name}"])
                for name in ("no_battery", "self_consumption",
                             "self_consumption_peak_shaving"))
    check("the monthly rule references reproduce the SI reference arm's "
          "checkpoint", worst < 1e-6, f"worst gap {worst:.2e} EUR")
    # The whole-year solve is a MIP once prices go negative often enough, and
    # then it stops inside a gap: two runs of the same model agree on the
    # objective to that gap, not to the cent (Ausgrid 138: 0.07 EUR of 404).
    gap = abs(ctrl["milp_full"]["cost_eur_closed"] - arm["ref_cost_milp_full"])
    check("...and the whole-year MILP to within its own solver gap",
          gap < 1e-3 * arm["ref_cost_milp_full"], f"gap {gap:.2e} EUR")

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
