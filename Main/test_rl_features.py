"""Invariant checks for the learned controllers' wider inputs
(rl_features.py, the FeatureSpec fields that switch them on, the re-issued AR
forecast, run_rl_features.py).

Like the other suites: plain script, one PASS/FAIL per check, non-zero exit on
failure. Every check is a way a wider observation can be quietly wrong -- a
column that reads the future, a history that restarts cold at the scored year,
a context cut from another window, a new field that makes every result on disk
stale, a screen that touches the test year.

    cd Main && python3 test_rl_features.py        (~1 min, 120-day slice)
"""

import json
import os
import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.simplefilter("ignore")

import hems_study as hs                                          # noqa: E402
import hbd_forecast as hbd                                       # noqa: E402
import rl_control as rl                                          # noqa: E402
import rl_features as rf                                         # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402
import run_rl_features as rx                                     # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Fixture: 120 days of one household on the SI settlement, and its last 60
# days as a second bundle -- the "scored year" that must not start cold
# --------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Input data", "Ausgrid")
N_DAYS, H = 120, 48
N = N_DAYS * H
TAIL = 60 * H

rb._calendar("SI")
frames = hs.load_study_frames(os.path.join(DATA, "Ausgrid 1.csv"))
df_kwh = frames["df_all_kwh"].iloc[:N + H]
df_kw = df_kwh.copy()
df_kw[["Energy_Generation", "Energy_Consumption"]] /= 0.5


def bundle(frame_kwh, frame_kw, n_steps):
    env = hs.align_envelope(
        hs.build_study_env(frame_kwh, battery_cap=10.0, soc_min_pct=0.10,
                           soc_max_pct=0.80, p_max=1.5, eff=0.95, delta_t=0.5,
                           H=48), 1.5, 0.95, 0.5)
    rates = hs.build_rate_vectors("SI", env, frame_kw.index,
                                  frame_kw["SMP"].values, 30)
    return {"env": env, "rates": rates,
            "settle": hs.build_settlement("SI", env, rates, 0.5),
            "sig": rbc.build_signals(env, n_steps=n_steps, rates=rates)}


whole = bundle(df_kwh, df_kw, N)
tail = bundle(df_kwh.iloc[TAIL:], df_kw.iloc[TAIL:], N - TAIL)
sig = whole["sig"]

load = df_kwh["Energy_Consumption"].to_numpy(dtype=float)
pv = df_kwh["Energy_Generation"].to_numpy(dtype=float)


def context(load, pv, offset=0):
    return rf.Context(load, pv, rl.median14_forecast(load, H),
                      rl.median14_forecast(pv, H), offset)


def forecaster(frame_kw, gamma):
    """The study's AR forecaster over `frame_kw`, its fit replaced by a fixed
    matrix: what is under test is which ROWS it reads, and a real fit is
    minutes of quantile regression that cannot change that."""
    f = hs.FITTED_KINDS["hbd_median14"](frame_kw, H, 60 * H)
    for col in ("Energy_Consumption", "Energy_Generation"):
        f._ars[(col, f.n_train)] = gamma
    return f


GAMMA = np.random.default_rng(0).normal(0.0, 0.02, size=(H, H))
FIRST = H + 14 * H                         # first anchor with a full lookback


def ar_tables(frame_kw):
    """(load, pv) tables as `run_rl_features.ar_forecasts` lays them out."""
    f = forecaster(frame_kw, GAMMA)
    n = len(frame_kw)
    out = []
    for col in ("Energy_Consumption", "Energy_Generation"):
        kwh = frame_kw[col].to_numpy(dtype=float) * 0.5
        med = rl.median14_forecast(kwh, H)
        table = med[np.minimum(np.arange(n)[:, None] + 1 + np.arange(H)[None, :],
                               n - 1)]
        table[FIRST - 1:] = f.rolling_forecasts(col, np.arange(FIRST, n + 1)) * 0.5
        out.append(table)
    return out


ALL = rl.FeatureSpec(horizon=48, peak_features=True, n_bins=24, forecast="ar",
                     groups=rf.GROUPS)
BASE = rl.FeatureSpec(horizon=48, peak_features=True)

# --------------------------------------------------------------------------
# 1. Nothing on disk moved: the new fields are inert at their defaults
# --------------------------------------------------------------------------
cfg_keys = set(BASE.config())
check("a default spec's config carries none of the new keys",
      not {"forecast", "groups"} & cfg_keys and BASE.config()["n_bins"] == 4,
      f"keys {sorted(cfg_keys)}")
cfg = rb.make_config(500_000, 0)
stale = []
for tariff in ("AU", "SI"):
    for method in ("bc", "dqn", "bc_dqn"):
        path = os.path.join(rb.OUT, tariff, f"fc_h24__{method}", "138.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                stored = json.load(fh)
            # A straggler re-trained at 1M carries that budget's digest.
            steps = stored["train_config"]["total_steps"]
            if stored["digest"] != rb.run_digest(
                    rb.make_config(steps, 0), rb.variant_specs(tariff)["fc_h24"],
                    tariff, method):
                stale.append((tariff, method))
check("the panel's stored fc_h24 digests are still current", not stale,
      f"stale: {stale}" if stale else "Ausgrid 138, both tariffs, three methods")

arr = np.asarray(sig.import_rate, dtype=float)
old_mean = rl._lookahead_bins(arr, 48, N, mean=True)
old_sum = rl._lookahead_bins(load[:N], 48, N, mean=False)
check("the general lookahead agrees with the panel's four-bin path",
      np.allclose(old_mean, rf.lookahead(arr, 48, N, rl._BIN_EDGES, "mean"),
                  atol=1e-6)
      and np.allclose(old_sum, rf.lookahead(load[:N], 48, N, rl._BIN_EDGES,
                                            "sum"), atol=1e-5))

# --------------------------------------------------------------------------
# 2. Spec plumbing
# --------------------------------------------------------------------------
check("group order is not part of a spec's identity",
      rl.FeatureSpec(groups=("history", "rank"))
      == rl.FeatureSpec(groups=("rank", "history"))
      and rl.FeatureSpec(groups=["rank"]).groups == ("rank",))
for label, kw in (("an unknown group", dict(groups=("weather",))),
                  ("the peak group without the contract features",
                   dict(groups=("peak",))),
                  ("an unknown forecast source", dict(forecast="prophet"))):
    try:
        rl.FeatureSpec(horizon=48, **kw)
        check(f"{label} is refused", False)
    except ValueError:
        check(f"{label} is refused", True)

specs = {t: rx.feature_specs(t) for t in ("AU", "SI")}
digs = {(t, v, m): rb.run_digest(cfg, s, t, m)
        for t in specs for v, s in specs[t].items() for m in rx.methods_for(v)}
panel = {rb.run_digest(cfg, rb.variant_specs(t)[v], t, m)
         for t in ("AU", "SI") for v in rb.variant_specs(t)
         if v != rb.WIDE_VARIANT for m in ("bc", "dqn", "bc_dqn")}
check("every feature-study contract has a digest of its own",
      len(set(digs.values())) == len(digs) and not set(digs.values()) & panel,
      f"{len(digs)} runs per household pair")
check("the panel's wide contract IS the study's best follow-up, digest for digest",
      all(rb.variant_specs(t)[rb.WIDE_VARIANT] == specs[t]["all_coarse_rel"]
          and rb.run_digest(cfg, rb.variant_specs(t)[rb.WIDE_VARIANT], t, m)
          == digs[(t, "all_coarse_rel", m)]
          for t in ("AU", "SI") for m in ("bc", "dqn", "bc_dqn")))
check("each single-input contract is fc_h24 plus exactly that input",
      specs["SI"]["fine"] == replace(BASE, n_bins=24)
      and specs["SI"]["ar"] == replace(BASE, forecast="ar")
      and specs["SI"]["rank"] == replace(BASE, groups=("rank",))
      and specs["SI"]["net"] == replace(BASE, groups=("net",))
      and specs["SI"]["peak"] == replace(BASE, groups=("peak",))
      and "peak" not in specs["AU"] and "peak" not in specs["AU"]["all"].groups
      and specs["SI"]["all"] == ALL)

# --------------------------------------------------------------------------
# 3. The re-issued AR forecast: what it is, and what it may read
# --------------------------------------------------------------------------
frame_kw = df_kw[["Energy_Consumption", "Energy_Generation"]]
f = forecaster(frame_kw, GAMMA)
anchors = np.array([FIRST, 40 * H, 40 * H + 17, 90 * H + 5, N])
rolled = f.rolling_forecasts("Energy_Consumption", anchors)
y = f._column("Energy_Consumption")
lo, hi = f._range("Energy_Consumption", f.n_train)
by_hand = []
for i in anchors:
    past_bl = f._baseline_at("Energy_Consumption", np.arange(i - H, i), f.n_train, None)
    fut_bl = f._baseline_at("Energy_Consumption", np.arange(i, i + H), f.n_train, int(i))
    by_hand.append(hbd.compose_forecast(y[i - H:i], past_bl, fut_bl, GAMMA, lo, hi))
check("rolling_forecasts is the port's own composition at each anchor",
      np.allclose(rolled, np.array(by_hand), atol=1e-9),
      f"max diff {np.abs(rolled - np.array(by_hand)).max():.2e}")

CUT = 80 * H
poison = frame_kw.copy()
poison.iloc[CUT:] = poison.iloc[CUT:] * 3.0 + 1.0
f_p = forecaster(poison, GAMMA)
before_cut = np.arange(FIRST, CUT + 1)       # anchors that have seen < CUT only
# The clip range is read off the training block (60 days), which the poison
# does not reach, so the two forecasters differ in the poisoned rows alone.
check("an AR forecast reads nothing at or after its anchor",
      all(np.array_equal(f.rolling_forecasts(c, before_cut),
                         f_p.rolling_forecasts(c, before_cut))
          for c in ("Energy_Consumption", "Energy_Generation")))
try:
    f.rolling_forecasts("Energy_Consumption", np.array([FIRST - 1]))
    check("an anchor without a full lookback is refused", False)
except ValueError:
    check("an anchor without a full lookback is refused", True)

# --------------------------------------------------------------------------
# 4. Causality of every column: poison the meter from CUT on
# --------------------------------------------------------------------------
ar_load, ar_pv = ar_tables(frame_kw)
ctx = context(load, pv)
fb = rl.FeatureBuilder(ALL)
static = fb.build_static(sig, load_fc=ar_load[:N], pv_fc=ar_pv[:N], context=ctx)

load_p, pv_p = load.copy(), pv.copy()
load_p[CUT:] = load_p[CUT:] * 3.0 + 1.0
pv_p[CUT:] = pv_p[CUT:] * 3.0 + 1.0
sig_p = replace(sig, consumption=load_p[:N], generation=pv_p[:N],
                surplus=np.maximum(pv_p[:N] - load_p[:N], 0.0),
                deficit=np.maximum(load_p[:N] - pv_p[:N], 0.0))
arp_load, arp_pv = ar_tables(poison)
static_p = fb.build_static(sig_p, load_fc=arp_load[:N], pv_fc=arp_pv[:N],
                           context=context(load_p, pv_p))
moved = np.abs(static[:CUT] - static_p[:CUT]).max(axis=0)
check("no column of the full contract moves before the poisoned interval",
      float(moved.max()) == 0.0,
      f"{static.shape[1]} columns, worst {moved.max():.2e} at column "
      f"{int(moved.argmax())}")
check("the poison does reach the rows after it (the test can fail)",
      float(np.abs(static[CUT:] - static_p[CUT:]).max()) > 0.1)

rank_cols = np.column_stack(rf.rank_columns(sig, 48))
price_p = np.asarray(sig.import_rate, dtype=float).copy()
probe = 40 * H
price_p[probe + 1 + 48:] = 9.0
rank_p = np.column_stack(rf.rank_columns(replace(sig, import_rate=price_p), 48))
# The day rank reads the local day, which the day-ahead list covers.
day_end = max(s for s in sig.day_steps[sig.day_idx[probe]])
check("price-rank columns read at most the horizon (and the local day)",
      np.array_equal(np.delete(rank_cols[probe], 1), np.delete(rank_p[probe], 1))
      and day_end <= probe + 48)

# --------------------------------------------------------------------------
# 5. No cold start at the scored year, and no borrowed history
# --------------------------------------------------------------------------
hist_whole = np.column_stack(rf.history_columns(sig, ctx, rl._BIN_EDGES))
hist_tail = np.column_stack(rf.history_columns(
    tail["sig"], context(load, pv, offset=TAIL), rl._BIN_EDGES))
# The envelope's lookahead stops at the bundle's end, which both share.
check("history at a later bundle's first step is the series' own, not a restart",
      np.allclose(hist_whole[TAIL:], hist_tail, atol=1e-12),
      f"first row max diff {np.abs(hist_whole[TAIL] - hist_tail[0]).max():.2e}")
cold = np.column_stack(rf.history_columns(
    tail["sig"], context(load[TAIL:], pv[TAIL:], offset=0), rl._BIN_EDGES))
check("...and a context WITHOUT the rows before it would have differed",
      float(np.abs(cold[0] - hist_tail[0]).max()) > 1e-3)
try:
    rf.history_columns(tail["sig"], context(load, pv, offset=TAIL - H),
                       rl._BIN_EDGES)
    check("a context cut from another window is refused", False)
except ValueError:
    check("a context cut from another window is refused", True)

cal_whole = np.column_stack(rf.calendar_columns(sig, with_window=False))
cal_tail = np.column_stack(rf.calendar_columns(tail["sig"], with_window=False))
days = pd.DatetimeIndex(sig.env.dataset.index[:N])
weekend = np.array([hs._si_cas.v_lokalni_cas(ts).weekday() >= 5 for ts in days])
check("work-free days are the settlement's calendar (weekends at least)",
      np.array_equal(cal_whole[TAIL:], cal_tail)
      and bool(np.all(cal_whole[weekend, 0] == 1.0))
      and 0.0 < cal_whole[:, 0].mean() < 0.5
      # tomorrow's flag today is today's flag tomorrow
      and np.array_equal(cal_whole[:-H, 1], cal_whole[H:, 0]))
share = rf.calendar_columns(sig, with_window=True)[2]
w = np.asarray(sig.windows[:N])
check("the ratchet-window share restarts with every window and stays in [0, 1)",
      float(share.min()) == 0.0 and float(share.max()) < 1.0
      and bool(np.all(share[1:][w[1:] != w[:-1]] == 0.0)))

# --------------------------------------------------------------------------
# 6. Layout the rest of the code counts on, and a well-scaled matrix
# --------------------------------------------------------------------------
fb.fit_norm(static)
check("the agreed power is still where agreed_kw_column says",
      np.allclose(static[:, fb.agreed_kw_column], sig.agreed_kw[:N]))
check("price_columns covers the rate, the credit and every price bin",
      len(fb.price_columns) == 2 + ALL.n_bins
      and np.allclose(static[:, fb.price_columns[0]], sig.import_rate[:N])
      and rl.FeatureBuilder(BASE).price_columns == list(range(8, 14)))
n_base = rl.FeatureBuilder(BASE).build_static(
    sig, load_fc=ctx.load_m14[:N], pv_fc=ctx.pv_m14[:N]).shape[1]
check("the full contract is the panel's plus the announced columns",
      static.shape[1] == n_base + 3 * 20 + 8 + 12 + 18 + 3 + 7,
      f"{n_base} -> {static.shape[1]} columns")
check("no column is constant or non-finite on the fixture",
      bool(np.isfinite(static).all()) and float(fb.norm_std.min()) > 1e-4,
      f"smallest std {fb.norm_std.min():.3g}")
# The group columns sit between the lookahead bins and the three contract
# columns: 8 + 2 + three channels of 24 bins = 82 before them.
z = np.abs(fb.normalize(static))[:, 82:static.shape[1] - 3]
check("no added column runs away once normalised",
      z.shape[1] == 8 + 12 + 18 + 3 + 7 and float(z.max()) < 40.0,
      f"largest |z| {z.max():.1f}")

# --------------------------------------------------------------------------
# 7. One evaluator: the wide contract is still a Policy like any other
# --------------------------------------------------------------------------
net = rl.QNet(fb.dim, 32)
policy = rl.LearnedPolicy(net, fb, True, load_fc=ar_load[:N], pv_fc=ar_pv[:N],
                          context=ctx)
out = rbc.run_policy(whole["env"], policy, n_steps=N, settle=whole["settle"],
                     soc_init_kwh=4.0, rates=whole["rates"])
whole["env"].clear_achieved_power()
check("a wide-contract policy runs through run_policy, contract converged",
      bool(np.isfinite(out["Cost_EUR_Total"])) and out["Agreed_Power_Converged"],
      f"{out['Agreed_Power_Iters']} contract pass(es)")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "m.pt")
    rl.save_model(path, net, fb, ALL, rl.TrainConfig(hidden=32), {})
    net2, fb2, spec2, _ = rl.load_model(path)
check("a saved wide model comes back with its contract",
      spec2 == ALL and np.array_equal(fb2.norm_mean, fb.norm_mean))

# --------------------------------------------------------------------------
# 8. The driver: validation first, the test year once
# --------------------------------------------------------------------------
check("a result is current under either legitimate budget, and only those",
      len(rx.digests("AU", "all", "dqn")) == 2
      and len(rx.digests("AU", "all", "bc")) == 1)
jobs = rx.jobs_for([138], ["AU", "SI"])
check("leave-one-out contracts are trained for the clone alone",
      all(m == "bc" for _, v, m, _ in jobs if v.startswith("all_no_"))
      and {m for _, v, m, _ in jobs if v == "all"} == set(rx.METHODS))
check("SI is chosen on the contract-aware measure, AU on the panel's",
      rx.CONTRACT_MEASURED == ("SI",)
      and [j[0] for j in rx.baseline_jobs([138], ["AU", "SI"])] == ["SI"] * 3)
with tempfile.TemporaryDirectory() as tmp:
    prep = {"ident": 1, "tariff": "SI", "train": whole, "sim": tail,
            "fc_train": (ctx.load_m14[:N], ctx.pv_m14[:N]),
            "fc_ar_train": (ar_load[:N], ar_pv[:N]), "context_train": ctx}
    inputs = rb.feature_inputs(prep, ALL, "train")
    check("feature_inputs hands a spec the source and the context it names",
          inputs["load_fc"] is prep["fc_ar_train"][0] and inputs["context"] is ctx
          and rb.feature_inputs(prep, BASE, "train")["context"] is None
          and rb.feature_inputs(prep, BASE, "train")["load_fc"]
          is prep["fc_train"][0])

# --------------------------------------------------------------------------
# 9. The pooled clone: same pool, same fit, wider static columns
# --------------------------------------------------------------------------
pool = {t: rx.pool_specs(t) for t in ("AU", "SI")}
check("no pooled contract asks for an AR that 270 households do not have",
      all(s.forecast == "median14" for t in pool for s in pool[t].values())
      and pool["SI"][rx.BASELINE] == BASE
      and pool["SI"]["all_no_ar"] == replace(ALL, forecast="median14"))
check("every pooled contract keeps the panel's dynamic features",
      all(rl.FeatureBuilder(s).n_dynamic == rl.FeatureBuilder(
          rb.variant_specs(t)[rx.BASELINE]).n_dynamic
          for t in pool for s in pool[t].values()))
d_a = rx.pool_digest("AU", "groups", 64, 0, [1, 2, 3])
check("a pooled clone's identity covers its pool, width, seed and contract",
      len({d_a, rx.pool_digest("AU", "groups", 64, 0, [1, 2]),
           rx.pool_digest("AU", "groups", 256, 0, [1, 2, 3]),
           rx.pool_digest("AU", "groups", 64, 1, [1, 2, 3]),
           rx.pool_digest("AU", "all_no_ar", 64, 0, [1, 2, 3]),
           rx.pool_digest("SI", "groups", 64, 0, [1, 2, 3])}) == 6)
cfg_p, clone_p = rx.pool_config("AU", 256, 0)
check("a pooled clone trains with the panel's tuned clone, at the width asked",
      cfg_p.hidden == 256 and clone_p == rb.effective_clone("AU", "bc")
      and cfg_p.lr == rb.effective_config(rb.make_config(1, 0), "AU", "bc").lr)

# --------------------------------------------------------------------------
# 10. The contract-paid arm: the contract study's run, a wider observation
# --------------------------------------------------------------------------
import run_rl_contract as rc                                     # noqa: E402

cs = rx.contract_specs()
wide_arm = rc.spec_for("contract", cs["peak"])
check("a wider contract-paid spec is the arm's own plus the inputs, nothing else",
      wide_arm == replace(rc.spec_for("contract"), groups=("peak",))
      and rc.spec_for("excess", cs["groups"]).contract_features is False
      and all(s.peak_features for s in cs.values()))
d0 = rc.run_digest("monthly", "contract", "dqn", cfg)
check("the contract study's own digests do not move, and a wider run's differs",
      d0 == rc.run_digest("monthly", "contract", "dqn", cfg, base=None)
      and d0 != rc.run_digest("monthly", "contract", "dqn", cfg, base=cs["peak"])
      and rc.run_digest("monthly", "excess", "bc", cfg)
      != rc.run_digest("monthly", "excess", "bc", cfg, base=cs["peak"]))

# --------------------------------------------------------------------------
# 11. Prices against the trailing week: same information, no level
# --------------------------------------------------------------------------
REL = replace(BASE, price_reference="trailing")
check("the price reference is inert at its default and refused when unknown",
      "price_reference" not in BASE.config()
      and REL.config()["price_reference"] == "trailing"
      and rb.run_digest(cfg, REL, "SI", "bc") != rb.run_digest(cfg, BASE, "SI", "bc"))
fb_l, fb_r = rl.FeatureBuilder(BASE), rl.FeatureBuilder(REL)
fc = (ctx.load_m14[:N], ctx.pv_m14[:N])
st_l = fb_l.build_static(sig, load_fc=fc[0], pv_fc=fc[1])
st_r = fb_r.build_static(sig, load_fc=fc[0], pv_fc=fc[1])
ref = rf.trailing_price(sig)
pc = fb_r.price_columns
other = [c for c in range(st_l.shape[1]) if c not in pc]
check("the trailing reference moves the price columns by one number and nothing else",
      np.allclose(st_l[:, pc] - st_r[:, pc], ref[:, None], atol=1e-6)
      and np.array_equal(st_l[:, other], st_r[:, other]))
p_cut = np.asarray(sig.import_rate, dtype=float).copy()
p_cut[CUT:] += 5.0
check("the trailing reference reads strictly past prices",
      np.array_equal(ref[:CUT + 1],
                     rf.trailing_price(replace(sig, import_rate=p_cut))[:CUT + 1])
      and ref[0] == sig.import_rate[0])
lifted = replace(sig, import_rate=sig.import_rate + 0.02,
                 export_credit=sig.export_credit + 0.02)
st_lift = fb_r.build_static(lifted, load_fc=fc[0], pv_fc=fc[1])
st_lift_l = fb_l.build_static(lifted, load_fc=fc[0], pv_fc=fc[1])
# The bins of a bundle's last day run past its end and read zero there; the
# level cancels wherever a price is read at all.
body = slice(1, N - H)
check("a price list lifted by a constant looks the same against its own week",
      np.allclose(st_lift[body][:, pc], st_r[body][:, pc], atol=1e-6)
      and float(np.abs(st_lift_l[body][:, pc] - st_l[body][:, pc]).min()) > 0.019)
check("follow-up contracts exist beside the choice and are never candidates",
      set(rx.FOLLOW_UP) <= set(specs["SI"]) and set(rx.FOLLOW_UP) <= set(rx.contract_specs())
      and specs["SI"]["rel"] == REL
      and specs["SI"]["all_coarse_rel"] == replace(specs["SI"]["all_coarse"],
                                                    price_reference="trailing")
      and not set(rx.FOLLOW_UP) & set(rx.SINGLES + ["all", "all_coarse"])
      and set(rx.POOL_FOLLOW_UP) <= set(rx.pool_specs("SI")))

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL CHECKS PASSED")
