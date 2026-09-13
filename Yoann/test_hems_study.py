"""The invariants the ERK 2026 comparison rests on.

Not a unit-test suite. Every check here is one of the ways this study has
actually been wrong: two batteries, two evaluators, two windows, two starting
states of charge. Each was found by measurement rather than by reading, so each
is pinned here where a re-run says so rather than a figure quietly moving.

    python test_hems_study.py

Runs in about a minute on a short slice; no sweep, no forecast cache.
"""

import sys
import warnings

import numpy as np
import pandas as pd

import hems_study as hs
import Rule_Based_Control as rbc
from Basic_Functions import cumulative_interval_price_series
from MILP_Household import build_household_env

BATTERY = dict(battery_cap=10.0, soc_min_pct=0.10, soc_max_pct=0.80,
               p_max=1.5, eff=0.95)
DELTA_T, H = 0.5, 48
DATA = "../Input data/Ausgrid/Ausgrid 127.csv"

_passed, _failed = [], []


def check(name, ok, detail=""):
    (_passed if ok else _failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


def load(n_days=10, start="2012-07-01 00:30:00"):
    """A slice in BOTH conventions: kWh per interval, and kW."""
    raw = pd.read_csv(DATA)
    raw.index = pd.to_datetime(raw["Timestamp_UTC"], format="ISO8601")
    cols = ["SMP", "Energy_Generation", "Energy_Consumption"]
    kwh = raw.loc[start:, cols].iloc[:H * n_days].copy()
    kw = kwh.copy()
    kw[["Energy_Generation", "Energy_Consumption"]] /= DELTA_T
    return kwh, kw


def make_env(kwh):
    return hs.align_envelope(
        hs.build_study_env(kwh, delta_t=DELTA_T, H=H, **BATTERY),
        BATTERY["p_max"], BATTERY["eff"], DELTA_T)


def test_one_battery(kwh):
    """A1/A2. The MILP and every rule must drive the SAME battery.

    They did not. MILPScheduler bounded AC power at +-p_max while the rules read
    env.max_charge_kwh, which upstream applies to STORED energy -- +1.579/-1.425
    kW against +-1.500, a 5.3 % larger charge rating and a 5.0 % smaller
    discharge rating, live in every published figure.
    """
    env = make_env(kwh)
    milp = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    rule = rbc.build_signals(env, n_steps=H)
    check("one battery: MILP charge limit == rule charge limit",
          abs(milp.max_ch_kw - rule.max_charge_ac / DELTA_T) < 1e-9,
          f"{milp.max_ch_kw:.4f} vs {rule.max_charge_ac / DELTA_T:.4f} kW")
    check("one battery: MILP discharge limit == rule discharge limit",
          abs(milp.max_dis_kw - rule.max_discharge_ac / DELTA_T) < 1e-9,
          f"{milp.max_dis_kw:.4f} vs {rule.max_discharge_ac / DELTA_T:.4f} kW")

    # An unaligned environment must be refused, not silently scored.
    raw_env = hs.build_study_env(kwh, delta_t=DELTA_T, H=H, **BATTERY)
    try:
        hs.UpstreamMILPScheduler(raw_env, delta_t=DELTA_T, parity=True, **BATTERY)
        check("one battery: unaligned environment is refused", False, "no error raised")
    except ValueError:
        check("one battery: unaligned environment is refused", True)


def test_milp_parity(kwh, kw):
    """A2. parity=True must reproduce the hand-rolled model it replaced.

    This is what makes the previously published numbers reproducible rather than
    merely plausible. If it ever fails, the swap changed the answer.
    """
    env = make_env(kwh)
    old = hs.MILPScheduler(delta_t=DELTA_T, **BATTERY)
    new = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    buy, sell, _ = hs.au_rate_vectors(kwh.index, kwh["SMP"].values, 30)
    gen = kw["Energy_Generation"].tolist()
    con = kw["Energy_Consumption"].tolist()
    worst = 0.0
    for day in range(3):
        a, b = day * H, (day + 1) * H
        ro = old.solve(5.0, buy[a:b], sell[a:b], gen[a:b], con[a:b])
        rn = new.solve(5.0, buy[a:b], sell[a:b], gen[a:b], con[a:b])
        worst = max(worst, abs(ro["cost"] - rn["cost"]))
    check("MILP parity: upstream physics reproduces the old model",
          worst < 1e-6, f"worst cost delta {worst:.2e} EUR over 3 daily solves")


def test_one_evaluator(kwh, kw):
    """The AU settlement must reproduce the controller's own costing exactly.

    Adopting a shared evaluator must move no published number; if it does, the
    change is not "one evaluator" but "a different bill".
    """
    env = make_env(kwh)
    rates = hs.build_rate_vectors("AU", env, kwh.index, kwh["SMP"].values, 30)
    settle = hs.build_settlement("AU", env, rates, DELTA_T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
    sched = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    ctrl = hs.ReactiveController(
        scheduler=sched, forecaster=None, real_data=kw, soc_init=5.0,
        horizon_steps=H, steps_per_day=H, reoptimize_every=1, freq="30min",
        rate_vectors=rates)
    df = ctrl.run(num_days=len(kwh) // H - 1, use_forecast=False)
    net = (df["Buy_kW"] - df["Sell_kW"]).to_numpy() * DELTA_T
    s = hs.settle_trajectory(env, net, settle, sig)
    own = float(df["Step_Cost"].sum())
    check("one evaluator: AU settlement == controller's own costing",
          abs(own - s["Cost_EUR"]) < 1e-9, f"delta {abs(own - s['Cost_EUR']):.2e} EUR")
    check("one evaluator: AU carries no capacity charge",
          s["Power_EUR"] == 0.0)
    check("one evaluator: the standing charge is recovered, not dropped",
          s["Fixed_EUR"] > 0, f"{s['Fixed_EUR']:.2f} EUR over {len(df)} steps")


def test_si_settlement_matches_upstream(kwh):
    """The SI walk must agree with the shared one for a fixed profile.

    price_interval and Basic_Functions.cumulative_interval_price_series carry the
    same running peak against the same contract; the only difference is that one
    prices a controller's trajectory and the other a fixed profile. On the
    no-battery profile they are the same thing, so they must agree.
    """
    env = make_env(kwh)
    n = len(kwh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        windows = rbc.reset_windows(env, n)
        peak_state = {b: 0.0 for b in rbc._BLOCKS}
        total = 0.0
        for idx in range(n):
            peak_state = rbc._drop_peak_on_window_start(peak_state, windows, idx)
            net = float(env.arr_consumption[idx] - env.arr_generation[idx])
            var, _, _, fixed, peak_state = rbc.price_interval(env, idx, net, peak_state)
            total += var + fixed
        upstream = cumulative_interval_price_series(
            kwh["Energy_Consumption"], kwh["Energy_Generation"], env, kwh)[-1]
    check("SI settlement: agrees with cumulative_interval_price_series",
          abs(total - upstream) < 1e-6, f"{total:.6f} vs {upstream:.6f} EUR")


def test_rules_stay_in_the_envelope(kwh):
    """Every rule is re-clamped by the runner, so drift is 0 by construction."""
    env = make_env(kwh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
        worst_drift, bad_decomp = 0.0, 0.0
        for pol in [rbc._Idle()] + hs.rule_roster("SI"):
            out = rbc.run_policy(env, pol, signals=sig)
            worst_drift = max(worst_drift, out["SOC_Drift_kWh"])
            bad_decomp = max(bad_decomp, abs(
                out["Cost_EUR"] - out["Energy_EUR"] - out["Power_EUR"]))
    check("rules: no policy escapes the battery envelope",
          worst_drift < 1e-9, f"worst SOC drift {worst_drift:.2e} kWh")
    check("rules: Cost_EUR == Energy_EUR + Power_EUR",
          bad_decomp < 1e-6, f"worst residual {bad_decomp:.2e} EUR")


def test_shared_starting_soc(kwh):
    """Every controller must start the year at the same state of charge.

    The rules began at SOC_FRACTION * capacity = 3.5 kWh stored while the MILP
    began at soc_init = 5.0 absolute, which is 4.0 stored. Half a kWh of free
    energy handed to one side of the comparison.
    """
    env = make_env(kwh)
    soc_init, soc_min = 5.0, BATTERY["battery_cap"] * BATTERY["soc_min_pct"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
        out = rbc.run_policy(env, rbc._Idle(), signals=sig,
                             soc_init_kwh=soc_init - soc_min)
    check("shared start: an idle pack ends where the MILP started",
          abs(out["Final_SOC_kWh"] + soc_min - soc_init) < 1e-9,
          f"{out['Final_SOC_kWh'] + soc_min:.4f} vs {soc_init:.4f} kWh")


def test_site_selection():
    """F5. The documented rule must be the one that produced the results."""
    published = [138, 127, 65, 148, 223, 179, 261, 142, 128, 168,
                 249, 81, 104, 172, 29, 27, 156, 180, 21, 290,
                 66, 240, 113, 67, 5, 158, 1, 204, 247, 137]
    check("site selection: derived from the clustering == the published list",
          hs.dataset_ids() == published)


def test_wear_cost():
    """F2. The wear rate must be the pack price over the rated cycle life."""
    import Battery_Economics as be
    check("wear: 10 kWh pack prices one full cycle at 0.4167 EUR",
          abs(be.cycle_cost_eur_per_efc(10.0)
              - be.CAPEX_EUR_PER_KWH * 10.0 / be.BATTERY_CYCLE_LIMIT_EFC) < 1e-12,
          f"{be.cycle_cost_eur_per_efc(10.0):.4f} EUR/EFC")


def test_naive_forecasters():
    """F9. A fit-free forecaster must read the past and only the past.

    The naive roster is the study's control group: every claim of the form
    "Prophet is/is not worth it" is a comparison against these. A baseline that
    peeks is not a weak forecaster, it is a strong one wearing the wrong label,
    and it would make the expensive model look worse than it is. So causality is
    checked by measurement here rather than by reading the slice arithmetic.
    """
    _, kw = load(n_days=40)
    anchor_i = 20 * H                       # 20 days of history behind it
    anchor = kw.index[anchor_i]
    built = {k: b(kw, H) for k, b in hs.SIMPLE_KINDS.items()}

    # The fitted kinds belong in this test too, and are the reason it matters
    # most: they are the only forecasters here that BOTH fit a model and read
    # recent actuals, so they have two ways to peek instead of one. The upstream
    # method this port comes from seeds its forecast with the realised value AT
    # the anchor -- which is `leak_current_interval`, the thing this study keeps
    # behind a flag and measures. If that seeding survived the port it would
    # show up here and nowhere else.
    #
    # Fit on the first 10 days so the AR has a training block, and keep it tiny:
    # this is a causality check, not an accuracy one.
    fitted_kw = dict(n_train=10 * H, max_ar_samples=400, refit_every_days=None)
    built.update({
        "hbd":          hs.HbdForecaster(kw, H, use_ar=True, **fitted_kw),
        "hbd_baseline": hs.HbdForecaster(kw, H, use_ar=False, **fitted_kw),
    })

    def rebuild(kind, frame):
        if kind in hs.SIMPLE_KINDS:
            return hs.SIMPLE_KINDS[kind](frame, H)
        return hs.HbdForecaster(frame, H, use_ar=(kind == "hbd"), **fitted_kw)

    # --- causality. Poison everything from the anchor onwards; a forecaster
    # that reads any of it produces NaN, and NaN != NaN survives any comparison.
    poisoned = kw.copy()
    poisoned.iloc[anchor_i:, poisoned.columns.get_indexer(
        ["Energy_Consumption", "Energy_Generation"])] = np.nan
    peeked = []
    for kind, fc in built.items():
        clean = fc.predict_next_day(anchor, H)
        blind = rebuild(kind, poisoned).predict_next_day(anchor, H)
        same = np.allclose(clean[["yhat_con", "yhat_gen"]].to_numpy(),
                           blind[["yhat_con", "yhat_gen"]].to_numpy(),
                           equal_nan=False)
        if not same:
            peeked.append(kind)
    check("forecasters: none reads at or after its anchor",
          not peeked, f"peeked: {', '.join(peeked)}" if peeked else
          f"{len(built)} kinds blinded from {anchor}")

    # --- the contract every caller relies on.
    bad = []
    for kind, fc in built.items():
        day = fc.predict_next_day(anchor, H)
        ok = (list(day.columns) == ["ds", "yhat_con", "yhat_gen"]
              and len(day) == H
              and pd.DatetimeIndex(day["ds"]).tz is None
              and (day[["yhat_con", "yhat_gen"]].to_numpy() >= 0).all()
              and (pd.DatetimeIndex(day["ds"])
                   == kw.index[anchor_i:anchor_i + H].tz_localize(None)).all())
        if not ok:
            bad.append(kind)
    check("forecasters: H rows, tz-naive ds, non-negative, named channels",
          not bad, f"violated by: {', '.join(bad)}" if bad else f"{len(built)} kinds")

    # --- the lags are the lags the names claim.
    for kind, days in (("persistence", 1), ("weekly", 7)):
        day = built[kind].predict_next_day(anchor, H)
        src = kw.iloc[anchor_i - days * H:anchor_i - days * H + H]
        check(f"{kind}: copies the day {days} day(s) before the anchor",
              np.allclose(day["yhat_con"], src["Energy_Consumption"])
              and np.allclose(day["yhat_gen"], src["Energy_Generation"]))

    # --- day-type match: the source day is the same type as the anchor day, and
    # it is the most recent such day.
    wrong = []
    for offset in range(7):                 # one anchor per weekday
        a_i = anchor_i + offset * H
        a = kw.index[a_i]
        day = built["daytype"].predict_next_day(a, H)
        lag = next((d for d in range(1, 8)
                    if np.allclose(day["yhat_con"],
                                   kw["Energy_Consumption"]
                                   .iloc[a_i - d * H:a_i - d * H + H])), None)
        weekend = kw.index[a_i].dayofweek in (5, 6)
        if lag is None or (kw.index[a_i - lag * H].dayofweek in (5, 6)) != weekend:
            wrong.append(f"{a:%a}")
    check("daytype: weekday from weekday, weekend from weekend",
          not wrong, f"mismatched on {', '.join(wrong)}" if wrong else
          "7 consecutive anchors, each matched within 7 days")

    # --- climatology: the median rejects one anomalous day, the mean does not.
    idx = pd.date_range("2012-01-01 00:30", periods=15 * H, freq="30min",
                        tz="UTC")
    shape = np.tile(np.linspace(0.2, 1.2, H), 15)
    synth = pd.DataFrame({"SMP": 0.1, "Energy_Consumption": shape,
                          "Energy_Generation": shape}, index=idx)
    a_i = 14 * H
    synth.iloc[(a_i - 3 * H):(a_i - 2 * H),
               synth.columns.get_indexer(["Energy_Consumption",
                                          "Energy_Generation"])] = 50.0
    normal = np.linspace(0.2, 1.2, H)
    med = hs.ClimatologyForecaster(synth, H, 7, "median").predict_next_day(
        synth.index[a_i], H)
    avg = hs.ClimatologyForecaster(synth, H, 7, "mean").predict_next_day(
        synth.index[a_i], H)
    check("median7: one anomalous day does not move the forecast",
          np.allclose(med["yhat_gen"], normal),
          f"max deviation {np.abs(med['yhat_gen'] - normal).max():.3e}")
    check("mean7: the same day does move it -- the two are not the same method",
          not np.allclose(avg["yhat_gen"], normal),
          f"mean is {avg['yhat_gen'].mean():.2f} vs {normal.mean():.2f} kW")

    # --- an unregistered kind must fail loudly. It used to fall through to
    # Prophet, so a typo'd arm ran a whole sweep and reported Prophet's numbers
    # under a name nobody had implemented.
    import tempfile
    raised = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hs.load_or_build_forecasts(
                "unit-test", kw.iloc[:10 * H], kw.iloc[10 * H:], H, "30min",
                hs.EnergyForecaster(), cache_dir=tmp, kind="no-such-method",
                history=kw.iloc[:10 * H])
    except ValueError:
        raised = True
    check("an unknown forecaster kind raises instead of serving Prophet", raised)


def test_one_clock(kwh):
    """F10. The rules and the tariff must read the SAME clock, and it must be
    the household's own.

    Two conversions were stacked on a series that needed neither. The Ausgrid
    files are stamped `Timestamp_UTC` with a `+00:00` suffix they never earned:
    they are local NSW wall-clock readings, DST included -- measured, the PV
    centroid steps 12.49 -> 13.13 on 2012-10-07 and 13.12 -> 12.32 on
    2013-04-07, the first Sunday in October and the first Sunday in April.
    `TariffCalculator` then converted them to Australia/Sydney (+10/+11 h) and
    `si_cas` gave every rule a Europe/Ljubljana clock (+1/+2 h), so a clock rule
    and the bill it was scored against sat ~9 hours apart on the same interval.

    Two things are checked, and the second is the one that bit: that the rules'
    clock is the stamp, and that the EA025 peak rate lands on the evening.
    """
    import si_cas as sc
    sc.nastavi_koledar(drzava="AU", podrocje="NSW", visja_sezona_meseci={5, 6, 7, 8},
                       casovni_pas="naive")
    hs.TariffCalculator.LOCAL_TZ = None
    hs.TariffCalculator.HOLIDAY_COUNTRY, hs.TariffCalculator.HOLIDAY_SUBDIV = "AU", "NSW"

    env = make_env(kwh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env, n_steps=len(kwh))
    stamp_hour = kwh.index.hour + kwh.index.minute / 60.0
    check("the rules' clock is the household's own stamp, unconverted",
          np.allclose(sig.local_hour, stamp_hour),
          f"max drift {np.max(np.abs(sig.local_hour - stamp_hour)):.2f} h")

    imp = np.asarray(hs.build_rate_vectors(
        "AU", env, kwh.index, kwh["SMP"].values, 30)[0], dtype=float)
    hour = np.floor(sig.local_hour).astype(int)
    by_hour = pd.Series(imp).groupby(hour).mean()
    dearest = set(by_hour.nlargest(6).index)
    check("the EA025 peak rate lands on local 15:00-21:00",
          dearest == {15, 16, 17, 18, 19, 20}, f"dearest hours {sorted(dearest)}")
    # And the roof is overhead at midday, which is the physical cross-check that
    # says the clock is the right one rather than merely a consistent one.
    peak_gen = int(pd.Series(sig.generation).groupby(hour).mean().idxmax())
    check("PV generation peaks around local noon", 11 <= peak_gen <= 14,
          f"peak at {peak_gen}:00")


def test_rules_read_the_price_they_pay(kwh):
    """F14. A price rule must decide against the tariff it is billed under.

    `rbc.build_signals` derived `sig.import_rate` from `env.pricing_scheme`,
    which the Ausgrid arm also sets to `si_samooskrba` -- its environment exists
    for the battery and the calendar, not for its prices. So `price_threshold`,
    `price_rank_daily`, `tariff_arbitrage` and `price_oracle` ranked their
    intervals on a Slovenian dynamic list while paying Ausgrid EA025, and could
    not see the 0.0270 / 0.0720 / 0.2360 network steps that are the whole of
    that tariff's signal. Measured on the fixture window below: the two series
    correlate 0.66 -- the number this test prints, and the one the notebook's
    caveats quote -- and the rules were worth ~9x more once shown the right one.
    """
    for tariff in ("AU", "SI"):
        env = make_env(kwh)
        rates = hs.build_rate_vectors(tariff, env, kwh.index, kwh["SMP"].values, 30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = rbc.build_signals(env, n_steps=len(kwh), rates=rates)
        check(f"{tariff}: the rules' price signal IS the arm's delivered rate",
              np.allclose(sig.import_rate, np.asarray(rates[0][:len(kwh)])),
              f"worst gap {np.max(np.abs(sig.import_rate - np.asarray(rates[0][:len(kwh)]))):.2e}")

    # And the settlement charges what the rule was shown. `price_interval` on SI
    # and `make_au_settlement` on AU are different functions; the check that
    # matters is that neither is fed a series the rule never saw.
    env = make_env(kwh)
    au = hs.build_rate_vectors("AU", env, kwh.index, kwh["SMP"].values, 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env_derived = rbc.build_signals(env, n_steps=len(kwh))
        arm = rbc.build_signals(env, n_steps=len(kwh), rates=au)
    check("AU: the old env-derived signal was a different series entirely",
          np.corrcoef(env_derived.import_rate, arm.import_rate)[0, 1] < 0.9,
          f"correlation {np.corrcoef(env_derived.import_rate, arm.import_rate)[0, 1]:.2f}, "
          f"means {env_derived.import_rate.mean():.4f} vs {arm.import_rate.mean():.4f}")


def test_tuned_parameters_reach_the_rules(kwh):
    """F15. What `RULE_PARAMS` says a rule is tuned to, the rule must be built with.

    A tuning table nothing reads is worse than none: it documents a claim about
    the study that the study does not implement.
    """
    for tariff in ("AU", "SI"):
        built = {p.name: p for p in hs.rule_roster(tariff)}
        for name, params in hs.RULE_PARAMS.get(tariff, {}).items():
            pol = built.get(name)
            if pol is None:
                check(f"{tariff}: {name} is tuned but not on this roster", False)
                continue
            wrong = {k: (getattr(pol, k, None), v) for k, v in params.items()
                     if getattr(pol, k, None) != v}
            check(f"{tariff}: {name} is built with its tuned parameters",
                  not wrong, str(wrong) if wrong else "")
    fs = {p.name: p for p in hs.rule_roster("AU")}["fixed_schedule"]
    check("AU: fixed_schedule is built with its tuned windows",
          (fs.charge_hours, fs.discharge_hours)
          == (hs.FIXED_SCHEDULE_WINDOWS["AU"]["charge_hours"],
              hs.FIXED_SCHEDULE_WINDOWS["AU"]["discharge_hours"]))


def test_discharge_window_is_live(kwh):
    """F11. `FixedSchedule.discharge_hours` must change the answer.

    It could not. Outside both windows the rule fell back to
    `_self_consumption`, and `_cover_load` is a strict subset of that -- so the
    pack was already emptied into whatever deficit came first, and naming a
    discharge window moved no discharge into it. Measured over 60 days, sliding
    that window across the day changed the discharge timing in 1 interval of
    2880 and the bill by 0.82 EUR; with `hold_between` it moves the bill by
    58.80. The ratio is the invariant, not either number.
    """
    env = make_env(kwh)
    rates = hs.build_rate_vectors("AU", env, kwh.index, kwh["SMP"].values, 30)
    settle = hs.build_settlement("AU", env, rates, DELTA_T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env, n_steps=len(kwh))

    def cost(dis, hold=True):
        pol = rbc.FixedSchedule(charge_hours=(11.0, 15.0), discharge_hours=dis,
                                respect_peak=False, hold_between=hold)
        return rbc.run_policy(env, pol, signals=sig, settle=settle,
                              soc_init_kwh=4.0)["Cost_EUR_Closed"]

    peak_window, off_window = cost((15.0, 21.0)), cost((2.0, 6.0))
    check("moving the discharge window moves the bill",
          abs(peak_window - off_window) > 0.01,
          f"peak {peak_window:.2f} vs off-peak {off_window:.2f} EUR")
    check("discharging into the EA025 peak beats discharging off-peak",
          peak_window < off_window)
    # The old behaviour stays reachable, and stays nearly deaf to the window,
    # which is the evidence that `hold_between` is what makes it mean anything.
    old_spread = abs(cost((15.0, 21.0), hold=False) - cost((2.0, 6.0), hold=False))
    new_spread = abs(peak_window - off_window)
    check("hold_between is what the discharge window acts through",
          new_spread > 10 * max(old_spread, 1e-9),
          f"old behaviour spreads {old_spread:.2f} EUR, new one {new_spread:.2f}")


def test_wear_reaches_the_objective(kwh, kw):
    """F12. A wear price must change what the MILP DOES, not just what is
    reported.

    `cycle_cost_eur_per_efc` was set on the environment, recorded in the run
    config and described as a shadow price "in its objective", while
    `UpstreamMILPScheduler.solve` built its objective out of `buy` and `sell`
    alone. Turning it on changed the checkpoint key and nothing else.
    """
    rates = None
    cycled = {}
    for rate in (0.0, 5.0):
        env = hs.align_envelope(
            hs.build_study_env(kwh, delta_t=DELTA_T, H=H,
                               cycle_cost_eur_per_efc=(rate or None), **BATTERY),
            BATTERY["p_max"], BATTERY["eff"], DELTA_T)
        rates = hs.build_rate_vectors("AU", env, kwh.index, kwh["SMP"].values, 30)
        sched = hs.UpstreamMILPScheduler(
            env, delta_t=DELTA_T, parity=False, exclusivity="auto",
            allow_spill=False, metering_bounds=True, **BATTERY)
        out = sched.solve(
            soc_init=5.0, buy_rate=list(rates[0][:H]), sell_rate=list(rates[1][:H]),
            p_gen=list(kw["Energy_Generation"].values[:H]),
            p_con=list(kw["Energy_Consumption"].values[:H]))
        cycled[rate] = sum(out["x_ch"]) * DELTA_T
    check("a wear price makes the MILP cycle less",
          cycled[5.0] < cycled[0.0] - 1e-6,
          f"{cycled[0.0]:.3f} kWh charged unpriced, {cycled[5.0]:.3f} kWh at 5 EUR/EFC")


def test_full_period_is_a_bound(kwh):
    """F13. The whole-period solve must be below everything it is the bound for.

    It is the denominator of every `gain_share_pct`, so a controller beating it
    is not an interesting result, it is a missing term in its objective. Two
    were missing when it was first written, and both are checked here by the
    fact that this passes: the terminal close-out (priced at the evaluator's own
    mean rate, not the arm's) and, on SI, the endogenous contract, which
    `settle_trajectory` was not converging for the MILP arms.

    SI is checked on a THREE-MONTH slice, not the ten-day one. Below the
    contract lag no month in the window reads its line from another month in it,
    so the solve cannot price its own standing charge while the rules converge
    theirs by re-running -- and it loses by 0.79 EUR for that reason alone. The
    study's arms are 365 days; `full_period_bound_check` reports the short case
    as a note rather than claiming a bound it cannot have.
    """
    for tariff, window in (("AU", kwh), ("SI", load(n_days=92)[0])):
        env = make_env(window)
        kwh = window
        rates = hs.build_rate_vectors(tariff, env, kwh.index, kwh["SMP"].values, 30)
        settle = hs.build_settlement(tariff, env, rates, DELTA_T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = rbc.build_signals(env, n_steps=len(kwh))
        soc_min = BATTERY["battery_cap"] * BATTERY["soc_min_pct"]

        full = hs.solve_full_period(
            env, rates, tariff, n_steps=len(kwh), soc_init_kwh=5.0,
            delta_t=DELTA_T, soc_min_kwh=soc_min,
            closeout_rate=float(np.mean(sig.import_rate)), verbose=False)
        net = (np.asarray(full["p_buy"]) - np.asarray(full["p_sell"])) * DELTA_T
        s = hs.settle_trajectory(env, net, settle, sig, soc_start=5.0,
                                 soc_end=full["soc_plan"][-1])
        efc = (sum(full["x_ch"]) * DELTA_T * BATTERY["eff"]
               + sum(full["x_dis"]) * DELTA_T / BATTERY["eff"]) / (2 * BATTERY["battery_cap"])
        wear = float(env.cycle_cost_eur_per_efc or 0.0)
        opt = s["Cost_EUR_Closed"] + wear * efc + (
            s["Fixed_EUR"] if tariff == "SI" else 0.0)

        metrics, _ = hs.run_rules(env, settle, tariff, signals=sig, soc_init_kwh=4.0)
        worst = None
        for name in sorted(k[5:] for k in metrics if k.startswith("cost_")):
            total = metrics[f"cost_{name}"] + wear * metrics[f"efc_{name}"] + (
                metrics[f"fixed_{name}"] if tariff == "SI" else 0.0)
            if worst is None or total < worst[1]:
                worst = (name, total)
        check(f"{tariff}: the whole-period solve bounds every rule",
              opt <= worst[1] + 0.01,
              f"optimum {opt:.2f} vs best rule {worst[0]} {worst[1]:.2f}")


if __name__ == "__main__":
    kwh, kw = load()
    print(f"Ausgrid 127, {len(kwh)} steps ({len(kwh) // H} days)\n")
    test_site_selection()
    test_wear_cost()
    test_one_battery(kwh)
    test_milp_parity(kwh, kw)
    test_one_evaluator(kwh, kw)
    test_si_settlement_matches_upstream(kwh)
    test_rules_stay_in_the_envelope(kwh)
    test_shared_starting_soc(kwh)
    test_naive_forecasters()
    test_one_clock(kwh)
    test_rules_read_the_price_they_pay(kwh)
    test_tuned_parameters_reach_the_rules(kwh)
    test_discharge_window_is_live(kwh)
    test_wear_reaches_the_objective(kwh, kw)
    test_full_period_is_a_bound(kwh)
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED: " + ", ".join(_failed))
    sys.exit(1 if _failed else 0)
