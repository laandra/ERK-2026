"""Tune every rule that reads a price to the price it is actually billed under.

    python tune_rules.py AU
    python tune_rules.py SI

Companion to `tune_fixed_schedule.py`, which does the same job for the one rule
whose parameters are a clock rather than a quantile; the environment harness is
imported from there so both tune against identically built arms.

WHY THIS EXISTS. `rbc.build_signals` derived `sig.import_rate` from
`env.pricing_scheme`, which the Ausgrid arm also sets to `si_samooskrba` because
its environment exists for the battery and the calendar rather than for its
prices. So `price_threshold`, `price_rank_daily`, `tariff_arbitrage` and
`price_oracle` chose their intervals against a Slovenian dynamic list while
being billed Ausgrid EA025 -- blind to the 0.0270 / 0.0720 / 0.2360 network
steps that are the whole of that tariff's signal. `build_signals` now takes the
arm's own rates, and every quantile and share tuned against the old signal was
tuned against the wrong distribution, so they are all re-searched here.

WHAT IS TUNED, and what is not. Only parameters that describe a price
distribution or a peak distribution: quantiles, trailing windows, the share of a
day a rule is willing to act on, the peak margin. Not `respect_peak` and not
`ratchet_aware`, which are statements about which tariff the rule is on rather
than settings to fit. `self_consumption`, `delayed_pv_charge` and
`tariff_arbitrage` have no parameters at all and appear here only as references.

HONESTY ABOUT FITTING. This is in-sample: a handful of households, the same year
the study scores. With one or two parameters per rule and grids this coarse the
overfitting risk is small, but it is not zero, and a rule tuned on four
households and reported on thirty is a rule with a small optimistic bias. The
per-household spread is printed for exactly that reason -- a parameter that wins
on the mean while losing on half the households has not been learned.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hems_study as hs                                        # noqa: E402
import Rule_Based_Control as rbc                               # noqa: E402
from tune_fixed_schedule import WEAR, contexts, HOUSEHOLDS, SI_HOUSEHOLDS  # noqa: E402


def _quantile_grid():
    """Trailing window x (cheap, dear) quantile pair.

    Both ends are open to 0.50/0.50, where the rule charges below the median and
    discharges above it, because the search hit an edge in each direction on the
    way here and an edge is not an answer.

    (The tight end is a relic of a run made against the wrong signal. Tuned on
    the Slovenian series -- 0.074-0.088 EUR/kWh, nearly flat -- no trade covered
    its cycle and the search ran to the quiet end of every grid. Against EA025's
    0.0270 / 0.0720 / 0.2360 the same rules want to trade an order of magnitude
    more. The tight rows are kept so that contrast stays visible in the table.)
    """
    return [dict(window_days=w, q_low=lo, q_high=hi)
            for w in (3, 7, 14, 30)
            for lo, hi in ((0.02, 0.98), (0.05, 0.95), (0.10, 0.90),
                           (0.15, 0.85), (0.20, 0.80), (0.25, 0.75),
                           (0.30, 0.70), (0.35, 0.65), (0.40, 0.60),
                           (0.45, 0.55), (0.50, 0.50))]


# One grid per rule. Every entry is a complete kwargs dict, so what was searched
# is legible rather than reconstructed from nested loops.
GRIDS = {
    "price_threshold": _quantile_grid(),
    "price_oracle": _quantile_grid(),
    "price_rank_daily": [dict(max_share=s)
                         for s in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20,
                                   0.25, 0.33, 0.50)],
    "peak_shaving": [dict(q_peak=q, margin=m, window_days=w)
                     for q in (0.90, 0.95, 0.98, 0.99)
                     for m in (0.8, 1.0, 1.2)
                     for w in (14, 30)],
    "self_consumption_peak_shaving": [
        dict(q_peak=q, margin=m, reserve_cap_frac=r)
        for q in (0.90, 0.95, 0.98, 0.99)
        for m in (0.8, 1.0, 1.2)
        for r in (0.3, 0.5, 0.7)],
}


def score_policy(ctx, tariff, name, params):
    """`(saving net of wear per household, share of intervals traded)`.

    The second return value is not decoration. Every one of these rules degrades
    into `self_consumption` at the quiet end of its own parameter -- a
    `max_share` under 1/48 makes `int(max_share * 48) == 0` and the day's plan
    comes back empty; a `q_low` far enough down never clears its own threshold.
    And at 0.417 EUR/EFC on a 1.5 kW pack that degenerate setting WINS, because
    on both tariffs the arbitrage available to a rule does not cover the cycle it
    costs. Left to itself the search therefore returns "do not trade" for every
    price rule, and the study ends up with four columns of self-consumption under
    four different names -- which answers a question nobody asked and hides the
    one this roster exists to ask.

    So trading is measured, and `tune` picks the best setting at which the rule
    is still the rule. The degenerate optimum is reported next to it, because
    "not trading beats trading here" is the actual finding and should be stated
    rather than buried in a parameter.
    """
    fixed_args = {}
    if name in hs._PEAK_AWARE_ARGS:
        fixed_args[hs._PEAK_AWARE_ARGS[name]] = (tariff == "SI")
    saving, traded = [], []
    for _n, env, settle, sig, base in ctx:
        pol = rbc.make_policy(name, **fixed_args, **params)
        r = rbc.run_policy(env, pol, signals=sig, settle=settle, soc_init_kwh=4.0,
                           keep_traces=True)
        ref = base["Cost_EUR_Closed"] + base["Fixed_EUR"]
        got = (r["Cost_EUR_Closed"] + r["Fixed_EUR"]
               + WEAR * r["Equivalent_Full_Cycles"])
        saving.append(ref - got)
        # A traded interval is one the rule would not have taken on
        # self-consumption alone: it charges from the grid (there is no surplus
        # to soak) or discharges while the house has no deficit to cover.
        sp = np.asarray(r["_setpoints"], dtype=float)
        grid_charge = (sp > 1e-9) & (sig.surplus[:len(sp)] <= 1e-9)
        export_dis = (sp < -1e-9) & (sig.deficit[:len(sp)] <= 1e-9)
        traded.append(float(np.mean(grid_charge | export_dis)))
    return np.array(saving), float(np.mean(traded))


# A rule that acts on fewer intervals than this is not distinguishable from
# self-consumption at the resolution the study reports, so it is not a setting
# of that rule -- it is a different controller.
MIN_TRADED_SHARE = 0.005


def tune(tariff, households=None, n_sim=365, rules=None):
    households = households or (SI_HOUSEHOLDS if tariff == "SI" else HOUSEHOLDS)
    roster = [p.name for p in hs.rule_roster(tariff)]
    rules = [r for r in (rules or list(GRIDS)) if r in roster and r in GRIDS]

    print(f"Tuning {len(rules)} rule(s) on {tariff}: {', '.join(rules)}")
    print(f"{len(households)} households, {n_sim} days, wear {WEAR:.4f} EUR/EFC")
    ctx = contexts(tariff, households, n_sim=n_sim)

    chosen = {}
    for name in rules:
        rows = []
        for params in GRIDS[name]:
            s, traded = score_policy(ctx, tariff, name, params)
            rows.append({**params, "mean": s.mean(), "worst": s.min(),
                         "traded": traded})
            print(".", end="", flush=True)
        print()
        d = pd.DataFrame(rows).sort_values("mean", ascending=False)
        live = d[d["traded"] >= MIN_TRADED_SHARE]

        print(f"\n=== {tariff} / {name}: best 6 of {len(rows)} ===")
        print(d.head(6).round(3).to_string(index=False))
        if len(live) < len(d):
            top = d.iloc[0]
            if top["traded"] < MIN_TRADED_SHARE:
                print(f"  ! the unconstrained optimum trades on "
                      f"{top['traded'] * 100:.2f} % of intervals -- it is "
                      f"self-consumption wearing this rule's name, and scores "
                      f"{top['mean']:.2f}. Not trading beats trading here.")
        if live.empty:
            print(f"  ! no setting keeps {name} trading; leaving it at its default")
            continue

        best = live.iloc[0]
        chosen[name] = {k: (int(best[k]) if k == "window_days" else float(best[k]))
                        for k in GRIDS[name][0]}
        default = hs.RULE_PARAMS.get(tariff, {}).get(name, {})
        if default:
            base_s, _ = score_policy(ctx, tariff, name, default)
            print(f"  in force {default} -> {base_s.mean():.2f} "
                  f"(worst household {base_s.min():.2f})")
        print(f"  chosen   {chosen[name]} -> {best['mean']:.2f} "
              f"(worst household {best['worst']:.2f}, "
              f"trades on {best['traded'] * 100:.1f} % of intervals)")
    print(f"\nRULE_PARAMS[{tariff!r}] = {chosen}")
    return chosen


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tariff = sys.argv[1] if len(sys.argv) > 1 else "AU"
    n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    tune(tariff, n_sim=n_sim, rules=sys.argv[3:] or None)
