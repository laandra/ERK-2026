"""Where `FIXED_SCHEDULE_WINDOWS` came from, and how to get it again.

    python tune_fixed_schedule.py AU        # full (start x length) sweep
    python tune_fixed_schedule.py SI        # shortlist; the full sweep costs hours

`fixed_schedule` is the one controller in the study whose behaviour is a free
parameter -- two windows on a clock -- and it was carrying a pair nobody had
searched, chosen moreover against a broken clock (see F10 in `hems_study`). This
is the search that replaced them, kept in the repo rather than in a notebook
cell so the numbers in that table can be checked rather than believed.

Scored on saving NET OF WEAR, at the pack price over its rated cycle life, and
on SI including the standing charge -- because the dogovorjena moc is endogenous
there, so a controller's peaks move its own fixed charge. A window that buys
20 EUR of energy with sixty extra cycles is not an improvement, and on SI a
window that raises the contract is a loss however cheap its energy was.

AU is swept exhaustively (96 windows per stage, two stages, four households).
SI is not: every SI policy run converges the agreed power, which makes the same
sweep a multi-hour job, so it crosses the windows the block schedule and the PV
profile actually nominate. The conclusion there did not turn out to be sensitive
to the resolution -- every window in the shortlist loses money net of wear.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hems_study as hs                                        # noqa: E402
import Rule_Based_Control as rbc                               # noqa: E402
import Battery_Economics as be                                 # noqa: E402
import si_cas as sc                                            # noqa: E402

HOUSEHOLDS = ["Ausgrid 104", "Ausgrid 113", "Ausgrid 127", "Ausgrid 128"]
BATTERY = dict(battery_cap=10.0, soc_min_pct=0.10, soc_max_pct=0.80,
               p_max=1.5, eff=0.95)
WEAR = be.cycle_cost_eur_per_efc(BATTERY["battery_cap"])


def contexts(tariff, households, n_sim=365, data_dir=None):
    """One priced environment per household, built exactly as the sweep builds it."""
    data_dir = data_dir or os.path.join("..", "Input data", "Ausgrid")
    # The study's calendar and the study's clock: tuning under any other is
    # tuning a different controller.
    sc.nastavi_koledar(drzava="AU", podrocje="NSW",
                       visja_sezona_meseci={5, 6, 7, 8}, casovni_pas="naive")
    hs.TariffCalculator.LOCAL_TZ = None
    hs.TariffCalculator.HOLIDAY_COUNTRY = "AU"
    hs.TariffCalculator.HOLIDAY_SUBDIV = "NSW"

    out = []
    for name in households:
        fr = hs.load_study_frames(os.path.join(data_dir, f"{name}.csv"),
                                  n_sim=n_sim, verbose=False)
        env = hs.align_envelope(
            hs.build_study_env(fr["df_ctrl_kwh"], delta_t=0.5, H=48,
                               cycle_cost_eur_per_efc=WEAR, **BATTERY),
            BATTERY["p_max"], BATTERY["eff"], 0.5)
        rates = hs.build_rate_vectors(tariff, env, fr["df_ctrl"].index,
                                      fr["df_ctrl"]["SMP"].values, 30)
        settle = hs.build_settlement(tariff, env, rates, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The ARM's rates. Without this the bundle prices itself off
            # `env.pricing_scheme`, which is Slovenian on both arms, and every
            # price rule would be tuned against a tariff it is not billed under
            # -- the exact bug this harness exists to re-tune after.
            sig = rbc.build_signals(env, n_steps=len(fr["df_sim"]), rates=rates)
        base = rbc.run_policy(env, rbc._Idle(), signals=sig, settle=settle,
                              soc_init_kwh=4.0)
        out.append((name, env, settle, sig, base))
        print(f"  {name}: no battery {base['Cost_EUR_Closed']:.2f} "
              f"+ fixed {base['Fixed_EUR']:.2f}", flush=True)
    return out


def score(ctx, tariff, charge_hours, discharge_hours):
    """Mean saving net of wear over the households, and the mean wear and peak."""
    rows = []
    for _name, env, settle, sig, base in ctx:
        pol = rbc.FixedSchedule(charge_hours=charge_hours,
                                discharge_hours=discharge_hours,
                                respect_peak=(tariff == "SI"), hold_between=True)
        out = rbc.run_policy(env, pol, signals=sig, settle=settle, soc_init_kwh=4.0)
        # On SI the standing charge is part of what a controller decides, so it
        # is part of what a controller is scored on. On AU it is a constant and
        # cancels; including it either way keeps one formula.
        ref = base["Cost_EUR_Closed"] + base["Fixed_EUR"]
        got = (out["Cost_EUR_Closed"] + out["Fixed_EUR"]
               + WEAR * out["Equivalent_Full_Cycles"])
        rows.append((ref - got, out["Equivalent_Full_Cycles"], out["Peak_Import_kW"]))
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()


def window(start, length):
    return float(start), float((start + length) % 24)


def sweep_au(ctx):
    """Two stages: charge given a seed discharge, then discharge given the winner.

    Crossing both exhaustively is 9,216 combinations per household; two stages
    is 192 and found the pair the tariff's own shape nominates, which is the
    evidence that the coarser search did not miss anything structural.
    """
    seed = (15.0, 21.0)
    stage1 = [{"start": s, "len": L,
               **dict(zip(("saving_net", "EFC", "peak_kW"),
                          score(ctx, "AU", window(s, L), seed)))}
              for s in range(24) for L in (3, 4, 5, 6)]
    best = max(stage1, key=lambda r: r["saving_net"])
    charge = window(best["start"], best["len"])
    stage2 = [{"start": s, "len": L,
               **dict(zip(("saving_net", "EFC", "peak_kW"),
                          score(ctx, "AU", charge, window(s, L))))}
              for s in range(24) for L in (3, 4, 5, 6)]
    return charge, stage1, stage2


# The windows SI's block schedule and the PV profile nominate. Blocks 07-14 and
# 16-20 are the dear ones on a working day; 00-06 and 22-24 the cheap ones; the
# roof runs 10-16.
SI_CHARGE = [(11., 15.), (10., 15.), (0., 6.), (2., 6.), (5., 8.), (13., 16.),
             (22., 4.), (1., 5.)]
SI_DISCHARGE = [(16., 20.), (7., 14.), (18., 22.), (15., 21.), (20., 22.)]


def sweep_si(ctx):
    rows = []
    for ch in SI_CHARGE:
        for dis in SI_DISCHARGE:
            s, efc, pk = score(ctx, "SI", ch, dis)
            rows.append({"charge": f"{int(ch[0]):02d}-{int(ch[1]):02d}",
                         "discharge": f"{int(dis[0]):02d}-{int(dis[1]):02d}",
                         "saving_net": s, "EFC": efc, "peak_kW": pk})
            print(".", end="", flush=True)
    print()
    return rows


# The SI table in `hems_study.FIXED_SCHEDULE_WINDOWS` was produced on the first
# TWO households, not four: each SI policy run converges the agreed power, so the
# shortlist alone is already a ~40-minute job at four. Reproduce the committed
# numbers by leaving this alone; widen it if the ranking is ever in doubt.
SI_HOUSEHOLDS = HOUSEHOLDS[:2]


def main(tariff, households=None, n_sim=365):
    households = households or (SI_HOUSEHOLDS if tariff == "SI" else HOUSEHOLDS)
    print(f"Tuning fixed_schedule on {tariff}, {len(households)} households, "
          f"{n_sim} days, wear at {WEAR:.4f} EUR/EFC")
    ctx = contexts(tariff, households, n_sim=n_sim)

    if tariff == "AU":
        charge, stage1, stage2 = sweep_au(ctx)
        for title, rows in (("charge window", stage1), ("discharge window", stage2)):
            d = pd.DataFrame(rows).sort_values("saving_net", ascending=False)
            print(f"\n=== AU {title}, best 8 ===")
            print(d.head(8).round(2).to_string(index=False))
        best = max(stage2, key=lambda r: r["saving_net"])
        chosen = (charge, window(best["start"], best["len"]))
    else:
        d = pd.DataFrame(sweep_si(ctx)).sort_values("saving_net", ascending=False)
        print("\n=== SI shortlist, best 12 ===")
        print(d.head(12).round(2).to_string(index=False))
        top = d.iloc[0]
        chosen = (tuple(float(x) for x in top["charge"].split("-")),
                  tuple(float(x) for x in top["discharge"].split("-")))
        # The comparison that decides whether a clock is worth running at all.
        for name, kw in (("self_consumption", {}),
                         ("self_consumption_peak_shaving", {"ratchet_aware": True})):
            pol = rbc.make_policy(name, **kw)
            rows = []
            for _n, env, settle, sig, base in ctx:
                o = rbc.run_policy(env, pol, signals=sig, settle=settle, soc_init_kwh=4.0)
                rows.append(base["Cost_EUR_Closed"] + base["Fixed_EUR"]
                            - (o["Cost_EUR_Closed"] + o["Fixed_EUR"]
                               + WEAR * o["Equivalent_Full_Cycles"]))
            print(f"  {name:32s} saving net of wear {np.mean(rows):7.2f}")

    old = score(ctx, tariff, (1.0, 5.0), (18.0, 22.0))
    new = score(ctx, tariff, *chosen)
    print(f"\n=== {tariff} chosen: charge {chosen[0]} discharge {chosen[1]} ===")
    print(f"  old 01-05 / 18-22   saving net of wear {old[0]:8.2f}  EFC {old[1]:6.1f}")
    print(f"  new                 saving net of wear {new[0]:8.2f}  EFC {new[1]:6.1f}")
    print(f"\n  in force: {hs.FIXED_SCHEDULE_WINDOWS[tariff]}")
    return chosen


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1] if len(sys.argv) > 1 else "AU",
         n_sim=int(sys.argv[2]) if len(sys.argv) > 2 else 365)
