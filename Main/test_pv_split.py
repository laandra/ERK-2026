"""The invariants the three-notebook split rests on.

    python test_pv_split.py

`CODE.ipynb` swept all 35 arms in one kernel, so "which notebook owns this arm"
was not a question anyone could get wrong. It is now, and getting it wrong does
not fail loudly: two notebooks that both claim an arm write one checkpoint from
two kernels, and what comes out is whichever process finished last. So the
partition is checked here rather than trusted.

Every check is about one of two things:

  THE PARTITION   exactly one notebook owns each arm, and between them they own
                  all of it. That is what makes concurrent runs safe.
  THE PAIRING     a `perfect` arm and its `forecast` twin differ in the
                  generation channel and in nothing else. That is what makes the
                  difference between them a measurement rather than a comparison
                  of two studies.

No sweep, no cache, no data: this reads `STUDY_ARMS` and the functions over it.
"""

import sys

import hems_study as hs          # first: it puts the repo root on sys.path
import pv_split as ps

_passed, _failed = [], []


def check(name, ok, detail=""):
    (_passed if ok else _failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


def test_the_partition_is_a_partition():
    """Every arm in exactly one notebook -- the concurrency invariant.

    Disjoint AND covering. Disjoint is the one that costs a result if it breaks;
    covering is the one that loses an arm silently, since a notebook only ever
    asks for its own track and nothing enumerates what nobody asked for.
    """
    print("\n-- the partition")
    owners = {}
    for track in ps.TRACKS:
        for name in ps.arm_names(track):
            owners.setdefault(name, []).append(track)
    roster = [a["name"] for a in hs.STUDY_ARMS]

    shared = {n: t for n, t in owners.items() if len(t) > 1}
    check("no arm is owned by two notebooks", not shared, str(shared))

    orphans = [n for n in roster if n not in owners]
    check("no arm is owned by none", not orphans, ", ".join(orphans))

    check("the tracks cover the roster exactly",
          set(owners) == set(roster),
          f"{len(owners)} owned vs {len(roster)} in STUDY_ARMS")

    total = sum(len(ps.arm_names(t)) for t in ps.TRACKS)
    check("the counts add up", total == len(roster), f"{total} vs {len(roster)}")

    # A TRACK PER NOTEBOOK, and a figure directory per track. Two notebooks
    # sharing a subdir is not a data race, but it is the same failure to a
    # reader: they draw the same figure NAMES over different arms, so the last
    # one run owns every export in it.
    subdirs = [ps.FIGURE_SUBDIR[t] for t in ps.TRACKS]
    check("each track exports to its own directory",
          len(set(subdirs)) == len(subdirs), ", ".join(subdirs))


def test_the_classification_is_what_it_claims():
    """A track's name has to be true of every arm in it."""
    print("\n-- what each track means")
    for name in ps.arm_names(ps.FORECAST):
        con, gen = ps.channel_sources(ps.arm_forecast_kind(name))
        check(f"{name}: one method answers both channels", con == gen,
              f"load={con} roof={gen}")
    for name in ps.arm_names(ps.PERFECT):
        con, gen = ps.channel_sources(ps.arm_forecast_kind(name))
        check(f"{name}: the roof is the truth and the load is not",
              gen == "truth" and con != "truth", f"load={con} roof={gen}")
    for name in ps.arm_names(ps.MIXED):
        con, gen = ps.channel_sources(ps.arm_forecast_kind(name))
        check(f"{name}: channels differ and neither is the truth",
              con != gen and gen != "truth", f"load={con} roof={gen}")


def test_the_pairs_differ_in_one_channel():
    """The measurement in `CODE_PV_VALUE.ipynb`, checked on the SPEC.

    The notebook also verifies it on the RESULTS -- the forecast-blind columns
    must come out bit-identical across a pair -- which is the stronger check but
    needs a finished sweep. This one needs nothing and catches the same mistake
    at the point it would be made: an arm added to the roster whose "twin"
    differs in the horizon, the tariff or the wear price as well as in the roof.
    """
    print("\n-- the pairs")
    ignore = {"name", "forecaster_kind"}
    for tariff in sorted(hs.REFERENCE_ARM):
        pairs = ps.pv_pairs(tariff)
        check(f"{tariff}: the pairing is non-empty", bool(pairs), f"{len(pairs)} pairs")
        for kind, base, truth in pairs:
            a = {k: v for k, v in ps.arm_spec(base).items() if k not in ignore}
            b = {k: v for k, v in ps.arm_spec(truth).items() if k not in ignore}
            check(f"{base} vs {truth}: same tariff, horizon and objective",
                  a == b, f"{a} vs {b}")
            check(f"{base} vs {truth}: the same load model on both sides",
                  ps.load_kind(ps.arm_forecast_kind(base))
                  == ps.load_kind(ps.arm_forecast_kind(truth)) == kind)
            check(f"{base} vs {truth}: one forecast roof, one perfect roof",
                  ps.channel_sources(ps.arm_forecast_kind(base))[1] != "truth"
                  and ps.channel_sources(ps.arm_forecast_kind(truth))[1] == "truth")
        # Both tariffs have to pair the SAME load models, or a figure with one
        # panel per tariff has different rows in each and the rows stop lining up.
        other = sorted(set(hs.REFERENCE_ARM) - {tariff})
        for t2 in other:
            check(f"{tariff} and {t2} pair the same load models",
                  [k for k, _, _ in pairs] == [k for k, _, _ in ps.pv_pairs(t2)])


def test_the_ladder_pins_the_load_channel():
    """The roof-quality ladder moves ONE channel, which is the whole of it."""
    print("\n-- the ladder")
    for tariff in sorted(hs.REFERENCE_ARM):
        rungs = ps.roof_ladder(tariff)
        check(f"{tariff}: the ladder has rungs", len(rungs) >= 2, f"{len(rungs)}")
        pinned = {ps.channel_sources(ps.arm_forecast_kind(a))[0] for _, a in rungs}
        check(f"{tariff}: the load channel is the same on every rung",
              len(pinned) == 1, str(pinned))
        roofs = [g for g, _ in rungs]
        check(f"{tariff}: no roof appears twice", len(set(roofs)) == len(roofs),
              ", ".join(roofs))
        check(f"{tariff}: the top rung is the truth", roofs[-1] == "truth",
              ", ".join(roofs))


def test_the_reference_arms_are_the_same_load_model():
    """The two tracks have to be answering the same question.

    `reference_arm` derives the perfect track's reference from
    `hs.REFERENCE_ARM` so that the two notebooks' headline figures differ in the
    roof and in nothing else. Hardcoding either one is how they drift into
    comparing a Prophet study against a median-14 study.
    """
    print("\n-- the reference arms")
    for tariff, base in hs.REFERENCE_ARM.items():
        f = ps.reference_arm(ps.FORECAST)[tariff]
        p = ps.reference_arm(ps.PERFECT)[tariff]
        check(f"{tariff}: the forecast track keeps hs.REFERENCE_ARM", f == base,
              f"{f} vs {base}")
        check(f"{tariff}: the perfect track carries the same load model",
              ps.load_kind(ps.arm_forecast_kind(p))
              == ps.load_kind(ps.arm_forecast_kind(f)),
              f"{p} vs {f}")
        check(f"{tariff}: and it is owned by the perfect track",
              p in set(ps.arm_names(ps.PERFECT)), p)


def test_the_labels_name_the_right_forecaster():
    """`prophet` is a CONTROLLER key, not a claim that the arm ran Prophet.

    This is the mislabel the split created: `CONTROLLER_ALGORITHM` spells the
    forecast-driven MPC "MPC-MILP {horizon}, Prophet forecast", which is false on
    every arm of the perfect track and on most of the forecast track.
    """
    print("\n-- the labels")
    for tariff, base in hs.REFERENCE_ARM.items():
        check(f"{tariff}: the deployable reference still says Prophet",
              ps.mpc_name(base) == "MILP+Prophet", ps.mpc_name(base))
        p = ps.reference_arm(ps.PERFECT)[tariff]
        check(f"{tariff}: the perfect reference says so in its name",
              "perfect PV" in ps.mpc_name(p), ps.mpc_name(p))
    for name in ps.arm_names(ps.PERFECT):
        check(f"{name}: names a perfect roof",
              ps.forecast_label(ps.arm_forecast_kind(name)).endswith("perfect PV"),
              ps.forecast_label(ps.arm_forecast_kind(name)))


def test_the_screens_have_one_writer():
    """Both shared screens resolve to a path under the one results tree.

    They are keyed on the kind and the household and know nothing about a tariff
    or an arm, so three notebooks rescoring them would write three identical
    files at one path -- and two at once would write a truncated one.
    """
    print("\n-- the shared screens")
    import os
    for which in ps.SCREEN_FILES:
        path = ps.screen_path(which)
        check(f"{which}: lives under the shared results dir",
              os.path.dirname(os.path.abspath(path))
              == os.path.abspath(hs.RESULTS_DIR), path)
    # And the caches the three notebooks SHARE are anchored to the module, not
    # to the cwd -- a cache that moves with the working directory is a cache that
    # misses the moment a notebook is opened from somewhere else.
    here = os.path.dirname(os.path.abspath(hs.__file__))
    for label, path in (("forecast", hs.FORECAST_CACHE_DIR),
                        ("oracle", hs.ORACLE_CACHE_DIR),
                        ("hbd params", hs.HbdForecaster.PARAM_CACHE_DIR)):
        check(f"the {label} cache is anchored to hems_study.py, not the cwd",
              os.path.abspath(path).startswith(here), path)


def test_the_sweep_summary_is_named_for_its_arms():
    """Three notebooks writing "summary_all_arms.csv" is a name that lies."""
    print("\n-- the sweep roll-up")
    full = hs.sweep_summary_path("out", hs.STUDY_ARMS)
    check("the full roster keeps the old name",
          full.endswith("summary_all_arms.csv"), full)
    paths = {t: hs.sweep_summary_path("out", ps.arm_specs(t)) for t in ps.TRACKS}
    check("a subset does not", all(not p.endswith("summary_all_arms.csv")
                                   for p in paths.values()), str(paths))
    check("and no two tracks collide", len(set(paths.values())) == len(paths),
          str(paths))


if __name__ == "__main__":
    test_the_partition_is_a_partition()
    test_the_classification_is_what_it_claims()
    test_the_pairs_differ_in_one_channel()
    test_the_ladder_pins_the_load_channel()
    test_the_reference_arms_are_the_same_load_model()
    test_the_labels_name_the_right_forecaster()
    test_the_screens_have_one_writer()
    test_the_sweep_summary_is_named_for_its_arms()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED: " + ", ".join(_failed))
    sys.exit(1 if _failed else 0)
