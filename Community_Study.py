"""Tariff comparison for a heterogeneous energy community.

A community of households is assembled from several Fluvius datasets -- some
with PV, some with PV and an EV or a heat pump, some with a battery -- and the
whole community is settled under four commercial arrangements:

    redni      GEN-I Redni       (self-supply list for PV owners)
    aktivni    GEN-I Aktivni     (4-tariff, time-of-use)
    dinamicni  GEN-I Dinamicni   (SIPX + margin)
    souporaba  ZOEE energy sharing between the members

The question is what the community pays as a whole, what its annual peak draw
is, who the extremes are, and how much of the answer depends on owning a
battery.

Each scenario runs the self-supply list for the PV owners and the supplier's
plain supply twin for everyone else, because `si_paketi.preveri_paket` rejects a
samooskrba list for a household with no PV device.

The battery households are dispatched by `MILP_Household.solve_household` and
the trajectory is handed to the settlement as an equivalent (consumption,
generation) pair reproducing its net grid flow. That is exact for the bill and
collapses the `lastna_raba_kwh` diagnostic to zero for those members.

The agreed billing power is a per-block vector that changes every month, and both
the dispatch and the settlement read the same schedule. The solver gap comes from
`Horizon_Comparison` rather than being re-typed.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import Horizon_Comparison as hc
import multi_household_tools as mht
from Environment import agreed_power_schedule_for_profile
from MILP_Household import (
    build_household_env,
    effective_profile as _effective_profile,
    full_period_solver,
    solve_household,
)

# --- Study configuration ---------------------------------------------------
SMP_COUNTRY_ID = hc.SMP_COUNTRY_ID
PRICE_COLUMN = hc.PRICE_COLUMN
GENERATION_COLUMN = hc.GENERATION_COLUMN
CONSUMPTION_COLUMN = hc.CONSUMPTION_COLUMN

PRICING_REFERENCE_YEAR = hc.PRICING_REFERENCE_YEAR
PEAK_RESET_MONTHS = hc.PEAK_RESET_MONTHS

BATTERY_CAPACITY_KWH = 30.0
SOC_FRACTION = hc.SOC_FRACTION
CHARGE_EFFICIENCY = hc.CHARGE_EFFICIENCY
DISCHARGE_EFFICIENCY = hc.DISCHARGE_EFFICIENCY
C_RATE = hc.C_RATE
INVERTER_MAX_KW = hc.INVERTER_MAX_KW
STEPS_PER_DAY = hc.STEPS_PER_DAY

# A community battery is not modelled; the per-member `battery_kwh` field is
# the hook.
COMMUNITY_BATTERY_KWH = 0.0

# Agreed billing power: re-set every month to the peak the previous month
# realized in each block, unbounded, leading month bootstrapped from the last
# complete one. Both the dispatch and the settlement read it.

# --- Community composition -------------------------------------------------
# (dataset, household ids, has_pv, ids that get a battery)
COMMUNITY_SPEC = [
    ("Fluvius", range(1, 21), False, ()),
    ("Fluvius_PV", range(1, 11), True, (1, 2, 3, 4, 5)),
    ("Fluvius_PV_EV", range(1, 11), True, ()),
    ("Fluvius_PV_HP", range(1, 11), True, (1, 2, 3, 4, 5)),
]

# --- Scenarios -------------------------------------------------------------
# Each scenario names the self-supply list for PV owners and its plain supply
# twin for everyone else.
SCENARIOS = {
    "redni": {
        "pv_paket": "GENI_SAMO_REDNI",
        "nonpv_paket": "GENI_REDNI",
        "label": "Redni",
    },
    "aktivni": {
        "pv_paket": "GENI_SAMO_AKTIVNI",
        "nonpv_paket": "GENI_AKTIVNI",
        "label": "Aktivni",
    },
    "dinamicni": {
        "pv_paket": "GENI_SAMO_DINAMICNI",
        "nonpv_paket": "GENI_DINAMICNI",
        "label": "Dinamicni",
    },
    "souporaba": {
        "pv_paket": "GENI_SAMO_DINAMICNI",
        "nonpv_paket": "GENI_DINAMICNI",
        "label": "Souporaba",
    },
}

SCHEME_PV = "si_samooskrba"
SCHEME_NONPV = "si_dobava"

# The souporaba scenario reuses the dinamicni dispatch: only the settlement
# differs.
DISPATCH_ALIAS = {"souporaba": "dinamicni"}

# --- Souporaba parameters --------------------------------------------------
SOUPORABA_DELEZ = 0.4
SOUPORABA_CENA_EUR_KWH = 0.05
SOUPORABA_SERVICE_ID = "GENI_SOUPORABA"

# Re-exported so the notebook can quote the organizer's monthly fees.
STORITVE_SOUPORABE = mht.STORITVE_SOUPORABE

# --- Cost categories -------------------------------------------------------
# Both settlement paths return one VAT-inclusive figure per billing item; this
# groups the items by who is paid, which the "energy + network / fixed" split
# cannot show because the network power charge sits inside "fixed".
COST_CATEGORIES = {
    # Energy bought from a neighbour is still energy, not network.
    "Supply_EUR": ("energija", "energija_skupnost", "energija_souporaba"),
    "Network_EUR": (
        "omreznina_energija",
        "omreznina_energija_skupnost",
        "omreznina_moc",
        "omreznina_presezna_moc",
    ),
    "Levies_EUR": (
        "trosarina",
        "prispevek_ure",
        "prispevek_operater_trga",
        "prispevek_ove_spte",
    ),
    "Supplier_Fee_EUR": ("mesecno_nadomestilo", "nadomestilo_souporaba"),
}

RESULTS_DIR = Path(__file__).resolve().parent / "Results" / "Community_Study"
DISPATCH_DIR = RESULTS_DIR / "dispatch"

_BLOCKS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------
@dataclass
class Member:
    key: str
    dataset: str
    household_id: int
    has_pv: bool
    battery_kwh: float
    segment: str
    contracted_power_kw: Dict[int, float] = field(default_factory=dict)
    # {month id: {block: kW}} -- what actually prices.
    agreed_power_schedule: Dict[int, Dict[int, float]] = field(default_factory=dict)

    @property
    def has_battery(self) -> bool:
        return self.battery_kwh > 0.0


def build_community(spec=None) -> List[Member]:
    """The community roster. Keys are stable across runs and sortable."""
    spec = spec if spec is not None else COMMUNITY_SPEC
    members = []
    for dataset, ids, has_pv, battery_ids in spec:
        battery_ids = set(battery_ids)
        for hid in ids:
            members.append(
                Member(
                    key=f"{dataset}_{hid:03d}",
                    dataset=dataset,
                    household_id=int(hid),
                    has_pv=has_pv,
                    battery_kwh=BATTERY_CAPACITY_KWH if hid in battery_ids else 0.0,
                    segment=dataset,
                )
            )
    return members


def scenario_paket(scenario: str, member: Member) -> str:
    cfg = SCENARIOS[scenario]
    return cfg["pv_paket"] if member.has_pv else cfg["nonpv_paket"]


def scenario_scheme(member: Member) -> str:
    return SCHEME_PV if member.has_pv else SCHEME_NONPV


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_member(member: Member) -> pd.DataFrame:
    """Household profile with the SMP series patched in (EUR/kWh)."""
    return hc.load_user(member.household_id, member.dataset)


def _mean_agreed_power(schedule: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """Mean agreed power per block over a schedule. Reporting only."""
    if not schedule:
        return {b: 0.0 for b in _BLOCKS}
    # A month carries no agreed power in a block it is not billed for, and
    # counting those months as 0 kW would drag the seasonal blocks (1 and 5)
    # far below what is ever actually contracted. So the mean is taken over
    # the months that contract the block, and is 0.0 only if none do.
    out = {}
    for b in _BLOCKS:
        vals = [v[b] for v in schedule.values() if b in v]
        out[b] = sum(vals) / len(vals) if vals else 0.0
    return out


def agreed_power_schedule(data: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """The member's dogovorjena obracunska moc, month by month and block by block.
    
    Exactly the schedule `HouseholdEnvironment` builds for the same profile, so
    the settlement bills against the contract the dispatch was optimized under.

    Which profile matters: passed the raw one it is the no-battery contract, and
    passed the profile a dispatch achieved it is the contract that dispatch
    re-agreed to. `contracted_power_map` below always passes the achieved one,
    because that is the household the bill is written for.
    """
    return agreed_power_schedule_for_profile(
        data,
        consumption_column=CONSUMPTION_COLUMN,
        generation_column=GENERATION_COLUMN,
        steps_per_day=STEPS_PER_DAY,
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        n_peaks=hc.AGREED_POWER_N_PEAKS,
        n_months_window=hc.AGREED_POWER_N_MONTHS_WINDOW,
    )


def contracted_power_map(profiles) -> Dict[str, Dict[int, Dict[int, float]]]:
    """Each member's contract, rolled from the profile it actually presents.

    A battery household re-agrees its dogovorjena moc down to the peaks its
    dispatch leaves at the meter, so the settlement has to read the effective
    profile, not the raw load behind the inverter. For a member with no battery
    -- and for every member of a no-battery scenario -- the two are the same
    profile and this is the no-battery contract, unchanged.
    """
    return {key: agreed_power_schedule(df) for key, df in profiles.items()}


def contracted_power_kw(data: pd.DataFrame) -> Dict[int, float]:
    """Mean agreed power per block over the year. Reporting only -- everything that
    bills reads `agreed_power_schedule`.
    """
    schedule = agreed_power_schedule(data)
    if not schedule:
        return {b: 0.0 for b in _BLOCKS}
    # A month carries no agreed power in a block it is not billed for, and
    # counting those months as 0 kW would drag the seasonal blocks (1 and 5)
    # far below what is ever actually contracted. So the mean is taken over
    # the months that contract the block, and is 0.0 only if none do.
    out = {}
    for b in _BLOCKS:
        vals = [v[b] for v in schedule.values() if b in v]
        out[b] = sum(vals) / len(vals) if vals else 0.0
    return out


def load_community(members=None, verbose=False):
    """Load every member's profile and attach its contracted power."""
    members = members if members is not None else build_community()
    data = {}
    for member in members:
        df = load_member(member)
        member.agreed_power_schedule = agreed_power_schedule(df)
        member.contracted_power_kw = _mean_agreed_power(member.agreed_power_schedule)
        data[member.key] = df
        if verbose:
            print(f"  {member.key}: {len(df)} intervals", flush=True)
    return members, data


# ---------------------------------------------------------------------------
# Battery dispatch
# ---------------------------------------------------------------------------
def solver():
    """The whole-period solver, with the study's relative gap and time limit.
    
    Shared with every other study rather than re-typed, so they stay comparable.
    """
    return full_period_solver()


def build_env(data, member: Member, scenario: str):
    """This study's household environment: one member, one scenario.

    The agreed power is deliberately not pinned -- left to itself the
    environment builds exactly the schedule `agreed_power_schedule` hands the
    settlement path.
    """
    return build_household_env(
        data,
        capacity_kwh=member.battery_kwh,
        scheme=scenario_scheme(member),
        paket_id=scenario_paket(scenario, member),
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        peak_reset_months=PEAK_RESET_MONTHS,
        price_column=PRICE_COLUMN,
        generation_column=GENERATION_COLUMN,
        consumption_column=CONSUMPTION_COLUMN,
    )


def dispatch_path(scenario: str, key: str) -> Path:
    return DISPATCH_DIR / f"{scenario}_{key}.csv"


def effective_profile(df_milp: pd.DataFrame, index) -> pd.DataFrame:
    """MILP trajectory -> the (consumption, generation) pair the settlement sees."""
    return _effective_profile(
        df_milp, index,
        consumption_column=CONSUMPTION_COLUMN,
        generation_column=GENERATION_COLUMN,
    )


def dispatch_member(member: Member, scenario: str, data=None, force=False, verbose=True):
    """Solve one battery household's whole year and cache the trajectory."""
    scenario = DISPATCH_ALIAS.get(scenario, scenario)
    path = dispatch_path(scenario, member.key)
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    if data is None:
        data = load_member(member)
        member.agreed_power_schedule = agreed_power_schedule(data)
        member.contracted_power_kw = _mean_agreed_power(member.agreed_power_schedule)

    env = build_env(data, member, scenario)
    soc = SOC_FRACTION * member.battery_kwh
    t0 = time.time()
    df_milp = solve_household(
        env,
        initial_soc_kwh=soc,
        final_soc_kwh=soc,
        verbose=False,
        solver=solver(),
        problem_name=f"Community_{scenario}_{member.key}",
    )
    runtime = time.time() - t0

    df_milp = df_milp.set_index("Date")
    path.parent.mkdir(parents=True, exist_ok=True)
    df_milp.to_csv(path)
    if verbose:
        print(
            f"  {scenario:10s} {member.key:22s} "
            f"cost {df_milp['Cum_Cost'].iloc[-1]:9.2f} EUR  ({runtime:5.1f}s)",
            flush=True,
        )
    return df_milp


def _dispatch_job(args):
    member, scenario, force = args
    try:
        dispatch_member(member, scenario, force=force, verbose=True)
        return (scenario, member.key, None)
    except Exception as exc:  # noqa: BLE001 - reported, batch continues
        return (scenario, member.key, repr(exc))


def dispatch_units(members=None, scenarios=None):
    """The (member, scenario) pairs that actually need a solve."""
    members = members if members is not None else build_community()
    scenarios = scenarios or [s for s in SCENARIOS if s not in DISPATCH_ALIAS]
    return [
        (m, s)
        for s in scenarios
        for m in members
        if m.has_battery
    ]


def run_batch_dispatch(members=None, scenarios=None, n_workers=10, force=False):
    """Solve every battery household under every scenario. Resumable."""
    import multiprocessing as mp

    units = dispatch_units(members, scenarios)
    pending = [
        (m, s, force)
        for (m, s) in units
        if force or not dispatch_path(s, m.key).exists()
    ]
    print(f"{len(units)} dispatch units, {len(pending)} to solve.", flush=True)
    if not pending:
        return []

    DISPATCH_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if n_workers and n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            outcomes = pool.map(_dispatch_job, pending)
    else:
        outcomes = [_dispatch_job(job) for job in pending]

    failures = [o for o in outcomes if o[2] is not None]
    print(f"Dispatch finished in {time.time() - t0:.1f}s, {len(failures)} failures.")
    for scenario, key, err in failures:
        print(f"  FAILED {scenario} {key}: {err}")
    return outcomes


def align_profiles(profiles):
    """Truncate every member to the shortest history in the group.
    
    `episode_length = len(data) - 1`, so a dispatched trajectory is one interval
    shorter than the raw profile and every member is cut to the common horizon.
    """
    n = min(len(df) for df in profiles.values())
    return {key: (df if len(df) == n else df.iloc[:n]) for key, df in profiles.items()}


def scenario_profiles(members, data, scenario: str):
    """Per-member profiles as the settlement should see them for this scenario.
    
    Battery members are replaced by their dispatched equivalent; everyone else
    passes through untouched, and all are aligned to a common horizon.
    """
    dispatch_scenario = DISPATCH_ALIAS.get(scenario, scenario)
    profiles = {}
    for member in members:
        df = data[member.key]
        if not member.has_battery:
            profiles[member.key] = df
            continue
        path = dispatch_path(dispatch_scenario, member.key)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached dispatch for {member.key} under {dispatch_scenario!r}. "
                f"Run Community_Study.run_batch_dispatch() first."
            )
        df_milp = pd.read_csv(path, index_col=0, parse_dates=True)
        eff = effective_profile(df_milp, df.index)
        eff["SMP"] = df[PRICE_COLUMN].to_numpy(dtype=float)[: len(eff)]
        profiles[member.key] = eff
    return align_profiles(profiles)


def dispatch_horizon(members, data, scenario: str) -> int:
    """How many intervals the battery members were actually dispatched over.
    
    Anything that has to be comparable with a dispatched run must be settled over
    exactly this many intervals; read off the cached files directly.
    """
    dispatch_scenario = DISPATCH_ALIAS.get(scenario, scenario)
    lengths = [len(data[m.key]) for m in members]
    for member in members:
        if not member.has_battery:
            continue
        path = dispatch_path(dispatch_scenario, member.key)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached dispatch for {member.key} under {dispatch_scenario!r}. "
                f"Run Community_Study.run_batch_dispatch() first."
            )
        with path.open() as handle:
            lengths.append(sum(1 for _ in handle) - 1)  # minus the header row
    return min(lengths)


def no_battery_profiles(members, data, scenario: str):
    """The same community with every battery taken out -- the baseline.
    
    Same households, load, PV yield, price list and contracted power, cut to
    `dispatch_horizon` so the two runs are subtractable interval by interval.
    """
    horizon = dispatch_horizon(members, data, scenario)
    return {m.key: data[m.key].iloc[:horizon] for m in members}


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def run_scenario(members, data, scenario: str, collect_flows=False, verbose=True,
                 no_battery=False):
    """Settle the whole community under one scenario.
    
    `no_battery=True` settles the same community with the batteries removed, so
    the difference between the two runs is the batteries and nothing else.
    """
    profiles = (
        no_battery_profiles(members, data, scenario)
        if no_battery
        else scenario_profiles(members, data, scenario)
    )
    by_key = {m.key: m for m in members}

    if scenario == "souporaba":
        senders = [m.key for m in members if m.has_pv]
        receivers = [m.key for m in members if not m.has_pv]
        if verbose:
            print(
                f"souporaba: {len(senders)} oddajnikov, {len(receivers)} prejemnikov, "
                f"delez {SOUPORABA_DELEZ:.0%}, cena {SOUPORABA_CENA_EUR_KWH} EUR/kWh",
                flush=True,
            )
        out = mht.run_souporaba_period_scenario(
            profiles,
            scenario_name=scenario,
            oddajnik_ids=senders,
            prejemnik_ids=receivers,
            delez_souporabe=SOUPORABA_DELEZ,
            cena_souporabe_eur_kwh=SOUPORABA_CENA_EUR_KWH,
            organizer_service_id=SOUPORABA_SERVICE_ID,
            oddajnik_paket_id=SCENARIOS[scenario]["pv_paket"],
            prejemnik_paket_id=SCENARIOS[scenario]["nonpv_paket"],
            contracted_power_map=contracted_power_map(profiles),
            generation_column=GENERATION_COLUMN,
            consumption_column=CONSUMPTION_COLUMN,
            pricing_reference_year=PRICING_REFERENCE_YEAR,
            progress=verbose,
        )
        summary = out["summary"].rename(columns={"za_placilo_eur": "total_cost_eur"})
        summary["scenario"] = scenario
        out["summary"] = _attach_member_columns(summary, by_key, profiles, data)
        out["community_profile"] = _community_profile(profiles)
        peak = out["community_profile"]["community_import_kw"]
        out["community_peak_kw"] = float(peak.max())
        out["community_peak_at"] = peak.idxmax()
        out["sum_of_individual_peaks_kw"] = float(
            out["summary"]
            .loc[out["summary"]["household_id"] != "GROUP", "peak_import_kw"]
            .sum()
        )
        return out

    out = mht.run_interval_scenario(
        profiles,
        scenario_name=scenario,
        scheme=SCHEME_NONPV,
        paket_id=SCENARIOS[scenario]["nonpv_paket"],
        scheme_map={m.key: scenario_scheme(m) for m in members},
        paket_id_map={m.key: scenario_paket(scenario, m) for m in members},
        contracted_power_map=contracted_power_map(profiles),
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        generation_column=GENERATION_COLUMN,
        consumption_column=CONSUMPTION_COLUMN,
        collect_flows=collect_flows,
    )
    out["summary"] = _attach_member_columns(out["summary"], by_key, profiles, data)
    return out


def _community_profile(profiles) -> pd.DataFrame:
    """Community net import/export per interval, summed over members.
    
    Summed positionally: the index carries duplicate DST labels, so pandas
    alignment would refuse.
    """
    index = next(iter(profiles.values())).index
    imp = np.zeros(len(index), dtype=float)
    exp = np.zeros(len(index), dtype=float)
    for key, df in profiles.items():
        net = df[CONSUMPTION_COLUMN].to_numpy(dtype=float) - df[
            GENERATION_COLUMN
        ].to_numpy(dtype=float)
        if len(net) != len(index):
            raise ValueError(
                f"member {key!r} has {len(net)} intervals, expected {len(index)}."
            )
        imp += np.maximum(net, 0.0)
        exp += np.maximum(-net, 0.0)
    interval_hours = 24.0 / STEPS_PER_DAY
    return pd.DataFrame(
        {
            "community_import_kwh": imp,
            "community_export_kwh": exp,
            "community_import_kw": imp / interval_hours,
        },
        index=index,
    )


def _attach_member_columns(summary: pd.DataFrame, by_key, profiles, data=None) -> pd.DataFrame:
    """Add roster attributes so every table can slice by segment/battery.
    
    `Actual_Load_kWh` is the physical load from the raw profile, identical across
    scenarios -- for a battery member the settled consumption is post-battery net
    import and is the wrong denominator for a unit price.
    """
    summary = summary.copy()
    interval_hours = 24.0 / STEPS_PER_DAY

    def attr(hid, name, default=None):
        member = by_key.get(hid)
        return getattr(member, name) if member is not None else default

    summary["segment"] = [attr(h, "segment") for h in summary["household_id"]]
    summary["has_pv"] = [attr(h, "has_pv") for h in summary["household_id"]]
    summary["battery_kwh"] = [attr(h, "battery_kwh") for h in summary["household_id"]]
    summary["dataset"] = [attr(h, "dataset") for h in summary["household_id"]]

    if "contracted_power_kw" not in summary.columns:
        summary["contracted_power_kw"] = [
            (attr(h, "contracted_power_kw") or {}).get(2, np.nan)
            for h in summary["household_id"]
        ]

    # The souporaba path never sees a peak in kW; derive it from the profiles.
    if "peak_import_kw" not in summary.columns:
        peaks = {}
        for key, df in profiles.items():
            net = df[CONSUMPTION_COLUMN].to_numpy(dtype=float) - df[
                GENERATION_COLUMN
            ].to_numpy(dtype=float)
            peaks[key] = float(np.max(np.maximum(net, 0.0))) / interval_hours
        summary["peak_import_kw"] = [
            peaks.get(h, np.nan) for h in summary["household_id"]
        ]

    if "total_consumption_kwh" not in summary.columns:
        summary["total_consumption_kwh"] = [
            float(profiles[h][CONSUMPTION_COLUMN].sum()) if h in profiles else np.nan
            for h in summary["household_id"]
        ]
        summary["total_generation_kwh"] = [
            float(profiles[h][GENERATION_COLUMN].sum()) if h in profiles else np.nan
            for h in summary["household_id"]
        ]

    if data is not None:
        horizon = len(next(iter(profiles.values())))
        load, pv = {}, {}
        for key in profiles:
            raw = data[key].iloc[:horizon]
            load[key] = float(raw[CONSUMPTION_COLUMN].sum())
            pv[key] = float(raw[GENERATION_COLUMN].sum())
        summary["Actual_Load_kWh"] = [load.get(h, np.nan) for h in summary["household_id"]]
        summary["Actual_PV_kWh"] = [pv.get(h, np.nan) for h in summary["household_id"]]
        group = summary["household_id"] == "GROUP"
        summary.loc[group, "Actual_Load_kWh"] = sum(load.values())
        summary.loc[group, "Actual_PV_kWh"] = sum(pv.values())

    return summary


def run_all_scenarios(members, data, scenarios=None, verbose=True, no_battery=False):
    scenarios = scenarios or list(SCENARIOS)
    results = {}
    for scenario in scenarios:
        if verbose:
            print(f"--- {scenario}{' (no battery)' if no_battery else ''} ---", flush=True)
        t0 = time.time()
        results[scenario] = run_scenario(members, data, scenario, verbose=verbose,
                                         no_battery=no_battery)
        if verbose:
            print(f"    settled in {time.time() - t0:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Community-level read-outs
# ---------------------------------------------------------------------------
def cost_categories(out) -> Dict[str, float]:
    """The community's gross bill grouped by who is paid, VAT included.
    
    Per-item figures come out of the settlement engines, so an unmapped item
    raises rather than vanishing from the total.
    """
    components = out.get("components")
    if components is None or not len(components):
        return {name: np.nan for name in COST_CATEGORIES}

    by_item = components.groupby("component")["eur"].sum()
    known = {item for items in COST_CATEGORIES.values() for item in items}
    unknown = sorted(set(by_item.index) - known)
    if unknown:
        raise ValueError(f"unmapped billing items {unknown}; extend COST_CATEGORIES.")
    return {
        name: float(by_item.reindex(items).fillna(0.0).sum())
        for name, items in COST_CATEGORIES.items()
    }


def community_totals(results) -> pd.DataFrame:
    """One row per scenario: what the community pays, and its peak."""
    rows = []
    for scenario, out in results.items():
        summary = out["summary"]
        group = summary[summary["household_id"] == "GROUP"]
        members_only = summary[summary["household_id"] != "GROUP"]
        consumed = float(members_only["Actual_Load_kWh"].sum())
        total = float(
            group["total_cost_eur"].iloc[0]
            if len(group)
            else members_only["total_cost_eur"].sum()
        )
        peak_kw = float(out["community_peak_kw"])
        sum_peaks = float(out["sum_of_individual_peaks_kw"])

        # The two paths report the credit differently -- the interval path nets
        # it inside energy_eur, a Racun deducts it after VAT -- so both are
        # normalized to "gross bill minus export credit". Split_Residual_EUR
        # asserts Variable_Gross + Fixed - Credit == Total.
        if "energy_eur" in group.columns:
            energy = float(group["energy_eur"].iloc[0])
            power = float(group["power_eur"].iloc[0])
            fixed = float(group["fixed_eur"].iloc[0])
            credit = float(group["credit_eur"].iloc[0])
            variable = energy + power + credit
        else:
            ddv = 1.22
            energy = power = np.nan
            variable = float(group["spremenljivi_del_eur"].iloc[0]) * ddv
            fixed = float(group["fiksni_del_eur"].iloc[0]) * ddv
            credit = float(group["dobropis_eur"].iloc[0])

        categories = cost_categories(out)
        rows.append(
            {
                "Scenario": SCENARIOS[scenario]["label"],
                "scenario": scenario,
                "Total_EUR": total,
                "EUR_per_MWh": total / (consumed / 1000.0) if consumed else np.nan,
                "Consumed_MWh": consumed / 1000.0,
                "Energy_EUR": energy,
                "Power_EUR": power,
                "Variable_Gross_EUR": variable,
                "Fixed_EUR": fixed,
                "Credit_EUR": credit,
                "Gross_EUR": variable + fixed,
                "Split_Residual_EUR": variable + fixed - credit - total,
                **categories,
                "Category_Residual_EUR": sum(categories.values()) - credit - total,
                "Community_Peak_kW": peak_kw,
                "Peak_At": out["community_peak_at"],
                "Sum_Individual_Peaks_kW": sum_peaks,
                "Diversity_Factor": peak_kw / sum_peaks if sum_peaks else np.nan,
            }
        )
    return pd.DataFrame(rows)


def community_kpis(results) -> pd.DataFrame:
    """Self-sufficiency, self-consumption and the community's net position."""
    rows = []
    for scenario, out in results.items():
        profile = out["community_profile"]
        summary = out["summary"]
        members_only = summary[summary["household_id"] != "GROUP"]
        consumed = float(members_only["Actual_Load_kWh"].sum())
        produced = float(members_only["Actual_PV_kWh"].sum())
        imported = float(profile["community_import_kwh"].sum())
        exported = float(profile["community_export_kwh"].sum())
        # What the community's own PV covered of its own load.
        self_consumed = produced - exported
        rows.append(
            {
                "Scenario": SCENARIOS[scenario]["label"],
                "scenario": scenario,
                "Consumed_MWh": consumed / 1000.0,
                "Produced_MWh": produced / 1000.0,
                "Imported_MWh": imported / 1000.0,
                "Exported_MWh": exported / 1000.0,
                "Self_Sufficiency_pct": 100.0 * self_consumed / consumed if consumed else np.nan,
                "Self_Consumption_pct": 100.0 * self_consumed / produced if produced else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_household(results) -> pd.DataFrame:
    """Long table: one row per (scenario, household)."""
    frames = []
    for scenario, out in results.items():
        summary = out["summary"]
        df = summary[summary["household_id"] != "GROUP"].copy()
        df["scenario"] = scenario
        df["Scenario"] = SCENARIOS[scenario]["label"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def by_segment(results) -> pd.DataFrame:
    """Mean annual cost per segment and battery status, per scenario."""
    df = per_household(results)
    df["battery"] = np.where(df["battery_kwh"] > 0, "battery", "no battery")
    grouped = (
        df.groupby(["Scenario", "segment", "battery"])["total_cost_eur"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .rename(columns={"count": "N", "mean": "Mean_EUR", "min": "Min_EUR", "max": "Max_EUR"})
    )
    return grouped


def battery_value(results) -> pd.DataFrame:
    """What the battery members pay vs the no-battery members of the same dataset.
    
    A peer comparison; the counterfactual is `battery_counterfactual`.
    """
    df = per_household(results)
    df = df[df["segment"].isin({m.dataset for m in build_community() if m.battery_kwh > 0})]
    df["battery"] = np.where(df["battery_kwh"] > 0, "battery", "no battery")
    return (
        df.groupby(["Scenario", "segment", "battery"])["total_cost_eur"]
        .mean()
        .unstack("battery")
        .assign(Saving_EUR=lambda d: d["no battery"] - d["battery"])
        .reset_index()
    )


def battery_counterfactual(members, data, results, scenarios=None) -> pd.DataFrame:
    """Re-settle the battery households with no battery, same everything else.
    
    Also verification check 4: a battery must never make a household's bill
    worse, and if it does the two contracted-power dicts have diverged.
    """
    scenarios = scenarios or [s for s in SCENARIOS if s != "souporaba"]
    battery_members = [m for m in members if m.has_battery]
    if not battery_members:
        return pd.DataFrame()

    rows = []
    for scenario in scenarios:
        # Settled over exactly the horizon the battery run was.
        baseline = no_battery_profiles(members, data, scenario)
        profiles = {m.key: baseline[m.key] for m in battery_members}
        out = mht.run_interval_scenario(
            profiles,
            scenario_name=f"{scenario}_nobattery",
            scheme=SCHEME_NONPV,
            paket_id=SCENARIOS[scenario]["nonpv_paket"],
            scheme_map={m.key: scenario_scheme(m) for m in battery_members},
            paket_id_map={m.key: scenario_paket(scenario, m) for m in battery_members},
            contracted_power_map=contracted_power_map(profiles),
            pricing_reference_year=PRICING_REFERENCE_YEAR,
            generation_column=GENERATION_COLUMN,
            consumption_column=CONSUMPTION_COLUMN,
            collect_flows=False,
        )
        base = out["summary"].set_index("household_id")["total_cost_eur"]
        actual = (
            results[scenario]["summary"].set_index("household_id")["total_cost_eur"]
        )
        for m in battery_members:
            rows.append(
                {
                    "Scenario": SCENARIOS[scenario]["label"],
                    "scenario": scenario,
                    "household_id": m.key,
                    "segment": m.segment,
                    "No_Battery_EUR": float(base[m.key]),
                    "With_Battery_EUR": float(actual[m.key]),
                    "Saving_EUR": float(base[m.key]) - float(actual[m.key]),
                }
            )
    return pd.DataFrame(rows)


def extremes(results, n=1) -> pd.DataFrame:
    """The cheapest and dearest household in each scenario, with the why.
    
    Annual load, PV yield, battery, how far the realized peak sits above the
    contracted power, and the average price actually paid.
    """
    df = per_household(results)
    rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        ordered = group.sort_values("total_cost_eur")
        for kind, subset in (
            ("lowest", ordered.head(n)),
            ("highest", ordered.tail(n).iloc[::-1]),
        ):
            for _, row in subset.iterrows():
                consumed = float(row["Actual_Load_kWh"])
                contracted = float(row.get("contracted_power_kw", np.nan))
                peak = float(row.get("peak_import_kw", np.nan))
                rows.append(
                    {
                        "Scenario": SCENARIOS[scenario]["label"],
                        "scenario": scenario,
                        "Extreme": kind,
                        "household_id": row["household_id"],
                        "segment": row["segment"],
                        "Cost_EUR": float(row["total_cost_eur"]),
                        "EUR_per_MWh": (
                            float(row["total_cost_eur"]) / (consumed / 1000.0)
                            if consumed
                            else np.nan
                        ),
                        "Consumed_kWh": consumed,
                        "Produced_kWh": float(row["Actual_PV_kWh"]),
                        "Net_Import_kWh": float(row["total_consumption_kwh"]),
                        "Battery_kWh": float(row["battery_kwh"]),
                        "Contracted_kW": contracted,
                        "Peak_kW": peak,
                        "Peak_over_Contract": (
                            peak / contracted if contracted else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def peak_day(profiles, key: str, timestamp) -> pd.DataFrame:
    """One household's profile on the local day containing `timestamp`."""
    df = profiles[key]
    day = pd.Timestamp(timestamp).normalize()
    mask = (df.index >= day) & (df.index < day + pd.Timedelta(days=1))
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def check_dispatch_balance(scenario=None, tol=1e-4) -> pd.DataFrame:
    """Sanity-check every cached MILP trajectory.
    
    The net flow is defined by the energy balance in `effective_profile`, so what
    is checked is the SOC recursion, that SOC stays inside [0, capacity], that
    curtailment never exceeds the interval's generation, that charge and
    discharge are never simultaneous, and that the year closes where it opened.
    """
    scenarios = [scenario] if scenario else [s for s in SCENARIOS if s not in DISPATCH_ALIAS]
    rows = []
    for sc in scenarios:
        for path in sorted(DISPATCH_DIR.glob(f"{sc}_*.csv")):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            soc = df["SOC_kWh"].to_numpy(dtype=float)
            ch = df["Charge_kW"].to_numpy(dtype=float)
            dis = df["Discharge_kW"].to_numpy(dtype=float)
            gen = df["Generation"].to_numpy(dtype=float)
            spill = df["Spill_kW"].to_numpy(dtype=float)

            predicted = soc[:-1] + ch[:-1] * CHARGE_EFFICIENCY - dis[:-1] / DISCHARGE_EFFICIENCY
            soc_residual = float(np.max(np.abs(predicted - soc[1:]))) if len(soc) > 1 else 0.0
            capacity = BATTERY_CAPACITY_KWH

            rows.append(
                {
                    "scenario": sc,
                    "household_id": path.stem[len(sc) + 1 :],
                    "Max_SOC_Residual_kWh": soc_residual,
                    "SOC_Drift_kWh": float(soc[-1] - soc[0]),
                    "SOC_in_Bounds": bool((soc >= -tol).all() and (soc <= capacity + tol).all()),
                    "Spill_within_Generation": bool((spill <= gen + tol).all()),
                    "No_Simultaneous_ChDis": bool((np.minimum(ch, dis) <= tol).all()),
                    "OK": bool(soc_residual < tol),
                }
            )
    return pd.DataFrame(rows)


def check_souporaba_conservation(out) -> pd.DataFrame:
    """Shared energy is conserved: what senders transfer, receivers use or forfeit.
    
    `MesecniObracun.zakljuci` rounds every diagnostic to 2 decimals, so each
    household contributes up to 0.005 kWh of rounding to its side of the balance.
    A residual above 0.005 x (households on the larger side) is a real leak.
    """
    monthly = out["monthly"]
    rows = []
    for (year, month), grp in monthly.groupby(["year", "month"]):
        senders = grp[grp["role"] == "oddajnik"]
        receivers = grp[grp["role"] == "prejemnik"]
        sent = float(senders["deljeno_kwh"].sum())
        received = float(
            receivers["deljeno_kwh"].sum() + receivers["neizrabljeno_kwh"].sum()
        )
        # Senders round one figure each; receivers round two each.
        rounding_bound = 0.005 * (len(senders) + 2 * len(receivers)) + 1e-9
        rows.append(
            {
                "year": year,
                "month": month,
                "Sent_kWh": sent,
                "Received_kWh": received,
                "Residual_kWh": sent - received,
                "Rounding_Bound_kWh": rounding_bound,
                "OK": abs(sent - received) <= rounding_bound,
            }
        )
    return pd.DataFrame(rows)


def check_community_peak(results) -> pd.DataFrame:
    """The coincident peak can never exceed the sum of the individual peaks."""
    rows = []
    for scenario, out in results.items():
        profile = out["community_profile"]
        series = profile["community_import_kw"]
        peak = float(out["community_peak_kw"])
        at = out["community_peak_at"]
        # Positional: the DST fall-back hour appears twice in the index.
        pos = int(np.argmax(series.to_numpy(dtype=float)))
        rows.append(
            {
                "scenario": scenario,
                "Community_Peak_kW": peak,
                "Sum_Individual_kW": float(out["sum_of_individual_peaks_kw"]),
                "Diversity_Factor": peak / float(out["sum_of_individual_peaks_kw"]),
                "Timestamp_Reproduces": bool(
                    series.index[pos] == at
                    and abs(float(series.iloc[pos]) - peak) < 1e-9
                ),
                "OK": peak <= float(out["sum_of_individual_peaks_kw"]) + 1e-9,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="re-solve cached units")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    members = build_community()
    print(f"Community: {len(members)} members, "
          f"{sum(1 for m in members if m.has_battery)} with a battery.")
    run_batch_dispatch(members, args.scenarios, n_workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
