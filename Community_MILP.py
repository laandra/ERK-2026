"""Perfect-foresight community MILP: skupnostna samooskrba vs souporaba.

One MILP over every member of a community, solving each member's battery and
how much of each producer's surplus is shared rather than sold to its supplier,
against one objective: the community's total electricity bill.

Both schemes are BILLING overlays on the same physical flows. Under skupnostna
samooskrba a shared kWh is billed as community energy price + reduced network
energy tariff + levies, and allocation nobody used returns to the pool and is
sold at the producer's buyback price. Under souporaba a shared kWh reduces only
the energy quantity the receiver's supplier invoices -- network charge, levies
and excise stay on the full metered offtake -- allocation unused inside the
interval is destroyed, and the organizer charges a monthly fee per metering
point. The per-kWh advantage of the first over the second is exactly the
distribution part of the network tariff.

Under ZOEE the share is a contract number, one static percentage applied to
every interval, which is why `SHARING_STATIC` is the contract as written and
`SHARING_DYNAMIC` -- allocation free per interval -- is the upper bound on any
sharing rule.

The model stays an LP almost everywhere. Receivers hold a price list with no
buyback, so the buy/sell complementarity is implied by the objective
(`assert_no_receiver_buyback` enforces the precondition), and simultaneous
charge and discharge is only profitable where the delivered import rate is
negative. Both are enforced lazily: the first pass is a pure LP, the trajectory
is checked for flows no meter could produce, and a binary is spent only where
one occurred. The exception is community self-supply with mismatched price lists
and an internal price too low for its VAT to hold the trade below profitable;
there the lazy loop converges badly and `exposed_intervals` get their binary up
front.

The horizon is decomposed by calendar month, which is exact for the tariff --
the power charge, the OVE+SPTE levy and the ratchet all reset monthly -- and an
approximation for the battery, which opens and closes each month at
`soc_fraction` of capacity.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp

from Pricing_Functions import (
    DDV,
    PAKETI,
    PRIVZETO_REFERENCNO_LETO,
    compute_prorated_fixed_charge_eur,
    moc_za_mesec,
)

# The household half of this model -- the battery, the balance and the ratchet --
# is the same one `MILP_Household.solve_household` builds; only the settlement
# below is this module's own.
from MILP_Household import (
    HouseholdNames,
    add_battery_exclusivity,
    add_excess_power_ratchet,
    add_grid_exclusivity,
    add_household_physics,
    agreed_power_by_month,
    floor_export_rates,
    month_calendar,
)

# The community settlement primitives live in the spaced folder.
_SI_DIR = Path(__file__).resolve().parent / "New pricing functions"
if str(_SI_DIR) not in sys.path:
    sys.path.append(str(_SI_DIR))

from si_cas import je_visja_sezona, v_lokalni_cas  # noqa: E402
from si_obracun import Pravila, samooskrba  # noqa: E402
from si_paketi import STORITVE_SOUPORABE  # noqa: E402
from si_tarife import (  # noqa: E402
    OPERATER_TRGA_EUR_KWH,
    TROSARINA_EUR_KWH,
    URE_EUR_KWH,
)

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
SCHEME_INDIVIDUAL = "individualno"
SCHEME_SOUPORABA = "souporaba"
SCHEME_SKUPNOST = "skupnostna_samooskrba"
SCHEMES = (SCHEME_INDIVIDUAL, SCHEME_SOUPORABA, SCHEME_SKUPNOST)

SHARING_STATIC = "staticni"
SHARING_DYNAMIC = "dinamicni"

ROLE_SENDER = "oddajnik"
ROLE_RECEIVER = "prejemnik"

# Excise + energy-efficiency + market-operator levies, EUR/kWh excluding VAT.
# Charged on every metered kWh under both schemes.
LEVIES_EUR_PER_KWH = TROSARINA_EUR_KWH + URE_EUR_KWH + OPERATER_TRGA_EUR_KWH
VAT_FACTOR = 1.0 + float(DDV)


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------
@dataclass
class CommunityMember:
    """One metering point in the community.
    
    `env` is a fully built `HouseholdEnvironment`: the profile, the battery, the
    price list and the agreed billing power all come off it. `role` is the ZOEE
    role, and a member is one or the other, never both. `key_weight` is a
    receiver's static share of the pool. `znacilni_primer` is the connection
    case 1-10 picking the community distribution postavka (1 = same transformer
    station, 0.00000 EUR/kWh; 4 = no reduction).
    """

    member_id: str
    env: object
    role: str
    contract_key: str = ""
    key_weight: float = 1.0
    znacilni_primer: int = 2
    label: str = ""

    def __post_init__(self):
        if self.role not in (ROLE_SENDER, ROLE_RECEIVER):
            raise ValueError(f"{self.member_id}: role must be {ROLE_SENDER!r} or {ROLE_RECEIVER!r}")
        if not self.label:
            self.label = self.member_id

    @property
    def paket_id(self) -> str:
        return self.env.pricing_options["paket_id"]

    @property
    def is_sender(self) -> bool:
        return self.role == ROLE_SENDER


# --------------------------------------------------------------------------
# Rate tables
# --------------------------------------------------------------------------
_RATE_CACHE: Dict[tuple, Dict[str, np.ndarray]] = {}


def paket_rate_table(paket_id, dates, smp_eur_kwh, interval_minutes, ref_year):
    """Per-interval ex-VAT rates for one price list, cached per (list, horizon).
    
    The supplier energy price, the buyback price and the tariff block -- what
    depends on the list rather than on the household. Rates come out of
    `si_obracun.samooskrba`, the function the invoices bill through.
    """
    key = (paket_id, int(ref_year), len(dates), dates[0], dates[-1], float(interval_minutes))
    cached = _RATE_CACHE.get(key)
    if cached is not None:
        return cached

    paket = PAKETI[paket_id]
    pravila = Pravila.za_leto(int(ref_year))
    n = len(dates)
    energy = np.empty(n, dtype=float)
    export = np.empty(n, dtype=float)
    blocks = np.empty(n, dtype=np.int32)

    for t in range(n):
        res = samooskrba(
            float(smp_eur_kwh[t]) * 1000.0,
            1.0,
            dates[t],
            interval_minutes,
            total_produced_kwh=0.0,
            paket=paket,
            pravila=pravila,
            meritve_15min=True,
        )
        energy[t] = res["cena_energije_eur_kwh"]
        export[t] = res["cena_oddaje_eur_kwh"]
        blocks[t] = int(res["blok"])

    table = {"energy": energy, "export": export, "blocks": blocks}
    _RATE_CACHE[key] = table
    return table


def network_rate_tables(blocks, ref_year, znacilni_primer):
    """Network energy postavke per interval, ex VAT.
    
    `full` is what a kWh out of the grid costs, `shared` what the same kWh costs
    as a community allocation. Their difference is the entire per-kWh advantage
    of skupnostna samooskrba over souporaba, and is zero at `znacilni_primer=4`.
    """
    om = Pravila.za_leto(int(ref_year)).omreznina
    full = np.array([om.energija[int(b)] for b in blocks], dtype=float)
    shared = np.array(
        [om.energija_prenos[int(b)] + om.energija_skupnost[int(znacilni_primer)][int(b)]
         for b in blocks],
        dtype=float,
    )
    return full, shared


def assert_no_receiver_buyback(members):
    """Receivers must hold a price list that credits an export nothing.
    
    With a positive buyback the LP could import a shared kWh at the reduced
    tariff and export it at the buyback price in the same interval -- a trade one
    meter cannot make. The precondition costs one assert; the alternative is one
    binary per member per interval.
    """
    offenders = []
    for m in members:
        if m.is_sender:
            continue
        paket = PAKETI[m.paket_id]
        # A household settling its exports once a year cannot be a receiver in
        # souporaba (`si_paketi.preveri_souporabo`).
        if paket.tip_odkupa.value != "ni":
            offenders.append(f"{m.label} ({m.paket_id}, odkup={paket.tip_odkupa.value})")
    if offenders:
        raise ValueError(
            "Receivers must be on a price list with no per-interval buyback "
            "(tip_odkupa=NI). Offending members: " + ", ".join(offenders)
        )


# --------------------------------------------------------------------------
# One solve
# --------------------------------------------------------------------------
def solve_community_milp(
    members: Sequence[CommunityMember],
    *,
    scheme: str,
    start_idx: int = 0,
    n_steps: Optional[int] = None,
    sharing_mode: str = SHARING_DYNAMIC,
    share_of_surplus: float = 1.0,
    internal_price_eur_kwh: float = 0.0,
    pay_for_unused: bool = True,
    service_id: Optional[str] = None,
    soc_fraction: float = 0.5,
    solver=None,
    ref_year: Optional[int] = None,
    max_lazy_rounds: int = 6,
    lazy_gap_rel: float = 1e-4,
    lazy_time_limit_s: float = 300.0,
    strict: bool = True,
    verbose: bool = False,
):
    """One perfect-foresight community solve over `n_steps` intervals.
    
    Returns the per-member settlement, the community total and the per-interval
    flow table, VAT-inclusive on the charge side and VAT-free on the credit side.
    
    `scheme=SCHEME_INDIVIDUAL` disables sharing and gives the baseline bill.
    `sharing_mode` is the static contract share or a free per-interval allocation.
    `internal_price_eur_kwh` is a transfer between members, taxable at the
    receiver and VAT-free at the producer, so every euro of it leaks 22 cents.
    `pay_for_unused` is the ZOEE default and affects souporaba only.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if sharing_mode not in (SHARING_STATIC, SHARING_DYNAMIC):
        raise ValueError(f"sharing_mode must be {SHARING_STATIC!r} or {SHARING_DYNAMIC!r}")

    members = list(members)
    sharing = scheme != SCHEME_INDIVIDUAL
    if sharing:
        assert_no_receiver_buyback(members)

    env0 = members[0].env
    ref_year = int(ref_year if ref_year is not None else (env0.pricing_reference_year or PRIVZETO_REFERENCNO_LETO))
    pravila = Pravila.za_leto(ref_year)
    interval_minutes = int(round(env0.interval_minutes))
    hours = interval_minutes / 60.0

    n_steps = int(len(env0.dataset) - start_idx if n_steps is None else n_steps)
    horizon = slice(start_idx, start_idx + n_steps)
    dates = env0.dataset.index[horizon]
    smp = env0.arr_price[horizon]

    # Calendar bookkeeping, shared by every member.
    _, _, months_sorted, month_idx_t = month_calendar(dates)

    senders = [m for m in members if m.is_sender]
    receivers = [m for m in members if not m.is_sender]
    if sharing and (not senders or not receivers):
        raise ValueError("a sharing scheme needs at least one sender and one receiver")

    weight_sum = sum(max(0.0, m.key_weight) for m in receivers) or 1.0
    weights = {m.member_id: max(0.0, m.key_weight) / weight_sum for m in receivers}

    alpha = float(share_of_surplus)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"share_of_surplus must be in [0, 1], got {alpha}")
    p_int = float(internal_price_eur_kwh)

    # ---- pass 1: the rates, before any variable exists -- whether a producer
    # needs a complementarity binary depends on what a shared kWh is worth at
    # the receivers.
    rates: Dict[str, dict] = {}
    for m in members:
        table = paket_rate_table(m.paket_id, dates, smp, interval_minutes, ref_year)
        r_net_full, r_net_shared = network_rate_tables(table["blocks"], ref_year, m.znacilni_primer)
        # The delivered cost of one imported kWh, VAT included. A buyback worth
        # more than that is an unbounded loop, hence the floor.
        delivered = VAT_FACTOR * (table["energy"] + r_net_full + LEVIES_EUR_PER_KWH)
        r_export, n_floored = floor_export_rates(delivered, table["export"])
        rates[m.member_id] = dict(
            r_energy=table["energy"], blocks=table["blocks"],
            r_net_full=r_net_full, r_net_shared=r_net_shared, delivered=delivered,
            r_export=r_export, n_floored=n_floored,
        )

    # What one shared kWh displaces at the best-placed receiver, ex VAT: under
    # souporaba the supplier's energy price, under community self-supply the
    # distribution part of the network tariff as well.
    if sharing:
        displaced = np.max(np.stack([
            rates[m.member_id]["r_energy"] if scheme == SCHEME_SOUPORABA else
            (rates[m.member_id]["r_energy"] + rates[m.member_id]["r_net_full"]
             - rates[m.member_id]["r_net_shared"])
            for m in receivers
        ]), axis=0)
    else:
        displaced = np.zeros(n_steps)

    prob = pulp.LpProblem("Community_Sharing_MILP", pulp.LpMinimize)
    obj_terms = []
    n_upfront_binaries = 0
    # Per-member state kept for the extraction pass below.
    state: Dict[str, dict] = {}

    for m in members:
        env = m.env
        gen = env.arr_generation[horizon]
        con = env.arr_consumption[horizon]
        rt = rates[m.member_id]
        r_energy, blocks = rt["r_energy"], rt["blocks"]
        r_net_full, r_net_shared = rt["r_net_full"], rt["r_net_shared"]
        delivered, r_export, n_floored = rt["delivered"], rt["r_export"], rt["n_floored"]

        cap = float(env.battery_capacity_kwh)
        soc_target = soc_fraction * cap
        # `exclusivity="inverter"` keeps the build a pure LP: charging and
        # discharging at once is bounded by the inverter rating here and
        # forbidden outright only where the solve turns out to have done it,
        # in the lazy loop below.
        hh = add_household_physics(
            prob, env, n_steps=n_steps, gen=gen, con=con,
            names=HouseholdNames.for_member(m.member_id),
            initial_soc_kwh=soc_target, final_soc_kwh=soc_target,
            exclusivity="inverter", metering_bounds=True,
        )
        buy, sell, spill = hh.buy, hh.sell, hh.spill
        ch, dis, soc = hh.charge, hh.discharge, hh.soc
        max_ch, max_dis = hh.max_charge_ac, hh.max_discharge_ac

        # Where the metering bounds are not enough. Importing and exporting
        # the same kWh in one interval is normally self-punishing, but a shared
        # kWh can be worth more at the receiver than a bought one costs at the
        # producer -- under community self-supply, with mismatched price lists
        # and an internal price whose VAT no longer holds the trade below
        # profitable. Those intervals get their binary up front, because fixing
        # one hands the same trade to the next and the lazy loop converges
        # badly. Every other configuration has an empty `exposed`.
        exposed = (np.flatnonzero(
            (VAT_FACTOR * (r_energy + r_net_full + LEVIES_EUR_PER_KWH - displaced)
             + (VAT_FACTOR - 1.0) * p_int < 0.0) & (gen > 0.0))
            if (sharing and m.is_sender) else np.array([], dtype=int))
        for t in exposed:
            add_grid_exclusivity(
                prob, buy_t=buy[t], sell_t=sell[t], gen_t=gen[t], con_t=con[t],
                max_charge_ac=max_ch, max_discharge_ac=max_dis,
                flag=pulp.LpVariable(f"bgrid0_{m.member_id}_{t}", cat="Binary"),
            )
        n_upfront_binaries += len(exposed)

        # ---- the ratchet excess-power charge, one variable per (block, month).
        # The peak is measured on the METERED import `buy`, which no billing
        # overlay can move.
        agreed_by_month = agreed_power_by_month(env, start_idx, months_sorted)
        _, _, ratchet_terms = add_excess_power_ratchet(
            prob, env, buy,
            blocks=blocks, month_idx_t=month_idx_t, months_sorted=months_sorted,
            agreed_by_month=agreed_by_month, start_idx=start_idx, n_steps=n_steps,
            hours=hours, pravila=pravila,
            peak_name=f"peak_{m.member_id}_b{{b}}_m{{m}}",
            excess_name=f"exc_{m.member_id}_b{{b}}_m{{m}}",
        )
        obj_terms.extend(ratchet_terms)

        # ---- the decision-independent fixed monthly charge, prorated.
        fixed_by_month = {}
        for mi, (y, mo) in enumerate(months_sorted):
            first_t = month_idx_t.index(mi)
            fixed_by_month[mi] = compute_prorated_fixed_charge_eur(
                dates[first_t], interval_minutes, scheme=env.pricing_scheme,
                dogovorjena_moc=agreed_by_month[mi], paket_id=m.paket_id,
                pricing_reference_year=ref_year,
            )
        fixed_total = sum(fixed_by_month[month_idx_t[t]] for t in range(n_steps))

        # ---- the organizer's monthly fee, souporaba only: a taxable item per
        # metering point per month.
        service_fee = 0.0
        if scheme == SCHEME_SOUPORABA and service_id:
            service = STORITVE_SOUPORABE[service_id]
            per_month = (service.nadomestilo_oddajnik if m.is_sender
                         else service.nadomestilo_prejemnik)
            service_fee = VAT_FACTOR * per_month * len(months_sorted)

        state[m.member_id] = dict(
            member=m, gen=gen, con=con, buy=buy, sell=sell, ch=ch, dis=dis, soc=soc,
            spill=spill, r_energy=r_energy, r_export=r_export, r_net_full=r_net_full,
            r_net_shared=r_net_shared, delivered=delivered, blocks=blocks, cap=cap,
            fixed_total=fixed_total, service_fee=service_fee, n_floored=n_floored,
            max_ch=max_ch, max_dis=max_dis, exposed=exposed,
            n_grid_binaries=len(exposed),
            n_battery_binaries=0,
            used=None, shared_out=None,
        )
        obj_terms.append(fixed_total + service_fee)

    # ----------------------------------------------------------------------
    # Sharing
    # ----------------------------------------------------------------------
    used_vars: Dict[str, list] = {}
    shared_vars: Dict[str, list] = {}
    if sharing:
        for m in receivers:
            used_vars[m.member_id] = [
                pulp.LpVariable(f"used_{m.member_id}_{t}", lowBound=0) for t in range(n_steps)
            ]
            state[m.member_id]["used"] = used_vars[m.member_id]
        if sharing_mode == SHARING_DYNAMIC:
            for m in senders:
                shared_vars[m.member_id] = [
                    pulp.LpVariable(f"share_{m.member_id}_{t}", lowBound=0) for t in range(n_steps)
                ]
                state[m.member_id]["shared_out"] = shared_vars[m.member_id]

        for t in range(n_steps):
            if sharing_mode == SHARING_STATIC:
                # The contract as written: a fixed fraction of the interval's
                # surplus, split by a fixed key. `alpha` is a parameter and
                # `sell` a variable, so the product stays linear.
                pool_t = alpha * pulp.lpSum(state[s.member_id]["sell"][t] for s in senders)
                for m in receivers:
                    prob += used_vars[m.member_id][t] <= weights[m.member_id] * pool_t
            else:
                for s in senders:
                    prob += shared_vars[s.member_id][t] <= state[s.member_id]["sell"][t]
                # Equality, not <=: under `pay_for_unused` an unconsumed
                # transfer would otherwise be paid for by nobody.
                prob += (pulp.lpSum(used_vars[m.member_id][t] for m in receivers)
                         == pulp.lpSum(shared_vars[s.member_id][t] for s in senders))
            for m in receivers:
                # A shared kWh is an allocation ON the metered offtake, never on
                # top of it.
                prob += used_vars[m.member_id][t] <= state[m.member_id]["buy"][t]

    # ----------------------------------------------------------------------
    # Objective: the community's bill
    # ----------------------------------------------------------------------
    # ---- what every member pays on its own offtake, and what the producers
    # are credited for what they exported.
    no_incentive = {}
    for m in members:
        st = state[m.member_id]
        buy, sell = st["buy"], st["sell"]
        r_energy, r_export = st["r_energy"], st["r_export"]
        r_net_full, r_net_shared = st["r_net_full"], st["r_net_shared"]
        used = st["used"]

        # The per-kWh discount a shared kWh carries at this member; negative
        # means the member is better off with the allocation. It turns
        # non-negative only where a SIPX price has fallen below the internal
        # price, and those intervals are counted rather than silently allowed.
        if used is not None:
            delta = (p_int - r_energy) if scheme == SCHEME_SOUPORABA else (
                p_int - r_energy - r_net_full + r_net_shared)
            no_incentive[m.member_id] = int(np.sum(delta >= 0.0))

        for t in range(n_steps):
            # Taxable base, ex VAT, then grossed up, as si_obracun.Racun
            # settles. The internal price is settled on the pool below.
            taxable = buy[t] * (r_energy[t] + r_net_full[t] + LEVIES_EUR_PER_KWH)
            if used is not None:
                if scheme == SCHEME_SOUPORABA:
                    # Only the supplier's energy quantity falls.
                    taxable -= used[t] * r_energy[t]
                else:
                    # Community self-supply: the energy price AND the
                    # distribution part of the network tariff are replaced.
                    taxable -= used[t] * (r_energy[t] + r_net_full[t] - r_net_shared[t])
            obj_terms.append(VAT_FACTOR * taxable)

        if m.is_sender:
            for t in range(n_steps):
                obj_terms.append(-sell[t] * r_export[t])

    # ---- what the schemes take out of the producers' buyback contract, and
    # what the receivers pay for it instead. Settled on the POOL rather than per
    # producer: with one price list on the sending side (asserted here) the
    # community total does not depend on the split.
    if sharing:
        sender_paketi = {s.paket_id for s in senders}
        if len(sender_paketi) > 1:
            raise ValueError(
                "the sharing pool settles unused and consumed allocation at ONE "
                "buyback price -- every sender must hold the same price list, got "
                f"{sorted(sender_paketi)}"
            )
        r_export_pool = state[senders[0].member_id]["r_export"]
        for t in range(n_steps):
            used_total = pulp.lpSum(used_vars[m.member_id][t] for m in receivers)
            if scheme == SCHEME_SKUPNOST:
                # Allocation a member could not use returns to the pool, so
                # only the consumed part leaves the buyback contract.
                transferred_t = used_total
                paid_t = used_total
            elif sharing_mode == SHARING_STATIC:
                # ZOEE: `alpha` of the surplus leaves the buyback contract
                # whether or not a receiver could use it.
                transferred_t = alpha * pulp.lpSum(state[s.member_id]["sell"][t] for s in senders)
                paid_t = transferred_t if pay_for_unused else used_total
            else:
                transferred_t = pulp.lpSum(shared_vars[s.member_id][t] for s in senders)
                paid_t = transferred_t          # dynamic transfers nothing unusable
            obj_terms.append(transferred_t * r_export_pool[t])   # buyback given up
            # Taxable at the receiver, VAT-free at the producer, so the
            # transfer nets to a 22 % leak and the optimal internal price is 0.
            obj_terms.append(paid_t * p_int * (VAT_FACTOR - 1.0))

    prob += pulp.lpSum(obj_terms)

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    # Solve, look for metering points that imported and exported in the same
    # interval, force those to pick a side, solve again. Adding the binaries up
    # front costs two orders of magnitude more time for a model that usually
    # needs none. The first pass is a pure LP and exact; every later pass is a
    # MIP solved to `lazy_gap_rel`, because what the binaries remove is a
    # fraction of a euro out of a bill in the hundreds.
    lazy_solver = pulp.PULP_CBC_CMD(msg=False, gapRel=float(lazy_gap_rel),
                                    timeLimit=float(lazy_time_limit_s))
    solve_s = 0.0
    rounds = 0
    n_binaries = 0
    while True:
        t0 = time.time()
        # A build that already carries binaries is a MIP on the first pass, so
        # it gets the gapped solver too.
        prob.solve(solver if (rounds == 0 and n_upfront_binaries == 0) else lazy_solver)
        solve_s += time.time() - t0
        status = pulp.LpStatus[prob.status]
        # A gapped MIP returns "Optimal" when it closed the gap and an incumbent
        # otherwise; anything with no solution has no trajectory to extract.
        if status not in ("Optimal", "Not Solved") or prob.objective.value() is None:
            raise RuntimeError(f"community solve ended {status}, no usable solution")

        violations = _complementarity_violations(members, state)
        if not violations:
            break
        if rounds >= max_lazy_rounds:
            # Fail rather than return a trajectory no meter could have
            # recorded: the objective wants the trade faster than binaries can
            # forbid it. Raise the internal price or drop the storage.
            if strict:
                raise RuntimeError(
                    f"{len(violations)} metering points still import and export in the "
                    f"same interval after {rounds} lazy rounds and {n_binaries} binaries. "
                    f"This configuration is not solvable to a physical trajectory; see "
                    f"the note in the module docstring."
                )
            break
        for (mid, t, kind) in violations:
            st = state[mid]
            z = pulp.LpVariable(f"b{kind}_{mid}_{t}", cat="Binary")
            if kind == "grid":
                add_grid_exclusivity(
                    prob, buy_t=st["buy"][t], sell_t=st["sell"][t],
                    gen_t=st["gen"][t], con_t=st["con"][t],
                    max_charge_ac=st["max_ch"], max_discharge_ac=st["max_dis"], flag=z,
                )
                st["n_grid_binaries"] += 1
            else:
                add_battery_exclusivity(
                    prob, charge_t=st["ch"][t], discharge_t=st["dis"][t],
                    max_charge_ac=st["max_ch"], max_discharge_ac=st["max_dis"], flag=z,
                )
                st["n_battery_binaries"] += 1
            n_binaries += 1
        rounds += 1

    result = _extract(
        members=members, state=state, senders=senders, receivers=receivers,
        scheme=scheme, sharing_mode=sharing_mode, alpha=alpha, p_int=p_int,
        pay_for_unused=pay_for_unused, weights=weights, n_steps=n_steps,
        dates=dates, hours=hours, month_idx_t=month_idx_t, months_sorted=months_sorted,
        no_incentive=no_incentive,
    )
    result["objective_eur"] = float(pulp.value(prob.objective))
    result["solve_s"] = solve_s
    result["status"] = status
    result["lazy_rounds"] = rounds
    result["grid_binaries"] = n_binaries + n_upfront_binaries
    result["exposed_intervals"] = int(sum(len(state[m.member_id]["exposed"]) for m in members))
    result["unresolved_violations"] = len(_complementarity_violations(members, state))
    if verbose:
        print(f"  {scheme}/{sharing_mode} alpha={alpha:.2f}: "
              f"{result['community_cost_eur']:,.2f} EUR in {solve_s:.1f} s")
    return result


def _v(x):
    """Variable value, constant, or zero."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(x.varValue or 0.0)


def _safe_ratio(part, whole, tol=1e-12):
    """`part / whole`, zero where the whole is zero."""
    whole = np.asarray(whole, dtype=float)
    return np.divide(np.asarray(part, dtype=float), whole,
                     out=np.zeros_like(whole), where=np.abs(whole) > tol)


def _complementarity_violations(members, state, tol=1e-4):
    """Solved flows that no single metering point or inverter could produce.
    
    "grid" imported and exported in the same interval, which a meter nets before
    billing; "battery" charged and discharged in the same interval. The
    tolerance is in kWh, above solver noise and below anything that moves a bill.
    """
    out = []
    for m in members:
        st = state[m.member_id]
        for t in range(len(st["buy"])):
            if min(_v(st["buy"][t]), _v(st["sell"][t])) > tol:
                out.append((m.member_id, t, "grid"))
            if min(_v(st["ch"][t]), _v(st["dis"][t])) > tol:
                out.append((m.member_id, t, "battery"))
    return out


def _verify_no_simultaneous_battery(ch, dis, tol=1e-6):
    """Check that no battery charged and discharged in the same interval."""
    return int(sum(1 for c, d in zip(ch, dis) if _v(c) > tol and _v(d) > tol))


def _extract(*, members, state, senders, receivers, scheme, sharing_mode, alpha,
             p_int, pay_for_unused, weights, n_steps, dates, hours,
             month_idx_t, months_sorted, no_incentive=None):
    """Re-price the solved trajectory member by member.
    
    The bill is rebuilt from the flows rather than read off the objective, so a
    mistake shows up as a mismatch instead of propagating.
    """
    sharing = scheme != SCHEME_INDIVIDUAL
    rows, flows = [], []

    used_by_member = {
        m.member_id: np.array([_v(u) for u in state[m.member_id]["used"]])
        for m in receivers if state[m.member_id]["used"] is not None
    }
    used_total_t = (sum(used_by_member.values()) if used_by_member
                    else np.zeros(n_steps))

    # What each producer put INTO the scheme, per interval.
    sell_by_sender = {s.member_id: np.array([_v(x) for x in state[s.member_id]["sell"]])
                      for s in senders}
    sell_total_t = sum(sell_by_sender.values()) if senders else np.zeros(n_steps)
    transferred_by_sender = {}
    if sharing:
        for s in senders:
            if scheme == SCHEME_SKUPNOST:
                # Only the consumed part leaves the buyback contract, split
                # over the producers in proportion to what each put in.
                transferred_by_sender[s.member_id] = used_total_t * _safe_ratio(
                    sell_by_sender[s.member_id], sell_total_t)
            elif sharing_mode == SHARING_STATIC:
                transferred_by_sender[s.member_id] = alpha * sell_by_sender[s.member_id]
            else:
                transferred_by_sender[s.member_id] = np.array(
                    [_v(x) for x in state[s.member_id]["shared_out"]])
    transferred_total_t = (sum(transferred_by_sender.values()) if transferred_by_sender
                           else np.zeros(n_steps))

    community = dict(cost=0.0, energy=0.0, network=0.0, levies=0.0, fixed=0.0,
                     power=0.0, credit=0.0, internal_paid=0.0, internal_received=0.0)

    for m in members:
        st = state[m.member_id]
        buy = np.array([_v(x) for x in st["buy"]])
        sell = np.array([_v(x) for x in st["sell"]])
        ch = np.array([_v(x) for x in st["ch"]])
        dis = np.array([_v(x) for x in st["dis"]])
        spill = np.array([_v(x) for x in st["spill"]])
        gen, con = st["gen"], st["con"]
        r_energy, r_export = st["r_energy"], st["r_export"]
        r_net_full, r_net_shared = st["r_net_full"], st["r_net_shared"]

        used = used_by_member.get(m.member_id, np.zeros(n_steps))

        if sharing and m.is_sender:
            transferred = transferred_by_sender[m.member_id]
            # What the receivers could absorb out of what this producer put in.
            consumed = used_total_t * _safe_ratio(transferred, transferred_total_t)
            wasted = transferred - consumed
            paid_for = (transferred if (scheme == SCHEME_SOUPORABA and pay_for_unused)
                        else consumed)
            sold = sell - transferred          # the buyback contract keeps the rest
        else:
            transferred = consumed = paid_for = wasted = np.zeros(n_steps)
            sold = sell if m.is_sender else np.zeros(n_steps)

        # What the receiver is INVOICED for: the allocation it was given, which
        # under souporaba is not the allocation it used.
        if sharing and not m.is_sender:
            if scheme == SCHEME_SOUPORABA and sharing_mode == SHARING_STATIC and pay_for_unused:
                charged_internal = weights[m.member_id] * transferred_total_t
            else:
                charged_internal = used
        else:
            charged_internal = np.zeros(n_steps)

        # ---- the bill
        energy_eur = float(np.sum((buy - used) * r_energy))
        if scheme == SCHEME_SKUPNOST:
            network_eur = float(np.sum((buy - used) * r_net_full + used * r_net_shared))
        else:
            network_eur = float(np.sum(buy * r_net_full))
        levies_eur = float(np.sum(buy) * LEVIES_EUR_PER_KWH)
        internal_paid_eur = float(np.sum(charged_internal) * p_int)
        taxable = energy_eur + network_eur + levies_eur + internal_paid_eur
        variable_eur = VAT_FACTOR * taxable

        credit_eur = float(np.sum(sold * r_export))
        internal_received_eur = float(np.sum(paid_for) * p_int)

        # The excess-power charge, re-derived from the trajectory as the
        # objective built it: peak per (block, month), less the agreed power.
        power_eur = _excess_charge_eur(
            buy / hours, st["blocks"], month_idx_t, months_sorted, m.env,
            pravila=Pravila.za_leto(int(m.env.pricing_reference_year or PRIVZETO_REFERENCNO_LETO)),
        )

        total = variable_eur + power_eur + st["fixed_total"] + st["service_fee"] \
            - credit_eur - internal_received_eur

        rows.append({
            "Member": m.label,
            "Role": m.role,
            "Contract": m.contract_key or m.paket_id,
            "Paket": m.paket_id,
            "Capacity_kWh": st["cap"],
            "Consumption_kWh": float(np.sum(con)),
            "Generation_kWh": float(np.sum(gen)),
            "Import_kWh": float(np.sum(buy)),
            "Export_kWh": float(np.sum(sell)),
            "Sold_kWh": float(np.sum(sold)),
            "Curtailed_kWh": float(np.sum(spill)),
            "Charged_kWh": float(np.sum(ch)),
            "Discharged_kWh": float(np.sum(dis)),
            "Self_Consumed_kWh": float(np.sum(np.minimum(gen, con))),
            "Shared_Out_kWh": float(np.sum(transferred)),
            "Shared_Out_Used_kWh": float(np.sum(consumed)),
            "Shared_Wasted_kWh": float(np.sum(wasted)),
            "Shared_In_kWh": float(np.sum(used)),
            "Energy_EUR": VAT_FACTOR * energy_eur,
            "Network_EUR": VAT_FACTOR * network_eur,
            "Levies_EUR": VAT_FACTOR * levies_eur,
            "Internal_Paid_EUR": VAT_FACTOR * internal_paid_eur,
            "Internal_Received_EUR": internal_received_eur,
            "Fixed_EUR": st["fixed_total"],
            "Service_Fee_EUR": st["service_fee"],
            "Power_EUR": power_eur,
            "Buyback_Credit_EUR": credit_eur,
            "Cost_EUR": total,
            "Peak_Import_kW": float(np.max(buy) / hours) if n_steps else 0.0,
            "Simultaneous_Battery_Intervals": _verify_no_simultaneous_battery(st["ch"], st["dis"]),
            # Import and export in the same interval: anything left here is a
            # place one meter could not have recorded the flows.
            "Simultaneous_Grid_Intervals": int(np.sum((buy > 1e-6) & (sell > 1e-6))),
            "Grid_Binaries": st["n_grid_binaries"],
            "Battery_Binaries": st["n_battery_binaries"],
            "Export_Rate_Floored_Intervals": st["n_floored"],
            "No_Sharing_Incentive_Intervals": int((no_incentive or {}).get(m.member_id, 0)),
        })

        community["cost"] += total
        community["energy"] += VAT_FACTOR * energy_eur
        community["network"] += VAT_FACTOR * network_eur
        community["levies"] += VAT_FACTOR * levies_eur
        community["fixed"] += st["fixed_total"] + st["service_fee"]
        community["power"] += power_eur
        community["credit"] += credit_eur
        community["internal_paid"] += VAT_FACTOR * internal_paid_eur
        community["internal_received"] += internal_received_eur

        flows.append(pd.DataFrame({
            "Date": dates,
            "Member": m.label,
            "Import_kWh": buy,
            "Export_kWh": sell,
            "Shared_In_kWh": used,
            "Shared_Out_kWh": transferred,
            "Shared_Wasted_kWh": wasted,
            "Charge_kWh": ch,
            "Discharge_kWh": dis,
            "Curtailed_kWh": spill,
        }))

    per_member = pd.DataFrame(rows)
    return {
        "per_member": per_member,
        "flows": pd.concat(flows, ignore_index=True),
        "community_cost_eur": community["cost"],
        "community": community,
        "scheme": scheme,
        "sharing_mode": sharing_mode,
        "share_of_surplus": alpha,
        "internal_price_eur_kwh": p_int,
        "shared_kwh": float(per_member["Shared_In_kWh"].sum()),
        "shared_wasted_kwh": float(per_member["Shared_Wasted_kWh"].sum()),
        "transferred_kwh": float(per_member["Shared_Out_kWh"].sum()),
    }


def _excess_charge_eur(power_kw, blocks, month_idx_t, months_sorted, env, pravila):
    """The ratchet excess-power charge for one solved trajectory, ex VAT.
    
    Per block and month, the highest quarter-hour power above the agreed billing
    power, at that month's season-correct postavka.
    """
    total = 0.0
    peaks: Dict[tuple, float] = {}
    for t in range(len(power_kw)):
        key = (int(blocks[t]), month_idx_t[t])
        if power_kw[t] > peaks.get(key, 0.0):
            peaks[key] = float(power_kw[t])
    for (b, mi), peak in peaks.items():
        y, mo = months_sorted[mi]
        agreed = moc_za_mesec(env.agreed_power_for_run(0), y * 12 + mo - 1).get(b, 0.0)
        if peak <= agreed:
            continue
        rate = pravila.omreznina.postavka_moc(b, je_visja_sezona(date(y, mo, 1)))
        total += (peak - agreed) * rate * pravila.omreznina.faktor_presezne_moci
    return total


# --------------------------------------------------------------------------
# A whole year, month by month
# --------------------------------------------------------------------------
def month_slices(index, min_intervals=0):
    """(start_idx, n_steps, (year, month)) per calendar month, Slovenian local time.
    
    The month is the billing unit -- power charge, OVE+SPTE and the ratchet all
    reset on the 1st -- so cutting the year here costs nothing on the tariff side.
    `min_intervals` folds a short slice into its neighbour: the Fluvius stamps are
    Brussels-local despite the `Z` suffix, so a year spills a few intervals into
    the following January.
    """
    local_months = np.array([(v_lokalni_cas(ts).year, v_lokalni_cas(ts).month) for ts in index])
    keys = [tuple(int(v) for v in k) for k in local_months]
    out = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            out.append((start, i - start, keys[start]))
            start = i

    if min_intervals > 0:
        merged = []
        for (s, n, key) in out:
            if merged and n < min_intervals:
                ps, pn, pkey = merged[-1]
                merged[-1] = (ps, pn + n, pkey)
            else:
                merged.append((s, n, key))
        # A short first slice folds its successor into itself instead.
        while len(merged) > 1 and merged[0][1] < min_intervals:
            s, n, key = merged.pop(0)
            ns, nn, nkey = merged[0]
            merged[0] = (s, n + nn, key)
        out = merged
    return out


def run_community_year(
    members: Sequence[CommunityMember],
    *,
    scheme: str,
    sharing_mode: str = SHARING_DYNAMIC,
    share_of_surplus: float = 1.0,
    internal_price_eur_kwh: float = 0.0,
    pay_for_unused: bool = True,
    service_id: Optional[str] = None,
    soc_fraction: float = 0.5,
    solver_kwargs: Optional[dict] = None,
    ref_year: Optional[int] = None,
    slices: Optional[Sequence] = None,
    keep_flows: bool = False,
    strict: bool = True,
    verbose: bool = False,
):
    """The whole horizon, one solve per calendar month, summed.
    
    Each month opens and closes at `soc_fraction` of every battery's capacity, so
    no month can raid another's storage.
    """
    members = list(members)
    slices = slices if slices is not None else month_slices(members[0].env.dataset.index)
    solver_kwargs = solver_kwargs or dict(msg=False)

    monthly = []
    for (start, length, key) in slices:
        res = solve_community_milp(
            members, scheme=scheme, start_idx=start, n_steps=length,
            sharing_mode=sharing_mode, share_of_surplus=share_of_surplus,
            internal_price_eur_kwh=internal_price_eur_kwh,
            pay_for_unused=pay_for_unused, service_id=service_id,
            soc_fraction=soc_fraction, solver=pulp.PULP_CBC_CMD(**solver_kwargs),
            ref_year=ref_year, strict=strict, verbose=False,
        )
        res["month"] = f"{key[0]:04d}-{key[1]:02d}"
        monthly.append(res)
        if verbose:
            print(f"    {res['month']}: {res['community_cost_eur']:>10,.2f} EUR  "
                  f"({res['solve_s']:.1f} s)", flush=True)

    return combine_months(monthly, keep_flows=keep_flows)


def combine_months(monthly, keep_flows=False):
    """Add up monthly settlements into one period result."""
    keys = ["Member", "Role", "Contract", "Paket", "Capacity_kWh"]
    frames = pd.concat([r["per_member"] for r in monthly], ignore_index=True)
    # Monthly totals add up; the peak is the highest month's.
    how = {c: ("max" if c == "Peak_Import_kW" else "sum")
           for c in frames.columns if c not in keys}
    per_member = frames.groupby(keys, as_index=False, sort=False).agg(how)

    first = monthly[0]
    community = {k: sum(r["community"][k] for r in monthly) for k in first["community"]}
    out = {
        "per_member": per_member,
        "community_cost_eur": float(per_member["Cost_EUR"].sum()),
        "community": community,
        "scheme": first["scheme"],
        "sharing_mode": first["sharing_mode"],
        "share_of_surplus": first["share_of_surplus"],
        "internal_price_eur_kwh": first["internal_price_eur_kwh"],
        "shared_kwh": float(per_member["Shared_In_kWh"].sum()),
        "shared_wasted_kwh": float(per_member["Shared_Wasted_kWh"].sum()),
        "transferred_kwh": float(per_member["Shared_Out_kWh"].sum()),
        "solve_s": sum(r["solve_s"] for r in monthly),
        # How much of the year needed a binary, and what the loop left unfixed.
        "lazy_rounds": max(r.get("lazy_rounds", 0) for r in monthly),
        "grid_binaries": sum(r.get("grid_binaries", 0) for r in monthly),
        "unresolved_violations": sum(r.get("unresolved_violations", 0) for r in monthly),
        "months": [r["month"] for r in monthly],
        "monthly": pd.DataFrame([
            {"Month": r["month"], "Cost_EUR": r["community_cost_eur"],
             "Shared_kWh": r["shared_kwh"], "Wasted_kWh": r["shared_wasted_kwh"],
             "Solve_s": r["solve_s"]}
            for r in monthly
        ]),
    }
    if keep_flows:
        out["flows"] = pd.concat([r["flows"] for r in monthly], ignore_index=True)
    return out


def summarize_run(result, label=""):
    """One row per run, for the comparison table."""
    pm = result["per_member"]
    return {
        "Scenario": label,
        "Scheme": result["scheme"],
        "Sharing": result["sharing_mode"],
        "Share": result["share_of_surplus"],
        "Internal_price": result["internal_price_eur_kwh"],
        "Community_cost_EUR": result["community_cost_eur"],
        "Shared_kWh": result["shared_kwh"],
        "Transferred_kWh": result["transferred_kwh"],
        "Wasted_kWh": result["shared_wasted_kwh"],
        "Import_kWh": float(pm["Import_kWh"].sum()),
        "Export_kWh": float(pm["Export_kWh"].sum()),
        "Curtailed_kWh": float(pm["Curtailed_kWh"].sum()),
        "Peak_Import_kW": float(pm["Peak_Import_kW"].max()),
        "Sum_Peaks_kW": float(pm["Peak_Import_kW"].sum()),
        "Solve_s": result.get("solve_s", float("nan")),
        "Lazy_rounds": result.get("lazy_rounds", 0),
        "Binaries": result.get("grid_binaries", 0),
        "Unresolved": result.get("unresolved_violations", 0),
    }
