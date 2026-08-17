"""Perfect-foresight community MILP: skupnostna samooskrba vs souporaba.

`MILP_Benchmark.run_milp_benchmark` optimizes ONE household against ONE price
list. Nothing in the repository optimizes a GROUP, and the two Slovenian
group schemes are not the same scheme with a different name -- they settle
different quantities, at different tariffs, with different rules about what
happens to energy nobody used. This module is the group version: one MILP over
every member of a community, solving

  * every member's battery (the same continuous formulation the single-household
    MILP uses -- charge, discharge, curtailment, SOC, one energy balance per
    member per interval), and
  * how much of each producer's surplus is SHARED rather than sold to its
    supplier, and who receives it,

against one objective -- the community's total electricity bill.

---------------------------------------------------------------------------
THE TWO SCHEMES
---------------------------------------------------------------------------
Both move a producing member's surplus to a consuming member over the public
grid. Neither moves an electron differently from the no-sharing case: the
meters register exactly what they registered before, and the schemes are
BILLING overlays on top of the same physical flows. What they change is what
those metered kWh cost.

SKUPNOSTNA SAMOOSKRBA (community self-supply, `si_obracun.skupnost`)
  A shared kWh at the receiver is billed as
      community energy price + REDUCED network energy tariff + levies
  where the reduced tariff is the transmission postavka plus the *community*
  distribution postavka `energija_skupnost[znacilni_primer][blok]` -- for
  `znacilni_primer = 1` (members behind the same transformer) that second term
  is 0.00000 EUR/kWh, and the distribution component of the network charge
  disappears entirely. The grid remainder is billed at the full tariff.
  Allocation that a member cannot use in the interval is NOT destroyed: it goes
  back to the pool and is sold at the producer's own buyback price.

SOUPORABA ELEKTRICNE ENERGIJE (energy sharing under ZOEE, `si_obracun.souporaba_*`)
  A shared kWh reduces the ENERGY quantity the receiver's supplier invoices --
  and nothing else. Network charge, levies and excise stay on the FULL metered
  offtake. Allocation the receiver cannot consume inside the same 15-minute
  interval is destroyed: it does not carry to the next interval, earns no
  credit, and (by default) the producer is still paid for it while the
  receiver's supplier keeps the energy. The organizer charges a monthly fee per
  metering point (`si_paketi.STORITVE_SOUPORABE`).

So the per-kWh advantage of skupnostna samooskrba over souporaba is exactly the
distribution part of the network tariff that souporaba keeps charging, and its
second advantage is that it wastes nothing. Souporaba's advantages are
regulatory rather than arithmetic: no shared production device is required, the
members may sit anywhere and hold different suppliers, and -- the one that
decides the answer in this repository -- GEN-I's *dinamicni samooskrba* price
list, the cheapest list in the battery-sizing study, carries
`dovoljuje_skupnostno=False` and cannot be held inside a community self-supply
scheme at all.

---------------------------------------------------------------------------
THE SHARE OF SHARED ENERGY (`delez deljene energije`)
---------------------------------------------------------------------------
Under ZOEE the share is a CONTRACT NUMBER: one static percentage of the
producer's surplus, split over the receivers by a static key, applied to every
15-minute interval of the year. It cannot react to whether a receiver is
actually drawing power right now, which is the whole reason souporaba wastes
energy. This module therefore offers two sharing modes:

  SHARING_STATIC   one share `alpha` of every producer's surplus, split by
                   fixed weights. This is the contract as it is actually
                   written, and `alpha` is what the sweep in the notebook
                   solves for -- one MILP per candidate share, the community
                   optimum read off the resulting curve.
  SHARING_DYNAMIC  the allocation is a decision variable per interval, bounded
                   by the surplus that exists and by what each receiver can
                   absorb. No static key can beat it, so it is the upper bound
                   on any sharing rule, and the gap between it and the best
                   static share is the price of the contract being static.

---------------------------------------------------------------------------
WHY THIS STAYS AN LP ALMOST EVERYWHERE
---------------------------------------------------------------------------
Two structural facts keep a community of sixteen households solvable:

1. RECEIVERS CANNOT EXPORT. A receiver holds a plain (brez odkupa) price list,
   whose export credit is identically zero, so no solution ever wants
   `sell > 0` at a receiver and the buy/sell complementarity that would
   otherwise need one binary per member per interval is implied by the
   objective. `assert_no_receiver_buyback` enforces the precondition rather
   than assuming it -- with a positive buyback at a receiver the model could
   import a shared kWh at the reduced tariff and export it at the full buyback
   price in the same interval, which is not something a single meter can do.
2. SIMULTANEOUS CHARGE AND DISCHARGE is only ever profitable where the
   delivered import rate is negative, which happens on the SIPX-linked lists
   and in about 140 intervals of 2024. `ch + dis <= inverter rating` -- one
   inverter, one rating -- is free and holds always; whatever is left is caught
   after the solve.

Both are normally enforced LAZILY: the first pass is a pure LP, the solved
trajectory is checked for flows no meter or inverter could produce, and a binary
is spent only where one actually occurred. Sixteen members over a month is a
~270 000-column model, and handing it a few thousand binaries up front turns a
17-second LP into an hour of branching for a correction worth about one euro.

The exception is the one case where the objective wants the trade EVERYWHERE it
is allowed -- community self-supply, a producer on a single-rate list beside a
receiver on a VT/MT one, and an internal price too low for its VAT to hold the
trade below profitable. Fixing one interval there hands the same trade to the
next, so the lazy loop converges badly and those intervals are given their
binary up front instead. `exposed_intervals` reports how many that was; in
every other configuration it is zero and the model is a pure LP.

The horizon is decomposed by CALENDAR MONTH, which is not an approximation for
the tariff: the network power charge, the OVE+SPTE levy and the excess-power
ratchet all reset monthly (`peak_reset_months=1`), so a month is a complete
billing unit. It IS an approximation for the battery -- each month opens and
closes at `soc_fraction` of capacity, so no month can borrow storage from
December -- and that is the same convention the annual single-household solves
already use at the year boundary.
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

# The community settlement primitives live in the spaced folder, exactly as
# `multi_household_tools` reaches them.
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
# Charged on every metered kWh under both schemes -- souporaba says so
# explicitly, and `si_obracun.skupnost` charges them on `iz_omrezja + deljena`,
# which is the same total.
LEVIES_EUR_PER_KWH = TROSARINA_EUR_KWH + URE_EUR_KWH + OPERATER_TRGA_EUR_KWH
VAT_FACTOR = 1.0 + float(DDV)


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------
@dataclass
class CommunityMember:
    """One metering point in the community.

    `env` is a fully built `HouseholdEnvironment` -- the profile, the battery,
    the price list and the agreed billing power all come off it, so a member
    here is priced by exactly the same machinery as a household in the
    single-household study and the two are comparable line by line.

    `role` is the ZOEE role. A sender puts surplus into the scheme; a receiver
    takes allocation out of it. A member is not both: see the module docstring
    for why receiving at a metering point that can also export needs a binary
    per interval to stay physical.

    `key_weight` is that receiver's static share of the pool (the `delitev` of
    `obracun_souporabe`); weights are normalized over the receivers.

    `znacilni_primer` is the "typical connection case" 1-10 that picks the
    community distribution postavka in `si_tarife.Omreznina.energija_skupnost`.
    1 = members on the same transformer station (0.00000 EUR/kWh), 4 = the full
    distribution postavka, i.e. no reduction at all.
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

    Everything the objective needs that depends on the PRICE LIST rather than
    on the household: the supplier energy price, the buyback price and the
    tariff block. Households on the same list over the same calendar share one
    table -- which is the whole reason a sixteen-member year is affordable to
    price: sixteen households on four lists cost four passes, not sixteen.

    Rates come out of `si_obracun.samooskrba`, the same function the invoices
    and the RL environment bill through, so a per-kWh rate here is the per-kWh
    rate there.
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
    """Network ENERGY postavke per interval, ex VAT.

    `full` is what a kWh out of the grid costs; `shared` is what the same kWh
    costs when it arrives as a community allocation -- transmission plus the
    community distribution postavka for this connection case. The difference
    between them is the entire per-kWh advantage of skupnostna samooskrba over
    souporaba, and it is zero by construction at `znacilni_primer = 4`.
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

    With a positive buyback at a receiver the LP can import a shared kWh at the
    reduced community tariff and export it at the full buyback price inside the
    same interval, netting the difference -- a trade a single metering point
    cannot make, because the meter nets import against export within the
    interval before either is billed. Enforcing the precondition costs one
    assert; the alternative is one binary per member per interval.
    """
    offenders = []
    for m in members:
        if m.is_sender:
            continue
        paket = PAKETI[m.paket_id]
        # NET metering is excluded for a second, legal reason as well: a
        # household settling its exports once a year cannot be a receiver in
        # souporaba at all (`si_paketi.preveri_souporabo`).
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

    Returns a dict with the per-member settlement, the community total and the
    per-interval flow table. Every euro in it is VAT-inclusive on the charge
    side and VAT-free on the credit side, which is how `si_obracun.Racun`
    settles a bill: `za_placilo = neto * 1.22 - dobropis`.

    Parameters that carry the scheme
    --------------------------------
    scheme                  SCHEME_INDIVIDUAL disables sharing entirely and
                            gives every member the bill it would get on its own
                            -- the baseline the two schemes are measured
                            against, solved by the same code so nothing but the
                            settlement differs.
    sharing_mode            SHARING_STATIC applies `share_of_surplus` to every
                            producer's surplus in every interval and splits it
                            by the members' `key_weight`; SHARING_DYNAMIC lets
                            the solver place the allocation.
    internal_price_eur_kwh  What the receiver pays the producer per shared kWh.
                            It is a TRANSFER between two members, so it moves
                            the split of the community bill and not, in the
                            main, its total -- except that the receiver's side
                            is a taxable invoice item while the producer's side
                            is a VAT-free credit, exactly as the buyback credit
                            is ("odkup presezka ... ni predmet obdavcitve z
                            DDV"). Every euro of internal price therefore costs
                            the community 22 cents of VAT, which is why 0.00 is
                            the community-optimal price and any positive value
                            is a fairness choice, not an efficiency one.
    pay_for_unused          ZOEE default: the producer is paid for what it
                            transferred, whether or not the receiver could use
                            it. Only affects souporaba, and only the split.
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

    # Calendar bookkeeping, shared by every member: the block a kWh falls in and
    # the month it is billed in are properties of the clock, not of the
    # household.
    local_times = [v_lokalni_cas(dates[t]) for t in range(n_steps)]
    month_key_t = [(lt.year, lt.month) for lt in local_times]
    months_sorted = sorted(set(month_key_t))
    month_idx_t = [months_sorted.index(k) for k in month_key_t]

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

    # ---- pass 1: the rates. Every member's, before any variable exists,
    # because whether a producer needs a complementarity binary depends on what
    # a shared kWh is worth at the RECEIVERS.
    rates: Dict[str, dict] = {}
    for m in members:
        table = paket_rate_table(m.paket_id, dates, smp, interval_minutes, ref_year)
        r_net_full, r_net_shared = network_rate_tables(table["blocks"], ref_year, m.znacilni_primer)
        # The delivered cost of one imported kWh, VAT included -- the number the
        # single-household MILP calls `import_rates`.
        delivered = VAT_FACTOR * (table["energy"] + r_net_full + LEVIES_EUR_PER_KWH)
        # A buyback worth more than an import costs is an unbounded loop: buy and
        # sell the same kWh forever. Real on the SIPX-linked lists, where a deeply
        # negative market price makes the delivered import rate negative while the
        # buyback stays at zero. Flooring makes the round trip exactly neutral;
        # curtailment is free, so the optimum never exports there anyway.
        rates[m.member_id] = dict(
            r_energy=table["energy"], blocks=table["blocks"],
            r_net_full=r_net_full, r_net_shared=r_net_shared, delivered=delivered,
            r_export=np.minimum(table["export"], delivered),
            n_floored=int(np.sum(table["export"] > delivered)),
        )

    # What one shared kWh displaces at the best-placed receiver, ex VAT. Under
    # souporaba only the supplier's energy price; under community self-supply the
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
        max_ch = env.max_charge_kwh / env.charge_efficiency
        max_dis = env.max_discharge_kwh * env.discharge_efficiency

        buy = [pulp.LpVariable(f"buy_{m.member_id}_{t}", lowBound=0) for t in range(n_steps)]
        sell = [pulp.LpVariable(f"sell_{m.member_id}_{t}", lowBound=0) for t in range(n_steps)]
        spill = [pulp.LpVariable(f"spill_{m.member_id}_{t}", lowBound=0, upBound=float(gen[t]))
                 for t in range(n_steps)]
        if cap > 0:
            ch = [pulp.LpVariable(f"ch_{m.member_id}_{t}", lowBound=0, upBound=max_ch)
                  for t in range(n_steps)]
            dis = [pulp.LpVariable(f"dis_{m.member_id}_{t}", lowBound=0, upBound=max_dis)
                   for t in range(n_steps)]
            soc = [pulp.LpVariable(f"soc_{m.member_id}_{t}", lowBound=0, upBound=cap)
                   for t in range(n_steps + 1)]
            prob += soc[0] == soc_fraction * cap
            prob += soc[n_steps] >= soc_fraction * cap
            inverter_kwh = max(max_ch, max_dis)
            for t in range(n_steps):
                prob += soc[t + 1] == soc[t] + ch[t] * env.charge_efficiency - dis[t] / env.discharge_efficiency
                # One inverter, one rating: whatever the split between charging
                # and discharging, their sum cannot exceed what the unit can pass.
                # Free for any single-direction operating point, and it halves the
                # only thing a solver would use a simultaneous pair for -- burning
                # energy through the round-trip loss when a kWh is worth less than
                # nothing. The rest of that is caught after the solve.
                prob += ch[t] + dis[t] <= inverter_kwh
        else:
            ch = [0.0] * n_steps
            dis = [0.0] * n_steps
            soc = None

        for t in range(n_steps):
            prob += gen[t] + buy[t] + dis[t] == con[t] + sell[t] + ch[t] + spill[t]
            # A meter nets import against export INSIDE the interval, so a
            # household is either importing or exporting, never both. These two
            # inequalities are implied by that and hold at every physical
            # operating point -- the export cannot exceed the production left
            # after the member's own load plus what the battery gives back, and
            # the import cannot exceed the load the production did not cover
            # plus what the battery and a spill take. They also bound the LP,
            # which the energy balance alone does not: `buy` and `sell` can
            # otherwise both run away together.
            prob += sell[t] <= max(gen[t] - con[t], 0.0) + dis[t]
            prob += buy[t] <= max(con[t] - gen[t], 0.0) + ch[t] + spill[t]

        # Where those two inequalities are not enough. Importing and exporting
        # the same kWh in one interval is normally self-punishing
        # (`r_export <= delivered` by construction), but a shared kWh can be
        # worth more at the receiver than a bought one costs at the producer.
        # Buying one kWh, "exporting" it and having it consumed as allocation
        # moves the community bill by
        #
        #     1.22 x (energy + network + levies at the producer)
        #   - 1.22 x (displaced at the best-placed receiver)
        #   + 0.22 x internal price          <- the VAT the transfer leaks
        #
        # and wherever that is negative the model will do it: "export" PV it is
        # simultaneously buying back for its own load, a trade one meter cannot
        # make because it nets the two before billing either. It is not
        # hypothetical -- a producer on the single-rate samooskrba list beside a
        # receiver on the VT/MT plain list, under community self-supply, in VT
        # hours, is exactly it, and a zero internal price removes the VAT term
        # that otherwise holds it just below profitable.
        #
        # Those intervals get their binary UP FRONT. The lazy loop below is kept
        # as a backstop, but it converges badly here: fixing one interval hands
        # the same trade to the next, and a run left 991 of them unresolved after
        # six rounds and 8 500 binaries. Every other configuration has an empty
        # `exposed` and stays a pure LP.
        exposed = (np.flatnonzero(
            (VAT_FACTOR * (r_energy + r_net_full + LEVIES_EUR_PER_KWH - displaced)
             + (VAT_FACTOR - 1.0) * p_int < 0.0) & (gen > 0.0))
            if (sharing and m.is_sender) else np.array([], dtype=int))
        for t in exposed:
            z = pulp.LpVariable(f"bgrid0_{m.member_id}_{t}", cat="Binary")
            prob += sell[t] <= (float(gen[t]) + max_dis) * z
            prob += buy[t] <= (float(con[t]) + max_ch + float(gen[t])) * (1 - z)
        n_upfront_binaries += len(exposed)

        # ---- the ratchet excess-power charge, one variable per (block, month).
        # Lifted from MILP_Benchmark so the two agree euro for euro on the same
        # trajectory: the peak is measured on the METERED import `buy`, which no
        # billing overlay can move -- sharing changes what a kWh costs, never
        # when it crossed the meter.
        agreed_by_month = {
            i: moc_za_mesec(env.agreed_power_for_run(start_idx), y * 12 + mo - 1)
            for i, (y, mo) in enumerate(months_sorted)
        }
        seed_peak = env.compute_seed_peak_kw(start_idx)
        window_ids = env.reset_window_ids[horizon]
        seed_window = int(window_ids[0])
        month_window = {}
        for t in range(n_steps):
            month_window.setdefault(month_idx_t[t], int(window_ids[t]))

        occurring = sorted({(int(blocks[t]), month_idx_t[t]) for t in range(n_steps)})
        peak_var = {bm: pulp.LpVariable(f"peak_{m.member_id}_b{bm[0]}_m{bm[1]}", lowBound=0)
                    for bm in occurring}
        excess_var = {bm: pulp.LpVariable(f"exc_{m.member_id}_b{bm[0]}_m{bm[1]}", lowBound=0)
                      for bm in occurring}
        for t in range(n_steps):
            prob += peak_var[(int(blocks[t]), month_idx_t[t])] >= buy[t] / hours

        last_var, last_window = {}, {}
        for (b, mi) in occurring:
            w = month_window[mi]
            if b in last_var and last_window[b] == w:
                prob += peak_var[(b, mi)] >= last_var[b]
            elif w == seed_window:
                prob += peak_var[(b, mi)] >= seed_peak.get(b, 0.0)
            last_var[b] = peak_var[(b, mi)]
            last_window[b] = w
            prob += excess_var[(b, mi)] >= peak_var[(b, mi)] - agreed_by_month[mi].get(b, 0.0)

        prev_exc, prev_win = {}, {}
        for (b, mi) in occurring:
            y, mo = months_sorted[mi]
            w = month_window[mi]
            rate_bm = pravila.omreznina.postavka_moc(b, je_visja_sezona(date(y, mo, 1)))
            faktor = pravila.omreznina.faktor_presezne_moci
            if b in prev_exc and prev_win[b] == w:
                prev_term = prev_exc[b]
            elif w == seed_window:
                prev_term = max(0.0, seed_peak.get(b, 0.0) - agreed_by_month[mi].get(b, 0.0))
            else:
                prev_term = 0.0
            obj_terms.append((excess_var[(b, mi)] - prev_term) * rate_bm * faktor)
            prev_exc[b] = excess_var[(b, mi)]
            prev_win[b] = w

        # ---- the decision-independent fixed monthly charge, prorated. Same
        # helper the single-household MILP uses, so the two carry the same
        # network power fee, OVE+SPTE levy and supplier monthly fee.
        fixed_by_month = {}
        for mi, (y, mo) in enumerate(months_sorted):
            first_t = month_idx_t.index(mi)
            fixed_by_month[mi] = compute_prorated_fixed_charge_eur(
                dates[first_t], interval_minutes, scheme=env.pricing_scheme,
                dogovorjena_moc=agreed_by_month[mi], paket_id=m.paket_id,
                pricing_reference_year=ref_year,
            )
        fixed_total = sum(fixed_by_month[month_idx_t[t]] for t in range(n_steps))

        # ---- the organizer's monthly fee, souporaba only. A taxable invoice
        # item per metering point per month, charged on the role.
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
                # The contract as written: a fixed fraction of whatever surplus
                # the interval happens to produce, split by a fixed key. `alpha`
                # is a parameter, `sell` a variable, so the product stays linear
                # and the dispatch is still free to decide how big the surplus is.
                pool_t = alpha * pulp.lpSum(state[s.member_id]["sell"][t] for s in senders)
                for m in receivers:
                    prob += used_vars[m.member_id][t] <= weights[m.member_id] * pool_t
            else:
                for s in senders:
                    prob += shared_vars[s.member_id][t] <= state[s.member_id]["sell"][t]
                # Equality, not <=: energy put into the scheme but not consumed
                # would otherwise be an accounting free lunch under
                # `pay_for_unused`, paid for by nobody. A dynamic scheme has no
                # reason to transfer what cannot be used, so this only removes
                # solutions that are artefacts.
                prob += (pulp.lpSum(used_vars[m.member_id][t] for m in receivers)
                         == pulp.lpSum(shared_vars[s.member_id][t] for s in senders))
            for m in receivers:
                # A shared kWh is an allocation ON the metered offtake, never on
                # top of it: it can only displace energy the member actually drew.
                prob += used_vars[m.member_id][t] <= state[m.member_id]["buy"][t]

    # ----------------------------------------------------------------------
    # Objective: the community's bill
    # ----------------------------------------------------------------------
    # ---- what every member pays on its own metered offtake, and what the
    # producers are credited for what they exported.
    no_incentive = {}
    for m in members:
        st = state[m.member_id]
        buy, sell = st["buy"], st["sell"]
        r_energy, r_export = st["r_energy"], st["r_export"]
        r_net_full, r_net_shared = st["r_net_full"], st["r_net_shared"]
        used = st["used"]

        # The per-kWh discount a shared kWh carries at this member. Negative =
        # the member is better off with the allocation, which is what makes the
        # solver take as much of it as the rule allows. It can only turn
        # non-negative on a SIPX-linked list in an interval whose market price
        # has fallen below the internal price; counted, not silently allowed,
        # because there the model would decline an allocation the regulation
        # applies automatically.
        if used is not None:
            delta = (p_int - r_energy) if scheme == SCHEME_SOUPORABA else (
                p_int - r_energy - r_net_full + r_net_shared)
            no_incentive[m.member_id] = int(np.sum(delta >= 0.0))

        for t in range(n_steps):
            # Taxable base, ex VAT, then grossed up -- the order si_obracun.Racun
            # settles in. The internal price is NOT here: it is charged on the
            # allocation the member was given, which under souporaba is not the
            # same as the allocation it managed to use, and both sides of it are
            # settled on the pool below.
            taxable = buy[t] * (r_energy[t] + r_net_full[t] + LEVIES_EUR_PER_KWH)
            if used is not None:
                if scheme == SCHEME_SOUPORABA:
                    # Only the supplier's energy quantity falls. Network charge,
                    # excise and levies stay on the full metered offtake.
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
    # what the receivers pay them for it instead. Both are settled on the POOL
    # rather than per producer: which producer's electron a receiver consumed is
    # not a question the meters can answer, and with one price list on the
    # sending side (asserted here) the community total does not depend on the
    # answer either.
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
                # Nothing is destroyed: allocation a member could not use goes
                # back to the pool and is sold at the producers' own buyback
                # price, so only the CONSUMED part leaves the buyback contract.
                transferred_t = used_total
                paid_t = used_total
            elif sharing_mode == SHARING_STATIC:
                # ZOEE: `alpha` of the surplus leaves the buyback contract
                # whether or not a receiver could use it, and under the default
                # contract term the producer is paid for all of it.
                transferred_t = alpha * pulp.lpSum(state[s.member_id]["sell"][t] for s in senders)
                paid_t = transferred_t if pay_for_unused else used_total
            else:
                transferred_t = pulp.lpSum(shared_vars[s.member_id][t] for s in senders)
                paid_t = transferred_t          # dynamic transfers nothing unusable
            obj_terms.append(transferred_t * r_export_pool[t])   # buyback given up
            # The internal price is a taxable invoice item at the receiver and a
            # VAT-free credit at the producer, exactly as the buyback credit is
            # ("odkup presezka ... ni predmet obdavcitve z DDV"). The transfer
            # therefore nets to a 22 % VAT leak: the community pays it, neither
            # member keeps it, and the community-optimal internal price is 0.
            obj_terms.append(paid_t * p_int * (VAT_FACTOR - 1.0))

    prob += pulp.lpSum(obj_terms)

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    # Solve, look for metering points that imported and exported in the same
    # interval, force the ones that did to pick a side, solve again. Adding the
    # binaries up front instead costs two orders of magnitude more time for a
    # model that -- as the round counter reports -- usually needs none of them.
    #
    # The first pass is a pure LP and is solved exactly. Every later pass is a
    # MIP, and it is solved to `lazy_gap_rel` rather than to proven optimality:
    # what those binaries remove is a fraction of a euro of unphysical trade out
    # of a bill in the hundreds, and CBC will otherwise spend an hour closing a
    # gap that is already far below the thing being measured. Whichever bound it
    # reached is reported.
    #
    # The battery binaries are the other reason a pass can be a MIP: on the
    # SIPX-linked lists a handful of intervals a year have a negative delivered
    # import rate, and each of those needs one. Sixteen members times those
    # intervals is enough to keep CBC branching for an hour on a 270 000-column
    # model, so a build that created any binary at all uses the gapped solver
    # from the start.
    lazy_solver = pulp.PULP_CBC_CMD(msg=False, gapRel=float(lazy_gap_rel),
                                    timeLimit=float(lazy_time_limit_s))
    solve_s = 0.0
    rounds = 0
    n_binaries = 0
    while True:
        t0 = time.time()
        # A build that already carries binaries is a MIP on the first pass, and
        # proving optimality on one of those costs hours for a correction worth
        # about a euro -- so it gets the gapped solver too.
        prob.solve(solver if (rounds == 0 and n_upfront_binaries == 0) else lazy_solver)
        solve_s += time.time() - t0
        status = pulp.LpStatus[prob.status]
        # A gap-limited or time-limited MIP comes back "Optimal" from CBC when it
        # closed the gap and with an incumbent otherwise; anything with no
        # solution at all has no trajectory to extract and must not be reported.
        if status not in ("Optimal", "Not Solved") or prob.objective.value() is None:
            raise RuntimeError(f"community solve ended {status}, no usable solution")

        violations = _complementarity_violations(members, state)
        if not violations:
            break
        if rounds >= max_lazy_rounds:
            # Better to fail than to return a trajectory no meter could have
            # recorded. A caller that reaches this has found a configuration
            # where the objective wants the trade faster than binaries can
            # forbid it -- community self-supply, mismatched price lists and an
            # internal price too low for its VAT to hold the trade below
            # profitable is the known one. Raise the internal price, drop the
            # storage, or accept that this combination is not solvable here.
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
                prob += st["sell"][t] <= (float(st["gen"][t]) + st["max_dis"]) * z
                prob += st["buy"][t] <= (float(st["con"][t]) + st["max_ch"]
                                         + float(st["gen"][t])) * (1 - z)
                st["n_grid_binaries"] += 1
            else:
                prob += st["ch"][t] <= st["max_ch"] * z
                prob += st["dis"][t] <= st["max_dis"] * (1 - z)
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
    """`part / whole`, zero where the whole is zero. Attribution keys are ratios
    of solved quantities and the denominator is zero in every interval nobody
    exported, which is most of them."""
    whole = np.asarray(whole, dtype=float)
    return np.divide(np.asarray(part, dtype=float), whole,
                     out=np.zeros_like(whole), where=np.abs(whole) > tol)


def _complementarity_violations(members, state, tol=1e-4):
    """Solved flows that no single metering point or inverter could produce.

    Two kinds, both caught the same way and both fixed by the same lazy binary:

      "grid"     imported and exported in the same interval. A meter nets the
                 two before either is billed, so a settlement cannot be built
                 on it.
      "battery"  charged and discharged in the same interval. A battery cannot,
                 and the environment cannot express it either -- one signed
                 setpoint per step. A solver reaches for it only where a kWh is
                 worth less than nothing and the round-trip loss becomes a way
                 to get paid for burning energy.

    The tolerance is in kWh: well above solver noise, far below anything that
    moves a bill.
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
    """A battery cannot charge and discharge in the same interval, and the
    environment cannot express it either -- one signed setpoint per step. The
    binaries above are only created where the LP could want to, so this checks
    the assumption held everywhere else."""
    return int(sum(1 for c, d in zip(ch, dis) if _v(c) > tol and _v(d) > tol))


def _extract(*, members, state, senders, receivers, scheme, sharing_mode, alpha,
             p_int, pay_for_unused, weights, n_steps, dates, hours,
             month_idx_t, months_sorted, no_incentive=None):
    """Re-price the solved trajectory member by member.

    The bill is rebuilt from the flows rather than read off the objective, so a
    mistake in the objective shows up as a mismatch instead of propagating: the
    caller gets both and the notebook asserts they agree.
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
                # Only the consumed part leaves the buyback contract; it is
                # attributed to the producers in proportion to what each of them
                # put into the pool. With one price list on the sending side any
                # split gives the same community total, and this is the split
                # the members would agree to.
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

        # What the receiver is INVOICED for -- the allocation it was given, which
        # under souporaba is not the allocation it managed to use.
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

        # The excess-power charge, re-derived from the trajectory exactly as the
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
            # Import and export in the same interval. The binaries above are
            # created only where the objective could want it, so anything left
            # here is a place the assumption failed and the flows are not
            # something one meter could have recorded.
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
    """The ratchet excess-power charge for one solved trajectory.

    Same rule as the objective and as `si_konica`: per block and month, the
    highest quarter-hour power above the agreed billing power, priced at that
    month's season-correct power postavka and weighted by the transitional
    factor. Ex VAT, matching `Pricing_Functions._apply_peak_ratchet`.
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
    """(start_idx, n_steps, (year, month)) per calendar month, in Slovenian local time.

    The month is the billing unit -- power charge, OVE+SPTE and the excess-power
    ratchet all reset on the 1st -- so cutting the year here costs nothing on
    the tariff side and makes the twelve solves independent.

    `min_intervals` folds a slice shorter than that into its neighbour. A whole
    calendar year of data does not always produce twelve slices: the Fluvius
    profiles are stamped in Brussels local time with a `Z` suffix, so a file
    ending 31 Dec 23:45 spills its last hour into the following January. Those
    few intervals are a real part of the year and must not be dropped, but they
    are not a month, and solving them as one puts a whole month's worth of
    calendar into a horizon four intervals long. Folding them into December is
    exact: the fixed charge is prorated per interval and the peak is tracked per
    (block, month) inside the solve either way.
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
        # A short FIRST slice has no predecessor to fold into, so it folds the
        # successor into itself instead.
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

    Returns the same shape as `solve_community_milp` with the twelve monthly
    settlements added up. Each month opens and closes at `soc_fraction` of every
    battery's capacity, so no month can raid another's storage; that is the one
    place the decomposition costs anything, and the notebook measures it against
    an annual single-household solve.
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
    # Everything is a monthly total and adds up; the peak is the highest month's,
    # because the peak charge is settled per month against the same agreed power
    # and the annual figure that describes the connection is the largest of them.
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
        # How much of the year needed a binary at all, and whether anything was
        # left unfixed when the lazy loop ran out of rounds.
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
