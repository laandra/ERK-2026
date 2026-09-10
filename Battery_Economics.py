"""Battery lifetime economics: what a pack costs to own, against what it saves.

The MILP says what a battery *does*; this module says what it *costs*. The two
are deliberately kept apart -- `MILP_Household` prices electricity and nothing
else, and the wear price it can carry in its objective (see
`cycle_cost_eur_per_efc` below) is a dispatch signal, not a line on a bill.

The model is a level annuity: one solved year of savings, repeated unchanged
over the service life, against capital recovered at `DISCOUNT_RATE` plus a
yearly O&M charge. There is no year-by-year cash-flow vector, so anything that
varies over the life -- capacity fade, price escalation -- has to enter as a
level equivalent or not at all.

Three costs, three different lives:

    CAPEX     one-off, recovered over Service_Life_y at DISCOUNT_RATE
    OPEX      OPEX_FRAC_OF_CAPEX_PER_YEAR of that capital, every year
    wear      NOT charged here. It enters as the cycle-limited service life
              below, and optionally as a shadow price inside the MILP.

Every study imported these constants by retyping them until this module existed;
they are here for the same reason the battery's physical constants live in
`MILP_Household` -- so two studies cannot quietly price the same pack
differently.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# The quotes
# ---------------------------------------------------------------------------
CAPEX_EUR_PER_KWH = 250.0      # installed, storage only -- the assumed price
CAPEX_FIXED_EUR = 1000.0       # hybrid inverter / install, independent of size
BATTERY_CALENDAR_LIFE_Y = 12   # warranty band for residential Li-ion
BATTERY_CYCLE_LIMIT_EFC = 6000  # equivalent full cycles before end of life

# The one input that is not a quote: r is a choice, and it decides the answer as
# firmly as the storage price does. Real, not nominal -- the savings are one
# solved year repeated in the reference year's prices, so electricity-price
# inflation is already netted out.
DISCOUNT_RATE = 0.05

# Yearly O&M as a share of installed capital: monitoring, the occasional service
# call, insurance. Quoted on TOTAL installed cost (storage + install), which is
# the usual convention for a PV/storage retrofit. `Net_Annual_StorageOnly_EUR`
# charges it on the storage term alone, consistently with its own capex.
OPEX_FRAC_OF_CAPEX_PER_YEAR = 0.015


# ---------------------------------------------------------------------------
# Discounting primitives
# ---------------------------------------------------------------------------
# Both factors use the closed form for every rate except exactly zero, where it
# is 0/0. Short-circuiting on `rate <= 0` would flatten the left half of the IRR
# search into a constant.
def capital_recovery_factor(rate, years):
    """Constant yearly payment that repays 1 EUR of capital over `years`."""
    years = max(1.0, float(years))
    if rate == 0:
        return 1.0 / years
    return rate / (1.0 - (1.0 + rate) ** -years)


def present_value_factor(rate, years):
    """Present value of 1 EUR received yearly for `years` years."""
    years = max(1.0, float(years))
    if rate == 0:
        return years
    return (1.0 - (1.0 + rate) ** -years) / rate


def irr(capex, annual_savings, years, lo=-0.999, hi=5.0, tol=1e-12):
    """Rate where the NPV of a level annuity equals capex. Signed: may be < 0.

    A NEGATIVE IRR is an answer, not an error. It is what a pack that never
    repays its capital within its service life earns -- 217 EUR/a on 3500 EUR
    over 12 years is -5.2 %/a -- and it is the only figure that ranks two
    controllers that both lose money. `present_value_factor` is defined and
    strictly decreasing across the whole of r > -1, so the root is there to be
    found; the search simply used to stop above it.

    What used to happen: an early return on `annual_savings * years < capex`
    -- exactly the "does not repay undiscounted" case -- sent every such pack
    back as NaN, and `lo=-0.95` truncated whatever was left. On this study that
    is not an edge case, it is most of the table: a 10 kWh pack saving 46 EUR/a
    is uneconomic by a factor of six, and NaN reported that as "no answer"
    rather than as "-19 %/a".

    NaN is now reserved for the case where no rate solves it at all:

      capex <= 0            nothing to recover; an IRR is not defined.
      annual_savings <= 0   NPV(r) = savings * pvf(r) - capex is negative for
                            every r > -1, because pvf is positive throughout.
                            No root exists -- this is "never, at any discount
                            rate", which is a different statement from a large
                            negative rate and must not be rendered as one.
      npv(lo) < 0           the root is below -99.9 %/a, where pvf already
                            exceeds 1e30 and the answer is numerical noise.
    """
    if capex <= 0 or annual_savings <= 0:
        return np.nan
    npv = lambda r: annual_savings * present_value_factor(r, years) - capex
    if npv(lo) < 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def annualized_cost_factor(rate, years, opex_frac=OPEX_FRAC_OF_CAPEX_PER_YEAR):
    """Yearly cost of 1 EUR of installed capital: recovery plus O&M.

    This is the height of the cost lines the sizing charts draw against the
    marginal value of capacity. Before OPEX existed it was the bare CRF, so
    `opex_frac=0.0` reproduces every chart as it was.
    """
    return capital_recovery_factor(rate, years) + float(opex_frac)


# ---------------------------------------------------------------------------
# Wear
# ---------------------------------------------------------------------------
def service_life_years(efc_per_year,
                       calendar_life_y=BATTERY_CALENDAR_LIFE_Y,
                       cycle_limit_efc=BATTERY_CYCLE_LIMIT_EFC):
    """Service life, whichever of calendar and cycles runs out first.

    `cycle_limit_efc=None` disables cycle limiting and returns the calendar life
    flat, which is the simpler model the single-household study uses.

    `efc_per_year` is `Equivalent_Full_Cycles` from `summarize_trajectory` --
    counted against the NAMEPLATE pack, so reserving SOC headroom shows up here
    as a longer life, which is the real physics of shallow cycling.

    Returns `(service_life, cycle_life)`, as floats for a scalar input and as
    arrays for an array one.
    """
    efc = np.asarray(efc_per_year, dtype=float)
    if cycle_limit_efc is None:
        cycle_life = np.full(efc.shape, np.inf)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            cycle_life = np.where(efc > 0, float(cycle_limit_efc) / efc, np.inf)
    life = np.minimum(float(calendar_life_y), cycle_life)
    # A scalar in, a scalar out: callers that size one pack should not have to
    # unwrap a 0-d array, which does not even index.
    if efc.ndim == 0:
        return float(life), float(cycle_life)
    return life, cycle_life


def cycle_cost_eur_per_efc(nominal_capacity_kwh,
                           capex_eur_per_kwh=CAPEX_EUR_PER_KWH,
                           cycle_limit_efc=BATTERY_CYCLE_LIMIT_EFC):
    """Wear price of one equivalent full cycle: storage capex / cycle life.

    The standard normalisation -- what the pack cost, divided by how many cycles
    it is rated for. Handed to the MILP it becomes the marginal cost of the next
    cycle, which is the number a dispatch decision actually needs.

    The install fee is deliberately NOT in the base: a hybrid inverter does not
    wear per cycle the way cells do, and folding it in would make one cycle cost
    three times more on a 3 kWh pack than on a 30 kWh one.

    At the assumed 250 EUR/kWh and 6000 EFC a 10 kWh pack prices a full cycle at
    0.417 EUR, i.e. 0.0208 EUR per kWh through the store -- enough to stop
    marginal arbitrage without touching the 0.109 EUR/kWh Aktivni block spread.
    """
    return float(capex_eur_per_kwh) * float(nominal_capacity_kwh) / float(cycle_limit_efc)


# ---------------------------------------------------------------------------
# The whole picture, one call
# ---------------------------------------------------------------------------
def battery_economics(savings_eur, capacity_kwh, efc_per_year, *,
                      capex_eur_per_kwh=CAPEX_EUR_PER_KWH,
                      capex_fixed_eur=CAPEX_FIXED_EUR,
                      opex_frac=OPEX_FRAC_OF_CAPEX_PER_YEAR,
                      calendar_life_y=BATTERY_CALENDAR_LIFE_Y,
                      cycle_limit_efc=BATTERY_CYCLE_LIMIT_EFC,
                      discount_rate=DISCOUNT_RATE):
    """Capex, OPEX, service life and the ROI figures, vectorised over arrays.

    Returns a dict of numpy arrays, one entry per output column, aligned with the
    inputs. Assign them straight onto a DataFrame.

        capex(C)   = capex_eur_per_kwh * C + capex_fixed_eur   for C > 0, else 0
        opex       = opex_frac * capex, every year
        life       = min(calendar_life_y, cycle_limit_efc / EFC)
        net saving = savings - opex

    `capex_fixed_eur=0.0` with `cycle_limit_efc=None` is the simpler model the
    single-household study uses.
    """
    savings = np.asarray(savings_eur, dtype=float)
    cap = np.asarray(capacity_kwh, dtype=float)
    sized = cap > 0

    storage_capex = cap * float(capex_eur_per_kwh)
    capex = np.where(sized, storage_capex + float(capex_fixed_eur), 0.0)

    opex = capex * float(opex_frac)
    net_savings = savings - opex

    life, cycle_life = service_life_years(efc_per_year, calendar_life_y, cycle_limit_efc)
    crf = np.array([capital_recovery_factor(discount_rate, n) for n in life])
    pvf = np.array([present_value_factor(discount_rate, n) for n in life])

    # The same question with the one-off install cost taken out, so "is storage
    # worth its own price?" is answered separately from "does the whole package
    # pay?". Its OPEX is charged on its own capex, not the full one.
    storage_only = savings - storage_capex * (crf + float(opex_frac))

    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(capex > 0, 100.0 * (net_savings * pvf - capex) / capex, np.nan)
        payback = np.where((capex > 0) & (net_savings > 0), capex / net_savings, np.nan)
        # Net_Annual == 0 solved for the per-kWh price. OPEX is proportional to
        # the same capital, so it joins the recovery factor rather than the
        # numerator: savings = capex * (crf + opex_frac).
        break_even = np.where(
            sized,
            (savings / (crf + float(opex_frac)) - float(capex_fixed_eur)) / np.where(sized, cap, 1.0),
            np.nan,
        )

    return {
        "Capex_EUR": capex,
        "Annual_OPEX_EUR": opex,
        "Net_Savings_EUR": net_savings,
        "Cycle_Life_y": cycle_life,
        "Service_Life_y": life,
        "Life_Limited_By": np.where(cycle_life < float(calendar_life_y), "cycles", "calendar"),
        "Annuity_Factor": crf,
        "Net_Annual_EUR": net_savings - capex * crf,
        "Net_Annual_StorageOnly_EUR": storage_only,
        "NPV_EUR": net_savings * pvf - capex,
        "ROI_pct": roi,
        "IRR_pct": np.array([
            100.0 * irr(c, s, n) if c > 0 else np.nan
            for c, s, n in zip(capex, net_savings, life)
        ]),
        "Payback_y": payback,
        "Break_Even_Capex_EUR_kWh": break_even,
    }


# ---------------------------------------------------------------------------
# Self-checks. These travelled with the functions from the notebooks and are
# kept at import time: 1000 EUR returning 200 EUR/a for 10 years is a 15.1 %
# IRR, and an investment that exactly returns its capital undiscounted is 0 %.
# ---------------------------------------------------------------------------
assert abs(irr(1000.0, 200.0, 10) - 0.15098) < 1e-4, "IRR solver is off"
assert abs(irr(1000.0, 100.0, 10)) < 1e-6, "IRR of a break-even annuity should be 0"
# Capital that is NOT repaid inside the life has a negative IRR, and the solver
# has to return it rather than NaN -- 50 EUR/a on 1000 EUR over 10 years is a
# real -11.0 %/a, and it is what most of the HEMS study's controllers earn.
# Checked against the root of the annuity directly, so the assertion is a
# property of the answer and not a transcribed constant.
_r = irr(1000.0, 50.0, 10)
assert _r < 0, "an annuity that does not repay undiscounted has a NEGATIVE IRR"
assert abs(50.0 * present_value_factor(_r, 10) - 1000.0) < 1e-6, "IRR is not a root"
assert abs(_r - (-0.10956)) < 1e-4, "negative-IRR branch moved"
# The one case that genuinely has no root: pvf > 0 for every r > -1, so a
# non-positive annuity can never reach a positive capex at any discount rate.
assert np.isnan(irr(1000.0, 0.0, 10)), "a zero annuity has no IRR"
assert np.isnan(irr(1000.0, -20.0, 10)), "a negative annuity has no IRR"
assert np.isnan(irr(0.0, 200.0, 10)), "no capital, no IRR"
assert abs(present_value_factor(0.05, 10) - 7.72173) < 1e-4
assert abs(capital_recovery_factor(0.05, 12) - 0.11283) < 1e-4
assert abs(annualized_cost_factor(0.05, 12, 0.0) - capital_recovery_factor(0.05, 12)) < 1e-12
# A 10 kWh pack at the assumed quotes prices a full cycle at 2500/6000 EUR.
assert abs(cycle_cost_eur_per_efc(10.0) - 2500.0 / 6000.0) < 1e-12
# Scalar in, scalar out -- and the two limits each bind where they should.
assert service_life_years(267.0) == (12.0, 6000.0 / 267.0)
assert service_life_years(1000.0)[0] == 6.0, "1000 EFC/a must be cycle-limited"
assert service_life_years(0.0) == (12.0, float("inf")), "an idle pack is calendar-limited"
assert service_life_years(1000.0, cycle_limit_efc=None)[0] == 12.0, \
    "cycle_limit_efc=None must disable cycle limiting"
assert np.shape(service_life_years([267.0, 1000.0])[0]) == (2,)
