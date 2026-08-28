"""Gymnasium environments for household / energy-community battery dispatch.

`HouseholdEnvironment` is continuous-first: an action is a signed battery
setpoint in kWh (positive = charge), the quantity the MILP benchmark solves
for. `action_mode="discrete"` keeps the legacy five-entry action set.

With `allow_curtailment=True` the action gains the curtailed kWh, matching the
MILP's `P_spill`. Curtailment is applied to generation first.
"""

import calendar
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from Basic_Functions import battery_delta, max_charge_now, max_discharge_now, pv_surplus

from Pricing_Functions import (
    aggregate_household_invoices,
    PRIVZETO_REFERENCNO_LETO,
    InvoiceBuilder,
    calculate_interval_price,
    dogovorjena_moc_iz_konic,
    mesecni_razpored_moci,
    oznaka_razporeda_moci,
    povprecje_najvecjih,
    KONICNI_BLOKI,
    ST_KONIC,
    resolve_block_for_datetime,
    resolve_reset_window_id,
)

_DEFAULT_INVOICE_OUTPUT_DIR = Path(__file__).resolve().parent / "Results" / "Invoices"

N_BLOCKS = 5
_BLOCKS = range(1, N_BLOCKS + 1)

# Legacy discrete action set, kept so the DQN pipeline keeps its action space.
ACTION_CHARGE_ANY = 0     # charge from PV, top up from the grid
ACTION_CHARGE_PV = 1      # charge from PV surplus only
ACTION_DISCHARGE_HOME = 2  # discharge, serving household load only
ACTION_DISCHARGE_ANY = 3  # discharge fully, exporting whatever the house can't use
ACTION_IDLE = 4           # battery unused
N_DISCRETE_ACTIONS = 5

# Curtailment modes in discrete mode: the breakpoints of the piecewise-linear
# interval cost. `action_mode="continuous"` sets the curtailed kWh directly.
CURTAIL_NONE = 0     # export everything the house and battery don't take
CURTAIL_NO_EXPORT = 1  # curtail exactly the surplus that would be exported
CURTAIL_ALL = 2      # shut local production off completely
N_CURTAILMENT_MODES = 3


# ---------------------------------------------------------------------------
# Agreed billing power (dogovorjena obracunska moc). Module-level so callers can
# build a schedule without constructing an environment.
# ---------------------------------------------------------------------------
def monthly_top_peaks_by_block(power_kw, block_arr, month_id_arr,
                               n_peaks=ST_KONIC):
    """{month id: {block: [n highest kW, largest first]}} of a grid draw.

    The agreed power is the MEAN of the operator's `ST_KONIC` highest peaks per
    block, and pooling that statistic over a multi-month window needs the raw
    values -- the mean of the five highest over a year is not the mean of twelve
    monthly means. So the whole top-n list is carried, not a single peak.

    Blocks that never occur in a month are absent from that month's dict.
    """
    power = np.asarray(power_kw, dtype=np.float64)
    blocks = np.asarray(block_arr, dtype=np.int64)
    months = np.asarray(month_id_arr, dtype=np.int64)
    n = max(int(n_peaks), 1)

    # One key per (month, block); blocks are 1..5, so a stride of 16 is safe.
    key = months * 16 + blocks
    # Group by key, and inside each group put the largest power first.
    order = np.lexsort((-power, key))
    key_s, power_s = key[order], power[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    ends = np.r_[starts[1:], key_s.size]

    peaks = {}
    for s, e in zip(starts, ends):
        k = int(key_s[s])
        peaks.setdefault(k // 16, {})[k % 16] = power_s[s:min(s + n, e)].tolist()
    return peaks


def monthly_peak_kw_by_block(power_kw, block_arr, month_id_arr):
    """{month id: {block: single highest kW}}, for reporting beside the contract.

    The agreed power is not read from this -- `monthly_top_peaks_by_block` is
    what the rule uses -- but the realized maximum is what shows whether a month
    ran over its contract, so `agreed_power_frame` prints the two side by side.
    """
    return {
        month: {b: vals[0] for b, vals in by_block.items()}
        for month, by_block in monthly_top_peaks_by_block(
            power_kw, block_arr, month_id_arr, n_peaks=1
        ).items()
    }


def month_completeness(month_id_arr, steps_per_day):
    """{month id: observed intervals / intervals a full month would have}."""
    counts = Counter(int(m) for m in month_id_arr)
    out = {}
    for month, n in counts.items():
        days = calendar.monthrange(month // 12, month % 12 + 1)[1]
        out[month] = n / float(days * steps_per_day)
    return out


def bootstrap_peak_kw(peaks, month_id_arr, steps_per_day, *, mode="cyclic",
                      first_month=None):
    """Per-block peak LISTS standing in as the month before `first_month`.

    `"cyclic"` (default) reads the last complete month of the dataset, `"own"`
    the leading month's own peaks, `"flat_max"` its single largest draw in
    every block. Shaped like one month of `monthly_top_peaks_by_block`, because
    `si_moc.mesecni_razpored` pools it with real months whenever the lookback
    window reaches past the start of the data.
    """
    if mode not in {"cyclic", "own", "flat_max"}:
        raise ValueError(
            f"agreed_power_bootstrap must be 'cyclic', 'own' or 'flat_max', got {mode!r}"
        )
    if not peaks:
        return None
    first = min(peaks) if first_month is None else int(first_month)
    if first not in peaks:
        return None

    def _flat(month):
        # One conservative scalar in every block: the largest draw seen. As a
        # one-entry list it averages to itself whatever `st_konic` is.
        largest = max((vals[0] for vals in peaks[month].values()), default=0.0)
        return {b: [float(largest)] for b in _BLOCKS}

    if mode == "own":
        return None                      # si_moc falls back to the month itself
    if mode == "flat_max":
        return _flat(first)

    complete_by = month_completeness(month_id_arr, steps_per_day)
    complete = [m for m in sorted(peaks) if m != first and complete_by.get(m, 0.0) >= 0.9]
    if not complete:
        # No second complete month; fall back to the conservative scalar.
        return _flat(first)
    return {b: list(vals) for b, vals in peaks[complete[-1]].items()}


def agreed_power_schedule_for_profile(
    data,
    *,
    consumption_column,
    generation_column,
    steps_per_day=96,
    pricing_reference_year=None,
    min_agreed_power_kw=0.0,
    connection_power_kw=None,
    lag_months=1,
    bootstrap="cyclic",
    carry_missing_blocks=True,
    n_peaks=ST_KONIC,
    n_months_window=1,
):
    """{month id: {block: agreed kW}} for a raw profile, with no environment.
    
    The same schedule `HouseholdEnvironment` builds for itself, so a settlement
    bills against the contract the dispatch was optimized under.
    """
    hours = 24.0 / float(steps_per_day)
    naive_power_kw = np.maximum(
        data[consumption_column].to_numpy(dtype=float)
        - data[generation_column].to_numpy(dtype=float),
        0.0,
    ) / hours

    block_cache, month_cache = {}, {}
    n = len(data.index)
    block_arr = np.empty(n, dtype=np.int32)
    month_id_arr = np.empty(n, dtype=np.int64)
    for i in range(n):
        ts = data.index[i]
        key = (ts.year, ts.month, ts.day, ts.hour)
        if key not in block_cache:
            block_cache[key] = resolve_block_for_datetime(
                ts, pricing_reference_year=pricing_reference_year
            )
            month_cache[key] = resolve_reset_window_id(ts, 1)
        block_arr[i] = block_cache[key]
        month_id_arr[i] = month_cache[key]

    peaks = monthly_top_peaks_by_block(
        naive_power_kw, block_arr, month_id_arr, n_peaks=n_peaks
    )
    return mesecni_razpored_moci(
        peaks,
        minimalna_moc_kw=min_agreed_power_kw,
        prikljucna_moc_kw=connection_power_kw,
        zamik_mesecev=lag_months,
        prenesi_manjkajoce_bloke=carry_missing_blocks,
        zacetne_konice=bootstrap_peak_kw(
            peaks, month_id_arr, steps_per_day, mode=bootstrap
        ),
        st_konic=n_peaks,
        n_months_window=n_months_window,
    )


# --- Endogenous agreed power: the contract a dispatch sets for itself -------
# A household that flattens its profile with a battery re-agrees its
# dogovorjena moc down to the flattened peak, and the shaver is then billed
# against the line it created. That makes the contract a FIXED POINT of the
# dispatch, not an input to it: solve under a contract, read the peaks, re-roll
# the contract, solve again, until the two agree.
AGREED_POWER_MAX_ITER = 6
# Two contracts are the same contract when every block of every month agrees to
# within this; anything finer is solver noise, not a different bill.
AGREED_POWER_TOL_KW = 1e-6


def _schedules_agree(a, b, tol_kw=AGREED_POWER_TOL_KW):
    if set(a) != set(b):
        return False
    return all(
        abs(float(a[m].get(blk, 0.0)) - float(b[m].get(blk, 0.0))) <= tol_kw
        for m in a for blk in set(a[m]) | set(b[m])
    )


def _schedule_envelope(schedules):
    """The block-by-block maximum of several schedules: the contract that avoids
    an excess charge under any of them."""
    months = sorted({m for s in schedules for m in s})
    return {
        m: {b: max(float(s[m].get(b, 0.0)) for s in schedules if m in s)
            for b in sorted({b for s in schedules if m in s for b in s[m]})}
        for m in months
    }


def converge_agreed_power(env, dispatch, *, start_idx=0,
                          max_iter=AGREED_POWER_MAX_ITER,
                          tol_kw=AGREED_POWER_TOL_KW, verbose=False):
    """Run `dispatch` until it is billed under the contract its own peaks set.

    `dispatch()` runs the whole horizon under whatever contract is currently in
    force on `env` and returns `(result, grid_import_kw)` -- one kW entry per
    executed interval, from `start_idx`. It is called once per iteration, so an
    expensive solve is an expensive loop; `max_iter` bounds it.

    Returns `(result, info)`. `result` is the LAST dispatch run, and the
    environment is left holding exactly the contract that run was billed under,
    so a caller can price and report against `env.agreed_power_at` afterwards
    without re-deriving anything.

    `info` carries `iterations`, `converged`, and `cycle`. The map is not a
    contraction in general -- shaving to this month's line can lower next
    month's, which frees the shaver, which raises it again -- so a two-cycle is
    a real outcome. It is settled on the ENVELOPE of the alternating contracts,
    the conservative reading: the household keeps the higher agreed power rather
    than paying an excess charge every other month.
    """
    env.clear_achieved_power()
    if not getattr(env, "agreed_power_lag_months", None) or (
        getattr(env, "_explicit_contracted_power_kw", None) is not None
    ):
        # A fixed contract, or none that rolls: one run is the whole answer.
        result, _ = dispatch()
        return result, {"iterations": 1, "converged": True, "cycle": False}

    seen = [env.agreed_power_schedule]
    result = restore = None
    for it in range(1, int(max_iter) + 1):
        billed = env.agreed_power_schedule
        # What the contract in force was derived from, so the run can be put
        # back exactly as it was billed if the loop runs out of iterations.
        restore = env.achieved_power_kw
        result, power_kw = dispatch()
        proposed = env.set_achieved_power_kw(power_kw, start_idx=start_idx)
        if _schedules_agree(billed, proposed, tol_kw):
            if verbose:
                print(f"Agreed power: fixed point after {it} dispatch(es).")
            return result, {"iterations": it, "converged": True, "cycle": False}
        if any(_schedules_agree(proposed, s, tol_kw) for s in seen[:-1]):
            envelope = env.apply_agreed_power_schedule(
                _schedule_envelope([proposed, billed])
            )
            result, _ = dispatch()
            env.apply_agreed_power_schedule(envelope)
            if verbose:
                print(f"Agreed power: cycled at iteration {it}; settled on the "
                      f"envelope of the two contracts.")
            return result, {"iterations": it + 1, "converged": False, "cycle": True}
        seen.append(proposed)

    # Out of iterations: report the run whose bill we actually have, under the
    # contract it was actually billed under.
    env.clear_achieved_power()
    if restore is not None:
        env.set_achieved_power_kw(restore, start_idx=0)
    if verbose:
        print(f"Agreed power: no fixed point in {max_iter} dispatches; "
              f"reporting the last one under its own contract.")
    return result, {"iterations": int(max_iter), "converged": False, "cycle": False}


class HouseholdEnvironment(gym.Env):
    """Single household with PV and a battery, priced under a Slovenian tariff."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset,
        dataset_norm=None,
        price_column="SMP",
        generation_column="Energy_Generation",
        consumption_column="Energy_Consumption",
        observation_mode="sliding_window",
        reset_mode="deterministic",
        action_mode="continuous",
        action_scale="physical",
        n_discrete_actions=N_DISCRETE_ACTIONS,
        allow_curtailment=False,
        clip_penalty=0.0,
        episode_length=None,
        steps_per_day=96,
        battery_capacity_kwh=20.0,
        nominal_capacity_kwh=None,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        max_daily_cycles=None,
        cycle_cost_eur_per_efc=None,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        max_charge_kwh=1.5,
        max_discharge_kwh=1.5,
        reward_weight_soc=0.0,
        reward_weight_arbitrage=0.0,
        reward_weight_cost=1.0,
        median_window_days=30,
        pricing_scheme="si_samooskrba",
        pricing_include_raw=False,
        pricing_reference_year=2026,
        pricing_options=None,
        contracted_power_kw=None,
        peak_reset_months=None,
        connection_power_kw=None,
        min_agreed_power_kw=0.0,
        agreed_power_lag_months=1,
        agreed_power_carry_missing_blocks=True,
        agreed_power_bootstrap="cyclic",
        agreed_power_from_dispatch=True,
        agreed_power_n_peaks=ST_KONIC,
        agreed_power_n_months_window=1,
        pricing_validate_pv=True,
        generate_monthly_invoice=False,
        generate_period_invoice=False,
        invoice_eko_racun=True,
        invoice_output_dir=None,
        invoice_run_label=None,
    ):
        self.dataset = dataset
        self.dataset_norm = dataset_norm if dataset_norm is not None else dataset

        self.price_column = str(price_column)
        self.generation_column = str(generation_column)
        self.consumption_column = str(consumption_column)

        for col in (self.price_column, self.generation_column, self.consumption_column):
            if col not in self.dataset.columns:
                raise ValueError(f"Missing required column in dataset: {col}")
            if col not in self.dataset_norm.columns:
                raise ValueError(f"Missing required column in dataset_norm: {col}")

        self.observation_mode = str(observation_mode)
        if self.observation_mode not in {"sliding_window", "compact"}:
            raise ValueError("observation_mode must be 'sliding_window' or 'compact'")

        self.reset_mode = str(reset_mode)
        if self.reset_mode not in {"deterministic", "random", "sequential"}:
            raise ValueError("reset_mode must be 'deterministic', 'random', or 'sequential'")

        self.action_mode = str(action_mode)
        if self.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be 'continuous' or 'discrete'")

        self.action_scale = str(action_scale)
        if self.action_scale not in {"physical", "normalized"}:
            raise ValueError("action_scale must be 'physical' or 'normalized'")

        self.n_discrete_actions = int(n_discrete_actions)
        if not 1 <= self.n_discrete_actions <= N_DISCRETE_ACTIONS:
            raise ValueError(f"n_discrete_actions must be in 1..{N_DISCRETE_ACTIONS}")

        # Curtailment, matching the MILP's P_spill. Off by default.
        self.allow_curtailment = bool(allow_curtailment)
        self.n_curtailment_modes = N_CURTAILMENT_MODES if self.allow_curtailment else 1

        self.clip_penalty = float(clip_penalty)

        self.steps_per_day = int(steps_per_day)
        if self.steps_per_day <= 0:
            raise ValueError("steps_per_day must be > 0")
        self.interval_minutes = 1440.0 / self.steps_per_day

        # `battery_capacity_kwh` is the capacity the PHYSICS sees -- the usable
        # window. `nominal_capacity_kwh` is the pack on the invoice, which the
        # two differ by exactly when a SOC window is set. Everything that clamps,
        # bounds or balances energy reads the usable figure, so a derated pack
        # needs no further change anywhere; only the economics and the cycle
        # count against the nameplate care about the difference.
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.nominal_capacity_kwh = (
            self.battery_capacity_kwh if nominal_capacity_kwh is None
            else float(nominal_capacity_kwh)
        )
        self.soc_min_frac = float(soc_min_frac)
        self.soc_max_frac = float(soc_max_frac)
        if not 0.0 <= self.soc_min_frac < self.soc_max_frac <= 1.0:
            raise ValueError(
                f"need 0 <= soc_min_frac < soc_max_frac <= 1, got "
                f"{self.soc_min_frac} / {self.soc_max_frac}"
            )
        # Usable-window cycles per local day, and the EUR a full nameplate cycle
        # costs the MILP objective. Both None means "no envelope, no wear price".
        self.max_daily_cycles = None if max_daily_cycles is None else float(max_daily_cycles)
        if self.max_daily_cycles is not None and self.max_daily_cycles <= 0:
            raise ValueError("max_daily_cycles must be > 0 or None")
        self.cycle_cost_eur_per_efc = (
            None if cycle_cost_eur_per_efc is None else float(cycle_cost_eur_per_efc)
        )
        if self.cycle_cost_eur_per_efc is not None and self.cycle_cost_eur_per_efc < 0:
            raise ValueError("cycle_cost_eur_per_efc must be >= 0 or None")
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        for name, eff in (
            ("charge_efficiency", self.charge_efficiency),
            ("discharge_efficiency", self.discharge_efficiency),
        ):
            if not 0.0 < eff <= 1.0:
                raise ValueError(f"{name} must be in (0.0, 1.0]")

        self.max_charge_kwh = float(max_charge_kwh)
        self.max_discharge_kwh = float(max_discharge_kwh)
        self.reward_weight_soc = float(reward_weight_soc)
        self.reward_weight_arbitrage = float(reward_weight_arbitrage)
        self.reward_weight_cost = float(reward_weight_cost)

        self.pricing_scheme = str(pricing_scheme)
        if self.pricing_scheme not in {"si_dobava", "si_samooskrba"}:
            raise ValueError("pricing_scheme must be 'si_dobava' or 'si_samooskrba'")
        self.pricing_include_raw = bool(pricing_include_raw)

        # Fresh dict per instance: it is mutated below, so a mutable default
        # would be shared across every construction that omits it.
        self.pricing_options = {"pricing_mode": "dinamicni", "buyback_mode": "dinamicni"}
        self.pricing_options.update(pricing_options or {})

        # The profiles predate the published SI tariff acts, so the reference
        # year is pinned rather than resolved from the data date.
        self.pricing_reference_year = (
            PRIVZETO_REFERENCNO_LETO if pricing_reference_year is None
            else int(pricing_reference_year)
        )
        self.pricing_options["pricing_reference_year"] = self.pricing_reference_year

        # --- Invoice generation (monthly / whole-period line-item bills) -----------
        self.generate_monthly_invoice = bool(generate_monthly_invoice)
        self.generate_period_invoice = bool(generate_period_invoice)
        self._invoicing_enabled = self.generate_monthly_invoice or self.generate_period_invoice
        if self._invoicing_enabled and self.reset_mode != "deterministic":
            raise ValueError(
                "Invoice generation requires reset_mode='deterministic' -- invoicing "
                "only makes sense over a single chronological pass; random/sequential "
                "resets would interleave or overwrite invoice state across unrelated "
                "episodes."
            )
        self.invoice_eko_racun = bool(invoice_eko_racun)
        self.invoice_output_dir = (
            Path(invoice_output_dir) if invoice_output_dir is not None else _DEFAULT_INVOICE_OUTPUT_DIR
        )
        self.invoice_run_label = invoice_run_label
        self._invoice_builder = None

        self.data_length = len(self.dataset)
        if self.data_length == 0:
            raise ValueError("Dataset is empty. Please provide a valid dataset.")

        self.arr_price = self.dataset[self.price_column].to_numpy(dtype=np.float64)
        self.arr_generation = self.dataset[self.generation_column].to_numpy(dtype=np.float64)
        self.arr_consumption = self.dataset[self.consumption_column].to_numpy(dtype=np.float64)

        # --- PV presence + pricing_scheme validation -------------------------------
        self.pricing_warnings = []
        self._has_pv = bool(np.nanmax(self.arr_generation) > 0.0)

        if self.pricing_scheme == "si_dobava" and self._has_pv:
            msg = (
                "pricing_scheme='si_dobava' assumes no on-site production, but "
                f"{self.generation_column} has nonzero values. si_dobava has no export-netting "
                "logic and would mis-tax exported energy (full retail rate + network "
                "charges + VAT applied to exported kWh). Use pricing_scheme="
                "'si_samooskrba', or set pricing_validate_pv=False to bypass this check."
            )
            if pricing_validate_pv:
                raise ValueError(msg)
            self.pricing_warnings.append(msg)
        elif self.pricing_scheme == "si_samooskrba" and not self._has_pv:
            self.pricing_warnings.append(
                f"pricing_scheme='si_samooskrba' selected but {self.generation_column} is "
                "always zero; self-supply netting has no effect (numerically "
                "equivalent to si_dobava)."
            )

        # --- Contracted power (dogovorjena_moc) + peak-ratchet config --------------
        self.peak_reset_months = None if peak_reset_months is None else int(peak_reset_months)

        # The draw a dispatch achieved, once one is registered; until then every
        # peak in this environment is measured on the no-battery profile.
        self._achieved_power_kw = None
        self._forced_agreed_power = False
        self._block_arr, self._window_id_arr, self._peak_seed_history = (
            self._precompute_peak_seed_history()
        )
        self._month_id_arr = self._precompute_month_ids()

        # No floor and no ceiling by default: the profiles carry no connection
        # agreement, and inventing one only manufactures excess charges.
        self.connection_power_kw = (
            None if connection_power_kw is None else float(connection_power_kw)
        )
        self.min_agreed_power_kw = float(min_agreed_power_kw)
        self.agreed_power_bootstrap = str(agreed_power_bootstrap)
        self.agreed_power_lag_months = (
            None if (agreed_power_lag_months is None or contracted_power_kw is not None)
            else int(agreed_power_lag_months)
        )
        self.agreed_power_carry_missing_blocks = bool(agreed_power_carry_missing_blocks)
        # The operator's statistic: the mean of the `n_peaks` highest 15-minute
        # peaks per block, pooled over `n_months_window` months back from the
        # lag. (5, 1) is this study's rule; (1, 1) is the single previous-month
        # maximum; (5, 12) is the rolling form of the Akt's own annual window.
        self.agreed_power_n_peaks = max(int(agreed_power_n_peaks), 1)
        self.agreed_power_n_months_window = max(int(agreed_power_n_months_window), 1)
        # Whether a runner should converge the contract onto the peaks its own
        # dispatch achieves. Inert when an explicit `contracted_power_kw` was
        # signed, and read (not obeyed) here: the loop belongs to whoever owns
        # the dispatch, so this flag only tells them to run it.
        self.agreed_power_from_dispatch = bool(agreed_power_from_dispatch)
        self._explicit_contracted_power_kw = contracted_power_kw
        self._install_agreed_power_schedule(
            self._build_agreed_power_schedule(contracted_power_kw)
        )

        self._peak_kw = {b: 0.0 for b in _BLOCKS}
        self._peak_window_id = 0

        self.arr_price_norm = self.dataset_norm[self.price_column].to_numpy(dtype=np.float64)
        self.arr_generation_norm = self.dataset_norm[self.generation_column].to_numpy(dtype=np.float64)
        self.arr_consumption_norm = self.dataset_norm[self.consumption_column].to_numpy(dtype=np.float64)

        if hasattr(self.dataset.index, "hour"):
            self.arr_hour = self.dataset.index.hour.to_numpy()
            self.arr_minute = self.dataset.index.minute.to_numpy()
        else:
            self.arr_hour = np.zeros(self.data_length, dtype=np.int32)
            self.arr_minute = np.zeros(self.data_length, dtype=np.int32)

        window_size = max(1, int(median_window_days * self.steps_per_day))
        self.arr_median_price = (
            self.dataset[self.price_column]
            .rolling(window=window_size, min_periods=1)
            .median()
            .to_numpy(dtype=np.float64)
        )
        self.arr_relative_price = (self.arr_price - self.arr_median_price) / (
            self.arr_median_price + 1e-8
        )

        self.episode_length = max(
            1,
            int(episode_length) if episode_length is not None else int(self.data_length - 1),
        )

        self._sequential_counter = 0
        self._current_step = 0
        self._episode_steps = 0
        self._episode_start = 0
        self._episode_end_exclusive = min(self.data_length, self.episode_length)
        self._battery = max(0.0, self.battery_capacity_kwh / 2.0)
        self._cumulative_payment = 0.0

        self.window_past = 0
        self.window_future = 11 * (self.steps_per_day // 24)

        # Continuous actions carry a second component (curtailed kWh); discrete
        # actions flatten the (battery, curtailment) pair into one index.
        if self.action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(
                self.n_discrete_actions * self.n_curtailment_modes
            )
        elif self.action_scale == "normalized":
            self.action_space = gym.spaces.Box(
                low=np.float32(-1.0),
                high=np.float32(1.0),
                shape=(1 + self.allow_curtailment,),
                dtype=np.float32,
            )
        else:
            max_curtail = float(np.max(self.arr_generation)) if self.allow_curtailment else 0.0
            self.action_space = gym.spaces.Box(
                low=np.array(
                    [-self.max_discharge_kwh] + [0.0] * self.allow_curtailment, dtype=np.float32
                ),
                high=np.array(
                    [self.max_charge_kwh / self.charge_efficiency]
                    + [max_curtail] * self.allow_curtailment,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._build_observation(0, 0.0)),),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Peak-ratchet precomputes
    # ------------------------------------------------------------------
    @property
    def tariff_blocks(self):
        """Per-row tariff time-block (1-5). Read by the MILP benchmark."""
        return self._block_arr

    @property
    def reset_window_ids(self):
        """Per-row ratchet reset-window id. Read by the MILP benchmark."""
        return self._window_id_arr

    @property
    def month_ids(self):
        """Per-row absolute calendar-month id (year*12 + month - 1), local time."""
        return self._month_id_arr

    def _naive_import_power_kw(self):
        """Grid draw in kW the household would have had with no battery."""
        hours_per_interval = self.interval_minutes / 60.0
        naive_import_kwh = np.maximum(self.arr_consumption - self.arr_generation, 0.0)
        return naive_import_kwh / hours_per_interval

    def _contract_power_kw(self):
        """The grid draw the contract and the ratchet seed are measured on.

        The no-battery profile until `set_achieved_power_kw` registers what a
        dispatch actually drew, and that draw from then on. Everything that
        derives a peak from the household -- the monthly agreed power, the
        running peak seeding a mid-dataset episode -- reads it here, so the two
        can never be measured on different profiles.
        """
        if self._achieved_power_kw is None:
            return self._naive_import_power_kw()
        return self._achieved_power_kw

    def _precompute_month_ids(self):
        """Absolute calendar-month id per row, Slovenian local time -- the key the
        ratchet window and the network invoice also turn over on.
        """
        month_cache = {}
        out = np.empty(self.data_length, dtype=np.int64)
        for i in range(self.data_length):
            ts = self.dataset.index[i]
            key = (ts.year, ts.month, ts.day, ts.hour)
            mid = month_cache.get(key)
            if mid is None:
                mid = resolve_reset_window_id(ts, 1)
                month_cache[key] = mid
            out[i] = mid
        return out

    def _monthly_top_peaks_by_block(self):
        """{month id: {block: top-n kW}} of this household's billed grid draw."""
        return monthly_top_peaks_by_block(
            self._contract_power_kw(), self._block_arr, self._month_id_arr,
            n_peaks=self.agreed_power_n_peaks,
        )

    def _monthly_peak_kw_by_block(self):
        """{month id: {block: single highest kW}}, for reporting only."""
        return monthly_peak_kw_by_block(
            self._contract_power_kw(), self._block_arr, self._month_id_arr
        )

    def _month_completeness(self):
        """{month id: observed intervals / intervals a full month would have}."""
        return month_completeness(self._month_id_arr, self.steps_per_day)

    def _bootstrap_peak_kw(self, peaks, first_month=None):
        """Per-block peaks standing in for the month before `first_month`."""
        return bootstrap_peak_kw(
            peaks, self._month_id_arr, self.steps_per_day,
            mode=self.agreed_power_bootstrap, first_month=first_month,
        )

    def _build_agreed_power_schedule(self, contracted_power_kw):
        """{month id: {block: agreed kW}} in force each calendar month.
        
        An explicit `contracted_power_kw` is held constant over every month. Left
        at None, each month agrees the peak the previous month realized -- read
        off `_contract_power_kw()`, i.e. off the no-battery profile until a
        dispatch registers the draw it achieved and off that draw afterwards. A
        household that flattens its profile with a battery re-contracts down to
        it, and the shaver then has to keep it there.
        """
        if contracted_power_kw is None:
            peaks = self._monthly_top_peaks_by_block()
            self._bootstrap_kw = self._bootstrap_peak_kw(peaks)
            return mesecni_razpored_moci(
                peaks,
                minimalna_moc_kw=self.min_agreed_power_kw,
                prikljucna_moc_kw=self.connection_power_kw,
                zamik_mesecev=self.agreed_power_lag_months,
                prenesi_manjkajoce_bloke=self.agreed_power_carry_missing_blocks,
                zacetne_konice=self._bootstrap_kw,
                st_konic=self.agreed_power_n_peaks,
                n_months_window=self.agreed_power_n_months_window,
            )

        self._bootstrap_kw = None
        if isinstance(contracted_power_kw, (int, float)):
            fixed = {b: float(contracted_power_kw) for b in _BLOCKS}
        else:
            fixed = {b: float(contracted_power_kw.get(b, 0.0)) for b in _BLOCKS}
        months = sorted({int(m) for m in self._month_id_arr}) or [0]
        return {m: dict(fixed) for m in months}

    def _install_agreed_power_schedule(self, schedule):
        """Put a freshly built schedule in force and refresh what derives from it."""
        self._agreed_power_schedule = schedule
        self._agreed_power_months = sorted(schedule)
        # Single-vector summary (the mean over the months) for reporting only;
        # everything that bills reads `agreed_power_at` / `agreed_power_kw`.
        # Averaged over the months that CONTRACT each block: blocks 1 and 5 are
        # seasonal and absent from most months, and counting those as 0 kW would
        # report a figure the household never signs.
        self.contracted_power_kw = {}
        for b in _BLOCKS:
            vals = [m[b] for m in schedule.values() if b in m]
            self.contracted_power_kw[b] = sum(vals) / len(vals) if vals else 0.0
        return schedule

    @property
    def agreed_power_schedule(self):
        """{month id: {block: agreed kW}} currently in force. Read-only copy."""
        return {m: dict(v) for m, v in self._agreed_power_schedule.items()}

    @property
    def achieved_power_kw(self):
        """The registered with-battery draw in kW, or None. Read-only copy."""
        return None if self._achieved_power_kw is None else self._achieved_power_kw.copy()

    @property
    def agreed_power_is_endogenous(self):
        """True once the contract is rolled from a dispatch's own grid draw."""
        return self._achieved_power_kw is not None or self._forced_agreed_power

    def set_achieved_power_kw(self, power_kw, start_idx=0):
        """Re-roll the contract and the ratchet seed from a dispatch's OWN draw.

        `power_kw` is the grid import in kW the dispatch actually presented to
        the meter, one entry per executed interval starting at `start_idx`.
        Intervals the call does not cover keep whatever they already held -- an
        earlier registration, or the no-battery draw. That is what lets a study
        solved one month at a time register each month as it finishes and have
        the next month's contract follow from it, and it is why the contract of a
        month outside the horizon is not silently reset to the no-battery one.

        This makes the agreed power ENDOGENOUS -- the peak a controller shaves
        this month sets the line it is billed against next month -- so a caller
        that changes the dispatch has to call this again and re-run. Use
        `converge_agreed_power` rather than calling it once and trusting the
        answer.

        Returns the schedule now in force. A no-op returning the fixed schedule
        when the environment was built with an explicit `contracted_power_kw`:
        that household signed a number, and no dispatch moves it.
        """
        if self._explicit_contracted_power_kw is not None:
            return self.agreed_power_schedule

        power_kw = np.asarray(power_kw, dtype=np.float64).ravel()
        start = int(start_idx)
        if start < 0 or start + power_kw.size > self.data_length:
            raise ValueError(
                f"achieved power of {power_kw.size} intervals from {start} does not "
                f"fit the {self.data_length}-interval dataset."
            )
        if np.any(power_kw < -1e-9):
            raise ValueError("achieved power must be the grid IMPORT in kW, never negative.")

        achieved = (self._naive_import_power_kw() if self._achieved_power_kw is None
                    else self._achieved_power_kw.copy())
        achieved[start:start + power_kw.size] = np.maximum(power_kw, 0.0)
        self._achieved_power_kw = achieved
        self._forced_agreed_power = False

        self._peak_seed_history = self._peak_seed_from_power(
            achieved, self._block_arr, self._window_id_arr
        )
        return self._install_agreed_power_schedule(
            self._build_agreed_power_schedule(None)
        )

    def apply_agreed_power_schedule(self, schedule):
        """Force a specific contract, leaving the achieved-draw bookkeeping alone.

        For the contract a solve settled on when it owned the decision, and for
        `converge_agreed_power` settling a two-cycle on the conservative envelope
        of the schedules it alternates between. Either way the schedule no longer
        follows from the profiles this environment holds, so it is flagged as
        forced and `clear_achieved_power` puts the derived one back.
        """
        self._forced_agreed_power = True
        return self._install_agreed_power_schedule(
            {int(m): {int(b): float(kw) for b, kw in v.items()}
             for m, v in schedule.items()}
        )

    def clear_achieved_power(self):
        """Put the contract back on the no-battery profile.

        Every run that is not itself converging the contract has to start here,
        or it is billed under the peaks of whatever dispatch happened to run
        before it.
        """
        if self._achieved_power_kw is None and not self._forced_agreed_power:
            return self.agreed_power_schedule
        self._achieved_power_kw = None
        self._forced_agreed_power = False
        self._peak_seed_history = self._peak_seed_from_power(
            self._naive_import_power_kw(), self._block_arr, self._window_id_arr
        )
        return self._install_agreed_power_schedule(
            self._build_agreed_power_schedule(self._explicit_contracted_power_kw)
        )

    @property
    def agreed_power_bootstrap_kw(self):
        """The per-block peaks the leading month's contract was set from."""
        return self._bootstrap_kw

    @property
    def agreed_power_rule(self):
        """Tag for the agreed-power rule, stamped onto every result row."""
        return oznaka_razporeda_moci(
            minimalna_moc_kw=self.min_agreed_power_kw,
            prikljucna_moc_kw=self.connection_power_kw,
            zamik_mesecev=self.agreed_power_lag_months,
            zacetek=self.agreed_power_bootstrap,
            iz_dispecinga=self.agreed_power_is_endogenous,
            st_konic=self.agreed_power_n_peaks,
            n_months_window=self.agreed_power_n_months_window,
        )

    def agreed_power_kw(self, month_id):
        """Agreed power {block: kW} in force in an absolute calendar month."""
        month = int(month_id)
        schedule = self._agreed_power_schedule
        if month in schedule:
            return schedule[month]
        months = self._agreed_power_months
        return schedule[min(max(month, months[0]), months[-1])]

    def agreed_power_at(self, idx):
        """Agreed power {block: kW} in force at row `idx`."""
        return self.agreed_power_kw(int(self._month_id_arr[int(idx)]))

    def agreed_power_for_run(self, start_idx=0, do_not_use_previous_month=False):
        """{month id: {block: kW}} for an analysis that begins at `start_idx`.
        
        By default the run's first month is billed on its real predecessor in the
        data; `do_not_use_previous_month=True` bootstraps it as a cold start
        instead. Only the first month is affected.
        """
        schedule = self._agreed_power_schedule
        if not do_not_use_previous_month:
            return schedule
        first_month = int(self._month_id_arr[int(start_idx)])
        if first_month == self._agreed_power_months[0]:
            return schedule                  # already bootstrapped, nothing to refuse

        peaks = self._monthly_top_peaks_by_block()
        if self.agreed_power_bootstrap == "cyclic":
            # "cyclic" resolves to the last complete month of the dataset,
            # which for a mid-dataset start lies after the run.
            pass
        seeded = dict(schedule)
        vir = self._bootstrap_peak_kw(peaks, first_month=first_month) or peaks[first_month]
        seeded[first_month] = dogovorjena_moc_iz_konic(
            {b: povprecje_najvecjih(vir.get(b, ()), self.agreed_power_n_peaks)
             for b in KONICNI_BLOKI},
            minimalna_moc_kw=self.min_agreed_power_kw,
            prikljucna_moc_kw=self.connection_power_kw,
        )
        return seeded

    def agreed_power_for_timestamp(self, timestamp):
        """Agreed power {block: kW} in force at a timestamp (row index unknown)."""
        return self.agreed_power_kw(resolve_reset_window_id(timestamp, 1))

    def agreed_power_frame(self):
        """The schedule as a DataFrame beside the peaks it was set from."""
        import pandas as pd

        peaks = self._monthly_peak_kw_by_block()
        rows = []
        for month in self._agreed_power_months:
            agreed = self._agreed_power_schedule[month]
            row = {"Month": f"{month // 12:04d}-{month % 12 + 1:02d}"}
            row.update({f"Agreed_B{b}": agreed.get(b, np.nan) for b in _BLOCKS})
            row.update({f"Peak_B{b}": peaks.get(month, {}).get(b, np.nan) for b in _BLOCKS})
            rows.append(row)
        return pd.DataFrame(rows).set_index("Month")

    def _precompute_peak_seed_history(self):
        """Per-row tariff block, reset-window id and running peak of the draw.

        The peak history is measured on `_contract_power_kw()` -- the no-battery
        profile until a dispatch registers the draw it actually achieved.
        """
        block_cache = {}
        block_arr = np.empty(self.data_length, dtype=np.int32)
        window_id_arr = np.empty(self.data_length, dtype=np.int64)
        for i in range(self.data_length):
            ts = self.dataset.index[i]
            cache_key = (ts.year, ts.month, ts.day, ts.hour)
            block = block_cache.get(cache_key)
            if block is None:
                block = resolve_block_for_datetime(
                    ts, pricing_reference_year=self.pricing_reference_year
                )
                block_cache[cache_key] = block
            block_arr[i] = block
            window_id_arr[i] = resolve_reset_window_id(ts, self.peak_reset_months)

        seed = self._peak_seed_from_power(
            self._naive_import_power_kw(), block_arr, window_id_arr
        )
        return block_arr, window_id_arr, seed

    @staticmethod
    def _peak_seed_from_power(power_kw, block_arr, window_id_arr):
        """{block: running max kW}, restarting at every reset-window boundary."""
        n = len(window_id_arr)
        window_starts = np.flatnonzero(
            np.diff(window_id_arr, prepend=window_id_arr[0] - 1) != 0
        )
        peak_seed_history = {}
        for b in _BLOCKS:
            in_block = np.where(block_arr == b, power_kw, 0.0)
            seed = np.empty(n, dtype=np.float64)
            for start, stop in zip(window_starts, [*window_starts[1:], n]):
                seed[start:stop] = np.maximum.accumulate(in_block[start:stop])
            peak_seed_history[b] = seed
        return peak_seed_history

    def compute_seed_peak_kw(self, start_idx):
        """Per-block peak state seeding an episode that starts at `start_idx`."""
        if start_idx <= 0:
            return {b: 0.0 for b in _BLOCKS}
        ref_idx = start_idx - 1
        cur_idx = min(start_idx, self.data_length - 1)
        if self._window_id_arr[ref_idx] != self._window_id_arr[cur_idx]:
            return {b: 0.0 for b in _BLOCKS}
        return {b: float(self._peak_seed_history[b][ref_idx]) for b in _BLOCKS}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _current_block_features(self, idx):
        block = int(self._block_arr[idx])
        return float(self.contracted_power_kw.get(block, 0.0)), float(block) / N_BLOCKS

    def _build_observation(self, idx, soc_norm):
        # Cyclical time-of-day features.
        hour_fraction = (self.arr_hour[idx] + self.arr_minute[idx] / 60.0) / 24.0
        sin_time = np.sin(2 * np.pi * hour_fraction)
        cos_time = np.cos(2 * np.pi * hour_fraction)
        block_peak_kw, block_ratio = self._current_block_features(idx)

        if self.observation_mode == "compact":
            return np.array(
                [
                    soc_norm,
                    self.arr_generation_norm[idx],
                    self.arr_consumption_norm[idx],
                    self.arr_price_norm[idx],
                    sin_time,
                    cos_time,
                    block_peak_kw,
                    block_ratio,
                ],
                dtype=np.float32,
            )

        start_past = max(0, idx - self.window_past + 1)
        pad_left = self.window_past - (idx - start_past + 1)

        gen_slice = self.arr_generation_norm[start_past : idx + 1].astype(np.float32)
        gen_window = np.concatenate([np.zeros(pad_left, dtype=np.float32), gen_slice])

        con_slice = self.arr_consumption_norm[start_past : idx + 1].astype(np.float32)
        con_window = np.concatenate([np.zeros(pad_left, dtype=np.float32), con_slice])

        end_price = min(self.data_length, idx + self.window_future + 1)
        price_slice = self.arr_price_norm[start_past:end_price].astype(np.float32)
        pad_right = (self.window_past + self.window_future) - pad_left - len(price_slice)
        price_window = np.concatenate(
            [
                np.zeros(pad_left, dtype=np.float32),
                price_slice,
                np.zeros(max(0, pad_right), dtype=np.float32),
            ]
        )

        return np.concatenate(
            [
                np.array([soc_norm], dtype=np.float32),
                gen_window,
                con_window,
                price_window,
                np.array([sin_time, cos_time, block_peak_kw, block_ratio], dtype=np.float32),
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward_soc(self, soc_norm):
        """Penalize leaving the comfortable state-of-charge band."""
        low, high = 0.1, 0.8
        if soc_norm < low:
            return -5.0 * (low - soc_norm) * self.reward_weight_soc
        if soc_norm > high:
            return -5.0 * (soc_norm - high) * self.reward_weight_soc
        return 0.0

    def _reward_arbitrage(self, relative_price, battery_delta_kwh):
        """Reward charging below the rolling median price and discharging above it."""
        if self.battery_capacity_kwh <= 0:
            return 0.0
        gamma = 3.0
        norm_change = battery_delta_kwh / self.battery_capacity_kwh
        return -5.0 * gamma * norm_change * relative_price * self.reward_weight_arbitrage

    def _reward_cost(self, variable_cost_eur):
        """Penalize the decision-dependent part of this interval's bill."""
        return -(5.0 / 8.0) * variable_cost_eur * self.reward_weight_cost

    def compute_reward(self, soc_norm, relative_price, battery_delta_kwh, variable_cost_eur):
        """Weighted sum of the three reward components, normalized by weight."""
        if self.battery_capacity_kwh <= 0:
            return self._reward_cost(variable_cost_eur)
        denominator = (
            self.reward_weight_soc + self.reward_weight_arbitrage + self.reward_weight_cost
        )
        if denominator <= 0:
            return self._reward_cost(variable_cost_eur)
        return (
            self._reward_soc(soc_norm)
            + self._reward_arbitrage(relative_price, battery_delta_kwh)
            + self._reward_cost(variable_cost_eur)
        ) / denominator

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def battery_limits(self, soc_kwh):
        """(max charge, max discharge) setpoints, non-negative kWh at the AC side."""
        return (
            max_charge_now(
                soc_kwh, self.charge_efficiency, self.max_charge_kwh, self.battery_capacity_kwh
            ),
            max_discharge_now(soc_kwh, self.discharge_efficiency, self.max_discharge_kwh),
        )

    def discrete_action_to_setpoint(self, action_int, soc_kwh, generation_kwh, consumption_kwh):
        """Legacy discrete action -> battery setpoint (kWh, + charge / - discharge)."""
        limit_charge, limit_discharge = self.battery_limits(soc_kwh)
        if action_int == ACTION_CHARGE_ANY:
            return limit_charge
        if action_int == ACTION_CHARGE_PV:
            return min(pv_surplus(generation_kwh, consumption_kwh), limit_charge)
        if action_int == ACTION_DISCHARGE_HOME:
            return -min(max(consumption_kwh - generation_kwh, 0.0), limit_discharge)
        if action_int == ACTION_DISCHARGE_ANY:
            return -limit_discharge
        if action_int == ACTION_IDLE:
            return 0.0
        raise ValueError(f"Unsupported discrete action: {action_int}")

    def curtailment_for_mode(self, mode, setpoint_kwh, generation_kwh, consumption_kwh):
        """Curtailed kWh for a discrete curtailment mode, clipped to [0, generation]."""
        if mode == CURTAIL_NONE:
            return 0.0
        if mode == CURTAIL_ALL:
            return float(generation_kwh)
        if mode == CURTAIL_NO_EXPORT:
            # grid_import = consumption - generation + setpoint; curtail exactly
            # the amount that would otherwise be exported (i.e. drive it to 0).
            exported = generation_kwh - consumption_kwh - setpoint_kwh
            return float(np.clip(exported, 0.0, generation_kwh))
        raise ValueError(f"Unsupported curtailment mode: {mode}")

    def _resolve_action(self, action, soc_kwh, generation_kwh, consumption_kwh):
        """The agent's action as (setpoint_kwh, curtailed_kwh, action_int).
        
        Curtailment is resolved first; the setpoint then sees only what is left.
        """
        if self.action_mode == "discrete":
            action_int = int(getattr(action, "value", action))
            if not 0 <= action_int < self.action_space.n:
                raise ValueError(f"Unsupported action: {action}")
            battery_action = action_int % self.n_discrete_actions
            curtail_mode = action_int // self.n_discrete_actions

            # CURTAIL_NO_EXPORT is defined against the setpoint the uncurtailed
            # generation would produce; for every charging action that is a fixed
            # point, so the setpoint below is unchanged by the substitution.
            curtailed = self.curtailment_for_mode(
                curtail_mode,
                self.discrete_action_to_setpoint(
                    battery_action, soc_kwh, generation_kwh, consumption_kwh
                ),
                generation_kwh,
                consumption_kwh,
            )
            setpoint = self.discrete_action_to_setpoint(
                battery_action, soc_kwh, generation_kwh - curtailed, consumption_kwh
            )
            return setpoint, curtailed, action_int

        values = np.asarray(action, dtype=np.float64).reshape(-1)
        curtailed = (
            float(np.clip(values[1], 0.0, generation_kwh)) if self.allow_curtailment else 0.0
        )
        setpoint = float(values[0])
        if self.action_scale == "normalized":
            limit_charge, limit_discharge = self.battery_limits(soc_kwh)
            setpoint = setpoint * (limit_charge if setpoint >= 0 else limit_discharge)
            if self.allow_curtailment:
                curtailed = float(np.clip(values[1], 0.0, 1.0)) * generation_kwh
        return setpoint, curtailed, None

    # ------------------------------------------------------------------
    # Episode plumbing
    # ------------------------------------------------------------------
    def _resolve_start_index(self, options):
        mode = self.reset_mode
        if options is not None:
            mode = str(options.get("reset_mode", mode))

        max_valid_start = max(0, self.data_length - self.episode_length)
        if mode == "deterministic":
            return 0
        if mode == "random":
            return int(self.np_random.integers(0, max_valid_start + 1))
        if mode == "sequential":
            n = max(1, int(options.get("sequential_n", 10)) if options else 10)
            idx = self._sequential_counter % n
            self._sequential_counter += 1
            return int(max_valid_start * idx / n)
        raise ValueError("reset_mode must be 'deterministic', 'random', or 'sequential'")

    def _build_info(self, idx, soc_kwh, cumulative_payment, **extra):
        soc_norm = (
            soc_kwh / self.battery_capacity_kwh if self.battery_capacity_kwh > 0 else 0.0
        )
        info = {
            "step_idx": int(idx),
            "battery": float(soc_kwh),
            "battery_norm": float(soc_norm),
            "cumulative_payment": float(cumulative_payment),
            "price": float(self.arr_price[idx]),
            "price_norm": float(self.arr_price_norm[idx]),
            "relative_price": float(self.arr_relative_price[idx]),
            "generation": float(self.arr_generation[idx]),
            "consumption": float(self.arr_consumption[idx]),
        }
        info.update({k: v for k, v in extra.items() if v is not None})
        return info

    def _invoice_period_label(self, start_idx, end_idx):
        start_ts = self.dataset.index[start_idx]
        end_ts = self.dataset.index[min(end_idx, self.data_length - 1)]
        return f"{start_ts:%Y-%m-%d}_{end_ts:%Y-%m-%d}"

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self._invoicing_enabled:
            if self._invoice_builder is not None:
                # Defensive: a caller resetting before hitting terminated/truncated
                # would otherwise silently drop the previous episode's invoice.
                self._invoice_builder.finalize()
            self._invoice_builder = InvoiceBuilder(
                # Callable, not a dict: the agreed power changes on the 1st, so
                # each month's invoice has to be built against its own figure.
                dogovorjena_moc=lambda year, month: self.agreed_power_kw(year * 12 + month - 1),
                pricing_scheme=self.pricing_scheme,
                eko_racun=self.invoice_eko_racun,
                interval_minutes=self.interval_minutes,
                output_dir=self.invoice_output_dir,
                run_label=str(self.invoice_run_label or self.pricing_scheme),
                write_monthly=self.generate_monthly_invoice,
                write_period=self.generate_period_invoice,
                pricing_reference_year=self.pricing_reference_year,
            )

        self._episode_start = self._resolve_start_index(options)
        self._episode_end_exclusive = max(
            self._episode_start + 1,
            min(self.data_length, self._episode_start + self.episode_length),
        )

        self._current_step = self._episode_start
        self._episode_steps = 0
        self._battery = max(0.0, self.battery_capacity_kwh / 2.0)
        self._cumulative_payment = 0.0
        self._peak_kw = self.compute_seed_peak_kw(self._episode_start)
        self._peak_window_id = int(self._window_id_arr[self._episode_start])

        soc_norm = (
            self._battery / self.battery_capacity_kwh if self.battery_capacity_kwh > 0 else 0.0
        )
        obs = self._build_observation(self._current_step, soc_norm)
        return obs, self._build_info(self._current_step, self._battery, self._cumulative_payment)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def step(self, action):
        idx = self._current_step
        soc_kwh = float(np.clip(self._battery, 0.0, self.battery_capacity_kwh))
        soc_norm = soc_kwh / self.battery_capacity_kwh if self.battery_capacity_kwh > 0 else 0.0
        generation_available_kwh = self.arr_generation[idx]
        consumption_kwh = self.arr_consumption[idx]

        setpoint_requested, curtailed_kwh, action_int = self._resolve_action(
            action, soc_kwh, generation_available_kwh, consumption_kwh
        )
        # Curtailment simply removes production before anything else sees it,
        # mirroring how P_spill enters the MILP energy balance.
        generation_kwh = generation_available_kwh - curtailed_kwh

        next_idx = idx + 1
        if next_idx >= self.data_length:
            obs = self._build_observation(idx, soc_norm)
            if self._invoicing_enabled:
                self._invoice_builder.finalize(
                    period_label=self._invoice_period_label(self._episode_start, idx)
                )
            info = self._build_info(idx, soc_kwh, self._cumulative_payment, action=action_int)
            return obs, 0.0, True, False, info

        # --- Energy router -------------------------------------------------
        # One signed setpoint drives every flow; the discrete actions above are
        # just five particular choices of setpoint.
        limit_charge, limit_discharge = self.battery_limits(soc_kwh)
        setpoint = float(np.clip(setpoint_requested, -limit_discharge, limit_charge))

        surplus_kwh = pv_surplus(generation_kwh, consumption_kwh)
        deficit_kwh = max(consumption_kwh - generation_kwh, 0.0)

        if setpoint >= 0.0:
            pv_to_battery_kwh = min(setpoint, surplus_kwh)
            grid_to_battery_kwh = setpoint - pv_to_battery_kwh
            battery_to_home_kwh = battery_to_grid_kwh = 0.0
        else:
            battery_to_home_kwh = min(-setpoint, deficit_kwh)
            battery_to_grid_kwh = -setpoint - battery_to_home_kwh
            pv_to_battery_kwh = grid_to_battery_kwh = 0.0

        grid_import_kwh = consumption_kwh - generation_kwh + setpoint

        battery_delta_kwh = battery_delta(
            pv_to_battery_kwh + grid_to_battery_kwh,
            battery_to_home_kwh + battery_to_grid_kwh,
            self.charge_efficiency,
            self.discharge_efficiency,
        )

        balance_error = (
            (pv_to_battery_kwh + grid_to_battery_kwh)
            + consumption_kwh
            - (generation_available_kwh - curtailed_kwh)
            - grid_import_kwh
            - (battery_to_home_kwh + battery_to_grid_kwh)
        )
        if abs(balance_error) > 1e-8:
            raise ValueError(
                f"Energy balance error {balance_error}: "
                f"charge={pv_to_battery_kwh + grid_to_battery_kwh}, "
                f"discharge={battery_to_home_kwh + battery_to_grid_kwh}, "
                f"consumption={consumption_kwh}, "
                f"generation={generation_available_kwh}, curtailed={curtailed_kwh}, "
                f"grid_import={grid_import_kwh}"
            )

        # --- Pricing -------------------------------------------------------
        current_window_id = int(self._window_id_arr[idx])
        if current_window_id != self._peak_window_id:
            self._peak_kw = {b: 0.0 for b in _BLOCKS}
            self._peak_window_id = current_window_id

        price_result = calculate_interval_price(
            self.arr_price[idx],
            grid_import_kwh,
            utc_date=self.dataset.index[idx],
            interval_minutes=self.interval_minutes,
            scheme=self.pricing_scheme,
            include_raw=(self.pricing_include_raw or self._invoicing_enabled),
            dogovorjena_moc=self.agreed_power_at(idx),
            prev_peak_kw=self._peak_kw,
            **self.pricing_options,
        )
        fixed_cost_eur = float(price_result["constant_price_aud"])
        variable_cost_eur = float(price_result["variable_price_aud"])
        self._peak_kw = dict(price_result["new_peak_kw"])
        if self._invoicing_enabled:
            self._invoice_builder.add_interval(price_result)

        # --- State update --------------------------------------------------
        new_soc_kwh = soc_kwh + battery_delta_kwh
        if not -1e-8 <= new_soc_kwh <= self.battery_capacity_kwh + 1e-8:
            raise ValueError(
                f"Battery state out of bounds: current={soc_kwh}, "
                f"change={battery_delta_kwh}, new={new_soc_kwh}, "
                f"capacity={self.battery_capacity_kwh}"
            )
        new_soc_kwh = float(np.clip(new_soc_kwh, 0.0, self.battery_capacity_kwh))
        new_payment = self._cumulative_payment + variable_cost_eur + fixed_cost_eur

        # --- Reward --------------------------------------------------------
        relative_price = self.arr_relative_price[idx]
        r_soc = self._reward_soc(soc_norm)
        r_arbitrage = (
            self._reward_arbitrage(relative_price, battery_delta_kwh)
            if self.battery_capacity_kwh > 0
            else 0.0
        )
        r_cost = self._reward_cost(variable_cost_eur)

        penalty = self._infeasible_request_penalty(
            action_int, soc_kwh, surplus_kwh, setpoint_requested, setpoint
        )
        reward = float(
            np.clip(
                self.compute_reward(
                    soc_norm, relative_price, battery_delta_kwh, variable_cost_eur
                )
                + penalty,
                -10.0,
                5.0,
            )
        )

        self._current_step = next_idx
        self._battery = new_soc_kwh
        self._cumulative_payment = new_payment
        self._episode_steps += 1

        terminated = self._current_step >= (self.data_length - 1)
        truncated = self._current_step >= (self._episode_end_exclusive - 1)

        new_soc_norm = (
            new_soc_kwh / self.battery_capacity_kwh if self.battery_capacity_kwh > 0 else 0.0
        )
        obs = self._build_observation(next_idx, new_soc_norm)

        info = self._build_info(
            next_idx,
            new_soc_kwh,
            new_payment,
            action=action_int,
            battery_setpoint_kwh={
                "requested": float(setpoint_requested),
                "applied": float(setpoint),
            },
            energy_flows={
                "pv_to_battery_kwh": float(pv_to_battery_kwh),
                "grid_to_battery_kwh": float(grid_to_battery_kwh),
                "battery_to_home_kwh": float(battery_to_home_kwh),
                "battery_to_grid_kwh": float(battery_to_grid_kwh),
                "grid_import_kwh": float(grid_import_kwh),
                "battery_delta_kwh": float(battery_delta_kwh),
                "curtailed_kwh": float(curtailed_kwh),
            },
            reward_components={
                "total": float(reward),
                "r_soc": float(r_soc),
                "r_arbitrage": float(r_arbitrage),
                "r_cost": float(r_cost),
                "variable_cost_eur": float(variable_cost_eur),
                "energy_component_eur": float(price_result["energy_component_eur"]),
                "power_component_eur": float(price_result["power_component_eur"]),
                "fixed_monthly_charge_eur": float(price_result["fixed_monthly_charge_eur"]),
            },
            peak_kw=dict(self._peak_kw),
        )

        if self._invoicing_enabled and (terminated or truncated):
            self._invoice_builder.finalize(
                period_label=self._invoice_period_label(self._episode_start, next_idx)
            )

        return obs, reward, bool(terminated), bool(truncated), info

    def _infeasible_request_penalty(
        self, action_int, soc_kwh, surplus_kwh, setpoint_requested, setpoint_applied
    ):
        """Penalty for asking for something the battery cannot do."""
        if self.action_mode == "discrete":
            battery_action = action_int % self.n_discrete_actions
            if battery_action in (ACTION_DISCHARGE_HOME, ACTION_DISCHARGE_ANY) and soc_kwh <= 1e-8:
                return -0.5  # tried to discharge an empty battery
            if (
                battery_action in (ACTION_CHARGE_ANY, ACTION_CHARGE_PV)
                and soc_kwh >= (self.battery_capacity_kwh - 1e-8)
                and surplus_kwh > 0
            ):
                return -0.5  # tried to charge a full battery
            return 0.0

        if self.clip_penalty == 0.0:
            return 0.0
        return -self.clip_penalty * abs(setpoint_requested - setpoint_applied)

    def render(self):
        return None

    def close(self):
        return None


class CommunityEnvironment(gym.Env):
    """Runs multiple HouseholdEnvironment instances together, uncoupled.
    
    Grouped stepping, per-household flow history, separate and aggregate
    invoice views.
    """

    metadata = {"render_modes": []}

    def __init__(self, household_envs: Dict[str, HouseholdEnvironment]):
        if not household_envs:
            raise ValueError("household_envs must contain at least one household environment.")

        normalized: Dict[str, HouseholdEnvironment] = {}
        for household_id, env in household_envs.items():
            if not isinstance(env, HouseholdEnvironment):
                raise TypeError(
                    f"Environment for household {household_id!r} is not HouseholdEnvironment."
                )
            normalized[str(household_id)] = env

        self.household_envs = normalized
        self.household_ids = tuple(normalized.keys())

        self.action_space = gym.spaces.Dict(
            {hid: env.action_space for hid, env in self.household_envs.items()}
        )
        self.observation_space = gym.spaces.Dict(
            {hid: env.observation_space for hid, env in self.household_envs.items()}
        )

        self._flow_history: Dict[str, list] = {hid: [] for hid in self.household_ids}
        self._last_info: Dict[str, Dict[str, Any]] = {hid: {} for hid in self.household_ids}

    def _default_action(self, household_id: str):
        """Leave the battery idle and curtail nothing -- the safest fallback."""
        env = self.household_envs[household_id]
        if env.action_mode == "discrete":
            return min(ACTION_IDLE, env.n_discrete_actions - 1)
        return np.zeros(env.action_space.shape, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observations: Dict[str, Any] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for idx, hid in enumerate(self.household_ids):
            env_seed = None if seed is None else int(seed) + idx
            obs, info = self.household_envs[hid].reset(seed=env_seed, options=options)
            observations[hid] = obs
            infos[hid] = info
            self._last_info[hid] = dict(info)
            self._flow_history[hid] = []

        return observations, infos

    def step(self, actions: Dict[str, Any]):
        observations: Dict[str, Any] = {}
        rewards: Dict[str, float] = {}
        terminated: Dict[str, bool] = {}
        truncated: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for hid in self.household_ids:
            action = actions.get(hid, self._default_action(hid))
            obs, reward, done, cut, info = self.household_envs[hid].step(action)
            observations[hid] = obs
            rewards[hid] = float(reward)
            terminated[hid] = bool(done)
            truncated[hid] = bool(cut)
            infos[hid] = info
            self._last_info[hid] = dict(info)

            flow = info.get("energy_flows")
            if flow is not None:
                self._flow_history[hid].append(
                    {
                        "step_idx": int(info.get("step_idx", -1)),
                        "household_id": hid,
                        **{k: float(v) for k, v in flow.items()},
                    }
                )

        return observations, rewards, terminated, truncated, infos

    def get_flow_history(self) -> Dict[str, list]:
        """Returns energy-flow history captured from household info dicts."""
        return {hid: list(rows) for hid, rows in self._flow_history.items()}

    def get_cumulative_payment(self) -> Dict[str, float]:
        """Returns current cumulative payment per household."""
        return {
            hid: float(self._last_info.get(hid, {}).get("cumulative_payment", 0.0))
            for hid in self.household_ids
        }

    def get_invoice_views(self, period_label: Optional[str] = None) -> Dict[str, Any]:
        """Returns separate household invoices and one group aggregate invoice."""
        household_rows: Dict[str, list] = {}
        for hid in self.household_ids:
            builder = self.household_envs[hid]._invoice_builder
            household_rows[hid] = [] if builder is None else builder.get_monthly_line_items()

        return aggregate_household_invoices(household_rows, period_label or "Skupno_obdobje")

    def render(self):
        return None

    def close(self):
        return None
