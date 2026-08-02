"""Gymnasium environments for household / energy-community battery dispatch.

`HouseholdEnvironment` is continuous-first: an action is a signed battery
setpoint in kWh for the interval (positive = charge, negative = discharge),
the same quantity the MILP benchmark solves for. Discrete actions remain
available via `action_mode="discrete"` so the legacy DQN keeps working; they
are a five-entry lookup that produces a setpoint and then goes through exactly
the same energy router.
"""

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
        clip_penalty=0.0,
        episode_length=None,
        steps_per_day=96,
        battery_capacity_kwh=20.0,
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

        self.clip_penalty = float(clip_penalty)

        self.steps_per_day = int(steps_per_day)
        if self.steps_per_day <= 0:
            raise ValueError("steps_per_day must be > 0")
        self.interval_minutes = 1440.0 / self.steps_per_day

        self.battery_capacity_kwh = float(battery_capacity_kwh)
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

        # Fresh dict per instance -- a mutable default argument here would be
        # mutated in place below (pricing_reference_year) and shared/corrupted
        # across every subsequent construction that also omits pricing_options.
        self.pricing_options = {"pricing_mode": "dinamicni", "buyback_mode": "dinamicni"}
        self.pricing_options.update(pricing_options or {})

        # Dataset timestamps (Ausgrid 2010-2013, Fluvius) predate the published
        # SI tariff acts, so leaving this unset would make pricing fall back to
        # the 2026 regime anyway (see Pricing_Functions._resolve_pravila); pin
        # it explicitly so the RL environment, the MILP benchmark and the
        # invoice builder all agree on the same year.
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
        if contracted_power_kw is None:
            self.contracted_power_kw = self._default_contracted_power_kw()
        elif isinstance(contracted_power_kw, (int, float)):
            self.contracted_power_kw = {b: float(contracted_power_kw) for b in _BLOCKS}
        else:
            self.contracted_power_kw = {b: float(contracted_power_kw.get(b, 0.0)) for b in _BLOCKS}

        self._block_arr, self._window_id_arr, self._peak_seed_history = (
            self._precompute_peak_seed_history()
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

        if self.action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(self.n_discrete_actions)
        elif self.action_scale == "normalized":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(
                low=np.float32(-self.max_discharge_kwh),
                high=np.float32(self.max_charge_kwh / self.charge_efficiency),
                shape=(1,),
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

    def _naive_import_power_kw(self):
        """Grid draw in kW the household would have had with no battery."""
        hours_per_interval = self.interval_minutes / 60.0
        naive_import_kwh = np.maximum(self.arr_consumption - self.arr_generation, 0.0)
        return naive_import_kwh / hours_per_interval

    def _default_contracted_power_kw(self):
        """Default contracted power: set meaningfully BELOW the historical
        realized peak, so the agent has real room to improve via peak-shaving
        rather than starting already compliant with a contract sized to its
        worst historical moment."""
        naive_power_kw = self._naive_import_power_kw()
        historical_peak_kw = float(np.max(naive_power_kw)) if naive_power_kw.size else 0.0
        return {b: historical_peak_kw / 1.5 for b in _BLOCKS}

    def _precompute_peak_seed_history(self):
        """One-time O(n) precompute of per-row tariff block, reset-window id,
        and a per-block running 'historical no-battery peak so far' array, used
        to seed the ratchet peak tracker at arbitrary episode-start indices
        (see compute_seed_peak_kw)."""
        naive_power_kw = self._naive_import_power_kw()

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

        # Running max per block, restarting at every reset-window boundary.
        window_starts = np.flatnonzero(np.diff(window_id_arr, prepend=window_id_arr[0] - 1) != 0)
        peak_seed_history = {}
        for b in _BLOCKS:
            in_block = np.where(block_arr == b, naive_power_kw, 0.0)
            seed = np.empty(self.data_length, dtype=np.float64)
            for start, stop in zip(window_starts, [*window_starts[1:], self.data_length]):
                seed[start:stop] = np.maximum.accumulate(in_block[start:stop])
            peak_seed_history[b] = seed
        return block_arr, window_id_arr, peak_seed_history

    def compute_seed_peak_kw(self, start_idx):
        """Running peak state (per block) to seed an episode starting at
        `start_idx`, derived from historical no-battery grid draw up to (but
        not including) start_idx. Returns all-zero at the very start of the
        dataset or right after a reset-window boundary."""
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
        """Reward charging when the price is below its rolling median and
        discharging when it is above."""
        if self.battery_capacity_kwh <= 0:
            return 0.0
        gamma = 3.0
        norm_change = battery_delta_kwh / self.battery_capacity_kwh
        return -5.0 * gamma * norm_change * relative_price * self.reward_weight_arbitrage

    def _reward_cost(self, variable_cost_eur):
        """Penalize the decision-dependent part of this interval's bill."""
        return -(5.0 / 8.0) * variable_cost_eur * self.reward_weight_cost

    def compute_reward(self, soc_norm, relative_price, battery_delta_kwh, variable_cost_eur):
        """Weighted sum of the three reward components, normalized by the sum of
        the weights. Public so the MILP benchmark can score its own trajectory
        with exactly the environment's reward."""
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
        """(max charge, max discharge) setpoints feasible at this state of
        charge, both non-negative kWh at the AC side."""
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

    def _resolve_action(self, action, soc_kwh, generation_kwh, consumption_kwh):
        """Turn whatever the agent produced into (setpoint_kwh, action_int)."""
        if self.action_mode == "discrete":
            action_int = int(getattr(action, "value", action))
            if not 0 <= action_int < self.n_discrete_actions:
                raise ValueError(f"Unsupported action: {action}")
            setpoint = self.discrete_action_to_setpoint(
                action_int, soc_kwh, generation_kwh, consumption_kwh
            )
            return setpoint, action_int

        value = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        if self.action_scale == "normalized":
            limit_charge, limit_discharge = self.battery_limits(soc_kwh)
            value = value * (limit_charge if value >= 0 else limit_discharge)
        return value, None

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
                dogovorjena_moc=self.contracted_power_kw,
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
        generation_kwh = self.arr_generation[idx]
        consumption_kwh = self.arr_consumption[idx]

        setpoint_requested, action_int = self._resolve_action(
            action, soc_kwh, generation_kwh, consumption_kwh
        )

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
            - generation_kwh
            - grid_import_kwh
            - (battery_to_home_kwh + battery_to_grid_kwh)
        )
        if abs(balance_error) > 1e-8:
            raise ValueError(
                f"Energy balance error {balance_error}: "
                f"charge={pv_to_battery_kwh + grid_to_battery_kwh}, "
                f"discharge={battery_to_home_kwh + battery_to_grid_kwh}, "
                f"consumption={consumption_kwh}, generation={generation_kwh}, "
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
            dogovorjena_moc=self.contracted_power_kw,
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
        """Discourage asking for something the battery cannot do.

        Discrete mode keeps the original flat penalty so legacy DQN runs stay
        comparable. Continuous mode scales `clip_penalty` (0 by default) by how
        much of the request had to be clipped away -- clipping alone already
        removes any benefit from an infeasible setpoint.
        """
        if self.action_mode == "discrete":
            if action_int in (ACTION_DISCHARGE_HOME, ACTION_DISCHARGE_ANY) and soc_kwh <= 1e-8:
                return -0.5  # tried to discharge an empty battery
            if (
                action_int in (ACTION_CHARGE_ANY, ACTION_CHARGE_PV)
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
    """Wrapper that runs multiple HouseholdEnvironment instances together.

    This class intentionally keeps each household independent (no VPP/community
    optimization coupling yet). It provides grouped stepping, per-household flow
    history tracking, and invoice views (separate + group aggregate).
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
        """Leave the battery idle -- the safest fallback for a missing action."""
        env = self.household_envs[household_id]
        if env.action_mode == "discrete":
            return min(ACTION_IDLE, env.action_space.n - 1)
        return np.zeros(1, dtype=np.float32)

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
