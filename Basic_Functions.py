"""Battery model helpers shared by the RL environment and the MILP benchmark.

Sign and efficiency convention
------------------------------
The battery is driven by a signed *setpoint* `p`, in kWh for one interval,
measured at the grid/AC side of the inverter -- i.e. energy as the house and
the meter see it, before round-trip losses:

    p > 0   charge:    p kWh flows into the battery system this interval.
                       Stored energy rises by  charge_efficiency * p.
    p < 0   discharge: |p| kWh flows out of the battery system.
                       Stored energy falls by  |p| / discharge_efficiency.
    p = 0   idle.

Charge and discharge efficiency are independent parameters so asymmetric
batteries can be modelled; both default to 0.95 in `HouseholdEnvironment`.
"""

from Pricing_Functions import calculate_interval_price


def max_discharge_now(soc_kwh, discharge_efficiency, max_discharge_kwh):
    """Largest feasible discharge setpoint (kWh at the AC side) for this step.

    Bounded by the energy actually stored and by the per-interval discharge
    limit. Discharging exactly this much drains `min(soc, max_discharge_kwh)`
    from storage.
    """
    return discharge_efficiency * min(soc_kwh, max_discharge_kwh)


def max_charge_now(soc_kwh, charge_efficiency, max_charge_kwh, capacity_kwh):
    """Largest feasible charge setpoint (kWh at the AC side) for this step.

    Bounded by the headroom left in the battery and by the per-interval charge
    limit. Charging exactly this much adds `min(capacity - soc, max_charge_kwh)`
    to storage.
    """
    return (1 / charge_efficiency) * min(capacity_kwh - soc_kwh, max_charge_kwh)


def pv_surplus(generation_kwh, consumption_kwh):
    """PV energy left over after serving the house -- available to the battery
    or for export."""
    return max(generation_kwh - consumption_kwh, 0.0)


def battery_delta(charge_kwh, discharge_kwh, charge_efficiency, discharge_efficiency):
    """Change in stored energy (kWh) for one interval, given the AC-side charge
    and discharge amounts. Both arguments are non-negative magnitudes."""
    return charge_efficiency * charge_kwh - discharge_kwh / discharge_efficiency


def cumulative_interval_price_series(consumption, generation, pricing_env, dataset):
    """Cumulative electricity cost of a fixed consumption/generation profile,
    priced interval by interval exactly the way `pricing_env` prices a
    trajectory. Used for the no-battery / no-PV baselines.
    """
    cumulative_payment = 0.0
    cumulative_series = []
    interval_minutes = 1440.0 / pricing_env.steps_per_day
    peak_kw = pricing_env.compute_seed_peak_kw(0)
    window_ids = pricing_env.reset_window_ids
    peak_window_id = int(window_ids[0]) if window_ids is not None and len(dataset) > 0 else 0

    for i, timestamp in enumerate(dataset.index):
        if window_ids is not None:
            current_window_id = int(window_ids[i])
            if current_window_id != peak_window_id:
                peak_kw = {b: 0.0 for b in range(1, 6)}
                peak_window_id = current_window_id

        net_kwh = float(consumption.iloc[i] - generation.iloc[i])
        interval_cost = calculate_interval_price(
            smp_market_price_kwh=float(dataset[pricing_env.price_column].iloc[i]),
            total_consumed_kwh=net_kwh,
            utc_date=timestamp,
            interval_minutes=interval_minutes,
            scheme=pricing_env.pricing_scheme,
            dogovorjena_moc=pricing_env.agreed_power_for_timestamp(timestamp),
            prev_peak_kw=peak_kw,
            **pricing_env.pricing_options,
        )

        peak_kw = dict(interval_cost["new_peak_kw"])
        cumulative_payment += float(interval_cost["constant_price_aud"])
        cumulative_payment += float(interval_cost["variable_price_aud"])
        cumulative_series.append(cumulative_payment)

    return cumulative_series
