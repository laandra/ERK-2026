from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from Pricing_Functions import (
    InvoiceBuilder,
    aggregate_household_invoices,
    calculate_interval_price,
)

# Souporaba/community helpers live in "New pricing functions".
import sys

_SI_DIR = Path(__file__).resolve().parent / "New pricing functions"
if str(_SI_DIR) not in sys.path:
    sys.path.append(str(_SI_DIR))

from si_obracun import obracun_souporabe  # noqa: E402
from si_paketi import Gospodinjstvo, PAKETI, STORITVE_SOUPORABE, Shema, Vloga  # noqa: E402


def _step_minutes(df: pd.DataFrame) -> float:
    if len(df.index) < 2:
        return 30.0
    minutes = (
        df.index.to_series().sort_values().diff().dropna().dt.total_seconds().median() / 60.0
    )
    if pd.isna(minutes) or float(minutes) <= 0:
        return 30.0
    return float(minutes)


def _dogovorjena_moc(default_kw: float) -> Dict[int, float]:
    kw = float(default_kw)
    return {1: kw, 2: kw, 3: kw, 4: kw, 5: kw}


def run_interval_scenario(
    household_data: Dict[str, pd.DataFrame],
    *,
    scenario_name: str,
    scheme: str,
    paket_id: str,
    pv_scale_map: Optional[Dict[str, float]] = None,
    pricing_reference_year: int = 2026,
    contracted_power_kw: float = 5.0,
    generation_column: str = "Energy_Generation",
    consumption_column: str = "Energy_Consumption",
) -> Dict[str, Any]:
    """Run a no-optimization baseline over multiple households.

    Pricing is computed interval-by-interval with the same dispatcher used by
    the environment. Invoicing is generated separately for each household and
    then aggregated to a group view.
    """
    if not household_data:
        raise ValueError("household_data cannot be empty.")

    pv_scale_map = pv_scale_map or {}
    flow_rows = []
    summary_rows = []
    household_line_items: Dict[str, list] = {}

    for household_id, df in household_data.items():
        hid = str(household_id)
        interval_minutes = _step_minutes(df)
        dogovorjena = _dogovorjena_moc(contracted_power_kw)
        peak_kw = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        pv_scale = float(pv_scale_map.get(hid, 1.0))

        builder = InvoiceBuilder(
            dogovorjena_moc=dogovorjena,
            pricing_scheme=scheme,
            interval_minutes=interval_minutes,
            output_dir=Path("Results") / "Invoices",
            run_label=f"{scenario_name}_{hid}",
            write_monthly=False,
            write_period=False,
            pricing_reference_year=pricing_reference_year,
        )

        total_cost = 0.0
        imported_kwh = 0.0
        exported_kwh = 0.0
        total_generation_kwh = 0.0
        total_consumption_kwh = 0.0

        for ts, row in df.iterrows():
            generation_kwh = float(row[generation_column]) * pv_scale
            consumption_kwh = float(row[consumption_column])
            net_consumed_kwh = consumption_kwh - generation_kwh

            price_result = calculate_interval_price(
                smp_market_price_kwh=float(row["SMP"]),
                total_consumed_kwh=net_consumed_kwh,
                utc_date=ts,
                interval_minutes=interval_minutes,
                scheme=scheme,
                paket_id=paket_id,
                pricing_reference_year=pricing_reference_year,
                dogovorjena_moc=dogovorjena,
                prev_peak_kw=peak_kw,
                include_raw=True,
            )
            peak_kw = dict(price_result.get("new_peak_kw", peak_kw))
            builder.add_interval(price_result)

            interval_total = float(price_result["variable_price_aud"]) + float(
                price_result["constant_price_aud"]
            )
            imported = max(net_consumed_kwh, 0.0)
            exported = max(-net_consumed_kwh, 0.0)

            total_cost += interval_total
            imported_kwh += imported
            exported_kwh += exported
            total_generation_kwh += generation_kwh
            total_consumption_kwh += consumption_kwh

            flow_rows.append(
                {
                    "scenario": scenario_name,
                    "household_id": hid,
                    "timestamp": ts,
                    "consumption_kwh": consumption_kwh,
                    "generation_kwh": generation_kwh,
                    "imported_kwh": imported,
                    "exported_kwh": exported,
                    "net_consumed_kwh": net_consumed_kwh,
                    "interval_cost_eur": interval_total,
                }
            )

        period_label = f"{df.index.min():%Y-%m-%d}_{df.index.max():%Y-%m-%d}"
        builder.finalize(period_label=period_label)
        household_line_items[hid] = builder.get_monthly_line_items()

        summary_rows.append(
            {
                "scenario": scenario_name,
                "household_id": hid,
                "interval_minutes": interval_minutes,
                "pv_scale": pv_scale,
                "total_consumption_kwh": total_consumption_kwh,
                "total_generation_kwh": total_generation_kwh,
                "total_imported_kwh": imported_kwh,
                "total_exported_kwh": exported_kwh,
                "total_cost_eur": total_cost,
            }
        )

    period_label = "Skupno_obdobje"
    invoice_views = aggregate_household_invoices(household_line_items, period_label)
    summary_df = pd.DataFrame(summary_rows)
    flow_df = pd.DataFrame(flow_rows)

    group_summary = {
        "scenario": scenario_name,
        "household_id": "GROUP",
        "interval_minutes": None,
        "pv_scale": None,
        "total_consumption_kwh": float(summary_df["total_consumption_kwh"].sum()),
        "total_generation_kwh": float(summary_df["total_generation_kwh"].sum()),
        "total_imported_kwh": float(summary_df["total_imported_kwh"].sum()),
        "total_exported_kwh": float(summary_df["total_exported_kwh"].sum()),
        "total_cost_eur": float(summary_df["total_cost_eur"].sum()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([group_summary])], ignore_index=True)

    return {
        "summary": summary_df,
        "flows": flow_df,
        "invoice_views": invoice_views,
    }


def run_souporaba_monthly_scenario(
    household_data: Dict[str, pd.DataFrame],
    *,
    scenario_name: str,
    oddajnik_id: str,
    prejemnik_ids: Iterable[str],
    pv_scale_map: Optional[Dict[str, float]] = None,
    delez_souporabe: float = 0.4,
    cena_souporabe_eur_kwh: float = 0.05,
    organizer_service_id: str = "GENI_SOUPORABA",
    oddajnik_paket_id: str = "GENI_SAMO_DINAMICNI",
    prejemnik_paket_id: str = "GENI_DINAMICNI",
    contracted_power_kw: float = 5.0,
    generation_column: str = "Energy_Generation",
    consumption_column: str = "Energy_Consumption",
) -> pd.DataFrame:
    """Compute one-month souporaba settlement summary for a household group."""
    if oddajnik_id not in household_data:
        raise ValueError(f"oddajnik_id={oddajnik_id!r} not found in household_data.")

    prejemnik_ids = [str(i) for i in prejemnik_ids]
    for pid in prejemnik_ids:
        if pid not in household_data:
            raise ValueError(f"prejemnik household {pid!r} not found in household_data.")

    pv_scale_map = pv_scale_map or {}
    all_ids = [str(oddajnik_id), *prejemnik_ids]

    base_idx = household_data[str(oddajnik_id)].index
    month = int(base_idx[0].month)
    year = int(base_idx[0].year)
    month_mask = (base_idx.year == year) & (base_idx.month == month)
    selected_idx = base_idx[month_mask]

    interval_minutes = _step_minutes(household_data[str(oddajnik_id)])
    service = STORITVE_SOUPORABE[organizer_service_id]

    dogovorjena = _dogovorjena_moc(contracted_power_kw)

    udelezeni = {
        str(oddajnik_id): {
            "gospodinjstvo": Gospodinjstvo(
                str(oddajnik_id),
                dogovorjena,
                ima_pv=True,
                shema_samooskrbe=Shema.NOVA,
                vloga_souporaba=Vloga.ODDAJNIK,
                delez_souporabe=float(delez_souporabe),
            ),
            "paket": PAKETI[oddajnik_paket_id],
            "delitev": {pid: 1.0 / max(len(prejemnik_ids), 1) for pid in prejemnik_ids},
        }
    }

    for pid in prejemnik_ids:
        udelezeni[pid] = {
            "gospodinjstvo": Gospodinjstvo(
                pid,
                dogovorjena,
                vloga_souporaba=Vloga.PREJEMNIK,
            ),
            "paket": PAKETI[prejemnik_paket_id],
        }

    podatki = []
    for ts in selected_idx:
        poraba = {}
        proizvodnja = {}

        for hid in all_ids:
            row = household_data[hid].loc[ts]
            poraba[hid] = float(row[consumption_column])

            pv_scale = float(pv_scale_map.get(hid, 0.0))
            prod = float(row[generation_column]) * pv_scale
            if prod > 0:
                proizvodnja[hid] = prod

        # si_obracun expects market price in EUR/MWh.
        market_price_mwh = float(household_data[str(oddajnik_id)].loc[ts, "SMP"]) * 1000.0
        podatki.append(
            {
                "utc_date": ts,
                "interval_minutes": interval_minutes,
                "market_price_mwh": market_price_mwh,
                "poraba": poraba,
                "proizvodnja": proizvodnja,
            }
        )

    results = obracun_souporabe(
        udelezeni,
        podatki,
        year,
        month,
        storitev=service,
        cena_souporabe_eur_kwh=float(cena_souporabe_eur_kwh),
    )

    rows = []
    group_total = 0.0
    for hid, racun in results.items():
        group_total += float(racun.za_placilo)
        diag = dict(getattr(racun, "diagnostika", {}) or {})
        rows.append(
            {
                "scenario": scenario_name,
                "household_id": hid,
                "za_placilo_eur": float(racun.za_placilo),
                "neto_eur": float(racun.neto),
                "ddv_eur": float(racun.ddv),
                "bruto_eur": float(racun.bruto),
                "prevzeto_kwh": float(racun.prevzeto_kwh),
                "oddano_kwh": float(diag.get("oddano_kwh", 0.0)),
                "deljeno_kwh": float(diag.get("deljeno_kwh", 0.0)),
            }
        )

    rows.append(
        {
            "scenario": scenario_name,
            "household_id": "GROUP",
            "za_placilo_eur": group_total,
            "neto_eur": None,
            "ddv_eur": None,
            "bruto_eur": None,
            "prevzeto_kwh": None,
            "oddano_kwh": None,
            "deljeno_kwh": None,
        }
    )

    return pd.DataFrame(rows)
