from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
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

from si_obracun import Pravila, obracun_souporabe  # noqa: E402
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
    scheme_map: Optional[Dict[str, str]] = None,
    paket_id_map: Optional[Dict[str, str]] = None,
    contracted_power_map: Optional[Dict[str, Dict[int, float]]] = None,
    collect_flows: bool = True,
) -> Dict[str, Any]:
    """Run a no-optimization baseline over multiple households.

    Pricing is computed interval-by-interval with the same dispatcher used by
    the environment. Invoicing is generated separately for each household and
    then aggregated to a group view.

    `scheme_map`, `paket_id_map` and `contracted_power_map` override the scalar
    `scheme` / `paket_id` / `contracted_power_kw` per household. A mixed group
    needs them: a samooskrba price list requires a PV device (`zahteva_pv`), so
    households without one must be priced on the supplier's plain supply list
    inside the very same scenario.

    `collect_flows=False` skips the per-interval flow table, which is one row
    per household per interval and does not fit comfortably in memory for a
    community of tens of households over a year. The community net-import
    profile is accumulated either way, so nothing needed for the group peak or
    the duration curve is lost.
    """
    if not household_data:
        raise ValueError("household_data cannot be empty.")

    pv_scale_map = pv_scale_map or {}
    scheme_map = scheme_map or {}
    paket_id_map = paket_id_map or {}
    contracted_power_map = contracted_power_map or {}
    flow_rows = []
    summary_rows = []
    component_rows = []
    household_line_items: Dict[str, list] = {}

    # Community net position per interval, accumulated across households.
    # Accumulated POSITIONALLY as numpy arrays, not by adding pandas Series:
    # the Fluvius files record the DST fall-back hour twice, so the index has
    # duplicate labels and Series arithmetic would try to align on them.
    community_index = next(iter(household_data.values())).index
    community_import_kwh = np.zeros(len(community_index), dtype=float)
    community_export_kwh = np.zeros(len(community_index), dtype=float)

    for household_id, df in household_data.items():
        hid = str(household_id)
        interval_minutes = _step_minutes(df)
        dogovorjena = contracted_power_map.get(hid) or _dogovorjena_moc(contracted_power_kw)
        household_scheme = scheme_map.get(hid, scheme)
        household_paket_id = paket_id_map.get(hid, paket_id)
        peak_kw = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        pv_scale = float(pv_scale_map.get(hid, 1.0))

        builder = InvoiceBuilder(
            dogovorjena_moc=dogovorjena,
            pricing_scheme=household_scheme,
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
        energy_eur = 0.0
        power_eur = 0.0
        fixed_eur = 0.0
        credit_eur = 0.0
        # VAT-inclusive charge per billing item (energija, omreznina_*, dajatve,
        # ...), so the bill can also be read by who is paid rather than only by
        # decision-dependence. Items sum to energy + power + fixed + credit.
        components: Dict[str, float] = {}
        peak_import_kw = 0.0
        hours_per_interval = interval_minutes / 60.0

        # Vectorized once, then iterated -- df.iterrows() re-boxes every row and
        # dominates the runtime at community scale.
        gen_arr = df[generation_column].to_numpy(dtype=float) * pv_scale
        con_arr = df[consumption_column].to_numpy(dtype=float)
        smp_arr = df["SMP"].to_numpy(dtype=float)
        net_arr = con_arr - gen_arr
        index = df.index

        if len(index) != len(community_index):
            raise ValueError(
                f"household {hid!r} has {len(index)} intervals, expected "
                f"{len(community_index)} -- the community profile is summed "
                f"positionally and needs a common calendar."
            )
        community_import_kwh += np.maximum(net_arr, 0.0)
        community_export_kwh += np.maximum(-net_arr, 0.0)

        for i in range(len(index)):
            ts = index[i]
            generation_kwh = float(gen_arr[i])
            consumption_kwh = float(con_arr[i])
            net_consumed_kwh = float(net_arr[i])

            price_result = calculate_interval_price(
                smp_market_price_kwh=float(smp_arr[i]),
                total_consumed_kwh=net_consumed_kwh,
                utc_date=ts,
                interval_minutes=interval_minutes,
                scheme=household_scheme,
                paket_id=household_paket_id,
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
            energy_eur += float(price_result["energy_component_eur"])
            power_eur += float(price_result["power_component_eur"])
            fixed_eur += float(price_result["constant_price_aud"])
            credit_eur += float(price_result.get("dobropis_odkup_eur", 0.0))
            for item, value in price_result.get("postavke_eur", {}).items():
                components[item] = components.get(item, 0.0) + float(value)
            peak_import_kw = max(peak_import_kw, imported / hours_per_interval)

            if collect_flows:
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

        component_rows.extend(
            {
                "scenario": scenario_name,
                "household_id": hid,
                "component": item,
                "eur": value,
            }
            for item, value in sorted(components.items())
        )

        summary_rows.append(
            {
                "scenario": scenario_name,
                "household_id": hid,
                "interval_minutes": interval_minutes,
                "pv_scale": pv_scale,
                "scheme": household_scheme,
                "paket_id": household_paket_id,
                "contracted_power_kw": dogovorjena.get(2, float("nan")),
                "total_consumption_kwh": total_consumption_kwh,
                "total_generation_kwh": total_generation_kwh,
                "total_imported_kwh": imported_kwh,
                "total_exported_kwh": exported_kwh,
                "peak_import_kw": peak_import_kw,
                "energy_eur": energy_eur,
                "power_eur": power_eur,
                "fixed_eur": fixed_eur,
                "credit_eur": credit_eur,
                "total_cost_eur": total_cost,
            }
        )

    period_label = "Skupno_obdobje"
    invoice_views = aggregate_household_invoices(household_line_items, period_label)
    summary_df = pd.DataFrame(summary_rows)
    flow_df = pd.DataFrame(flow_rows)

    # The group peak is the COINCIDENT peak of the summed net import, not the
    # sum of the individual peaks -- the households do not peak at the same
    # instant. The billed power charge still uses the individual peaks; this
    # figure describes the community's grid connection.
    interval_hours = float(summary_df["interval_minutes"].iloc[0]) / 60.0
    community_profile = pd.DataFrame(
        {
            "community_import_kwh": community_import_kwh,
            "community_export_kwh": community_export_kwh,
        },
        index=community_index,
    )
    community_profile["community_import_kw"] = (
        community_profile["community_import_kwh"] / interval_hours
    )
    coincident_peak_kw = float(community_profile["community_import_kw"].max())
    coincident_peak_at = community_profile["community_import_kw"].idxmax()
    sum_of_individual_peaks_kw = float(summary_df["peak_import_kw"].sum())

    group_summary = {
        "scenario": scenario_name,
        "household_id": "GROUP",
        "interval_minutes": None,
        "pv_scale": None,
        "scheme": None,
        "paket_id": None,
        "contracted_power_kw": float(summary_df["contracted_power_kw"].sum()),
        "total_consumption_kwh": float(summary_df["total_consumption_kwh"].sum()),
        "total_generation_kwh": float(summary_df["total_generation_kwh"].sum()),
        "total_imported_kwh": float(summary_df["total_imported_kwh"].sum()),
        "total_exported_kwh": float(summary_df["total_exported_kwh"].sum()),
        "peak_import_kw": coincident_peak_kw,
        "energy_eur": float(summary_df["energy_eur"].sum()),
        "power_eur": float(summary_df["power_eur"].sum()),
        "fixed_eur": float(summary_df["fixed_eur"].sum()),
        "credit_eur": float(summary_df["credit_eur"].sum()),
        "total_cost_eur": float(summary_df["total_cost_eur"].sum()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([group_summary])], ignore_index=True)

    return {
        "summary": summary_df,
        "components": pd.DataFrame(component_rows),
        "flows": flow_df,
        "invoice_views": invoice_views,
        "community_profile": community_profile,
        "community_peak_kw": coincident_peak_kw,
        "community_peak_at": coincident_peak_at,
        "sum_of_individual_peaks_kw": sum_of_individual_peaks_kw,
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


def _souporaba_participants(
    oddajnik_ids,
    prejemnik_ids,
    *,
    dogovorjena_map: Dict[str, Dict[int, float]],
    delez_souporabe: float,
    oddajnik_paket_id: str,
    prejemnik_paket_id: str,
) -> Dict[str, Dict]:
    """Build the `udelezenci` mapping `obracun_souporabe` expects.

    Every sender shares with every receiver in equal parts. `delitev` weights
    are normalized inside `obracun_souporabe`, so equal weights are enough.
    """
    oddajnik_ids = [str(i) for i in oddajnik_ids]
    prejemnik_ids = [str(i) for i in prejemnik_ids]
    weight = 1.0 / max(len(prejemnik_ids), 1)

    udelezenci: Dict[str, Dict] = {}
    for oid in oddajnik_ids:
        udelezenci[oid] = {
            "gospodinjstvo": Gospodinjstvo(
                oid,
                dogovorjena_map[oid],
                ima_pv=True,
                shema_samooskrbe=Shema.NOVA,
                vloga_souporaba=Vloga.ODDAJNIK,
                delez_souporabe=float(delez_souporabe),
            ),
            "paket": PAKETI[oddajnik_paket_id],
            "delitev": {pid: weight for pid in prejemnik_ids},
        }

    for pid in prejemnik_ids:
        udelezenci[pid] = {
            "gospodinjstvo": Gospodinjstvo(
                pid,
                dogovorjena_map[pid],
                vloga_souporaba=Vloga.PREJEMNIK,
            ),
            "paket": PAKETI[prejemnik_paket_id],
        }

    return udelezenci


def run_souporaba_period_scenario(
    household_data: Dict[str, pd.DataFrame],
    *,
    scenario_name: str,
    oddajnik_ids: Iterable[str],
    prejemnik_ids: Iterable[str],
    pv_scale_map: Optional[Dict[str, float]] = None,
    delez_souporabe: float = 0.4,
    cena_souporabe_eur_kwh: float = 0.05,
    organizer_service_id: str = "GENI_SOUPORABA",
    oddajnik_paket_id: str = "GENI_SAMO_DINAMICNI",
    prejemnik_paket_id: str = "GENI_DINAMICNI",
    contracted_power_kw: float = 5.0,
    contracted_power_map: Optional[Dict[str, Dict[int, float]]] = None,
    generation_column: str = "Energy_Generation",
    consumption_column: str = "Energy_Consumption",
    pricing_reference_year: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Souporaba settlement over the whole period, month by month.

    `obracun_souporabe` settles one calendar month at a time -- the monthly
    power charge, the OVE/SPTE contribution and the excess-power penalty all
    reset monthly, so a year cannot be settled in one call. This runs every
    (year, month) present in the index and sums the resulting `Racun` objects
    per household.

    Unlike `run_souporaba_monthly_scenario` this takes *many* senders.
    `obracun_souporabe` has always supported that; only the wrapper was
    restrictive.
    """
    oddajnik_ids = [str(i) for i in oddajnik_ids]
    prejemnik_ids = [str(i) for i in prejemnik_ids]
    all_ids = [*oddajnik_ids, *prejemnik_ids]

    missing = [i for i in all_ids if i not in household_data]
    if missing:
        raise ValueError(f"households not found in household_data: {missing}")
    if not oddajnik_ids:
        raise ValueError("souporaba needs at least one oddajnik.")
    if not prejemnik_ids:
        raise ValueError("souporaba needs at least one prejemnik.")

    pv_scale_map = pv_scale_map or {}
    contracted_power_map = contracted_power_map or {}
    dogovorjena_map = {
        hid: contracted_power_map.get(hid) or _dogovorjena_moc(contracted_power_kw)
        for hid in all_ids
    }

    base_idx = household_data[all_ids[0]].index
    interval_minutes = _step_minutes(household_data[all_ids[0]])
    service = STORITVE_SOUPORABE[organizer_service_id]
    pravila = (
        Pravila.za_leto(int(pricing_reference_year))
        if pricing_reference_year is not None
        else None
    )

    udelezenci = _souporaba_participants(
        oddajnik_ids,
        prejemnik_ids,
        dogovorjena_map=dogovorjena_map,
        delez_souporabe=delez_souporabe,
        oddajnik_paket_id=oddajnik_paket_id,
        prejemnik_paket_id=prejemnik_paket_id,
    )

    # Pull every household's series out once; .loc per timestamp per household
    # is what makes the naive version unusable at community scale.
    consumption = {
        hid: household_data[hid][consumption_column].to_numpy(dtype=float)
        for hid in all_ids
    }
    generation = {
        hid: household_data[hid][generation_column].to_numpy(dtype=float)
        * float(pv_scale_map.get(hid, 1.0))
        for hid in all_ids
    }
    smp_mwh = household_data[all_ids[0]]["SMP"].to_numpy(dtype=float) * 1000.0

    months = sorted({(int(ts.year), int(ts.month)) for ts in base_idx})
    positions = {ym: [] for ym in months}
    for pos, ts in enumerate(base_idx):
        positions[(int(ts.year), int(ts.month))].append(pos)

    totals: Dict[str, Dict[str, float]] = {}
    # VAT-inclusive charge per billing item, same shape as the interval path.
    component_totals: Dict[str, Dict[str, float]] = {}
    monthly_rows = []

    for year, month in months:
        idx_positions = positions[(year, month)]
        podatki = []
        for pos in idx_positions:
            ts = base_idx[pos]
            poraba = {hid: float(consumption[hid][pos]) for hid in all_ids}
            proizvodnja = {
                hid: float(generation[hid][pos])
                for hid in all_ids
                if generation[hid][pos] > 0
            }
            podatki.append(
                {
                    "utc_date": ts,
                    "interval_minutes": interval_minutes,
                    "market_price_mwh": float(smp_mwh[pos]),
                    "poraba": poraba,
                    "proizvodnja": proizvodnja,
                }
            )

        results = obracun_souporabe(
            udelezenci,
            podatki,
            year,
            month,
            storitev=service,
            pravila=pravila,
            cena_souporabe_eur_kwh=float(cena_souporabe_eur_kwh),
        )

        for hid, racun in results.items():
            diag = dict(getattr(racun, "diagnostika", {}) or {})
            acc = totals.setdefault(
                hid,
                {
                    "za_placilo_eur": 0.0,
                    "neto_eur": 0.0,
                    "ddv_eur": 0.0,
                    "bruto_eur": 0.0,
                    "dobropis_eur": 0.0,
                    "fiksni_del_eur": 0.0,
                    "spremenljivi_del_eur": 0.0,
                    "prevzeto_kwh": 0.0,
                    "oddano_kwh": 0.0,
                    "deljeno_kwh": 0.0,
                    "neizrabljeno_kwh": 0.0,
                    "lastna_raba_kwh": 0.0,
                },
            )
            acc["za_placilo_eur"] += float(racun.za_placilo)
            acc["neto_eur"] += float(racun.neto)
            acc["ddv_eur"] += float(racun.ddv)
            acc["bruto_eur"] += float(racun.bruto)
            acc["dobropis_eur"] += float(racun.dobropis_odkup)
            acc["fiksni_del_eur"] += float(racun.fiksni_del)
            acc["spremenljivi_del_eur"] += float(racun.spremenljivi_del)
            acc["prevzeto_kwh"] += float(racun.prevzeto_kwh)
            acc["oddano_kwh"] += float(diag.get("oddano_kwh", 0.0))
            acc["deljeno_kwh"] += float(diag.get("deljeno_kwh", 0.0))
            acc["neizrabljeno_kwh"] += float(diag.get("neizrabljena_souporaba_kwh", 0.0))
            acc["lastna_raba_kwh"] += float(diag.get("lastna_raba_kwh", 0.0))

            ddv_factor = 1.0 + float(racun.ddv_stopnja)
            comp = component_totals.setdefault(hid, {})
            for item, value in racun.postavke.items():
                comp[item] = comp.get(item, 0.0) + float(value) * ddv_factor

            monthly_rows.append(
                {
                    "scenario": scenario_name,
                    "household_id": hid,
                    "year": year,
                    "month": month,
                    "role": (
                        "oddajnik" if hid in set(oddajnik_ids) else "prejemnik"
                    ),
                    "za_placilo_eur": float(racun.za_placilo),
                    "prevzeto_kwh": float(racun.prevzeto_kwh),
                    "oddano_kwh": float(diag.get("oddano_kwh", 0.0)),
                    "deljeno_kwh": float(diag.get("deljeno_kwh", 0.0)),
                    "neizrabljeno_kwh": float(
                        diag.get("neizrabljena_souporaba_kwh", 0.0)
                    ),
                }
            )

        if progress:
            print(f"  souporaba {year}-{month:02d} settled", flush=True)

    oddajnik_set = set(oddajnik_ids)
    rows = []
    for hid, acc in totals.items():
        rows.append(
            {
                "scenario": scenario_name,
                "household_id": hid,
                "role": "oddajnik" if hid in oddajnik_set else "prejemnik",
                **acc,
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("household_id").reset_index(drop=True)

    group_row = {
        "scenario": scenario_name,
        "household_id": "GROUP",
        "role": None,
        **{
            col: float(summary_df[col].sum())
            for col in summary_df.columns
            if col not in ("scenario", "household_id", "role")
        },
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([group_row])], ignore_index=True)

    components_df = pd.DataFrame(
        [
            {
                "scenario": scenario_name,
                "household_id": hid,
                "component": item,
                "eur": value,
            }
            for hid, comp in sorted(component_totals.items())
            for item, value in sorted(comp.items())
        ]
    )

    return {
        "summary": summary_df,
        "components": components_df,
        "monthly": pd.DataFrame(monthly_rows),
    }
