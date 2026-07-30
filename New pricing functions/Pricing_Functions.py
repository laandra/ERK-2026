"""Pricing_Functions.py — unified single-user interval pricing dispatcher for RL/MILP.

Two pricing families are supported:

  SCHEME_AUS_BASE:
    Legacy Australian benchmark pricing (`Aus_Base`), kept unchanged as the
    original default behavior for existing RL pipelines.

  SCHEME_SI_DOBAVA / SCHEME_SI_SAMOOSKRBA:
    Real Slovenian household electricity billing, built on top of the
    `si_tarife`/`si_cas`/`si_paketi`/`si_obracun` modules:
      - `si_dobava`      — plain grid supply, no on-site production (PV).
      - `si_samooskrba`  — PV self-supply with intra-interval netting.
    Both return a full per-interval breakdown: a decision-independent fixed
    monthly charge (network power-block fee, OVE+SPTE levy, supplier monthly
    fee — prorated per interval) plus a decision-dependent variable charge
    split into its linear energy component (per-kWh, time-of-use/block
    priced) and its power/peak component (a ratchet excess-power charge —
    see `si_konica.py`).

Multi-user schemes (`si_skupnost`, community/peer-sharing "souporaba") are
explicitly NOT supported here — single user only. They will get their own
separate pricing function later.
"""
from __future__ import annotations

import calendar
import datetime
import functools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Keep sibling imports working even when this file is loaded via a root shim.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from si_cas import bloki_v_mesecu, casovni_blok, je_visja_sezona, v_lokalni_cas
from si_obracun import Pravila, dobava, samooskrba
from si_paketi import PAKETI, TipCene, TipOdkupa
from si_tarife import DDV, PRIVZETO_REFERENCNO_LETO, ima_tarifne_postavke, ove_spte_eur_kw
from si_konica import marginal_excess_charge_eur, reset_window_id, update_running_peak

# -----------------------------------------------------------------------------
# Public scheme names
# -----------------------------------------------------------------------------
SCHEME_AUS_BASE = "aus_base"
SCHEME_SI_DOBAVA = "si_dobava"
SCHEME_SI_SAMOOSKRBA = "si_samooskrba"

SUPPORTED_SCHEMES: Tuple[str, ...] = (
    SCHEME_AUS_BASE,
    SCHEME_SI_DOBAVA,
    SCHEME_SI_SAMOOSKRBA,
)

# Multi-user modes are intentionally unsupported in this dispatcher.
SKIPPED_MULTI_USER_SCHEMES: Tuple[str, ...] = (
    "si_skupnost",
    "si_obracun_skupnosti",
    "si_obracun_souporabe",
)


def list_pricing_schemes(include_skipped: bool = False) -> Tuple[str, ...]:
    if include_skipped:
        return SUPPORTED_SCHEMES + SKIPPED_MULTI_USER_SCHEMES
    return SUPPORTED_SCHEMES


# -----------------------------------------------------------------------------
# Legacy Australian pricing (unchanged behavior)
# -----------------------------------------------------------------------------
def Aus_Base(
    smp_market_price_kwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: float = 30,
) -> dict:
    """Legacy Australian interval pricing function kept as default behavior."""

    GST_RATE = 0.10
    DAYS_IN_MONTH = 30

    monthly_subscription_ex_gst = 20.00
    daily_supply_ex_gst = 1.09

    intervals_per_day = (24 * 60) / float(interval_minutes)
    intervals_per_month = intervals_per_day * DAYS_IN_MONTH

    constant_cost_ex_gst = (daily_supply_ex_gst / intervals_per_day) + (
        monthly_subscription_ex_gst / intervals_per_month
    )
    constant_cost_inc_gst = constant_cost_ex_gst * (1 + GST_RATE)

    #Converted from EUR to AUD using 0.615 conversion rate
    spot_price_kwh = smp_market_price_kwh / 0.615

    MLF = 0.995
    DLF = 1.045
    adjusted_spot_kwh = spot_price_kwh * MLF * DLF

    nem_time = utc_date + datetime.timedelta(hours=10)
    hour = nem_time.hour

    if 15 <= hour < 21:
        network_rate_kwh = 0.2360
    elif 10 <= hour < 15:
        network_rate_kwh = 0.0270
    else:
        network_rate_kwh = 0.0720

    env_market_rate_kwh = 0.0250

    if total_consumed_kwh >= 0:
        total_rate_kwh_ex_gst = adjusted_spot_kwh + network_rate_kwh + env_market_rate_kwh
        variable_cost_ex_gst = total_consumed_kwh * total_rate_kwh_ex_gst
        variable_cost_inc_gst = variable_cost_ex_gst * (1 + GST_RATE)
    else:
        variable_cost_ex_gst = total_consumed_kwh * adjusted_spot_kwh
        variable_cost_inc_gst = variable_cost_ex_gst

    return {
        "constant_price": round(constant_cost_inc_gst, 10),
        "variable_price": round(variable_cost_inc_gst, 10),
    }


# -----------------------------------------------------------------------------
# Small resolution helpers
# -----------------------------------------------------------------------------
def _resolve_meritve_15min(
    interval_minutes: float, meritve_15min: Optional[bool], warnings: List[str]
) -> bool:
    if meritve_15min is not None:
        return bool(meritve_15min)
    if float(interval_minutes) != 15.0:
        warnings.append(
            f"interval_minutes={interval_minutes} != 15; meritve_15min auto-set to "
            f"False (AKTIVNI 4-tariff pricing falls back to the flat ET-equivalent rate)."
        )
        return False
    return True


def _resolve_pravila(
    utc_date: datetime.datetime,
    pravila: Any,
    pricing_reference_year: Optional[int],
    warnings: List[str],
) -> Pravila:
    """Regulatory regime for one interval.

    Datasets used here (Ausgrid 2010-2013) carry timestamps for which no SI
    tariff act exists, so both branches fall back to the default reference
    year (2026) instead of raising -- see `si_obracun.Pravila.za_leto` /
    `Pravila.privzeta`.
    """
    if pravila is not None:
        return pravila
    if pricing_reference_year is not None:
        year = int(pricing_reference_year)
        if year < 2027 and year != PRIVZETO_REFERENCNO_LETO and not ima_tarifne_postavke(
            datetime.date(year, 1, 1)
        ):
            warnings.append(
                f"pricing_reference_year={year} has no published SI tariff rates; "
                f"pricing falls back to the {PRIVZETO_REFERENCNO_LETO} regime."
            )
        return Pravila.za_leto(year)
    data_date = v_lokalni_cas(utc_date).date()
    if not ima_tarifne_postavke(data_date):
        warnings.append(
            f"No SI tariff rates published for the data timestamp {data_date}; "
            f"pricing falls back to the {PRIVZETO_REFERENCNO_LETO} regime. Set "
            f"pricing_reference_year explicitly to silence this."
        )
    return Pravila.privzeta(data_date)


def _resolve_dogovorjena_moc(
    dogovorjena_moc: Optional[Union[float, Dict[int, float]]]
) -> Dict[int, float]:
    if dogovorjena_moc is None:
        return {b: 0.0 for b in range(1, 6)}
    if isinstance(dogovorjena_moc, (int, float)):
        return {b: float(dogovorjena_moc) for b in range(1, 6)}
    return {b: float(dogovorjena_moc.get(b, 0.0)) for b in range(1, 6)}


def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _localized(utc_date: datetime.datetime, pravila: Pravila) -> datetime.datetime:
    lok = v_lokalni_cas(utc_date)
    if pravila.preslikaj_v_leto is not None:
        try:
            lok = lok.replace(year=pravila.preslikaj_v_leto)
        except ValueError:  # 29. februar
            lok = lok.replace(year=pravila.preslikaj_v_leto, day=28)
    return lok


# -----------------------------------------------------------------------------
# Supplier package selection
# -----------------------------------------------------------------------------
def _parse_tip_cene(value: Any) -> Optional[TipCene]:
    if value is None:
        return None
    if isinstance(value, TipCene):
        return value
    return TipCene(str(value))


def _parse_tip_odkupa(value: Any) -> Optional[TipOdkupa]:
    if value is None:
        return None
    if isinstance(value, TipOdkupa):
        return value
    return TipOdkupa(str(value))


@functools.lru_cache(maxsize=256)
def _select_si_package_cached(
    scheme: str,
    tip_cene_val: Optional[str],
    tip_odkupa_val: Optional[str],
    provider: Optional[str],
) -> str:
    want_pv = scheme == SCHEME_SI_SAMOOSKRBA
    candidates = []
    for pid, p in PAKETI.items():
        if want_pv and not p.dovoljuje_pv:
            continue
        if not want_pv and p.zahteva_pv:
            continue
        if tip_cene_val is not None and p.tip_cene.value != tip_cene_val:
            continue
        if want_pv and tip_odkupa_val is not None and p.tip_odkupa.value != tip_odkupa_val:
            continue
        if provider is not None and p.dobavitelj.lower() != provider.lower():
            continue
        candidates.append(pid)

    if not candidates:
        raise ValueError(
            f"No SI package matches scheme={scheme!r}, pricing_mode={tip_cene_val!r}, "
            f"buyback_mode={tip_odkupa_val!r}, provider={provider!r}."
        )
    # Deterministic pick: most recently introduced offer, cheapest fixed fee as tiebreak.
    candidates.sort(
        key=lambda pid: (-PAKETI[pid].velja_od.toordinal(), PAKETI[pid].mesecno_nadomestilo, pid)
    )
    return candidates[0]


def _select_si_package(
    scheme: str,
    *,
    paket_id: Optional[str],
    pricing_mode: Optional[Any],
    buyback_mode: Optional[Any],
    provider: Optional[str],
    warnings: List[str],
) -> str:
    if paket_id is not None:
        if paket_id not in PAKETI:
            raise ValueError(f"Unknown paket_id={paket_id!r}.")
        return paket_id
    tip_cene = _parse_tip_cene(pricing_mode)
    tip_odkupa = _parse_tip_odkupa(buyback_mode)
    return _select_si_package_cached(
        scheme,
        tip_cene.value if tip_cene is not None else None,
        tip_odkupa.value if tip_odkupa is not None else None,
        provider,
    )


def _infer_consumed_produced(
    total_consumed_kwh: float, total_produced_kwh: Optional[float]
) -> Tuple[float, float]:
    if total_produced_kwh is not None:
        return float(total_consumed_kwh), float(total_produced_kwh)
    net = float(total_consumed_kwh)
    if net >= 0:
        return net, 0.0
    return 0.0, -net


# -----------------------------------------------------------------------------
# Fixed (decision-independent) monthly charge, prorated per interval
# -----------------------------------------------------------------------------
def _prorated_fixed_charge_eur(
    utc_date: datetime.datetime,
    interval_minutes: float,
    *,
    pravila: Pravila,
    paket,
    dogovorjena_moc: Dict[int, float],
    apply_ddv: bool,
    eko_racun: bool,
) -> float:
    lok = _localized(utc_date, pravila)
    leto, mesec = lok.year, lok.month

    om = pravila.omreznina
    vs = je_visja_sezona(datetime.date(leto, mesec, 1))
    bloki = bloki_v_mesecu(leto, mesec, pravila.razpored)

    moc = sum(dogovorjena_moc.get(b, 0.0) * om.postavka_moc(b, vs) for b in bloki)
    ref_blok = min(bloki) if bloki else 2
    ove_spte = dogovorjena_moc.get(ref_blok, 0.0) * ove_spte_eur_kw(pravila.dajatve_datum)
    nadomestilo = paket.nadomestilo(eko_racun)

    fixed_ex_ddv = moc + ove_spte + nadomestilo
    fixed = fixed_ex_ddv * (1.0 + DDV) if apply_ddv else fixed_ex_ddv

    days = _days_in_month(leto, mesec)
    intervals_in_month = max(1.0, (days * 24.0 * 60.0) / float(interval_minutes))
    return fixed / intervals_in_month


# -----------------------------------------------------------------------------
# Ratchet peak/excess-power charge
# -----------------------------------------------------------------------------
def _apply_peak_ratchet(
    raw: Dict[str, Any],
    pravila: Pravila,
    dogovorjena_moc: Optional[Union[float, Dict[int, float]]],
    prev_peak_kw: Optional[Dict[int, float]],
) -> Tuple[float, Dict[int, float], Optional[int]]:
    if dogovorjena_moc is None:
        return 0.0, dict(prev_peak_kw or {}), None

    blok = raw.get("blok")
    moc_kw = float(raw.get("moc_kw", 0.0))
    resolved = _resolve_dogovorjena_moc(dogovorjena_moc)
    prev_kw = float((prev_peak_kw or {}).get(blok, 0.0))
    new_peak_kw = update_running_peak(prev_peak_kw or {}, blok, moc_kw)

    vs = je_visja_sezona(raw["lokalni_cas"].date())
    rate = pravila.omreznina.postavka_moc(blok, vs)
    charge = marginal_excess_charge_eur(
        prev_kw, new_peak_kw[blok], resolved.get(blok, 0.0),
        rate, pravila.omreznina.faktor_presezne_moci,
    )
    return charge, new_peak_kw, blok


# -----------------------------------------------------------------------------
# Normalization — unified result shape for every scheme
# -----------------------------------------------------------------------------
def _normalize_si_result(
    raw: Dict[str, Any],
    scheme: str,
    *,
    apply_ddv: bool,
    fixed_component_eur: float = 0.0,
    power_component_eur: float = 0.0,
    new_peak_kw: Optional[Dict[int, float]] = None,
    peak_blok: Optional[int] = None,
) -> Dict[str, Any]:
    postavke = raw.get("obdavcljive_postavke", {})
    taxable = float(sum(float(v) for v in postavke.values()))
    dobropis = float(raw.get("dobropis_odkup", 0.0))

    if apply_ddv:
        energy_component_eur = taxable * (1.0 + float(DDV)) - dobropis
    else:
        energy_component_eur = taxable - dobropis

    variable_total = energy_component_eur + float(power_component_eur)

    return {
        "scheme": scheme,
        "currency": "EUR",
        "constant_price_aud": round(float(fixed_component_eur), 10),
        "variable_price_aud": round(variable_total, 10),
        "energy_component_eur": round(energy_component_eur, 10),
        "power_component_eur": round(float(power_component_eur), 10),
        "fixed_monthly_charge_eur": round(float(fixed_component_eur), 10),
        "taxable_interval_eur": round(taxable, 10),
        "dobropis_odkup_eur": round(dobropis, 10),
        "ddv_included": bool(apply_ddv),
        "new_peak_kw": dict(new_peak_kw or {}),
        "peak_blok": peak_blok,
    }


def _normalize_aus_result(
    raw: Dict[str, Any], scheme: str, *, prev_peak_kw: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    constant_price = float(raw.get("constant_price", 0.0))
    variable_price = float(raw.get("variable_price", 0.0))
    return {
        "scheme": scheme,
        "currency": "AUD",
        "constant_price_aud": round(constant_price, 10),
        "variable_price_aud": round(variable_price, 10),
        "energy_component_eur": round(variable_price, 10),
        "power_component_eur": 0.0,
        "fixed_monthly_charge_eur": round(constant_price, 10),
        "taxable_interval_eur": 0.0,
        "dobropis_odkup_eur": 0.0,
        "ddv_included": True,
        "new_peak_kw": dict(prev_peak_kw or {}),
        "peak_blok": None,
    }


# -----------------------------------------------------------------------------
# Per-scheme resolvers
# -----------------------------------------------------------------------------
def _resolve_si_dobava(
    smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes, *,
    paket_id, pricing_mode, buyback_mode, provider, pravila, meritve_15min,
    apply_ddv, total_produced_kwh, dogovorjena_moc, prev_peak_kw, eko_racun, warnings,
):
    resolved_paket_id = _select_si_package(
        SCHEME_SI_DOBAVA, paket_id=paket_id, pricing_mode=pricing_mode,
        buyback_mode=buyback_mode, provider=provider, warnings=warnings,
    )
    paket = PAKETI[resolved_paket_id]
    raw = dobava(
        smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes,
        paket=paket, pravila=pravila, meritve_15min=meritve_15min,
    )

    fixed_component_eur = _prorated_fixed_charge_eur(
        utc_date, interval_minutes, pravila=pravila, paket=paket,
        dogovorjena_moc=_resolve_dogovorjena_moc(dogovorjena_moc),
        apply_ddv=apply_ddv, eko_racun=eko_racun,
    )
    power_component_eur, new_peak_kw, peak_blok = _apply_peak_ratchet(
        raw, pravila, dogovorjena_moc, prev_peak_kw,
    )

    normalized = _normalize_si_result(
        raw, SCHEME_SI_DOBAVA, apply_ddv=apply_ddv,
        fixed_component_eur=fixed_component_eur,
        power_component_eur=power_component_eur,
        new_peak_kw=new_peak_kw, peak_blok=peak_blok,
    )
    normalized["paket_id"] = resolved_paket_id
    return normalized, raw


def _resolve_si_samooskrba(
    smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes, *,
    paket_id, pricing_mode, buyback_mode, provider, pravila, meritve_15min,
    apply_ddv, total_produced_kwh, dogovorjena_moc, prev_peak_kw, eko_racun, warnings,
):
    consumed, produced = _infer_consumed_produced(total_consumed_kwh, total_produced_kwh)
    resolved_paket_id = _select_si_package(
        SCHEME_SI_SAMOOSKRBA, paket_id=paket_id, pricing_mode=pricing_mode,
        buyback_mode=buyback_mode, provider=provider, warnings=warnings,
    )
    paket = PAKETI[resolved_paket_id]
    raw = samooskrba(
        smp_market_price_mwh, consumed, utc_date, interval_minutes,
        total_produced_kwh=produced, paket=paket, pravila=pravila,
        meritve_15min=meritve_15min,
    )

    fixed_component_eur = _prorated_fixed_charge_eur(
        utc_date, interval_minutes, pravila=pravila, paket=paket,
        dogovorjena_moc=_resolve_dogovorjena_moc(dogovorjena_moc),
        apply_ddv=apply_ddv, eko_racun=eko_racun,
    )
    power_component_eur, new_peak_kw, peak_blok = _apply_peak_ratchet(
        raw, pravila, dogovorjena_moc, prev_peak_kw,
    )

    normalized = _normalize_si_result(
        raw, SCHEME_SI_SAMOOSKRBA, apply_ddv=apply_ddv,
        fixed_component_eur=fixed_component_eur,
        power_component_eur=power_component_eur,
        new_peak_kw=new_peak_kw, peak_blok=peak_blok,
    )
    normalized["paket_id"] = resolved_paket_id
    return normalized, raw


def _resolve_single_scheme(
    scheme, smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes, *,
    paket_id, pricing_mode, buyback_mode, provider, pravila, meritve_15min,
    apply_ddv, total_produced_kwh, dogovorjena_moc, prev_peak_kw, eko_racun, warnings,
):
    if scheme == SCHEME_AUS_BASE:
        raw = Aus_Base(smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes)
        return _normalize_aus_result(raw, scheme, prev_peak_kw=prev_peak_kw), raw
    if scheme == SCHEME_SI_DOBAVA:
        return _resolve_si_dobava(
            smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes,
            paket_id=paket_id, pricing_mode=pricing_mode, buyback_mode=buyback_mode,
            provider=provider, pravila=pravila, meritve_15min=meritve_15min,
            apply_ddv=apply_ddv, total_produced_kwh=total_produced_kwh,
            dogovorjena_moc=dogovorjena_moc, prev_peak_kw=prev_peak_kw,
            eko_racun=eko_racun, warnings=warnings,
        )
    if scheme == SCHEME_SI_SAMOOSKRBA:
        return _resolve_si_samooskrba(
            smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes,
            paket_id=paket_id, pricing_mode=pricing_mode, buyback_mode=buyback_mode,
            provider=provider, pravila=pravila, meritve_15min=meritve_15min,
            apply_ddv=apply_ddv, total_produced_kwh=total_produced_kwh,
            dogovorjena_moc=dogovorjena_moc, prev_peak_kw=prev_peak_kw,
            eko_racun=eko_racun, warnings=warnings,
        )
    if scheme in SKIPPED_MULTI_USER_SCHEMES:
        raise ValueError(
            f"scheme={scheme!r} is a multi-user scheme and is not supported by this "
            f"single-user dispatcher. Multi-user pricing will get its own separate "
            f"pricing function."
        )
    raise ValueError(f"Unknown scheme={scheme!r}. Supported: {SUPPORTED_SCHEMES}")


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------
def calculate_interval_price(
    smp_market_price_kwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: float = 30,
    *,
    scheme: str = SCHEME_AUS_BASE,
    paket_id: Optional[str] = None,
    pricing_mode: Optional[Any] = None,
    buyback_mode: Optional[Any] = None,
    provider: Optional[str] = None,
    pravila: Any = None,
    pricing_reference_year: Optional[int] = None,
    meritve_15min: Optional[bool] = None,
    apply_ddv: bool = True,
    total_produced_kwh: Optional[float] = None,
    dogovorjena_moc: Optional[Union[float, Dict[int, float]]] = None,
    prev_peak_kw: Optional[Dict[int, float]] = None,
    eko_racun: bool = True,
    compare_all: bool = False,
    include_raw: bool = False,
) -> Dict[str, Any]:
    """Unified single-user interval pricing dispatcher for RL/MILP.

    Key features:
    - Select an explicit SI package with `paket_id`, or by pricing model via
      `pricing_mode` (TipCene) / `buyback_mode` (TipOdkupa), optionally
      constrained by `provider`.
    - Non-15-minute intervals are supported: if `meritve_15min` is omitted
      and `interval_minutes != 15`, it's auto-set to False (a warning is
      returned in `warnings`).
    - `pricing_reference_year` forces SI regulatory rules from a chosen year
      (e.g. 2026 or 2027) even when input timestamps are older.
    - Pass `dogovorjena_moc` (contracted power per tariff block, kW — a
      float applies to all blocks) and `prev_peak_kw` (this caller's running
      peak state from the previous call, or None/empty at the start) to
      enable the ratchet excess-power charge; omit both to get pure per-kWh
      linear pricing (`power_component_eur` stays 0). The updated running
      peak is returned as `new_peak_kw` — the caller is responsible for
      storing and re-passing it on the next call (and for resetting it,
      e.g. at a configured reset-window boundary).
    - Returns `constant_price_aud` (prorated, decision-independent fixed
      monthly charge) and `variable_price_aud` (decision-dependent; equals
      `energy_component_eur + power_component_eur`, both also returned
      individually for isolated use).
    """
    smp_market_price_mwh = float(smp_market_price_kwh) * 1000.0
    warnings: List[str] = []
    resolved_pravila = _resolve_pravila(utc_date, pravila, pricing_reference_year, warnings)
    resolved_meritve = _resolve_meritve_15min(interval_minutes, meritve_15min, warnings)

    normalized, raw = _resolve_single_scheme(
        scheme, smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes,
        paket_id=paket_id, pricing_mode=pricing_mode, buyback_mode=buyback_mode,
        provider=provider, pravila=resolved_pravila, meritve_15min=resolved_meritve,
        apply_ddv=bool(apply_ddv), total_produced_kwh=total_produced_kwh,
        dogovorjena_moc=dogovorjena_moc, prev_peak_kw=prev_peak_kw,
        eko_racun=bool(eko_racun), warnings=warnings,
    )

    result: Dict[str, Any] = dict(normalized)
    result["comparison_enabled"] = bool(compare_all)
    result.setdefault("warnings", list(warnings))
    if include_raw:
        result["raw_result"] = raw

    if compare_all:
        comparisons: Dict[str, Any] = {}
        for other_scheme in SUPPORTED_SCHEMES:
            if other_scheme == scheme:
                continue
            try:
                other_pravila = _resolve_pravila(utc_date, pravila, pricing_reference_year, [])
                other_norm, _ = _resolve_single_scheme(
                    other_scheme, smp_market_price_mwh, total_consumed_kwh, utc_date,
                    interval_minutes, paket_id=None, pricing_mode=pricing_mode,
                    buyback_mode=buyback_mode, provider=provider, pravila=other_pravila,
                    meritve_15min=resolved_meritve, apply_ddv=bool(apply_ddv),
                    total_produced_kwh=total_produced_kwh, dogovorjena_moc=dogovorjena_moc,
                    prev_peak_kw=prev_peak_kw, eko_racun=bool(eko_racun), warnings=[],
                )
                comparisons[other_scheme] = other_norm
            except Exception as exc:  # noqa: BLE001 - comparison is best-effort/informational
                comparisons[other_scheme] = {"error": str(exc)}
        result["comparisons"] = comparisons

    return result


# -----------------------------------------------------------------------------
# Public helpers for Environment.py / the MILP notebook (avoid needing to
# import si_cas/si_konica directly — this module stays the single integration
# surface for RL/MILP).
# -----------------------------------------------------------------------------
def resolve_block_for_datetime(
    utc_date: datetime.datetime,
    *,
    pravila: Any = None,
    pricing_reference_year: Optional[int] = None,
) -> int:
    """Tariff time-block (1-5) for a single timestamp, using the same
    pravila-resolution rules as `calculate_interval_price`."""
    warnings: List[str] = []
    resolved = _resolve_pravila(utc_date, pravila, pricing_reference_year, warnings)
    lok = _localized(utc_date, resolved)
    return casovni_blok(lok, resolved.razpored)


def resolve_reset_window_id(
    utc_date: datetime.datetime, peak_reset_months: Optional[int]
) -> int:
    """Monotonic reset-window id for the ratchet peak tracker (see si_konica)."""
    lok = v_lokalni_cas(utc_date)
    return reset_window_id(lok.year, lok.month, peak_reset_months)


def compute_prorated_fixed_charge_eur(
    utc_date: datetime.datetime,
    interval_minutes: float,
    *,
    scheme: str,
    dogovorjena_moc: Optional[Union[float, Dict[int, float]]],
    paket_id: Optional[str] = None,
    pricing_mode: Optional[Any] = None,
    buyback_mode: Optional[Any] = None,
    provider: Optional[str] = None,
    pravila: Any = None,
    pricing_reference_year: Optional[int] = None,
    meritve_15min: Optional[bool] = None,
    apply_ddv: bool = True,
    eko_racun: bool = True,
) -> float:
    """Fixed-monthly-charge-only helper for the MILP benchmark: resolves the
    SI package exactly like `calculate_interval_price` does, but returns
    ONLY the prorated fixed component (no per-kWh energy, no peak/ratchet
    term), so the MILP can add a clean per-interval constant to its
    objective without a dummy `total_consumed_kwh` call. Returns 0.0 for
    `scheme == aus_base` (no fixed-monthly concept there)."""
    if scheme not in (SCHEME_SI_DOBAVA, SCHEME_SI_SAMOOSKRBA):
        return 0.0
    warnings: List[str] = []
    resolved_pravila = _resolve_pravila(utc_date, pravila, pricing_reference_year, warnings)
    resolved_paket_id = _select_si_package(
        scheme, paket_id=paket_id, pricing_mode=pricing_mode,
        buyback_mode=buyback_mode, provider=provider, warnings=warnings,
    )
    return _prorated_fixed_charge_eur(
        utc_date, interval_minutes, pravila=resolved_pravila, paket=PAKETI[resolved_paket_id],
        dogovorjena_moc=_resolve_dogovorjena_moc(dogovorjena_moc),
        apply_ddv=bool(apply_ddv), eko_racun=bool(eko_racun),
    )
