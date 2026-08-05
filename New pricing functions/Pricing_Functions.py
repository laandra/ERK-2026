"""Pricing_Functions.py — unified single-user interval pricing dispatcher for RL/MILP.

Two schemes are supported, both modelling real Slovenian household electricity
billing on top of the `si_tarife`/`si_cas`/`si_paketi`/`si_obracun` modules:

  - `si_dobava`      — plain grid supply, no on-site production (PV).
  - `si_samooskrba`  — PV self-supply with intra-interval netting.

Both return a full per-interval breakdown: a decision-independent fixed
monthly charge (network power-block fee, OVE+SPTE levy, supplier monthly fee —
prorated per interval) plus a decision-dependent variable charge split into its
linear energy component (per-kWh, time-of-use/block priced) and its power/peak
component (a ratchet excess-power charge — see `si_konica.py`).

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
    # Keep sibling imports resolvable without shadowing the root shim module.
    sys.path.append(str(_THIS_DIR))

from si_cas import bloki_v_mesecu, casovni_blok, je_visja_sezona, v_lokalni_cas
from si_obracun import Pravila, dobava, samooskrba
from si_paketi import PAKETI, TipCene, TipOdkupa
from si_tarife import DDV, PRIVZETO_REFERENCNO_LETO, ima_tarifne_postavke, ove_spte_eur_kw
from si_konica import marginal_excess_charge_eur, reset_window_id, update_running_peak

# -----------------------------------------------------------------------------
# Public scheme names
# -----------------------------------------------------------------------------
SCHEME_SI_DOBAVA = "si_dobava"
SCHEME_SI_SAMOOSKRBA = "si_samooskrba"

SUPPORTED_SCHEMES: Tuple[str, ...] = (
    SCHEME_SI_DOBAVA,
    SCHEME_SI_SAMOOSKRBA,
)

# Multi-user modes are intentionally unsupported in this dispatcher.
SKIPPED_MULTI_USER_SCHEMES: Tuple[str, ...] = (
    "si_skupnost",
    "si_obracun_skupnosti",
    "si_obracun_souporabe",
)


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
def _prorated_fixed_breakdown_eur(
    utc_date: datetime.datetime,
    interval_minutes: float,
    *,
    pravila: Pravila,
    paket,
    dogovorjena_moc: Dict[int, float],
    apply_ddv: bool,
    eko_racun: bool,
) -> Dict[str, float]:
    """The fixed monthly charge, prorated per interval and split by recipient.

    The three keys are the names `si_obracun.FIKSNE_POSTAVKE` uses, so a bill
    settled interval by interval here and one settled by `si_obracun` can be
    broken down into the same categories (network / levy / supplier).
    """
    lok = _localized(utc_date, pravila)
    leto, mesec = lok.year, lok.month

    om = pravila.omreznina
    vs = je_visja_sezona(datetime.date(leto, mesec, 1))
    bloki = bloki_v_mesecu(leto, mesec, pravila.razpored)

    moc = sum(dogovorjena_moc.get(b, 0.0) * om.postavka_moc(b, vs) for b in bloki)
    ref_blok = min(bloki) if bloki else 2
    ove_spte = dogovorjena_moc.get(ref_blok, 0.0) * ove_spte_eur_kw(pravila.dajatve_datum)
    nadomestilo = paket.nadomestilo(eko_racun)

    days = _days_in_month(leto, mesec)
    intervals_in_month = max(1.0, (days * 24.0 * 60.0) / float(interval_minutes))
    scale = (1.0 + DDV if apply_ddv else 1.0) / intervals_in_month
    return {
        "omreznina_moc": moc * scale,
        "prispevek_ove_spte": ove_spte * scale,
        "mesecno_nadomestilo": nadomestilo * scale,
    }


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
    return sum(
        _prorated_fixed_breakdown_eur(
            utc_date, interval_minutes, pravila=pravila, paket=paket,
            dogovorjena_moc=dogovorjena_moc, apply_ddv=apply_ddv,
            eko_racun=eko_racun,
        ).values()
    )


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
    fixed_breakdown_eur: Optional[Dict[str, float]] = None,
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

    # One VAT-inclusive line per billing item, so a caller can add up a bill by
    # who is paid rather than by decision-dependence. The credit stays out of
    # it -- it is not a charge -- so the items sum to
    # `energy_component + power_component + fixed_component + dobropis`.
    ddv_factor = 1.0 + float(DDV) if apply_ddv else 1.0
    items = {k: float(v) * ddv_factor for k, v in postavke.items()}
    for k, v in (fixed_breakdown_eur or {}).items():
        items[k] = items.get(k, 0.0) + float(v)
    if power_component_eur:
        items["omreznina_presezna_moc"] = (
            items.get("omreznina_presezna_moc", 0.0) + float(power_component_eur)
        )

    return {
        "postavke_eur": items,
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

    fixed_breakdown_eur = _prorated_fixed_breakdown_eur(
        utc_date, interval_minutes, pravila=pravila, paket=paket,
        dogovorjena_moc=_resolve_dogovorjena_moc(dogovorjena_moc),
        apply_ddv=apply_ddv, eko_racun=eko_racun,
    )
    fixed_component_eur = sum(fixed_breakdown_eur.values())
    power_component_eur, new_peak_kw, peak_blok = _apply_peak_ratchet(
        raw, pravila, dogovorjena_moc, prev_peak_kw,
    )

    normalized = _normalize_si_result(
        raw, SCHEME_SI_DOBAVA, apply_ddv=apply_ddv,
        fixed_component_eur=fixed_component_eur,
        power_component_eur=power_component_eur,
        fixed_breakdown_eur=fixed_breakdown_eur,
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

    fixed_breakdown_eur = _prorated_fixed_breakdown_eur(
        utc_date, interval_minutes, pravila=pravila, paket=paket,
        dogovorjena_moc=_resolve_dogovorjena_moc(dogovorjena_moc),
        apply_ddv=apply_ddv, eko_racun=eko_racun,
    )
    fixed_component_eur = sum(fixed_breakdown_eur.values())
    power_component_eur, new_peak_kw, peak_blok = _apply_peak_ratchet(
        raw, pravila, dogovorjena_moc, prev_peak_kw,
    )

    normalized = _normalize_si_result(
        raw, SCHEME_SI_SAMOOSKRBA, apply_ddv=apply_ddv,
        fixed_component_eur=fixed_component_eur,
        power_component_eur=power_component_eur,
        fixed_breakdown_eur=fixed_breakdown_eur,
        new_peak_kw=new_peak_kw, peak_blok=peak_blok,
    )
    normalized["paket_id"] = resolved_paket_id
    return normalized, raw


def _resolve_single_scheme(
    scheme, smp_market_price_mwh, total_consumed_kwh, utc_date, interval_minutes, *,
    paket_id, pricing_mode, buyback_mode, provider, pravila, meritve_15min,
    apply_ddv, total_produced_kwh, dogovorjena_moc, prev_peak_kw, eko_racun, warnings,
):
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
    scheme: str = SCHEME_SI_SAMOOSKRBA,
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
    result.setdefault("warnings", list(warnings))
    if include_raw:
        result["raw_result"] = raw

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
    objective without a dummy `total_consumed_kwh` call."""
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unknown scheme={scheme!r}. Supported: {SUPPORTED_SCHEMES}")
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


# -----------------------------------------------------------------------------
# Invoice API re-exports
# -----------------------------------------------------------------------------
# Compatibility: if this implementation file is imported directly as
# `Pricing_Functions` (for example due to path ordering in notebooks), expose
# the same invoice symbols as the root shim.
from si_invoice import (  # noqa: E402
    InvoiceBuilder,
    aggregate_household_invoices,
    aggregate_line_items,
    build_invoice_household,
    racun_to_line_items,
    write_rows_csv,
)
