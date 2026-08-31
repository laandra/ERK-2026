"""si_invoice.py — turns per-interval si_obracun raw results into human-readable,
line-item electricity invoices (monthly and whole-period), CSV-exportable.

Bridges `Pricing_Functions.calculate_interval_price(..., include_raw=True)` output
into calendar-month bills using `si_obracun.MesecniObracun`/`Racun` for all the
regulatory math (network power/energy fees, OVE+SPTE, URE, trošarina, operater
trga, monthly nadomestilo, sqrt-based excess-power penalty, DDV, dobropis
netting) — this module does not duplicate any of that math, it only adds the
per-tariff-group/per-block bookkeeping needed to flatten a `Racun` into the
line-item layout of a real Slovenian bill, and drives month-boundary
accumulation across a chronological run.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from si_obracun import MesecniObracun, Pravila, Racun  # noqa: E402
from si_paketi import PAKETI, Gospodinjstvo, Paket, Shema, TipCene  # noqa: E402


def resolve_invoice_pravila(pricing_reference_year: Optional[int]) -> Optional[Pravila]:
    """Mirrors Pricing_Functions._resolve_pravila's reference-year handling.

    MesecniObracun's month-end fixed-charge calc (_fiksne, for the network
    power fee/OVE+SPTE/excess-power ratchet) resolves its own Pravila from
    the literal calendar month when none is passed in -- independently of
    whatever regime calculate_interval_price used per interval. Without this,
    a historical dataset (e.g. 2012) prices each interval correctly under the
    pricing_reference_year override, but month-end finalization still tries
    to look up tariffs for 2012 and raises (only 2025/2026 tariff tables are
    loaded in si_tarife.py). Returns None (meaning: fall back to the literal
    calendar date, per-accumulator) when no reference year is set.
    """
    if pricing_reference_year is None:
        return None
    year = int(pricing_reference_year)
    if year >= 2027:
        return Pravila.od_2027()
    if year == 2026:
        return Pravila.od_2026()
    return Pravila.ob_datumu(dt.date(year, 1, 1))

DDV_ODSTOTEK = 22  # display label only; the actual rate is si_tarife.DDV


# ---------------------------------------------------------------------------
# Household construction
# ---------------------------------------------------------------------------
def build_invoice_household(
    *,
    dogovorjena_moc: Dict[int, float],
    pricing_scheme: str,
    eko_racun: bool = True,
    meritve_15min: bool = True,
    ime: str = "rl_gospodinjstvo",
) -> Gospodinjstvo:
    """Household config for MesecniObracun, matching the same ima_pv/scheme
    logic the calculate_interval_price dispatcher already uses internally."""
    ima_pv = pricing_scheme == "si_samooskrba"
    return Gospodinjstvo(
        ime=ime,
        dogovorjena_moc=dict(dogovorjena_moc),
        ima_pv=ima_pv,
        shema_samooskrbe=Shema.NOVA if ima_pv else Shema.BREZ,
        meritve_15min=meritve_15min,
        eko_racun=eko_racun,
    )


# ---------------------------------------------------------------------------
# Paket-aware energy-purchase grouping
# ---------------------------------------------------------------------------
_AKTIVNI_LABELS = {
    "soncna_ns": "Električna energija – Sončna NS",
    "soncna_vs": "Električna energija – Sončna VS",
    "osnovna": "Električna energija – Osnovna",
    "konicna": "Električna energija – Konična",
}


def _resolve_energy_group(paket: Paket, raw: Dict[str, Any]) -> str:
    """Returns the display label used to group this interval's energy-purchase
    cost, chosen according to the resolved package's pricing structure."""
    if paket.tip_cene is TipCene.TARIFNI:
        if paket.vt or paket.mt:
            return "Električna energija VT" if raw.get("vt") else "Električna energija MT"
        return "Električna energija ET"
    if paket.tip_cene is TipCene.AKTIVNI:
        return _AKTIVNI_LABELS.get(raw.get("tarifa"), "Električna energija")
    # DINAMICNI (or anything else without a discrete tariff bucket) falls back
    # to per-block grouping, matching the network-fee section below.
    return f"Električna energija – blok {raw.get('blok')}"


# ---------------------------------------------------------------------------
# Per-month accumulation
# ---------------------------------------------------------------------------
class InvoiceAccumulator:
    """Wraps MesecniObracun for one calendar month (or partial trailing
    period), additionally tracking the per-block network-energy breakdown and
    the paket-aware energy-purchase-group breakdown that MesecniObracun
    doesn't expose on its own, plus the actual first/last interval date seen
    (so a partial period reports its real date range, not the calendar
    month's)."""

    def __init__(self, leto, mesec, gospodinjstvo, paket, pravila=None, **kwargs):
        self._mo = MesecniObracun(leto, mesec, gospodinjstvo, paket, pravila, **kwargs)
        self._paket = paket
        self._gospodinjstvo = gospodinjstvo
        self._po_blokih_omreznina: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._energija_po_skupini: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._min_date = None
        self._max_date = None

    def dodaj(self, interval: Dict[str, Any]) -> None:
        blok = interval["blok"]
        post = interval.get("obdavcljive_postavke", {})
        prevzeto = interval.get("prevzeto_kwh", 0.0)

        b = self._po_blokih_omreznina[blok]
        b["omreznina_energija_eur"] += post.get("omreznina_energija", 0.0)
        b["prevzeto_kwh"] += prevzeto

        group = _resolve_energy_group(self._paket, interval)
        g = self._energija_po_skupini[group]
        g["energija_eur"] += post.get("energija", 0.0)
        g["kolicina"] += prevzeto
        g["enota"] = "kWh"

        lok = interval.get("lokalni_cas")
        if lok is not None:
            d = lok.date()
            if self._min_date is None or d < self._min_date:
                self._min_date = d
            if self._max_date is None or d > self._max_date:
                self._max_date = d

        self._mo.dodaj(interval)

    def zakljuci(self):
        racun = self._mo.zakljuci()
        return (
            racun,
            {b: dict(v) for b, v in self._po_blokih_omreznina.items()},
            {k: dict(v) for k, v in self._energija_po_skupini.items()},
            self._min_date,
            self._max_date,
            self._paket,
            self._gospodinjstvo,
        )


# ---------------------------------------------------------------------------
# Flattening a Racun into real-bill-style line items
# ---------------------------------------------------------------------------
def _row(
    obdobje: str,
    produkt: str,
    kategorija: str,
    *,
    kolicina: Optional[float] = None,
    enota: Optional[str] = None,
    cena_eur_em: Optional[float] = None,
    znesek_eur_brez_ddv: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "obdobje": obdobje,
        "produkt": produkt,
        "kolicina": round(kolicina, 3) if kolicina is not None else None,
        "enota": enota,
        "cena_eur_em": round(cena_eur_em, 6) if cena_eur_em is not None else None,
        "znesek_eur_brez_ddv": (
            round(znesek_eur_brez_ddv, 5) if znesek_eur_brez_ddv is not None else None
        ),
        "kategorija": kategorija,
    }


def _format_obdobje(min_date, max_date) -> str:
    if min_date is None or max_date is None:
        return ""
    return f"{min_date:%d.%m.%Y}–{max_date:%d.%m.%Y}"


def racun_to_line_items(
    racun: Racun,
    po_blokih_omreznina: Dict[int, Dict[str, float]],
    energija_po_skupini: Dict[str, Dict[str, float]],
    paket: Paket,
    gospodinjstvo: Gospodinjstvo,
    min_date,
    max_date,
) -> List[Dict[str, Any]]:
    obdobje = _format_obdobje(min_date, max_date)
    rows: List[Dict[str, Any]] = []

    # 0. Title row: which plan this bill is for.
    rows.append(_row(obdobje, racun.paket, "naslov"))

    # 1. Energy purchase, grouped per the paket's own pricing structure.
    skupaj_energija = 0.0
    for label in sorted(energija_po_skupini):
        vals = energija_po_skupini[label]
        znesek = vals.get("energija_eur", 0.0)
        kolicina = vals.get("kolicina", 0.0)
        cena = znesek / kolicina if kolicina else 0.0
        rows.append(
            _row(
                obdobje, label, "energija",
                kolicina=kolicina, enota=vals.get("enota", "kWh"),
                cena_eur_em=cena, znesek_eur_brez_ddv=znesek,
            )
        )
        skupaj_energija += znesek
    rows.append(_row(obdobje, "Skupaj električna energija", "skupaj",
                      znesek_eur_brez_ddv=skupaj_energija))

    # 2. Network fees (always block-based): contracted power + per-block
    # network energy fee + excess-power ratchet.
    skupaj_omreznina = 0.0
    moc_po_blokih = racun.diagnostika.get("omreznina_moc_po_blokih", {})
    for blok in sorted(moc_po_blokih):
        znesek = moc_po_blokih[blok]
        kolicina = gospodinjstvo.dogovorjena_moc.get(blok, 0.0)
        cena = znesek / kolicina if kolicina else 0.0
        rows.append(
            _row(
                obdobje, f"Dogovorjena moč za časovni blok {blok}", "omreznina",
                kolicina=kolicina, enota="kW", cena_eur_em=cena,
                znesek_eur_brez_ddv=znesek,
            )
        )
        skupaj_omreznina += znesek
    for blok in sorted(po_blokih_omreznina):
        vals = po_blokih_omreznina[blok]
        znesek = vals.get("omreznina_energija_eur", 0.0)
        kolicina = vals.get("prevzeto_kwh", 0.0)
        cena = znesek / kolicina if kolicina else 0.0
        rows.append(
            _row(
                obdobje, f"Prevzeta EE za časovni blok {blok}", "omreznina",
                kolicina=kolicina, enota="kWh", cena_eur_em=cena,
                znesek_eur_brez_ddv=znesek,
            )
        )
        skupaj_omreznina += znesek
    presezna_moc = racun.postavke.get("omreznina_presezna_moc", 0.0)
    if presezna_moc:
        rows.append(_row(obdobje, "Presežna moč", "omreznina",
                          znesek_eur_brez_ddv=presezna_moc))
        skupaj_omreznina += presezna_moc
    rows.append(_row(obdobje, "Skupaj omrežnina za elektroenergetski sistem", "skupaj",
                      znesek_eur_brez_ddv=skupaj_omreznina))

    # 3. Regulatory contributions (per-kWh on total intake, not block-split).
    prevzeto_kwh = racun.prevzeto_kwh
    prispevki = [
        ("Prispevek za delovanje operaterja trga", "prispevek_operater_trga"),
        ("Prispevek za energetsko učinkovitost", "prispevek_ure"),
    ]
    skupaj_prispevki = 0.0
    for label, key in prispevki:
        znesek = racun.postavke.get(key, 0.0)
        cena = znesek / prevzeto_kwh if prevzeto_kwh else 0.0
        rows.append(_row(obdobje, label, "prispevki", kolicina=prevzeto_kwh,
                          enota="kWh", cena_eur_em=cena, znesek_eur_brez_ddv=znesek))
        skupaj_prispevki += znesek
    ove_spte = racun.postavke.get("prispevek_ove_spte", 0.0)
    ove_spte_diag = racun.diagnostika.get("omreznina_moc_po_blokih", {})
    # Same reference block MesecniObracun._fiksne uses (min active block, or 2
    # as a fallback when no block is active that period).
    ref_blok = min(ove_spte_diag) if ove_spte_diag else 2
    ove_spte_kw = gospodinjstvo.dogovorjena_moc.get(ref_blok, 0.0)
    cena_ove = (ove_spte / ove_spte_kw) if ove_spte_kw else 0.0
    rows.append(_row(obdobje, "Prispevek za SPTE in OVE", "prispevki",
                      kolicina=ove_spte_kw, enota="kW" if ove_spte_kw else None,
                      cena_eur_em=cena_ove if ove_spte_kw else None,
                      znesek_eur_brez_ddv=ove_spte))
    skupaj_prispevki += ove_spte
    rows.append(_row(obdobje, "Skupaj prispevki in ostale dajatve", "skupaj",
                      znesek_eur_brez_ddv=skupaj_prispevki))

    # 4. Excise duty (trošarina).
    trosarina = racun.postavke.get("trosarina", 0.0)
    cena_tro = trosarina / prevzeto_kwh if prevzeto_kwh else 0.0
    rows.append(_row(obdobje, "Trošarina", "trosarina", kolicina=prevzeto_kwh,
                      enota="kWh", cena_eur_em=cena_tro, znesek_eur_brez_ddv=trosarina))
    rows.append(_row(obdobje, "Skupaj trošarina", "skupaj", znesek_eur_brez_ddv=trosarina))

    # 5. Static disclosure line (not modeled by si_tarife.py, always 0 EUR;
    # included only for visual/structural fidelity to a real bill).
    rows.append(_row(obdobje, "100% Jedrska energija", "poslovanje",
                      kolicina=1, enota="kos", cena_eur_em=0.0,
                      znesek_eur_brez_ddv=0.0))

    # 6. Monthly management fee, split back into base fee + eko discount
    # (Paket.nadomestilo() only returns the already-netted figure).
    skupaj_poslovanje = 0.0
    osnovno = paket.mesecno_nadomestilo
    if osnovno:
        rows.append(_row(obdobje, "Pavšalni strošek poslovanja", "poslovanje",
                          kolicina=1, enota="kos", cena_eur_em=osnovno,
                          znesek_eur_brez_ddv=osnovno))
        skupaj_poslovanje += osnovno
    if gospodinjstvo.eko_racun and paket.mesecno_nadomestilo_eko is not None:
        popust = paket.mesecno_nadomestilo_eko - paket.mesecno_nadomestilo
        if popust:
            rows.append(_row(obdobje, "E-popust", "poslovanje",
                              kolicina=1, enota="kos", cena_eur_em=popust,
                              znesek_eur_brez_ddv=popust))
            skupaj_poslovanje += popust
    if paket.dodatna_storitev:
        rows.append(_row(obdobje, "Dodatna storitev", "poslovanje",
                          kolicina=1, enota="kos", cena_eur_em=paket.dodatna_storitev,
                          znesek_eur_brez_ddv=paket.dodatna_storitev))
        skupaj_poslovanje += paket.dodatna_storitev
    rows.append(_row(obdobje, "Obračun storitev pogodbenega računa", "skupaj",
                      znesek_eur_brez_ddv=skupaj_poslovanje))

    # 7. Bill-level totals.
    rows.append(_row(obdobje, "Skupaj brez DDV", "skupaj", znesek_eur_brez_ddv=racun.neto))
    rows.append(_row(obdobje, f"DDV {DDV_ODSTOTEK}%", "ddv", znesek_eur_brez_ddv=racun.ddv))
    rows.append(_row(obdobje, "Skupaj z DDV", "skupaj", znesek_eur_brez_ddv=racun.bruto))
    if racun.dobropis_odkup:
        rows.append(_row(obdobje, "Dobropis za oddano energijo", "dobropis",
                          znesek_eur_brez_ddv=-racun.dobropis_odkup))
    rows.append(_row(obdobje, "ZA PLAČILO", "za_placilo", znesek_eur_brez_ddv=racun.za_placilo))

    for opozorilo in racun.opozorila:
        rows.append(_row(obdobje, opozorilo, "opozorilo"))

    return rows


# ---------------------------------------------------------------------------
# Whole-period aggregation
# ---------------------------------------------------------------------------
def aggregate_line_items(rows: List[Dict[str, Any]], obdobje_label: str) -> List[Dict[str, Any]]:
    """Collapses line items from multiple months into one whole-period set of
    line items, preserving row order of first appearance. Amounts are always
    linear (sum is exact); per-unit prices are re-derived as a blended
    average (znesek/kolicina) since unit prices are not linear across months
    (e.g. seasonal power-block rates)."""
    if not rows:
        return []

    order: List[str] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r["produkt"]
        if key not in grouped:
            order.append(key)
            grouped[key] = {
                "kategorija": r["kategorija"],
                "enota": r["enota"],
                "kolicina": 0.0,
                "znesek": 0.0,
                "has_kolicina": False,
                "has_znesek": False,
            }
        g = grouped[key]
        if r["kolicina"] is not None:
            g["kolicina"] += r["kolicina"]
            g["has_kolicina"] = True
        if r["znesek_eur_brez_ddv"] is not None:
            g["znesek"] += r["znesek_eur_brez_ddv"]
            g["has_znesek"] = True

    out: List[Dict[str, Any]] = []
    # Title row(s) collapse to a single occurrence at the top.
    title_keys = [k for k in order if grouped[k]["kategorija"] == "naslov"]
    other_keys = [k for k in order if grouped[k]["kategorija"] != "naslov"]
    if title_keys:
        out.append(_row(obdobje_label, title_keys[0], "naslov"))
    for key in other_keys:
        g = grouped[key]
        kolicina = g["kolicina"] if g["has_kolicina"] else None
        znesek = g["znesek"] if g["has_znesek"] else None
        cena = (znesek / kolicina) if (kolicina and znesek is not None and g["kategorija"] not in
                                        ("skupaj", "ddv", "za_placilo", "dobropis")) else None
        out.append(
            _row(
                obdobje_label, key, g["kategorija"],
                kolicina=kolicina, enota=g["enota"], cena_eur_em=cena,
                znesek_eur_brez_ddv=znesek,
            )
        )
    return out


def aggregate_household_invoices(
    household_rows: Dict[str, List[Dict[str, Any]]],
    obdobje_label: str,
) -> Dict[str, Any]:
    """Build separate and group invoice views from household line items.

    Args:
        household_rows: Mapping household_id -> monthly line-item rows.
        obdobje_label: Label used in the whole-period aggregated views.

    Returns:
        Dict with:
          - separate: aggregated line-items per household
          - group: one aggregated line-item list over all households
    """
    separate: Dict[str, List[Dict[str, Any]]] = {}
    combined_rows: List[Dict[str, Any]] = []

    for household_id, rows in household_rows.items():
        hh_rows = list(rows or [])
        separate[household_id] = aggregate_line_items(hh_rows, obdobje_label)
        combined_rows.extend(hh_rows)

    group = aggregate_line_items(combined_rows, obdobje_label)

    # Keep a deterministic title row for group-level reporting.
    if group and group[0].get("kategorija") == "naslov":
        group[0]["produkt"] = "Skupni obračun gospodinjstev"

    return {
        "separate": separate,
        "group": group,
    }


# ---------------------------------------------------------------------------
# Paper-invoice rounding
# ---------------------------------------------------------------------------
#: Decimals a quantity carries on a printed bill, per unit of measure.
INVOICE_DECIMALS = {"kWh": 0, "kW": 1, "kos": 0}

#: Line categories that are detail lines; everything else is a subtotal, a tax
#: line, a title or a warning.
DETAIL_KATEGORIJE = ("energija", "omreznina", "prispevki", "trosarina", "poslovanje")

#: Decimals kept on money. The quantity is what a bill rounds; the amount that
#: follows from it stays at source precision, as the printed line does
#: (`183 kWh x 0,124900 = 22,85670`).
ZNESEK_DECIMALS = 5


def sum_amount(rows, kategorije, enota: Optional[str] = None) -> float:
    """Total of the line items in the given categories.

    With `enota` set it sums the *quantity* column instead, which is how the
    intake is recovered from a rounded bill.
    """
    field = "kolicina" if enota else "znesek_eur_brez_ddv"
    return float(
        sum(
            r[field] or 0.0
            for r in rows
            if r["kategorija"] in kategorije and (enota is None or r["enota"] == enota)
        )
    )


def round_invoice_rows(rows: List[Dict[str, Any]], ddv_stopnja: Optional[float] = None):
    """Re-cast line items the way a printed bill states them.

    Checked line by line against a real Elektro energija invoice (Posavskega,
    06/2026), which fixes three rules that are easy to guess wrong:

      1. The QUANTITY is rounded, the amount is not — the amount is that rounded
         quantity times the unit price at full precision.
      2. The unit price is never rounded; it comes from the price list as published.
      3. One intake figure per bill: the levies and the excise are charged on the
         rounded energy quantity the bill already states (183 + 184 = 367 kWh),
         not on a separately rounded total (which would give 368).

    Subtotals, VAT and the total are then re-derived from the rounded lines.
    """
    if not rows:
        return []
    if ddv_stopnja is None:
        from si_tarife import DDV as _DDV  # local: keeps the module import light
        ddv_stopnja = float(_DDV)

    out = [dict(r) for r in rows]

    def _round_qty(row):
        decimals = INVOICE_DECIMALS.get(row["enota"])
        if row["kolicina"] is None or decimals is None:
            return
        row["kolicina"] = round(float(row["kolicina"]), decimals)
        if decimals == 0:
            row["kolicina"] = float(int(row["kolicina"]))

    def _rederive_amount(row):
        if row["znesek_eur_brez_ddv"] is None:
            return
        if row["kolicina"] is not None and row["cena_eur_em"] is not None:
            row["znesek_eur_brez_ddv"] = round(
                row["kolicina"] * row["cena_eur_em"], ZNESEK_DECIMALS
            )

    # 1. Energy lines first: they are what the bill states the intake to be.
    for r in out:
        if r["kategorija"] == "energija":
            _round_qty(r)
            _rederive_amount(r)
    billed_kwh = sum_amount(out, ("energija",), enota="kWh")

    # 2. Everything else; per-kWh levies reuse the intake already stated.
    for r in out:
        if r["kategorija"] not in DETAIL_KATEGORIJE or r["kategorija"] == "energija":
            continue
        if r["kategorija"] in ("prispevki", "trosarina") and r["enota"] == "kWh":
            r["kolicina"] = billed_kwh
        else:
            _round_qty(r)
        _rederive_amount(r)

    # 3. Section subtotals: the sum of the detail lines that precede them.
    pending: Dict[str, float] = {}
    tail: List[Dict[str, Any]] = []
    for r in out:
        kat = r["kategorija"]
        if kat in DETAIL_KATEGORIJE:
            pending[kat] = pending.get(kat, 0.0) + (r["znesek_eur_brez_ddv"] or 0.0)
        elif kat == "skupaj":
            if pending:
                r["znesek_eur_brez_ddv"] = round(sum(pending.values()), ZNESEK_DECIMALS)
                pending = {}
            else:
                tail.append(r)  # bill-level totals, filled in below

    # 4. Bill-level totals, from the rounded detail lines.
    neto = round(sum_amount(out, DETAIL_KATEGORIJE), ZNESEK_DECIMALS)
    ddv = round(neto * ddv_stopnja, ZNESEK_DECIMALS)
    bruto = round(neto + ddv, ZNESEK_DECIMALS)
    dobropis = -sum_amount(out, ("dobropis",))
    for r, value in zip(tail, (neto, bruto)):  # "Skupaj brez DDV", "Skupaj z DDV"
        r["znesek_eur_brez_ddv"] = value
    for r in out:
        if r["kategorija"] == "ddv":
            r["znesek_eur_brez_ddv"] = ddv
        elif r["kategorija"] == "za_placilo":
            r["znesek_eur_brez_ddv"] = round(bruto - dobropis, ZNESEK_DECIMALS)
    return out


def block_reconciliation_gap(rows) -> float:
    """Rounded per-block intake minus the intake the bill states, in kWh.

    A real bill reconciles; rounding each block independently does not guarantee
    it. Reported rather than silently reallocated.
    """
    return round(
        sum(
            r["kolicina"] or 0.0
            for r in rows
            if r["kategorija"] == "omreznina" and r["enota"] == "kWh"
        )
        - sum_amount(rows, ("energija",), enota="kWh"),
        3,
    )


# ---------------------------------------------------------------------------
# Views over a set of line items
# ---------------------------------------------------------------------------
def invoice_total(rows) -> float:
    """The `ZA PLAČILO` figure of a set of line items."""
    for r in rows:
        if r["kategorija"] == "za_placilo":
            return float(r["znesek_eur_brez_ddv"] or 0.0)
    return 0.0


#: What each bill line rolls up into. The five buckets are disjoint and, absent
#: a dobropis, add up to `za plačilo`.
BILL_PARTS = {
    "Energija": ("energija",),
    "Omrežnina": ("omreznina",),
    "Prispevki in trošarina": ("prispevki", "trosarina"),
    "Poslovanje": ("poslovanje",),
    "DDV": ("ddv",),
}

#: Column labels for `invoice_frame`, in the order a paper bill prints them.
_INVOICE_COLUMNS = {
    "produkt": "Postavka",
    "kolicina": "Količina",
    "enota": "EM",
    "cena_eur_em": "Cena [EUR/EM]",
    "znesek_eur_brez_ddv": "Znesek [EUR]",
}


def invoice_frame(rows, *, drop_warnings: bool = False):
    """Line items as a bill-shaped DataFrame for display in a notebook.

    Keeps row order (that order *is* the bill layout) and returns `kategorija`
    as the index name, which is what a caller styles on.
    """
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=list(_INVOICE_COLUMNS.values()))
    kept = [r for r in rows if not (drop_warnings and r["kategorija"] == "opozorilo")]
    frame = pd.DataFrame(kept)[list(_INVOICE_COLUMNS)].rename(columns=_INVOICE_COLUMNS)
    frame.index = pd.Index([r["kategorija"] for r in kept], name="kategorija")
    return frame


def composition_frame(by_month: Dict[Any, List[Dict[str, Any]]]):
    """Month-by-month bill composition, one row per invoiced month.

    Indexed by the `MM/YYYY` label, with the five `BILL_PARTS` buckets, the
    intake the bill states, and the resulting average price.
    """
    import pandas as pd

    bucket_of = {kat: name for name, kats in BILL_PARTS.items() for kat in kats}
    records = []
    for (leto, mesec) in sorted(by_month):
        rows = by_month[(leto, mesec)]
        totals = {name: 0.0 for name in BILL_PARTS}
        for row in rows:
            name = bucket_of.get(row["kategorija"])
            if name is not None:
                totals[name] += float(row["znesek_eur_brez_ddv"] or 0.0)
        prevzeto = sum_amount(rows, ("energija",), enota="kWh")
        za_placilo = invoice_total(rows)
        records.append({
            "obdobje": f"{mesec:02d}/{leto}",
            "leto": leto,
            "mesec": mesec,
            **{k: round(v, 2) for k, v in totals.items()},
            "Za plačilo": round(za_placilo, 2),
            "Prevzeto [kWh]": round(prevzeto, 2),
            "Povprečna cena [EUR/kWh]": (
                round(za_placilo / prevzeto, 4) if prevzeto else float("nan")
            ),
        })
    return pd.DataFrame(records).set_index("obdobje")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_rows_csv(rows: List[Dict[str, Any]], path) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Stateful driver used by Environment.py and by MILP replay code
# ---------------------------------------------------------------------------
class InvoiceBuilder:
    """The one settlement driver: per-interval price_result dicts in (as returned
    by `calculate_interval_price(..., include_raw=True)`), calendar-month
    line-item invoices out.

    Every invoicing path in the repo goes through this — the RL environment, the
    MILP benchmark, and the real meter profiles in `si_poraba_doma` — so a bill
    is built the same way whatever produced the intervals.
    """

    def __init__(
        self,
        *,
        dogovorjena_moc: Union[Dict[int, float], Callable[[int, int], Dict[int, float]]],
        pricing_scheme: str,
        eko_racun: bool = True,
        interval_minutes: float,
        output_dir=None,
        run_label: str = "racun",
        write_monthly: bool = False,
        write_period: bool = False,
        pricing_reference_year: Optional[int] = None,
        round_like_invoice: bool = False,
        obracunaj_presezno_moc: bool = True,
        strogo: bool = True,
        on_month: Optional[Callable[[Any, List[Dict[str, Any]], Racun], None]] = None,
    ):
        # A household that manages its dogovorjena obracunska moc can change it
        # every month (free, effective the 1st), so this accepts either a fixed
        # per-block dict or `f(leto, mesec) -> {blok: kW}`; in the latter case a
        # fresh Gospodinjstvo is built for each invoiced month.
        self._moc_fn = dogovorjena_moc if callable(dogovorjena_moc) else None
        self._build_household = lambda moc: build_invoice_household(
            dogovorjena_moc=moc,
            pricing_scheme=pricing_scheme,
            eko_racun=eko_racun,
            meritve_15min=abs(float(interval_minutes) - 15.0) < 1e-6,
        )
        self.gospodinjstvo = (
            None if self._moc_fn is not None else self._build_household(dogovorjena_moc)
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.run_label = run_label
        self.write_monthly = write_monthly
        self.write_period = write_period
        # `round_like_invoice` is applied PER MONTH, before any roll-up, because
        # that is the order the real bills happen in: a year is the sum of twelve
        # rounded invoices, not one rounding of the year.
        self.round_like_invoice = bool(round_like_invoice)
        self._acc_kwargs = {
            "obracunaj_presezno_moc": bool(obracunaj_presezno_moc),
            "strogo": bool(strogo),
        }
        self._on_month = on_month
        # Same regulatory regime the per-interval calculate_interval_price
        # calls use -- keeps month-end fixed-charge finalization (network
        # power fee, OVE+SPTE, excess-power ratchet) consistent with them,
        # instead of independently re-resolving tariffs from the literal
        # calendar date (which fails outside the loaded 2025/2026 tables).
        self._pravila = resolve_invoice_pravila(pricing_reference_year)

        self._paket: Optional[Paket] = None
        self._current_key = None
        self._accumulator: Optional[InvoiceAccumulator] = None
        #: `{(leto, mesec): rows}` — each month's bill kept separate, so a caller
        #: can roll them up any way it likes.
        self.by_month: Dict[Any, List[Dict[str, Any]]] = {}

    # -- state ------------------------------------------------------------
    @property
    def monthly_line_items(self) -> List[Dict[str, Any]]:
        """Every closed month's line items, concatenated chronologically."""
        return [r for key in sorted(self.by_month) for r in self.by_month[key]]

    @property
    def months(self) -> List[Any]:
        return sorted(self.by_month)

    @property
    def years(self) -> List[int]:
        return sorted({leto for leto, _ in self.by_month})

    @property
    def paket(self) -> Optional[Paket]:
        return self._paket

    # -- accumulation -----------------------------------------------------
    def add_interval(self, price_result: Dict[str, Any]) -> None:
        raw = price_result.get("raw_result")
        if raw is None:
            return
        if self._paket is None:
            self._paket = PAKETI[price_result["paket_id"]]

        lok = raw["lokalni_cas"]
        key = (lok.year, lok.month)
        if self._current_key is None or key != self._current_key:
            if self._current_key is not None:
                self._finalize_month()
            if self._moc_fn is not None:
                self.gospodinjstvo = self._build_household(self._moc_fn(lok.year, lok.month))
            self._accumulator = InvoiceAccumulator(
                lok.year, lok.month, self.gospodinjstvo, self._paket, self._pravila,
                **self._acc_kwargs,
            )
            self._current_key = key

        self._accumulator.dodaj(raw)

    def _finalize_month(self) -> None:
        if self._accumulator is None:
            return
        racun, po_blokih_omreznina, energija_po_skupini, min_date, max_date, paket, gosp = (
            self._accumulator.zakljuci()
        )
        rows = racun_to_line_items(
            racun, po_blokih_omreznina, energija_po_skupini, paket, gosp, min_date, max_date
        )
        if self.round_like_invoice:
            rows = round_invoice_rows(rows, racun.ddv_stopnja)
        # Appended, not assigned: with `pricing_reference_year` remapping every
        # timestamp into one year, two calendar months of a multi-year dataset
        # can land on the same (leto, mesec) key, and dropping one silently would
        # under-bill the run.
        self.by_month.setdefault(self._current_key, []).extend(rows)
        self._accumulator = None
        if self._on_month is not None:
            self._on_month(self._current_key, rows, racun)
        if self.write_monthly and self.output_dir is not None:
            write_rows_csv(
                self.monthly_line_items, self.output_dir / f"{self.run_label}_monthly.csv"
            )

    def finalize(self, period_label: Optional[str] = None) -> None:
        self._finalize_month()
        if self.write_period and self.output_dir is not None and self.by_month:
            agg = self.get_period_line_items(period_label)
            write_rows_csv(agg, self.output_dir / f"{self.run_label}_period.csv")

    # -- views ------------------------------------------------------------
    def get_monthly_line_items(self) -> List[Dict[str, Any]]:
        """All collected monthly line items, in chronological order."""
        self._finalize_month()
        return self.monthly_line_items

    def get_period_line_items(self, period_label: Optional[str] = None) -> List[Dict[str, Any]]:
        """One aggregated bill over every invoiced month."""
        rows = self.get_monthly_line_items()
        if not rows:
            return []
        return aggregate_line_items(rows, period_label or self.run_label)

    def get_month_line_items(self, leto: int, mesec: int) -> List[Dict[str, Any]]:
        """One month's line items, exactly as they were billed."""
        self._finalize_month()
        return list(self.by_month.get((int(leto), int(mesec)), []))

    def get_year_line_items(self, leto: int) -> List[Dict[str, Any]]:
        """One aggregated bill over every invoiced month of one calendar year."""
        self._finalize_month()
        rows = [r for key in sorted(self.by_month) if key[0] == int(leto)
                for r in self.by_month[key]]
        return aggregate_line_items(rows, str(int(leto))) if rows else []

    def get_composition(self):
        """Month-by-month bill composition as a DataFrame."""
        self._finalize_month()
        return composition_frame(self.by_month)

    def get_invoice_views(self, period_label: Optional[str] = None) -> Dict[str, Any]:
        """Monthly and period views in one object for API consumers."""
        return {
            "monthly": self.get_monthly_line_items(),
            "period": self.get_period_line_items(period_label=period_label),
        }
