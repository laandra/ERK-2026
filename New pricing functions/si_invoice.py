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
from typing import Any, Dict, List, Optional

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
    """Accumulates per-interval price_result dicts (as returned by
    calculate_interval_price(..., include_raw=True)) into calendar-month
    invoices, optionally writing a running monthly-line-items CSV and, on
    finalize(), a whole-period aggregate CSV."""

    def __init__(
        self,
        *,
        dogovorjena_moc: Dict[int, float],
        pricing_scheme: str,
        eko_racun: bool = True,
        interval_minutes: float,
        output_dir,
        run_label: str,
        write_monthly: bool = False,
        write_period: bool = False,
        pricing_reference_year: Optional[int] = None,
    ):
        self.gospodinjstvo = build_invoice_household(
            dogovorjena_moc=dogovorjena_moc,
            pricing_scheme=pricing_scheme,
            eko_racun=eko_racun,
            meritve_15min=abs(float(interval_minutes) - 15.0) < 1e-6,
        )
        self.output_dir = Path(output_dir)
        self.run_label = run_label
        self.write_monthly = write_monthly
        self.write_period = write_period
        # Same regulatory regime the per-interval calculate_interval_price
        # calls use -- keeps month-end fixed-charge finalization (network
        # power fee, OVE+SPTE, excess-power ratchet) consistent with them,
        # instead of independently re-resolving tariffs from the literal
        # calendar date (which fails outside the loaded 2025/2026 tables).
        self._pravila = resolve_invoice_pravila(pricing_reference_year)

        self._paket: Optional[Paket] = None
        self._current_key = None
        self._accumulator: Optional[InvoiceAccumulator] = None
        self.monthly_line_items: List[Dict[str, Any]] = []

    def add_interval(self, price_result: Dict[str, Any]) -> None:
        raw = price_result.get("raw_result")
        if raw is None:
            return
        if self._paket is None:
            self._paket = PAKETI[price_result["paket_id"]]

        lok = raw["lokalni_cas"]
        key = (lok.year, lok.month)
        if self._current_key is None:
            self._accumulator = InvoiceAccumulator(
                lok.year, lok.month, self.gospodinjstvo, self._paket, self._pravila
            )
            self._current_key = key
        elif key != self._current_key:
            self._finalize_month()
            self._accumulator = InvoiceAccumulator(
                lok.year, lok.month, self.gospodinjstvo, self._paket, self._pravila
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
        self.monthly_line_items.extend(rows)
        self._accumulator = None
        if self.write_monthly:
            write_rows_csv(self.monthly_line_items, self.output_dir / f"{self.run_label}_monthly.csv")

    def finalize(self, period_label: Optional[str] = None) -> None:
        self._finalize_month()
        if self.write_period and self.monthly_line_items:
            label = period_label or self.run_label
            agg = aggregate_line_items(self.monthly_line_items, label)
            write_rows_csv(agg, self.output_dir / f"{self.run_label}_period.csv")
