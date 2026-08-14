"""si_poraba_doma.py — bills for the real meter exports in `Input data/Poraba doma/`.

Loads a distributor 15-minute export, decides which agreed billing power was in
force each month, and feeds every interval to `si_invoice.InvoiceBuilder` — the
same driver the RL environment and the MILP benchmark invoice through. No
invoice math lives here.

Run:
    python "New pricing functions/si_poraba_doma.py" --from 2025-01 --verify-blocks
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from Pricing_Functions import (  # noqa: E402
    SCHEME_SI_DOBAVA,
    SCHEME_SI_SAMOOSKRBA,
    calculate_interval_price,
)
from si_cas import casovni_blok  # noqa: E402
from si_invoice import (  # noqa: E402
    InvoiceBuilder,
    block_reconciliation_gap,
    invoice_total,
    sum_amount,
    write_rows_csv,
)
from si_moc import (  # noqa: E402
    KONICNI_BLOKI,
    PRIKLJUCNE_MOCI_KW,
    ST_KONIC,
    administrativna_moc,
    minimalna_dogovorjena_moc,
    referencno_okno,
    uskladi_bloke,
    zaokrozi_moc,
)
from si_paketi import PAKETI  # noqa: E402

TZ_SI = "Europe/Ljubljana"
INTERVAL_MINUTES = 15
PROFILE_DIR = _REPO / "Input data" / "Poraba doma"
OUTPUT_DIR = _REPO / "Results" / "Invoices"

# `si_obracun._cena_energije` reads a TARIFNI package as two-tariff whenever vt
# or mt is non-zero. GEN-I's redni cenik publishes all three rates, so zeroing
# vt/mt is what selects the ET (single-tariff) column of the same price list.
GENI_REDNI_1T = replace(
    PAKETI["GENI_REDNI"],
    id="GENI_REDNI_1T",
    ime="Redni cenik za gospodinjske odjemalce (enotarifni)",
    vt=0.0,
    mt=0.0,
)
PAKETI[GENI_REDNI_1T.id] = GENI_REDNI_1T


@dataclass
class Household:
    """One metering point and the billing choices the profile does not carry."""

    name: str
    folder: str
    paket_id: str
    #: "si_dobava" (grid supply only) or "si_samooskrba" (PV self-supply); the
    #: paket has to match, and samooskrba settles the export's A- column too.
    scheme: str = SCHEME_SI_DOBAVA
    eko_racun: bool = True

    # --- connection, from the soglasje za priključitev ---------------------
    #: Either a fuse rating out of `si_moc.PRIKLJUCNE_MOCI_KW` ("3x16") or an
    #: explicit `prikljucna_moc_kw`. Sets the regulatory floor and ceiling on the
    #: agreed power, and the administrative power when there is no history.
    #: BOTH None is the "soglasje unknown" case: no floor and no ceiling, so the
    #: agreed power follows the measured peaks and the monotonicity rule alone —
    #: better than inventing a connection size and then billing the household
    #: against it. A year with no peaks to read then has nothing to fall back on
    #: and says so, because there is no administrative power without a connection.
    varovalka: Optional[str] = "3x16"
    prikljucna_moc_kw: Optional[float] = None
    faze: int = 3

    # --- agreed billing power ---------------------------------------------
    #: Where the vector comes from when it is not pinned:
    #:   "auto"            the export's `Dogovorjena moč` column, with the
    #:                     operator's rule filling blocks the column omits
    #:   "dso"             the column alone
    #:   "operater"        the operator's rule alone, ignoring the column
    #:   "administrativna" the no-history figure for this connection
    #: Every mode falls back to "administrativna" when its source is empty.
    vir_moci: str = "auto"
    #: Straight off the paper invoice, and billed VERBATIM — no floor, no
    #: ceiling, no monotonicity clipping. Overrides PER BLOCK, so a summer bill's
    #: `{2: 3.8, 3: 4.0, 4: 4.0, 5: 4.0}` still gets a resolved block 1 in
    #: winter instead of being billed at 0 kW. Either `{blok: kW}` for every
    #: month or `{(leto, mesec): {blok: kW}}`.
    dogovorjena_moc: Optional[Dict] = None
    #: A household whose agreed power the operator sets and which never changes
    #: it itself is permanently exempt from the excess-power charge (Ur. l. RS
    #: 27/25). True prices it anyway.
    obracunaj_presezno_moc: bool = False

    VIRI_MOCI = ("auto", "dso", "operater", "administrativna")

    def __post_init__(self):
        if self.scheme not in (SCHEME_SI_DOBAVA, SCHEME_SI_SAMOOSKRBA):
            raise ValueError(f"{self.name}: unknown scheme {self.scheme!r}")
        if self.vir_moci not in self.VIRI_MOCI:
            raise ValueError(
                f"{self.name}: unknown vir_moci {self.vir_moci!r}; "
                f"expected one of {self.VIRI_MOCI}"
            )
        if self.paket_id not in PAKETI:
            raise ValueError(f"{self.name}: unknown paket_id {self.paket_id!r}")
        paket = PAKETI[self.paket_id]
        pv = self.scheme == SCHEME_SI_SAMOOSKRBA
        if pv and not paket.dovoljuje_pv:
            raise ValueError(
                f"{self.name}: {self.paket_id} is not a self-supply price list; "
                f"pick a paket with dovoljuje_pv or use scheme={SCHEME_SI_DOBAVA!r}."
            )
        if not pv and paket.zahteva_pv:
            raise ValueError(
                f"{self.name}: {self.paket_id} requires PV; use "
                f"scheme={SCHEME_SI_SAMOOSKRBA!r}."
            )
        if self.prikljucna_moc_kw is None and self.varovalka is not None:
            if self.varovalka not in PRIKLJUCNE_MOCI_KW:
                raise ValueError(
                    f"{self.name}: unknown varovalka {self.varovalka!r}; known: "
                    f"{sorted(PRIKLJUCNE_MOCI_KW)} — or set prikljucna_moc_kw, "
                    f"or varovalka=None when the soglasje is unknown."
                )
            self.prikljucna_moc_kw = PRIKLJUCNE_MOCI_KW[self.varovalka]
            self.faze = int(self.varovalka.split("x")[0])


#: Dogovorjena obračunska moč copied off the paper invoices, per contract year.
#: A metering point whose export does not cover the operator's reference window
#: cannot have its figure reconstructed, so the bill is the answer — this is the
#: property of the metering point, not of one notebook, which is why it lives on
#: the catalogue where the CLI and the notebook both read it.
#:
#: Posavskega: the export starts 26. 7. 2025 and 07–08/2025 are unsettled, so
#: 10/2024–09/2025 holds 1 of 12 settled months. The rule reads that single
#: September and returns 4.0/4.3/4.3 against the 06/2026 bill's 3.8/4.0/4.0.
#: Widening the window is not the fix: over all months present blok 2 rises to
#: 5.7 kW, because the autumn 2025 peaks are far above what the operator read.
RACUNI_MOCI: Dict[str, Dict] = {
    "Posavskega": {(2026, m): {2: 3.8, 3: 4.0, 4: 4.0, 5: 4.0} for m in range(1, 13)},
}

HOUSEHOLDS: Tuple[Household, ...] = (
    Household("Adamiceva", "Adamiceva", "ELEN_ZANESLJIVA"),
    Household("Posavskega", "Posavskega", "ELEN_ZANESLJIVA",
              dogovorjena_moc=RACUNI_MOCI["Posavskega"]),
    Household("Koroska", "Koroska", "GENI_REDNI_1T"),
)


def opis_prikljucka(hh: Household) -> str:
    """The connection as a log line: fuse and kW, or "priključek neznan"."""
    if hh.prikljucna_moc_kw is None:
        return "priključek neznan (brez spodnje in zgornje meje)"
    if hh.varovalka is None:
        return f"{hh.prikljucna_moc_kw:g} kW"
    return f"{hh.varovalka} / {hh.prikljucna_moc_kw:g} kW"


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------
def load_profile(folder: str) -> pd.DataFrame:
    """All CSV exports for one metering point, as one clean chronological frame.

    The export stamps 15-minute *interval ends* in Ljubljana wall-clock time;
    `calculate_interval_price` wants the interval *start* in UTC. Each file is
    localized on its own before the files are merged, because the October
    fall-back hour repeats one wall-clock stamp inside a single file and has to
    become two distinct UTC instants before overlapping exports can be
    de-duplicated on the timestamp.

    Columns: `utc_start`, `local_start`, `leto`, `mesec`, `kwh` (A+),
    `oddano_kwh` (A-), `blok_dso`, `moc_dso`, `obracunski`.
    """
    files = sorted((PROFILE_DIR / folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV exports in {PROFILE_DIR / folder}")

    frames: List[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path, low_memory=False)
        local_end = pd.to_datetime(raw["Časovna značka"]).dt.tz_localize(
            TZ_SI, ambiguous="infer", nonexistent="shift_forward"
        )
        frames.append(pd.DataFrame({
            "utc_start": local_end.dt.tz_convert("UTC")
            - pd.Timedelta(minutes=INTERVAL_MINUTES),
            "kwh": pd.to_numeric(raw["Energija A+"], errors="coerce").fillna(0.0),
            "oddano_kwh": pd.to_numeric(raw["Energija A-"], errors="coerce").fillna(0.0),
            "blok_dso": pd.to_numeric(raw["Blok"], errors="coerce"),
            "moc_dso": pd.to_numeric(raw["Dogovorjena moč"], errors="coerce"),
            "obracunski": raw["Obračunski podatek"].astype(str),
        }))

    df = pd.concat(frames, ignore_index=True)
    # Prefer the settled ("Da") copy of any interval shipped more than once.
    df["_settled"] = df["obracunski"].eq("Da")
    df = (
        df.sort_values(["utc_start", "_settled"], ascending=[True, False])
        .drop_duplicates("utc_start", keep="first")
        .drop(columns="_settled")
        .reset_index(drop=True)
    )
    local_start = df["utc_start"].dt.tz_convert(TZ_SI)
    df["local_start"] = local_start
    df["leto"] = local_start.dt.year
    df["mesec"] = local_start.dt.month
    return df


def verify_blocks(df: pd.DataFrame, *, say=print) -> int:
    """Cross-check our block assignment against the distributor's own column.

    This is the test that pins the interval-start convention: computed from the
    raw (interval-end) stamp ~7 % of intervals mismatch, from the interval start
    none do. Returns the mismatch count.
    """
    ours = df["local_start"].map(lambda t: casovni_blok(t.to_pydatetime(), "2024"))
    bad = int((ours != df["blok_dso"]).sum())
    say(f"  block check   {len(df) - bad}/{len(df)} intervals agree with the DSO's "
        f"Blok column" + ("" if bad == 0 else f"  ({bad} MISMATCH)"))
    return bad


def _check_scheme(df: pd.DataFrame, hh: Household, *, say=print) -> None:
    """A plain-supply price list cannot settle a profile with real production.

    All three meters register a few 1 Wh A- readings — meter noise on a house
    with no PV — so the guard is on the share of intake, not on a strict zero.
    """
    oddano, prevzeto = float(df["oddano_kwh"].sum()), float(df["kwh"].sum())
    share = oddano / prevzeto if prevzeto else 1.0
    if hh.scheme == SCHEME_SI_SAMOOSKRBA:
        if oddano <= 5.0 and share <= 0.005:
            say(f"  warning       scheme={hh.scheme} but the profile feeds in only "
                f"{oddano:.3f} kWh; is this really a self-supply metering point?")
        return
    if oddano <= 0.0:
        return
    if oddano > 5.0 or share > 0.005:
        raise ValueError(
            f"{hh.name}: profile feeds in {oddano:.3f} kWh ({share:.2%} of intake); "
            f"scheme={SCHEME_SI_DOBAVA!r} cannot settle self-supply. Set "
            f"scheme={SCHEME_SI_SAMOOSKRBA!r} and a samooskrbni paket."
        )
    say(f"  note          {oddano:.3f} kWh of A- readings ({share:.4%} of intake) "
        f"ignored as meter noise")


# ---------------------------------------------------------------------------
# The agreed billing power in force in each month
# ---------------------------------------------------------------------------
def konice_v_oknu(
    df: pd.DataFrame, leto: int, *, settled_only: bool = True
) -> Tuple[Dict[int, float], int]:
    """Average of the `ST_KONIC` highest 15-minute powers per block, over the
    operator's reference window for contract year `leto`.

    `settled_only` keeps only intervals the distributor marks `Obračunski
    podatek = Da` — 15-minute settlement metering starts with the tariff reform
    on 1. 10. 2024, so anything before it is provisional and is not what the
    operator computed from. Also returns how many of the 12 months the window
    actually contains.
    """
    start, end = referencno_okno(leto)
    w = df[(df["local_start"] >= pd.Timestamp(start, tz=TZ_SI))
           & (df["local_start"] < pd.Timestamp(end, tz=TZ_SI))]
    if settled_only:
        w = w[w["obracunski"] == "Da"]
    if w.empty:
        return {}, 0
    konice = {
        int(blok): float((g["kwh"] * (60.0 / INTERVAL_MINUTES)).nlargest(ST_KONIC).mean())
        for blok, g in w.groupby("blok_dso")
        if int(blok) in KONICNI_BLOKI
    }
    return konice, int(w["local_start"].dt.strftime("%Y-%m").nunique())


def _dso_by_year(df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """`{leto: {blok: kW}}` as the export's own `Dogovorjena moč` column states it.

    Taken per contract year, not per month: the figure is set once a year, and
    blocks 1 and 5 are seasonal so no single month names them all.
    """
    out: Dict[int, Dict[int, float]] = {}
    for (leto, blok), kw in df.groupby(["leto", "blok_dso"])["moc_dso"].max().items():
        if pd.notna(kw) and kw > 0:
            out.setdefault(int(leto), {})[int(blok)] = float(kw)
    return out


@dataclass
class AgreedPower:
    """The per-block kW vector in force in one contract year, and where it came from."""

    leto: int
    vector: Dict[int, float]
    vir: str                       # dso | dso+operater | operater | administrativna
    mesecev_v_oknu: int = 0
    konice: Dict[int, float] = field(default_factory=dict)
    #: False when the reference window held only provisional (unsettled) readings.
    settled: bool = True

    def __str__(self) -> str:
        shown = {b: round(v, 1) for b, v in sorted(self.vector.items())}
        okno = ""
        if self.vir.endswith("operater"):
            okno = (f", {self.mesecev_v_oknu}/12 months in the reference window"
                    + ("" if self.settled else ", provisional"))
        return f"{self.leto}: {shown}  [{self.vir}{okno}]"


def _normalize_pinned(dogovorjena_moc) -> Optional[Dict]:
    """Accept either pinned shape, return `{(leto, mesec) or None: {blok: kW}}`.

    Told apart by their values the way `si_moc.je_mesecni_razpored` does it: a
    per-month schedule maps to dicts, a flat vector maps to numbers. A flat
    vector is stored under `None`, meaning "every month".
    """
    if not dogovorjena_moc:
        return None
    if all(isinstance(v, dict) for v in dogovorjena_moc.values()):
        return {(int(k[0]), int(k[1])): {int(b): float(kw) for b, kw in v.items()}
                for k, v in dogovorjena_moc.items()}
    return {None: {int(b): float(kw) for b, kw in dogovorjena_moc.items()}}


def agreed_power_by_year(
    df: pd.DataFrame, hh: Household, leta: List[int], *, say=print
) -> Dict[int, AgreedPower]:
    """Resolve the agreed power for each contract year, per `hh.vir_moci`.

    The two real sources are the export's own `Dogovorjena moč` column and the
    operator's rule (`si_moc`: the average of the five highest peaks per block
    over October(y-2)..September(y-1), blocks 1-4, rounded to 0.1 kW, floored at
    the connection's published minimum, then raised so no block sits below a
    lower one — which is what sets block 5). "auto" takes the column and lets the
    rule fill blocks it omits, which is how block 1 gets a figure once the export
    stops reporting it. Any mode with nothing to go on falls back to the
    administrative power.
    """
    want_dso = hh.vir_moci in ("auto", "dso")
    want_rule = hh.vir_moci in ("auto", "operater")
    dso = _dso_by_year(df) if want_dso else {}
    out: Dict[int, AgreedPower] = {}
    for leto in leta:
        # No connection known -> no floor and no ceiling; `uskladi_bloke` already
        # reads prikljucna_moc_kw=None as "unbounded".
        minimalna = (
            minimalna_dogovorjena_moc(hh.prikljucna_moc_kw, faze=hh.faze, leto=leto)
            if hh.prikljucna_moc_kw else 0.0
        )
        konice, mesecev, settled = {}, 0, True
        if want_rule:
            konice, mesecev = konice_v_oknu(df, leto)
            if not konice:
                # Nothing settled in the window. Provisional readings are a far
                # better estimate than the administrative power, but they are not
                # what the operator computed from, so say so.
                konice, mesecev = konice_v_oknu(df, leto, settled_only=False)
                settled = not konice
        derived = {b: zaokrozi_moc(v) for b, v in konice.items()}
        letni_dso = dso.get(leto, {})
        base = {b: letni_dso.get(b, derived.get(b)) for b in KONICNI_BLOKI
                if letni_dso.get(b, derived.get(b)) is not None}

        if not base:
            if not hh.prikljucna_moc_kw:
                raise ValueError(
                    f"{hh.name}: {leto} has no agreed power in the export and no "
                    f"history in the reference window (10/{leto - 2}–09/{leto - 1}), "
                    f"and with no connection known there is no administrative power "
                    f"to fall back on. Set varovalka= (or prikljucna_moc_kw=), or "
                    f"pin dogovorjena_moc from the bill."
                )
            admin = administrativna_moc(hh.prikljucna_moc_kw, faze=hh.faze, leto=leto)
            base, vir = {b: admin for b in KONICNI_BLOKI}, "administrativna"
        elif letni_dso and derived and set(letni_dso) < set(base):
            vir = "dso+operater"
        elif letni_dso:
            vir = "dso"
        else:
            vir = "operater"

        out[leto] = AgreedPower(
            leto=leto,
            vector=uskladi_bloke(base, minimalna_moc_kw=minimalna,
                                 prikljucna_moc_kw=hh.prikljucna_moc_kw),
            vir=vir,
            mesecev_v_oknu=mesecev,
            konice=konice,
            settled=settled,
        )
        say(f"  agreed power  {out[leto]}")
        if vir.endswith("operater") and (mesecev < 12 or not settled):
            say(f"  warning       {leto}: the operator's rule needs a full reference "
                f"year of settled readings; this window holds {mesecev}/12 month(s)"
                + ("" if settled else ", none of them settled")
                + " — pin dogovorjena_moc from the bill instead")
        elif vir == "administrativna":
            say(f"  warning       {leto}: no agreed power reported and no history in "
                f"the reference window (10/{leto - 2}–09/{leto - 1}); using the "
                f"administrative power for a "
                f"{hh.varovalka or f'{hh.prikljucna_moc_kw:g} kW'} connection")
        prazni = [b for b, kw in sorted(out[leto].vector.items()) if kw <= 0.0]
        if prazni:
            say(f"  warning       {leto}: block(s) {prazni} resolved to 0 kW — no "
                f"peaks in the reference window, and with the connection unknown "
                f"there is no floor to hold them up; they bill at 0 kW")
    return out


def agreed_power_schedule(
    df: pd.DataFrame, hh: Household, months: List[Tuple[int, int]], *, say=print
) -> Tuple[Dict[Tuple[int, int], Dict[int, float]], Dict[int, AgreedPower]]:
    """`({(leto, mesec): {blok: kW}}, {leto: AgreedPower})` for the months to invoice.

    A pinned block is billed exactly as written — no floor, no ceiling, no
    monotonicity clipping, because the figure on the bill is the answer, not an
    input to be corrected; a non-monotone result is reported, not reshaped.

    The pin applies PER BLOCK. Blocks 1 and 5 are seasonal, so a bill copied off
    a summer invoice names only blocks 2-5; replacing the whole vector with it
    would bill block 1 at 0 kW every November through February — the single most
    expensive block. Unnamed blocks therefore keep their resolved figure.
    """
    pinned = _normalize_pinned(hh.dogovorjena_moc)
    by_year = agreed_power_by_year(df, hh, sorted({leto for leto, _ in months}), say=say)
    schedule: Dict[Tuple[int, int], Dict[int, float]] = {}
    n_pinned = 0
    for key in months:
        vector = dict(by_year[key[0]].vector)
        fixed = None if pinned is None else pinned.get(key, pinned.get(None))
        if fixed is not None:
            vector.update(fixed)
            n_pinned += 1
            _report_non_monotone(vector, f"{key[1]:02d}/{key[0]}", say)
        schedule[key] = vector
    if pinned is not None:
        blocks = sorted({b for v in pinned.values() for b in v})
        say(f"  agreed power  blocks {blocks} pinned in {n_pinned}/{len(months)} "
            f"invoiced month(s); other blocks resolved as above")
    return schedule, by_year


def _report_non_monotone(vector: Dict[int, float], label: str, say) -> None:
    blocks = sorted(vector)
    breaks = [(a, b) for a, b in zip(blocks, blocks[1:]) if vector[a] > vector[b] + 1e-9]
    if breaks:
        a, b = breaks[0]
        say(f"  warning       pinned agreed power for {label} is not monotone "
            f"(P{a}={vector[a]:g} > P{b}={vector[b]:g}); billed as given")


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
@dataclass
class InvoiceResult:
    """One household's invoicing run: the builder holds every bill view."""

    household: Household
    paket_ime: str
    profile: pd.DataFrame
    schedule: Dict[Tuple[int, int], Dict[int, float]]
    agreed: Dict[int, AgreedPower]
    builder: InvoiceBuilder
    summary: List[Dict]

    @property
    def by_month(self):
        return self.builder.by_month

    @property
    def rows(self):
        return self.builder.monthly_line_items

    @property
    def months(self):
        return self.builder.months

    @property
    def years(self):
        return self.builder.years

    def month(self, leto: int, mesec: int):
        return self.builder.get_month_line_items(leto, mesec)

    def year(self, leto: int):
        return self.builder.get_year_line_items(leto)

    def period(self, label: Optional[str] = None):
        return self.builder.get_period_line_items(label or self._period_label())

    def composition(self):
        return self.builder.get_composition()

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary)

    def _period_label(self) -> str:
        if not self.summary:
            return "celotno obdobje"
        return (f"{min(s['od'] for s in self.summary):%d.%m.%Y}–"
                f"{max(s['do'] for s in self.summary):%d.%m.%Y}")


def invoice_household(
    hh: Household,
    *,
    month_from: Optional[Tuple[int, int]] = None,
    month_to: Optional[Tuple[int, int]] = None,
    complete_months_only: bool = False,
    round_like_invoice: bool = False,
    check_blocks: bool = False,
    verbose: bool = True,
) -> InvoiceResult:
    """Price one household month by month over the requested range.

    `complete_months_only` drops months the export does not fully cover — a
    partial month still bills a whole month of fixed charges.
    """
    say = print if verbose else (lambda *a, **k: None)
    paket = PAKETI[hh.paket_id]
    df = load_profile(hh.folder)
    say(f"\n{hh.name}  ({paket.dobavitelj} – {paket.ime}, {hh.scheme}, "
        f"{opis_prikljucka(hh)})")
    say(f"  profile       {len(df)} intervals, "
        f"{df['local_start'].min():%Y-%m-%d %H:%M} .. "
        f"{df['local_start'].max():%Y-%m-%d %H:%M}, {df['kwh'].sum():.1f} kWh in")
    _check_scheme(df, hh, say=say)
    if check_blocks:
        verify_blocks(df, say=say)

    months, skipped = [], []
    for key, month_df in _months(df):
        if (month_from is not None and key < month_from) or (
            month_to is not None and key > month_to
        ):
            continue
        expected = _intervals_in_month(key)
        if complete_months_only and len(month_df) != expected:
            skipped.append((key, len(month_df), expected))
            continue
        months.append(key)
    for key, seen, expected in skipped:
        say(f"  skipped       {key[1]:02d}/{key[0]}: {seen} of {expected} intervals")
    if not months:
        raise ValueError(f"{hh.name}: no months in the requested range.")

    schedule, agreed = agreed_power_schedule(df, hh, months, say=say)
    notes: List[str] = []

    def on_month(key, rows, racun):
        if round_like_invoice:
            gap = block_reconciliation_gap(rows)
            if gap:
                notes.append(f"{key[1]:02d}/{key[0]}: rounded block quantities sum to "
                             f"{gap:+.0f} kWh against the stated intake")

    builder = InvoiceBuilder(
        # The agreed power is a contract year figure, so it is resolved per
        # invoiced month rather than pinned once for the whole run.
        dogovorjena_moc=lambda leto, mesec: schedule[(leto, mesec)],
        pricing_scheme=hh.scheme,
        eko_racun=hh.eko_racun,
        interval_minutes=INTERVAL_MINUTES,
        run_label=f"{hh.name}_{hh.paket_id}",
        round_like_invoice=round_like_invoice,
        obracunaj_presezno_moc=hh.obracunaj_presezno_moc,
        strogo=False,
        on_month=on_month,
    )

    wanted = set(months)
    spans: Dict[Tuple[int, int], Tuple[int, dt.date, dt.date]] = {}
    for key, month_df in _months(df):
        if key not in wanted:
            continue
        spans[key] = (len(month_df), month_df["local_start"].min().date(),
                      month_df["local_start"].max().date())
        for row in month_df.itertuples():
            builder.add_interval(calculate_interval_price(
                0.0,  # market price: unused by a TARIFNI (VT/MT/ET) price list
                float(row.kwh),
                row.utc_start.to_pydatetime(),
                INTERVAL_MINUTES,
                scheme=hh.scheme,
                paket_id=hh.paket_id,
                total_produced_kwh=(
                    float(row.oddano_kwh) if hh.scheme == SCHEME_SI_SAMOOSKRBA else None
                ),
                dogovorjena_moc=schedule[key],
                eko_racun=hh.eko_racun,
                meritve_15min=True,
                include_raw=True,
            ))
    builder.finalize()
    for note in notes:
        say(f"  note          {note}")

    summary = [
        _month_summary(hh, key, builder.by_month[key], schedule[key],
                       *spans[key], _intervals_in_month(key))
        for key in builder.months
    ]
    return InvoiceResult(
        household=hh,
        paket_ime=f"{paket.dobavitelj} – {paket.ime}",
        profile=df,
        schedule=schedule,
        agreed=agreed,
        builder=builder,
        summary=summary,
    )


def _month_summary(hh, key, rows, moc, seen, od, do, expected) -> Dict:
    prevzeto = sum_amount(rows, ("energija",), enota="kWh")
    za_placilo = invoice_total(rows)
    return {
        "gospodinjstvo": hh.name,
        "paket": hh.paket_id,
        "shema": hh.scheme,
        "leto": key[0],
        "mesec": key[1],
        "obdobje": f"{key[1]:02d}/{key[0]}",
        "od": od,
        "do": do,
        "intervalov": seen,
        "polnih_intervalov": expected,
        "popolno": seen == expected,
        "prevzeto_kwh": round(prevzeto, 2),
        "dogovorjena_moc_kw": {b: round(v, 1) for b, v in sorted(moc.items())},
        "energija_eur": round(sum_amount(rows, ("energija",)), 2),
        "omreznina_eur": round(sum_amount(rows, ("omreznina",)), 2),
        "ddv_eur": round(sum_amount(rows, ("ddv",)), 2),
        "za_placilo_eur": round(za_placilo, 2),
        "povprecna_cena_eur_kwh": (
            round(za_placilo / prevzeto, 4) if prevzeto else float("nan")
        ),
    }


def _months(df: pd.DataFrame):
    for (leto, mesec), month_df in df.groupby(["leto", "mesec"], sort=True):
        yield (int(leto), int(mesec)), month_df


def _intervals_in_month(key: Tuple[int, int]) -> int:
    """15-minute intervals in one Ljubljana calendar month (DST-aware)."""
    start = pd.Timestamp(dt.datetime(key[0], key[1], 1), tz=TZ_SI)
    return int(((start + pd.offsets.MonthBegin(1)) - start).total_seconds()
               // (INTERVAL_MINUTES * 60))


# ---------------------------------------------------------------------------
# Catalogue helpers
# ---------------------------------------------------------------------------
def household_names() -> List[str]:
    return [h.name for h in HOUSEHOLDS]


def get_household(name: str) -> Household:
    for h in HOUSEHOLDS:
        if h.name.lower() == str(name).lower():
            return h
    raise KeyError(f"Unknown household {name!r}. Available: {household_names()}")


def unconfigured_folders() -> List[str]:
    """Profile folders on disk that no `Household` covers yet — dropping an
    export in is not enough, it also needs a price list only the user knows."""
    if not PROFILE_DIR.exists():
        return []
    configured = {h.folder.lower() for h in HOUSEHOLDS}
    return sorted(p.name for p in PROFILE_DIR.iterdir()
                  if p.is_dir() and p.name.lower() not in configured and any(p.glob("*.csv")))


def validate_agreed_power(name_or_df, hh: Optional[Household] = None, *, say=print):
    """Our reconstruction beside the figure the operator actually set.

    The export's own `Dogovorjena moč` column is the only ground truth available
    without the paper bill.
    """
    if isinstance(name_or_df, str):
        hh = hh or get_household(name_or_df)
        df = load_profile(hh.folder)
    else:
        df, hh = name_or_df, hh or HOUSEHOLDS[0]
    reported = _dso_by_year(df)
    rows = []
    for leto in sorted({int(y) for y in df["leto"]}):
        konice, mesecev = konice_v_oknu(df, leto)
        derived = uskladi_bloke(
            {b: zaokrozi_moc(v) for b, v in konice.items()},
            minimalna_moc_kw=(
                minimalna_dogovorjena_moc(hh.prikljucna_moc_kw, faze=hh.faze, leto=leto)
                if hh.prikljucna_moc_kw else 0.0),
            prikljucna_moc_kw=hh.prikljucna_moc_kw,
        ) if konice else {}
        rep = reported.get(leto, {})
        shared = sorted(set(derived) & set(rep))
        rows.append({
            "leto": leto,
            "mesecev_v_oknu": mesecev,
            "izracunano": {b: derived[b] for b in sorted(derived)} or None,
            "poroca_operater": {b: rep[b] for b in sorted(rep)} or None,
            "ujemanje": (all(abs(derived[b] - rep[b]) < 5e-2 for b in shared)
                         if shared else None),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_month(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    year, month = text.split("-")
    return int(year), int(month)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="month_from", metavar="YYYY-MM")
    ap.add_argument("--to", dest="month_to", metavar="YYYY-MM")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="restrict to one household (repeatable)")
    ap.add_argument("--complete-months-only", action="store_true")
    ap.add_argument("--round-like-invoice", action="store_true",
                    help="state whole kWh and one-decimal kW, and re-derive every "
                         "amount from them, the way a printed bill does")
    ap.add_argument("--verify-blocks", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    summaries: List[Dict] = []
    for hh in [h for h in HOUSEHOLDS if not args.only or h.name in args.only]:
        result = invoice_household(
            hh,
            month_from=_parse_month(args.month_from),
            month_to=_parse_month(args.month_to),
            complete_months_only=args.complete_months_only,
            round_like_invoice=args.round_like_invoice,
            check_blocks=args.verify_blocks,
        )
        label = f"{hh.name}_{hh.paket_id}"
        write_rows_csv(result.rows, out_dir / f"{label}_monthly.csv")
        write_rows_csv(result.period(), out_dir / f"{label}_period.csv")
        summaries.extend(result.summary)
        print(f"  {'obdobje':>9}  {'kWh':>8}  {'za plačilo':>11}")
        for s in result.summary:
            flag = "" if s["popolno"] else "  (delni mesec)"
            print(f"  {s['obdobje']:>9}  {s['prevzeto_kwh']:>8.1f}  "
                  f"{s['za_placilo_eur']:>11.2f}{flag}")
        print(f"  {'skupaj':>9}  {sum(s['prevzeto_kwh'] for s in result.summary):>8.1f}"
              f"  {sum(s['za_placilo_eur'] for s in result.summary):>11.2f}")
        print(f"  -> {out_dir / (label + '_monthly.csv')}")

    if summaries:
        write_rows_csv(summaries, out_dir / "poraba_doma_summary.csv")
        print(f"\n-> {out_dir / 'poraba_doma_summary.csv'}")


if __name__ == "__main__":
    main()
