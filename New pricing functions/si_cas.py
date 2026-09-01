"""
si_cas.py — koledar, sezone, časovni bloki in tarifna okna.

Prazniki: knjižnica `holidays` (pip install holidays), kategorija 'public'.
POZOR: SI ima tudi kategorijo 'workday' (dan Primoža Trubarja, 17. 8., 15. 9.,
23. 9., 25. 10., 23. 11.) — to so prazniki, ki NISO dela prosti, zato se za
omrežnino štejejo kot navadni delovni dnevi. Privzeta kategorija 'public'
vrne natanko dela proste dneve, zato je pravilna.

VIRI:
  [4] Akt o metodologiji za obračunavanje omrežnine za elektrooperaterje
      (Ur. l. RS 146/22 … 27/25, 76/25) — https://pisrs.si/pregledPredpisa?id=AKT_1266
  [9] Agencija za energijo / uro.si — razpored časovnih blokov
      https://www.uro.si/prenova-omre%C5%BEnine/novi-%C4%8Dasovni-bloki
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Dict, Optional, Tuple

import holidays

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:                                    # Python < 3.9
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore


class _EuropeLjubljanaFallback(dt.tzinfo):
    """Fallback tzinfo for Europe/Ljubljana when system tzdata is unavailable."""

    _STD_OFFSET = dt.timedelta(hours=1)
    _DST_OFFSET = dt.timedelta(hours=2)

    @staticmethod
    def _last_sunday(leto: int, mesec: int) -> dt.date:
        if mesec == 12:
            naslednji = dt.date(leto + 1, 1, 1)
        else:
            naslednji = dt.date(leto, mesec + 1, 1)
        d = naslednji - dt.timedelta(days=1)
        return d - dt.timedelta(days=(d.weekday() + 1) % 7)

    @classmethod
    def _dst_bounds_utc(cls, leto: int) -> tuple[dt.datetime, dt.datetime]:
        zacetek = dt.datetime.combine(cls._last_sunday(leto, 3), dt.time(1), tzinfo=dt.timezone.utc)
        konec = dt.datetime.combine(cls._last_sunday(leto, 10), dt.time(1), tzinfo=dt.timezone.utc)
        return zacetek, konec

    @classmethod
    def _is_dst_utc(cls, utc_dt: dt.datetime) -> bool:
        zacetek, konec = cls._dst_bounds_utc(utc_dt.year)
        return zacetek <= utc_dt < konec

    def fromutc(self, utc_dt: dt.datetime) -> dt.datetime:
        if utc_dt.tzinfo is not self:
            raise ValueError("fromutc: tzinfo mismatch")
        utc_naive = utc_dt.replace(tzinfo=dt.timezone.utc)
        offset = self._DST_OFFSET if self._is_dst_utc(utc_naive) else self._STD_OFFSET
        return (utc_naive + offset).replace(tzinfo=self)

    def utcoffset(self, lokalni_dt: dt.datetime | None) -> dt.timedelta:
        return self._STD_OFFSET + self.dst(lokalni_dt)

    def dst(self, lokalni_dt: dt.datetime | None) -> dt.timedelta:
        if lokalni_dt is None:
            return dt.timedelta(0)
        leto = lokalni_dt.year
        zacetek = dt.datetime.combine(self._last_sunday(leto, 3), dt.time(2))
        konec = dt.datetime.combine(self._last_sunday(leto, 10), dt.time(3))
        lokalni_naive = lokalni_dt.replace(tzinfo=None)
        return dt.timedelta(hours=1) if zacetek <= lokalni_naive < konec else dt.timedelta(0)

    def tzname(self, lokalni_dt: dt.datetime | None) -> str:
        return "CEST" if self.dst(lokalni_dt) else "CET"


try:
    TZ_SI = ZoneInfo("Europe/Ljubljana")
except ZoneInfoNotFoundError:
    TZ_SI = _EuropeLjubljanaFallback()

TZ_UTC = dt.timezone.utc

# Privzeto: Slovenija, severna polobla. Blokovni razporedi so definirani glede
# na "delovni / dela prost dan" in "visja / nizka sezona", ne glede na koledar
# dolocene drzave -- zato je mogoce isto tarifno strukturo uporabiti na profilu
# iz druge drzave ali z druge poloble, ce se koledar premakne z `nastavi_koledar`.
#
# Configurable because the tariff structure is portable but the calendar is not.
# Applying SI public holidays to an Australian load profile marks the wrong days
# as non-working (only 5 of 11 NSW and 14 SI dates coincide in 2013), and the
# northern high season puts the winter network peak on the southern summer.
DRZAVA_PRAZNIKOV = "SI"
PODROCJE_PRAZNIKOV = None                              # npr. "NSW"
VISJA_SEZONA_MESECI = frozenset({11, 12, 1, 2})        # nov, dec, jan, feb


def nastavi_koledar(drzava=None, podrocje=None, visja_sezona_meseci=None):
    """Premakni koledar na drugo drzavo / poloblo. Vrne prejsnjo nastavitev.

    Returns the previous (country, subdivision, months) so a caller can restore
    it -- these are module-level, so a study that changes them changes them for
    everything in the process.
    """
    global DRZAVA_PRAZNIKOV, PODROCJE_PRAZNIKOV, VISJA_SEZONA_MESECI
    prej = (DRZAVA_PRAZNIKOV, PODROCJE_PRAZNIKOV, VISJA_SEZONA_MESECI)
    if drzava is not None:
        DRZAVA_PRAZNIKOV, PODROCJE_PRAZNIKOV = drzava, podrocje
    elif podrocje is not None:
        PODROCJE_PRAZNIKOV = podrocje
    if visja_sezona_meseci is not None:
        VISJA_SEZONA_MESECI = frozenset(visja_sezona_meseci)
    return prej


@lru_cache(maxsize=None)
def _prazniki(leto: int, drzava: str = "SI", podrocje=None) -> frozenset:
    """Dela prosti prazniki (kategorija 'public') za dano drzavo/podrocje."""
    return frozenset(
        holidays.country_holidays(drzava, subdiv=podrocje, years=leto).keys()
    )


def je_dela_prost(d: dt.date) -> bool:
    return d.weekday() >= 5 or d in _prazniki(
        d.year, DRZAVA_PRAZNIKOV, PODROCJE_PRAZNIKOV
    )


def je_visja_sezona(d: dt.date) -> bool:
    return d.month in VISJA_SEZONA_MESECI


def v_lokalni_cas(utc_date: dt.datetime) -> dt.datetime:
    """Naive datetime se obravnava kot UTC; vrne čas v Europe/Ljubljana."""
    if utc_date.tzinfo is None:
        utc_date = utc_date.replace(tzinfo=TZ_UTC)
    return utc_date.astimezone(TZ_SI)


# ---------------------------------------------------------------------------
# RAZPORED ČASOVNIH BLOKOV
# ---------------------------------------------------------------------------
# Vzorec ur: [00-06, 06-07, 07-14, 14-16, 16-20, 20-22, 22-24]
_MEJE = (0, 6, 7, 14, 16, 20, 22, 24)


def _razsiri(vzorec: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(
        vzorec[i]
        for i in range(7)
        for _ in range(_MEJE[i + 1] - _MEJE[i])
    )


# --- razpored, veljaven do 31. 12. 2026 --------------------------------- [4][9]
RAZPORED_2024 = {
    ("visja", "delovni"): _razsiri((3, 2, 1, 2, 1, 2, 3)),
    ("visja", "prost"):   _razsiri((4, 3, 2, 3, 2, 3, 4)),
    ("nizka", "delovni"): _razsiri((4, 3, 2, 3, 2, 3, 4)),
    ("nizka", "prost"):   _razsiri((5, 4, 3, 4, 3, 4, 5)),
}

# --- razpored od 1. 1. 2027 --------------------------------------------------
# POTRJENO iz Akta (Ur. l. RS 76/25) je le, KATERI bloki nastopijo:
#   visja/delovni -> {1,2,3};  visja/prost -> {3,4}
#   nizka/delovni -> {3,4,5};  nizka/prost -> {4,5}
# (vira: agencija prek gen-i.si/podpora/elektricna-energija/omreznina/ in
#  rtvslo.si/.../762383 — "med dela prostimi dnevi bosta veljala le dva bloka")
#
# !!! URNE MEJE ZA NIŽJO SEZONO SO TU OCENA, NE CITAT. !!!
# Agencija napoveduje, da bo v nižji sezoni razpored "z drugačnimi urami, ki
# bolje sledijo proizvodnji iz sončnih elektrarn". Točnih ur v aktu nisem
# potrdil. Preveri Akt in po potrebi popravi ta slovar — zato je izpostavljen
# kot spremenljivka in ne zakodiran globlje.
RAZPORED_2027 = {
    ("visja", "delovni"): _razsiri((3, 2, 1, 2, 1, 2, 3)),   # potrjeno {1,2,3}
    ("visja", "prost"):   _razsiri((4, 4, 3, 4, 3, 4, 4)),   # potrjeno {3,4}
    ("nizka", "delovni"): _razsiri((5, 4, 3, 4, 3, 4, 5)),   # potrjeno {3,4,5}
    ("nizka", "prost"):   _razsiri((5, 5, 4, 5, 4, 5, 5)),   # potrjeno {4,5}
}
RAZPORED_2027_JE_OCENA = True   # zastavica: urne meje niso potrjene iz akta

RAZPOREDI = {
    "2024": RAZPORED_2024,
    "2027": RAZPORED_2027,
}


def razpored_za_datum(d: dt.date) -> str:
    return "2027" if d >= dt.date(2027, 1, 1) else "2024"


def casovni_blok(lokalni_cas: dt.datetime, razpored: str = "2024") -> int:
    d = lokalni_cas.date()
    sezona = "visja" if je_visja_sezona(d) else "nizka"
    dan = "prost" if je_dela_prost(d) else "delovni"
    return RAZPOREDI[razpored][(sezona, dan)][lokalni_cas.hour]


def bloki_v_mesecu(leto: int, mesec: int, razpored: str = "2024") -> set:
    """Bloki, ki se v mesecu sploh pojavijo — omrežnina za moč se plača za vsakega."""
    tabela = RAZPOREDI[razpored]
    d = dt.date(leto, mesec, 1)
    bloki = set()
    while d.month == mesec:
        sezona = "visja" if je_visja_sezona(d) else "nizka"
        dan = "prost" if je_dela_prost(d) else "delovni"
        bloki.update(tabela[(sezona, dan)])
        d += dt.timedelta(days=1)
    return bloki


# ---------------------------------------------------------------------------
# TARIFNA OKNA DOBAVITELJEV (ne omrežnine!)
# ---------------------------------------------------------------------------
def je_vt(lokalni_cas: dt.datetime) -> bool:
    """
    VT: delovni dnevi 06:00–22:00. MT: delovniki 22:00–06:00 + dela prosti dnevi.
    Vir: gen-i.si, petrol.si (oba enako; definirano v splošnem aktu Agencije).
    """
    if je_dela_prost(lokalni_cas.date()):
        return False
    return 6 <= lokalni_cas.hour < 22


def aktivna_tarifa(lokalni_cas: dt.datetime) -> str:
    """
    Okno 4-tarifnih "aktivnih" cenikov (GEN-I Aktivni / Fiksni samooskrbe).
    Enako za VSE dni, tudi vikende in praznike.
      sončna  09:00–16:00
      osnovna 21:00–09:00 in 16:00–18:00
      konična 18:00–21:00
    Sezoni: nižja = marec–oktober, višja = november–februar.
    Vir: gen-i.si/dom/elektricna-energija/ceniki-in-akcije/aktivni-cenik-elektrike-za-dom/
    """
    h = lokalni_cas.hour
    if 9 <= h < 16:
        return "soncna_vs" if je_visja_sezona(lokalni_cas.date()) else "soncna_ns"
    if 18 <= h < 21:
        return "konicna"
    return "osnovna"
