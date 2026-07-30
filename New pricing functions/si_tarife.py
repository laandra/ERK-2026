"""
si_tarife.py — konstante slovenskega obračuna električne energije (gospodinjstva).

VSI ZNESKI SO BREZ DDV, razen kjer je izrecno navedeno.

VIRI (preverjeno 22. 7. 2026):
  [1] Agencija za energijo — Obrazložitev računa
      https://www.agen-rs.si/gospodinjski/elektrika/obrazlozitev-racuna
      => struktura računa: energija + omrežnina (moč + energija) + prispevki
         (OVE+SPTE na kW obračunske moči; URE in operater trga na kWh) +
         trošarina na kWh + DDV 22 % na neto seštevek vseh postavk.
  [2] Akt o določitvi tarifnih postavk za omrežnine elektrooperaterjev
      (Ur. l. RS 97/24, 82/25) — obdobje 1. 1. 2025 – 28. 2. 2026
      https://www.agen-rs.si/documents/10926/32579/Tarifne+postavke+omre%C5%BEnine+za+leto+2025_20251209/53b466f6-736f-4f94-9d4c-abf4c7454021
  [3] Akt o določitvi tarifnih postavk za omrežnine elektrooperaterjev
      (Ur. l. RS 105/25) — obdobje 1. 3. 2026 – 31. 12. 2026
      https://www.agen-rs.si/documents/10926/32579/Tarifne+postavke+omre%C5%BEnine+za+leto+2026/e6d62320-5dbd-4f74-a5db-a260d6defacb
  [4] Akt o metodologiji za obračunavanje omrežnine za elektrooperaterje
      (Ur. l. RS 146/22 in spremembe … 27/25, 76/25) — časovni bloki, presežna moč
      https://pisrs.si/pregledPredpisa?id=AKT_1266
  [5] Portal Energetika — prispevek OVE+SPTE (časovnica za gospodinjstva)
      https://www.energetika-portal.si/podrocja/energetika/prispevek-za-obnovljive-vire/
  [6] Portal Energetika — prispevek URE = 0,080 €c/kWh
      https://www.energetika-portal.si/podrocja/energetika/prihranki-energije/prispevek-za-energetsko-ucinkovitost/
  [7] Borzen — prispevek za delovanje operaterja trga 0,00013 EUR/kWh
      https://www.borzen.si/Portals/0/SL/Splo%C5%A1no/Financiranje%20GJS.pdf
  [8] Portal Energetika — struktura cene Q1 2026: trošarina 1,53 EUR/MWh (znižana)
      https://www.energetika-portal.si/nc/novica/n/cene-elektricne-energije-v-prvem-cetrtletju-2026/
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# DAJATVE IN DAVKI
# ---------------------------------------------------------------------------
DDV = 0.22                      # [1] 22 % na neto seštevek vseh postavk
TROSARINA_EUR_KWH = 0.00153     # [8] znižana (polna bi bila 0,00305)
URE_EUR_KWH = 0.00080           # [6] 0,080 €c/kWh
OPERATER_TRGA_EUR_KWH = 0.00013 # [7] Borzen

# [5] OVE+SPTE za gospodinjske odjemalce na NN 400/230 V — EUR/kW obračunske moči
#     na mesec. Časovnica velja od datuma naprej.
OVE_SPTE_CASOVNICA = [
    (dt.date(2023, 11, 1), 0.00000),   # popolna oprostitev
    (dt.date(2025,  7, 1), 0.77562),   # ponovna uvedba
    (dt.date(2025, 11, 1), 0.38781),   # 50 % znižanje do konca feb. 2026
    (dt.date(2026,  3, 1), 0.77562),   # polni prispevek
]


def ove_spte_eur_kw(d: dt.date) -> float:
    """Prispevek OVE+SPTE [EUR/kW/mesec] za gospodinjstvo na dani datum. [5]"""
    vrednost = 0.0
    for od, v in OVE_SPTE_CASOVNICA:
        if d >= od:
            vrednost = v
    return vrednost


# ---------------------------------------------------------------------------
# TARIFNE POSTAVKE OMREŽNINE — uporabniška skupina 0
# (uporabniki priključeni na NN izvod nazivne napetosti 400/230 V = gospodinjstva)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Omreznina:
    """Tarifne postavke omrežnine za eno regulativno obdobje."""
    velja_od: dt.date
    velja_do: dt.date
    vir: str

    # skupaj = prenos + distribucija
    moc: Dict[int, float]                 # EUR/kW/mesec, po časovnih blokih
    moc_znizana_vs: Dict[int, float]      # EUR/kW/mesec v višji sezoni (ukrep postopnosti)
    energija: Dict[int, float]            # EUR/kWh, po časovnih blokih

    # ločeni komponenti (potrebni za obračun deljene energije v skupnosti)
    energija_prenos: Dict[int, float]
    energija_distribucija: Dict[int, float]

    # distribucijska postavka za energijo, deljeno med člani skupnosti,
    # po "značilnem primeru priključitve" 1–10 (glej opombo spodaj)
    energija_skupnost: Dict[int, Dict[int, float]]

    # obračun brez 15-minutnih meritev (starejši števci): VT/MT/ET namesto blokov
    brez_15min_moc: float = 0.0
    brez_15min_moc_znizana: float = 0.0
    brez_15min_vt: float = 0.0
    brez_15min_mt: float = 0.0
    brez_15min_et: float = 0.0

    # faktor utežitve presežne moči (Akt, prehodne določbe) [4]
    faktor_presezne_moci: float = 0.9

    def postavka_moc(self, blok: int, visja_sezona: bool) -> float:
        if visja_sezona and blok in self.moc_znizana_vs:
            return self.moc_znizana_vs[blok]
        return self.moc[blok]


# --- obdobje 1. 1. 2025 – 28. 2. 2026 -------------------------------------- [2]
OMREZNINA_2025 = Omreznina(
    velja_od=dt.date(2025, 1, 1),
    velja_do=dt.date(2026, 2, 28),
    vir="Ur. l. RS 97/24, 82/25",
    moc={1: 3.42250, 2: 0.91224, 3: 0.16297, 4: 0.00407, 5: 0.00000},
    moc_znizana_vs={1: 1.71126},           # VS 2025/26 = nov 25, dec 25, jan 26, feb 26
    energija={1: 0.01998, 2: 0.01833, 3: 0.01809, 4: 0.01855, 5: 0.01873},
    energija_prenos={1: 0.00662, 2: 0.00592, 3: 0.00553, 4: 0.00584, 5: 0.00593},
    energija_distribucija={1: 0.01336, 2: 0.01241, 3: 0.01256, 4: 0.01271, 5: 0.01280},
    energija_skupnost={
        1:  {1: 0.00000, 2: 0.00000, 3: 0.00000, 4: 0.00000, 5: 0.00000},
        2:  {1: 0.00550, 2: 0.00550, 3: 0.00550, 4: 0.00550, 5: 0.00550},
        3:  {1: 0.01220, 2: 0.01137, 3: 0.01154, 4: 0.01164, 5: 0.01174},
        4:  {1: 0.01336, 2: 0.01241, 3: 0.01256, 4: 0.01271, 5: 0.01280},
        5:  {1: 0.00707, 2: 0.00664, 3: 0.00683, 4: 0.00663, 5: 0.00672},
        6:  {1: 0.00829, 2: 0.00782, 3: 0.00798, 4: 0.00778, 5: 0.00786},
        7:  {1: 0.00480, 2: 0.00443, 3: 0.00457, 4: 0.00435, 5: 0.00441},
        8:  {1: 0.00598, 2: 0.00558, 3: 0.00570, 4: 0.00547, 5: 0.00552},
        9:  {1: 0.00116, 2: 0.00115, 3: 0.00114, 4: 0.00112, 5: 0.00111},
        10: {1: 0.00030, 2: 0.00030, 3: 0.00030, 4: 0.00030, 5: 0.00030},
    },
    brez_15min_moc=2.22012, brez_15min_moc_znizana=1.55409,
    brez_15min_vt=0.01875, brez_15min_mt=0.01836, brez_15min_et=0.01856,
    faktor_presezne_moci=0.9,
)

# --- obdobje 1. 3. 2026 – 31. 12. 2026 ------------------------------------- [3]
OMREZNINA_2026 = Omreznina(
    velja_od=dt.date(2026, 3, 1),
    velja_do=dt.date(2026, 12, 31),
    vir="Ur. l. RS 105/25",
    moc={1: 3.82301, 2: 1.09230, 3: 0.28902, 4: 0.02436, 5: 0.00245},
    moc_znizana_vs={1: 2.67611},           # VS 2026 = nov 2026, dec 2026 (70 %)
    energija={1: 0.02217, 2: 0.01998, 3: 0.01717, 4: 0.01805, 5: 0.01299},
    energija_prenos={1: 0.00774, 2: 0.00671, 3: 0.00551, 4: 0.00602, 5: 0.00440},
    energija_distribucija={1: 0.01443, 2: 0.01327, 3: 0.01166, 4: 0.01203, 5: 0.00859},
    energija_skupnost={
        1:  {1: 0.00000, 2: 0.00000, 3: 0.00000, 4: 0.00000, 5: 0.00000},
        2:  {1: 0.00547, 2: 0.00563, 3: 0.00482, 4: 0.00473, 5: 0.00330},
        3:  {1: 0.01307, 2: 0.01207, 3: 0.01065, 4: 0.01096, 5: 0.00783},
        4:  {1: 0.01456, 2: 0.01336, 3: 0.01176, 4: 0.01218, 5: 0.00870},
        5:  {1: 0.00627, 2: 0.00676, 3: 0.00578, 4: 0.00570, 5: 0.00405},
        6:  {1: 0.00750, 2: 0.00811, 3: 0.00688, 4: 0.00681, 5: 0.00483},
        7:  {1: 0.00424, 2: 0.00458, 3: 0.00389, 4: 0.00395, 5: 0.00259},
        8:  {1: 0.00547, 2: 0.00598, 3: 0.00501, 4: 0.00513, 5: 0.00335},
        9:  {1: 0.00115, 2: 0.00128, 3: 0.00118, 4: 0.00122, 5: 0.00102},
        10: {1: 0.00034, 2: 0.00179, 3: 0.00018, 4: 0.00046, 5: 0.00003},
    },
    brez_15min_moc=2.68165, brez_15min_moc_znizana=2.14532,
    brez_15min_vt=0.02004, brez_15min_mt=0.01741, brez_15min_et=0.01864,
    faktor_presezne_moci=1.05,
)

OMREZNINA_OBDOBJA = [OMREZNINA_2025, OMREZNINA_2026]

# Privzeti regulativni režim za podatke, ki nosijo datume izven objavljenih
# obdobij (npr. Ausgrid dataseti 2010–2013): obračuna se po letu 2026.
PRIVZETO_REFERENCNO_LETO = 2026
PRIVZETA_OMREZNINA = OMREZNINA_2026


def ima_tarifne_postavke(d: dt.date) -> bool:
    """True, če so za dani datum objavljene tarifne postavke omrežnine."""
    return any(o.velja_od <= d <= o.velja_do for o in OMREZNINA_OBDOBJA)


def omreznina_za_datum(d: dt.date) -> Omreznina:
    """Izbere veljavne tarifne postavke omrežnine za dani datum.

    Strogo: za datume izven objavljenih obdobij sproži ValueError. Kdor želi
    star dataset obračunati po danes veljavnih postavkah, naj uporabi
    `si_obracun.Pravila.za_leto(...)` / `Pravila.privzeta(...)`, ki padeta
    nazaj na `PRIVZETO_REFERENCNO_LETO`.
    """
    for o in OMREZNINA_OBDOBJA:
        if o.velja_od <= d <= o.velja_do:
            return o
    raise ValueError(
        f"Za {d} ni naloženih tarifnih postavk. Dodaj nov Omreznina objekt "
        f"iz Akta o določitvi tarifnih postavk (agen-rs.si)."
    )


# ---------------------------------------------------------------------------
# CENIKI DOBAVITELJEV (energija). Ceniki veljajo "do preklica" — preveri datum!
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cenik:
    ime: str
    vt: float = 0.0                 # EUR/kWh
    mt: float = 0.0
    et: float = 0.0
    mesecno_nadomestilo: float = 0.0  # EUR/mesec (fiksno)
    dinamicni_pribitek: float = 0.0   # EUR/kWh nad borzno ceno
    cap_eur_kwh: Optional[float] = None  # navzgor zamejena borzna cena
    vir: str = ""


CENIKI = {
    "GEN-I": Cenik("GEN-I redni", vt=0.11990, mt=0.09790, et=0.10890,
                   mesecno_nadomestilo=1.99,
                   vir="gen-i.si, redni cenik od 1. 3. 2025"),
    "GEN-I-EKO": Cenik("GEN-I redni + EKO popust", vt=0.11990, mt=0.09790, et=0.10890,
                       mesecno_nadomestilo=0.99,
                       vir="gen-i.si, redni cenik od 1. 3. 2025"),
    "PETROL": Cenik("Petrol redni", vt=0.12795, mt=0.10795, et=0.11795,
                    mesecno_nadomestilo=1.98,
                    vir="petrol.si, cenik marec 2025"),
    "ECE-FLEKS": Cenik("ECE FLEKS (dinamični)", dinamicni_pribitek=0.01400,
                       mesecno_nadomestilo=0.0,
                       vir="ece.si, pribitek 14,00 EUR/MWh brez DDV"),
}
