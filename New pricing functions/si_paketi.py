"""
si_paketi.py — katalog rednih paketov slovenskih dobaviteljev + validacija
združljivosti paketa z gospodinjstvom.

VSE CENE SO BREZ DDV, v EUR/kWh oz. EUR/mesec.

KLJUČNO PRAVILO O DDV (popravek prejšnje verzije):
  "Odkup presežka proizvedene električne energije v napravi za samooskrbo
   ni predmet obdavčitve z DDV."
  — gen-i.si (vsi samooskrbni ceniki), bisol-energija.si
  Dobropis za oddajo se torej NE vključi v osnovo za DDV; odšteje se od
  računa Z DDV.

VIRI (preverjeno 22. 7. 2026):
 [G1] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-elektricne-energije-za-gospodinjske-odjemalce/
 [G2] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-fiksni-gospodinjski-odjemalci/
 [G3] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/aktivni-cenik-elektrike-za-dom/
 [G4] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-dinamicni-gospodinjski-odjemalci/
 [G5] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-za-prevzem-in-oddajo-elektricne-energije-pri-samooskrbi-gospodinjskih-odjemalcev/
 [G6] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/aktivni-cenik-samooskrbe-za-dom-in-skupnosti/
 [G7] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/fiksni-cenik-samooskrbe-za-dom-in-skupnosti/
 [G8] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-dinamicni-samooskrba-za-gospodinjske-odjemalce/
 [G9] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-elektricne-energije-za-samooskrbo-gospodinjskih-odjemalcev/
 [B1] https://www.bisol-energija.si/fiksni-gospodinjstva
 [B2] https://www.bisol-energija.si/dinamicni-gospodinjstva
 [B3] https://www.bisol-energija.si/samooskrba/dinamicna
 [P1] https://www.petrol.si/binaries/content/assets/www/2025/dokumenti-in-obrazci/ee/go/cenik_go-odjem_marec-2025_f1.pdf
 [P2] https://www.petrol.si/binaries/content/assets/www/2026/dokumenti/ee/akcijski-cenik-elektricne-energije-za-gospodinjske-odjemalce-fiks-2026-9.-05.-2026.pdf
 [P3] https://www.petrol.si/za-dom/energenti/samooskrba
 [E1] https://www.elektro-energija.si/za-dom/dokumenti-in-ceniki
      (redni cenik ZANESLJIVA OSKRBA, DINAMIČNA OSKRBA, ZANESLJIVA OSKRBA –
       FIKSNI, E-popust, dodatek za izbiro vira energije) — preverjeno 29. 7. 2026
 [E2] https://www.elektro-energija.si/pomoc/souporaba-energije
      (Elektro energija v ponudbi NIMA rešitev samooskrbe; njeni odjemalci se
       v souporabo lahko vključijo LE kot prejemniki)
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TipCene(str, Enum):
    TARIFNI = "tarifni"          # VT/MT ali ET
    AKTIVNI = "aktivni"          # 4 tarife: sončna NS/VS, osnovna, konična
    DINAMICNI = "dinamicni"      # SIPX + pribitek


class TipOdkupa(str, Enum):
    NI = "ni"                    # dobavitelj presežkov ne odkupuje
    FIKSNI = "fiksni"            # ena cena EUR/kWh
    AKTIVNI = "aktivni"          # 4 tarife
    DINAMICNI = "dinamicni"      # SIPX − pribitek
    NET_METERING = "net_metering"  # letno netiranje, brez sprotnega dobropisa


class Shema(str, Enum):
    """Shema samooskrbe, v katero je merilno mesto vključeno."""
    BREZ = "brez"                    # ni naprave za samooskrbo
    NOVA = "nova"                    # soglasje po 1. 1. 2024, 15-min obračun
    NET_METERING = "net_metering"    # soglasje do 31. 12. 2023, letno netiranje


class Vloga(str, Enum):
    """Vloga merilnega mesta v souporabi električne energije (ZOEE)."""
    NI = "ni"                        # ni vključeno v souporabo
    ODDAJNIK = "oddajnik"            # deli presežke
    PREJEMNIK = "prejemnik"          # prejema deljeno energijo
    OBOJE = "oboje"                  # isto MM je lahko oddajnik in prejemnik


class NezdruzljivPaket(ValueError):
    """Paket ni združljiv s konfiguracijo gospodinjstva."""


@dataclass(frozen=True)
class Gospodinjstvo:
    """Konfiguracija odjemnega mesta — podlaga za validacijo paketa."""
    ime: str = "gospodinjstvo"
    dogovorjena_moc: Dict[int, float] = field(default_factory=dict)  # {blok: kW}
    ima_pv: bool = False
    shema_samooskrbe: Shema = Shema.BREZ
    skupnostna: bool = False          # član skupnostne samooskrbe / skupnosti OVE
    meritve_15min: bool = True
    eko_racun: bool = True            # elektronski račun -> EKO popust
    znacilni_primer: int = 2          # 1–10, za omrežnino deljene energije
    vloga_souporaba: Vloga = Vloga.NI
    delez_souporabe: float = 0.0      # delež ODDANE energije, namenjen souporabi

    def __post_init__(self):
        if self.ima_pv and self.shema_samooskrbe is Shema.BREZ:
            raise ValueError(
                f"{self.ime}: ima_pv=True zahteva shema_samooskrbe NOVA ali NET_METERING."
            )
        if not self.ima_pv and self.shema_samooskrbe is not Shema.BREZ:
            raise ValueError(
                f"{self.ime}: shema_samooskrbe={self.shema_samooskrbe.value} "
                f"brez naprave za samooskrbo (ima_pv=False)."
            )
        if self.skupnostna and not self.ima_pv:
            raise ValueError(
                f"{self.ime}: skupnostna=True zahteva vključitev v samooskrbo."
            )
        # --- souporaba (ZOEE) ---
        if self.vloga_souporaba in (Vloga.ODDAJNIK, Vloga.OBOJE) and not self.ima_pv:
            raise ValueError(
                f"{self.ime}: oddajnik v souporabi mora imeti proizvodno napravo "
                f"na OVE (ima_pv=True)."
            )
        if (self.vloga_souporaba in (Vloga.PREJEMNIK, Vloga.OBOJE)
                and self.shema_samooskrbe is Shema.NET_METERING):
            raise ValueError(
                f"{self.ime}: odjemalec v stari shemi samooskrbe z letnim "
                f"netiranjem (NET metering) NE more biti prejemnik v souporabi. "
                f"Kot oddajnik lahko sodeluje."
            )
        if not (0.0 <= self.delez_souporabe <= 1.0):
            raise ValueError(
                f"{self.ime}: delez_souporabe mora biti med 0 in 1, "
                f"je {self.delez_souporabe}."
            )


@dataclass(frozen=True)
class Paket:
    """Cenik dobavitelja za gospodinjske odjemalce."""
    id: str
    dobavitelj: str
    ime: str
    vir: str
    velja_od: dt.date

    tip_cene: TipCene
    tip_odkupa: TipOdkupa = TipOdkupa.NI

    # --- prevzem: tarifni ---
    vt: float = 0.0
    mt: float = 0.0
    et: float = 0.0
    # --- prevzem: aktivni (4 tarife) ---
    soncna_ns: float = 0.0
    soncna_vs: float = 0.0
    osnovna: float = 0.0
    konicna: float = 0.0
    # --- prevzem: dinamični ---
    pribitek_odjem: float = 0.0        # EUR/kWh, prišteje se SIPX
    cap_sipx: Optional[float] = None   # zgornja meja URNEGA SIPX (EUR/kWh)
    cap_mesecni: Optional[float] = None  # zgornja meja MESEČNE povprečne cene

    # --- oddaja (odkup presežkov) ---
    odkup_fiksni: float = 0.0
    odkup_soncna_ns: float = 0.0
    odkup_soncna_vs: float = 0.0
    odkup_osnovna: float = 0.0
    odkup_konicna: float = 0.0
    pribitek_oddaja: float = 0.0       # EUR/kWh, ODŠTEJE se od SIPX

    # --- fiksne postavke ---
    mesecno_nadomestilo: float = 0.0
    mesecno_nadomestilo_eko: Optional[float] = None
    dodatna_storitev: float = 0.0      # npr. BISOL DINAMIČNI+ 1,63 EUR/mes

    # --- pravila združljivosti ---
    zahteva_pv: bool = False           # samo za lastnike naprave za samooskrbo
    dovoljuje_pv: bool = False         # sme ga imeti gospodinjstvo s PV
    dovoljene_sheme: tuple = (Shema.BREZ,)
    dovoljuje_skupnostno: bool = False
    zahteva_15min: bool = False
    # dobavitelj podpira vlogo ODDAJNIK v souporabi (potrebna lastna proizvodnja
    # in samooskrbni cenik); npr. Elektro energija samooskrbe ne ponuja [E2]
    dovoljuje_oddajnika: bool = True
    opombe: str = ""

    def nadomestilo(self, eko: bool) -> float:
        if eko and self.mesecno_nadomestilo_eko is not None:
            return self.mesecno_nadomestilo_eko + self.dodatna_storitev
        return self.mesecno_nadomestilo + self.dodatna_storitev


# ===========================================================================
# KATALOG
# ===========================================================================
PAKETI: Dict[str, Paket] = {}


def _reg(p: Paket) -> Paket:
    PAKETI[p.id] = p
    return p


# --------------------------------------------------------------- GEN-I: dobava
_reg(Paket(
    id="GENI_REDNI", dobavitelj="GEN-I", ime="Redni cenik za gospodinjske odjemalce",
    vir="[G1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11990, mt=0.09790, et=0.10890,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    opombe="Ne velja za oskrbo skupnih delov večstanovanjskih stavb.",
))

_reg(Paket(
    id="GENI_FIKSNI", dobavitelj="GEN-I", ime="GEN-I Fiksni – Gospodinjski odjemalci",
    vir="[G2]", velja_od=dt.date(2025, 10, 23), tip_cene=TipCene.TARIFNI,
    vt=0.12490, mt=0.10290, et=0.11390,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="GENI_AKTIVNI", dobavitelj="GEN-I", ime="Aktivni cenik elektrike za dom",
    vir="[G3]", velja_od=dt.date(2026, 2, 6), tip_cene=TipCene.AKTIVNI,
    soncna_ns=0.03490, soncna_vs=0.08990, osnovna=0.11790, konicna=0.17990,
    et=0.10890,   # nadomestna cena ob pomanjkljivih 15-min podatkih = ET rednega
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_15min=True,
))

_reg(Paket(
    id="GENI_DINAMICNI", dobavitelj="GEN-I", ime="GEN-I Dinamični – Gospodinjski odjemalci",
    vir="[G4]", velja_od=dt.date(2024, 10, 2), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01199, cap_sipx=0.22000,
    mesecno_nadomestilo=2.97, mesecno_nadomestilo_eko=1.97,
    opombe="Zamejitev 220 EUR/MWh velja na URNI SIPX. Navzdol neomejeno.",
))

# ----------------------------------------------------- GEN-I: samooskrba (nova)
_reg(Paket(
    id="GENI_SAMO_REDNI", dobavitelj="GEN-I",
    ime="Redni cenik za prevzem in oddajo pri samooskrbi",
    vir="[G5]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.FIKSNI,
    et=0.10290, odkup_fiksni=0.05390,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    opombe="Pristop od 5. 2. 2026 le v sklopu paketa Pametna samooskrba.",
))

_reg(Paket(
    id="GENI_SAMO_AKTIVNI", dobavitelj="GEN-I",
    ime="Aktivni cenik samooskrbe za dom in skupnosti",
    vir="[G6]", velja_od=dt.date(2026, 2, 5),
    tip_cene=TipCene.AKTIVNI, tip_odkupa=TipOdkupa.AKTIVNI,
    soncna_ns=0.04090, soncna_vs=0.11490, osnovna=0.12990, konicna=0.19290,
    et=0.14090,                      # nadomestna cena ob pomanjkljivih meritvah
    odkup_soncna_ns=0.00190, odkup_soncna_vs=0.06990,
    odkup_osnovna=0.07490, odkup_konicna=0.14990,
    odkup_fiksni=0.01490,            # nadomestna cena oddaje
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True,
    opombe="Ne velja za stranke paketa Pametna samooskrba.",
))

_reg(Paket(
    id="GENI_SAMO_FIKSNI", dobavitelj="GEN-I",
    ime="Fiksni cenik samooskrbe za dom in skupnosti",
    vir="[G7]", velja_od=dt.date(2026, 2, 17),
    tip_cene=TipCene.AKTIVNI, tip_odkupa=TipOdkupa.AKTIVNI,
    soncna_ns=0.04590, soncna_vs=0.11990, osnovna=0.13490, konicna=0.19790,
    et=0.14090,
    odkup_soncna_ns=0.00190, odkup_soncna_vs=0.06490,
    odkup_osnovna=0.06990, odkup_konicna=0.14490,
    odkup_fiksni=0.01490,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True, opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="GENI_SAMO_DINAMICNI", dobavitelj="GEN-I",
    ime="GEN-I Dinamični – Samooskrba za gospodinjske odjemalce",
    vir="[G8]", velja_od=dt.date(2024, 10, 2),
    tip_cene=TipCene.DINAMICNI, tip_odkupa=TipOdkupa.DINAMICNI,
    pribitek_odjem=0.01199, pribitek_oddaja=0.01199,
    mesecno_nadomestilo=2.97, mesecno_nadomestilo_eko=1.97,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,),
    dovoljuje_skupnostno=False,      # cenik izrecno IZKLJUČUJE skupnostno samooskrbo
    zahteva_15min=True,
    opombe="Cenik izrecno ne velja za skupnostno samooskrbo.",
))

# --------------------------------------------- GEN-I: samooskrba (NET metering)
_reg(Paket(
    id="GENI_NETMETERING", dobavitelj="GEN-I",
    ime="Redni cenik električne energije za samooskrbo (NET metering)",
    vir="[G9]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.NET_METERING,
    et=0.12990,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NET_METERING,), dovoljuje_skupnostno=True,
    opombe="Letno netiranje; presežek nad letno porabo se prenese brezplačno, "
           "prizna se ugodnost v višini enega mesečnega nadomestila na celo MWh.",
))

# ------------------------------------------------------------- BISOL: dobava
_reg(Paket(
    id="BISOL_FIKSNI", dobavitelj="BISOL Energija", ime="Paket FIKSNI",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.12700, mt=0.10700, et=0.11700, mesecno_nadomestilo=1.63,
    opombe="Brez vezave.",
))

_reg(Paket(
    id="BISOL_FIKSNI_VEZAVA", dobavitelj="BISOL Energija", ime="Paket FIKSNI z vezavo",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11600, mt=0.09600, et=0.10500, mesecno_nadomestilo=1.63,
    opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="BISOL_FIKSNI99", dobavitelj="BISOL Energija", ime="Paket FIKSNI99",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.10900, mt=0.09400, et=0.09900, mesecno_nadomestilo=1.63,
    opombe="Akcijski cenik, vezava do konca 2026.",
))

_reg(Paket(
    id="BISOL_DINAMICNI", dobavitelj="BISOL Energija", ime="Paket DINAMIČNI",
    vir="[B2]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01300, mesecno_nadomestilo=1.63,
    opombe="Brez vezave; navzdol neomejeno (tudi negativne cene).",
))

_reg(Paket(
    id="BISOL_DINAMICNI_PLUS", dobavitelj="BISOL Energija",
    ime="Paket DINAMIČNI + storitev DINAMIČNI+",
    vir="[B2]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01300, cap_mesecni=0.14700,
    mesecno_nadomestilo=1.63, dodatna_storitev=1.63,
    opombe="Zamejitev velja na MESEČNO povprečno obračunsko ceno SIPX "
           "(0,147 EUR/kWh brez pribitka), ne na urno ceno. 12-mesečna vezava.",
))

# --------------------------------------------------------- BISOL: samooskrba
_reg(Paket(
    id="BISOL_SAMO_DINAMICNA", dobavitelj="BISOL Energija", ime="DINAMIČNA Samooskrba",
    vir="[B3]", velja_od=dt.date(2026, 1, 1),
    tip_cene=TipCene.DINAMICNI, tip_odkupa=TipOdkupa.DINAMICNI,
    pribitek_odjem=0.01300, pribitek_oddaja=0.01300,
    mesecno_nadomestilo=1.63,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True,
    opombe="Simetričen razmik ±0,013 EUR/kWh okoli SIPX. Za soglasja po 1. 1. 2024. "
           "Za net metering le ob nadgradnji s hranilnikom.",
))

# ------------------------------------------------------------------- PETROL
_reg(Paket(
    id="PETROL_REDNI", dobavitelj="Petrol", ime="Redni cenik za gospodinjske odjemalce",
    vir="[P1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.12795, mt=0.10795, et=0.11795, mesecno_nadomestilo=1.98,
))

_reg(Paket(
    id="PETROL_AKCIJSKI", dobavitelj="Petrol", ime="Akcijski cenik (marec 2025)",
    vir="[P1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11995, mt=0.09995, et=0.10995, mesecno_nadomestilo=1.98,
))

_reg(Paket(
    id="PETROL_FIKS2026", dobavitelj="Petrol", ime="Akcijski cenik FIKS 2026",
    vir="[P2]", velja_od=dt.date(2026, 5, 9), tip_cene=TipCene.TARIFNI,
    vt=0.12395, mt=0.10195, et=0.11295, mesecno_nadomestilo=1.98,
    opombe="12 mesecev zagotovljene cene.",
))

_reg(Paket(
    id="PETROL_SAMOOSKRBA", dobavitelj="Petrol", ime="Samooskrba (brez odkupa presežkov)",
    vir="[P3]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.NI,
    vt=0.12795, mt=0.10795, et=0.11795, mesecno_nadomestilo=1.98,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA, Shema.NET_METERING), dovoljuje_skupnostno=True,
    opombe="Petrol viškov NE odkupuje in ne omogoča prenosa — oddana energija "
           "je za odjemalca brez vrednosti.",
))

# ---------------------------------------------------------- ELEKTRO ENERGIJA
# Vsi trije gospodinjski ceniki imajo isto strukturo fiksnega dela:
#   pavšalni strošek poslovanja − E-popust 0,81 EUR/mm/mesec (brez DDV) za
#   račun v elektronski obliki. Zato mesecno_nadomestilo_eko = nadomestilo − 0,81.
# Elektro energija NE ponuja samooskrbnega cenika in ne odkupuje presežkov [E2],
# zato noben paket ne dovoljuje PV in noben ne omogoča vloge oddajnika.
ELEN_E_POPUST = 0.81          # EUR/merilno mesto/mesec, brez DDV [E1]
ELEN_DODATEK_VIR = 0.82       # 100 % SONCE ali 100 % VODA, EUR/mm/mesec [E1]

_reg(Paket(
    id="ELEN_ZANESLJIVA", dobavitelj="Elektro energija",
    ime="Zanesljiva oskrba (redni cenik)",
    vir="[E1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.12490, mt=0.10290, et=0.11390,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=1.99 - ELEN_E_POPUST,
    dovoljuje_oddajnika=False,
    opombe="Brez vezave, velja do spremembe/preklica. Ne velja za oskrbo "
           "skupnih delov večstanovanjskih stavb.",
))

_reg(Paket(
    id="ELEN_FIKSNI", dobavitelj="Elektro energija",
    ime="Zanesljiva oskrba – Fiksni",
    vir="[E1]", velja_od=dt.date(2025, 10, 23), tip_cene=TipCene.TARIFNI,
    vt=0.12990, mt=0.10790, et=0.11890,
    # e-račun je pri tem ceniku obvezen, zato E-popust velja vedno
    mesecno_nadomestilo=1.99 - ELEN_E_POPUST,
    mesecno_nadomestilo_eko=1.99 - ELEN_E_POPUST,
    dovoljuje_oddajnika=False,
    opombe="Cena zajamčena 12 mesecev od pričetka uporabe aneksa; med vezavo "
           "prehod na drug cenik ni mogoč. Račun se izda v elektronski obliki, "
           "zato je E-popust vedno priznan.",
))

_reg(Paket(
    id="ELEN_DINAMICNA", dobavitelj="Elektro energija", ime="Dinamična oskrba",
    vir="[E1]", velja_od=dt.date(2024, 10, 2), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01199, cap_sipx=0.22000,
    mesecno_nadomestilo=2.97, mesecno_nadomestilo_eko=2.97 - ELEN_E_POPUST,
    dovoljuje_oddajnika=False,
    opombe="Zamejitev 220 EUR/MWh velja na URNI indeks SIPX (povprečje "
           "15-min poslov znotraj ure). Navzdol neomejeno. Brez vezave.",
))


def z_izbiro_vira(paket: Paket, vir: str = "jedrska") -> Paket:
    """Vrne kopijo paketa z dodatkom za izbiro vira energije [E1].

    Privzeti vir je brezogljična jedrska energija (0,00 EUR/mesec) — to je
    tudi dejanski privzetek Elektro energije od 1. 1. 2021 in najcenejša
    izbira, zato `z_izbiro_vira(p)` vrne paket nespremenjen. '100 % SONCE'
    ali '100 % VODA' se lahko doda h kateremu koli ceniku za 0,82 EUR/merilno
    mesto/mesec brez DDV. Na merilnem mestu je mogoče izbrati le en vir.
    """
    vir = vir.lower()
    if vir in ("jedrska", "jedrski", "privzeto"):
        return paket
    if vir not in ("sonce", "soncni", "voda", "vodni"):
        raise ValueError("vir mora biti 'sonce', 'voda' ali 'jedrska'")
    oznaka = "100 % SONCE" if vir.startswith("son") else "100 % VODA"
    return dataclasses.replace(
        paket,
        id=f"{paket.id}_{'SONCE' if vir.startswith('son') else 'VODA'}",
        ime=f"{paket.ime} + {oznaka}",
        dodatna_storitev=paket.dodatna_storitev + ELEN_DODATEK_VIR,
    )


# ===========================================================================
# VALIDACIJA
# ===========================================================================
def preveri_paket(paket: Paket, g: Gospodinjstvo,
                  datum: Optional[dt.date] = None,
                  strogo: bool = True) -> List[str]:
    """
    Preveri združljivost paketa z gospodinjstvom.
    Vrne seznam opozoril; ob strogo=True ob nezdružljivosti sproži NezdruzljivPaket.
    """
    napake: List[str] = []
    opozorila: List[str] = []

    # 1) PV vs. paket brez samooskrbe
    if g.ima_pv and not paket.dovoljuje_pv:
        napake.append(
            f"{g.ime} ima napravo za samooskrbo, paket '{paket.ime}' "
            f"({paket.dobavitelj}) pa je cenik za dobavo brez samooskrbe. "
            f"Uporabi samooskrbni cenik istega dobavitelja."
        )

    # 2) samooskrbni paket brez PV
    if paket.zahteva_pv and not g.ima_pv:
        napake.append(
            f"Paket '{paket.ime}' ({paket.dobavitelj}) je namenjen izključno "
            f"odjemalcem z napravo za samooskrbo, {g.ime} pa je nima."
        )

    # 3) shema samooskrbe (nova vs. NET metering)
    if g.ima_pv and g.shema_samooskrbe not in paket.dovoljene_sheme:
        dovoljene = ", ".join(s.value for s in paket.dovoljene_sheme)
        napake.append(
            f"{g.ime} je v shemi '{g.shema_samooskrbe.value}', paket "
            f"'{paket.ime}' pa velja za: {dovoljene}."
        )

    # 4) skupnostna samooskrba
    if g.skupnostna and not paket.dovoljuje_skupnostno:
        napake.append(
            f"{g.ime} je v skupnostni samooskrbi, paket '{paket.ime}' "
            f"pa je izrecno omejen na individualno samooskrbo."
        )

    # 5) 15-minutne meritve
    if paket.zahteva_15min and not g.meritve_15min:
        opozorila.append(
            f"Paket '{paket.ime}' predpostavlja 15-min meritve; brez njih se "
            f"uporabi nadomestna cena (ET)."
        )

    # 6) veljavnost cenika
    if datum is not None and datum < paket.velja_od:
        opozorila.append(
            f"Paket '{paket.ime}' velja šele od {paket.velja_od.isoformat()}, "
            f"obračunavaš pa {datum.isoformat()} — cene so ekstrapolirane nazaj."
        )

    # 7) souporaba: vloga oddajnika zahteva samooskrbni cenik dobavitelja
    if (g.vloga_souporaba in (Vloga.ODDAJNIK, Vloga.OBOJE)
            and not paket.dovoljuje_oddajnika):
        napake.append(
            f"{g.ime} nastopa kot oddajnik v souporabi, {paket.dobavitelj} "
            f"pa v ponudbi nima rešitev samooskrbe — po ceniku '{paket.ime}' "
            f"je mogoča le vloga prejemnika."
        )

    # 8) PV brez odkupa
    if g.ima_pv and paket.tip_odkupa is TipOdkupa.NI:
        opozorila.append(
            f"Paket '{paket.ime}' ne odkupuje presežkov — oddana energija se "
            f"ovrednoti z 0 EUR."
        )

    if napake and strogo:
        raise NezdruzljivPaket(" | ".join(napake))
    return napake + opozorila


def zdruzljivi_paketi(g: Gospodinjstvo,
                      datum: Optional[dt.date] = None) -> List[Paket]:
    """Vsi paketi iz kataloga, ki so združljivi z danim gospodinjstvom."""
    out = []
    for p in PAKETI.values():
        try:
            preveri_paket(p, g, datum, strogo=True)
            out.append(p)
        except NezdruzljivPaket:
            pass
    return out


# ===========================================================================
# SOUPORABA ELEKTRIČNE ENERGIJE (ZOEE)
# ===========================================================================
"""
Souporaba ni isto kot skupnostna samooskrba. Ključne razlike (viri [S1]–[S4]):

  1. Deli se ODDANA energija (presežek po lastni rabi), po 15-min intervalih.
     Delež se nanaša na oddano energijo, NE na celotno proizvodnjo in NE na
     letni presežek.
  2. Souporaba zniža SAMO obračunsko količino energije pri prejemniku.
     Omrežnina, prispevki in trošarina se pri prejemniku še naprej obračunajo
     od CELOTNE energije, prevzete iz omrežja. (Pri skupnostni samooskrbi se
     za deljeno energijo uporabi znižana distribucijska postavka — glej
     `energija_skupnost` v si_tarife.py.)
  3. Neizrabljena deljena energija se NE prenese v naslednji interval in ne
     ustvari dobropisa — pripade dobavitelju prejemnika.
  4. Odjemalci v stari shemi (letni NET metering) so lahko oddajniki,
     NE morejo pa biti prejemniki.
  5. Lokacija ni omejitev; udeleženci so lahko pri različnih dobaviteljih
     (razen če organizator to omejuje).
  6. Cena med udeležencema ni zakonsko določena (lahko 0), po GEN-I pa
     ne sme presegati tržno veljavne cene električne energije.
  7. Organizator (dobavitelj, agregator) zaračuna mesečno nadomestilo na
     merilno mesto.

VIRI:
 [S1] Petrol: https://www.petrol.si/znanje-in-podpora/2026/clanki/souporaba-elektricne-energije-kako-deliti-soncno-energijo.html
 [S2] Petrol cenik storitve: https://www.petrol.si/binaries/content/assets/www/2026/dokumenti/ee/cenik-storitev-souporabe-elektricne-energije-od-9.-06.-2026.pdf
 [S3] GEN-I: https://gen-i.si/dom/trajnostne-resitve/souporaba-energije/
 [S4] GEN-I cenik storitve: https://www.gen-i.si/media/nmtipmxa/cenik-storitve-souporaba-elektri%c4%8dne-energije-gos_v2.pdf
"""


@dataclass(frozen=True)
class StoritevSouporabe:
    """Ponudba organizatorja souporabe. Zneski brez DDV, EUR/merilno mesto/mesec."""
    id: str
    organizator: str
    ime: str
    vir: str
    velja_od: dt.date
    nadomestilo_oddajnik: float = 0.0
    nadomestilo_prejemnik: float = 0.0
    zahteva_istega_dobavitelja: bool = False
    cena_omejena_na_trzno: bool = False
    opombe: str = ""

    def nadomestilo(self, vloga: Vloga) -> float:
        if vloga is Vloga.ODDAJNIK:
            return self.nadomestilo_oddajnik
        if vloga is Vloga.PREJEMNIK:
            return self.nadomestilo_prejemnik
        if vloga is Vloga.OBOJE:
            return self.nadomestilo_oddajnik + self.nadomestilo_prejemnik
        return 0.0


STORITVE_SOUPORABE: Dict[str, StoritevSouporabe] = {}


def _regs(s: StoritevSouporabe) -> StoritevSouporabe:
    STORITVE_SOUPORABE[s.id] = s
    return s


_regs(StoritevSouporabe(
    id="BREZ_ORGANIZATORJA", organizator="—",
    ime="Samoorganizirana souporaba prek portala Moj Elektro",
    vir="[S1]", velja_od=dt.date(2026, 1, 1),
    opombe="Oddajnik in prejemnik se registrirata sama; dobavitelj je dolžan "
           "upoštevati količine souporabe brez dodatnega plačila.",
))

_regs(StoritevSouporabe(
    id="GENI_SOUPORABA", organizator="GEN-I", ime="Souporaba energije",
    vir="[S4]", velja_od=dt.date(2026, 7, 13),
    nadomestilo_oddajnik=4.99, nadomestilo_prejemnik=0.99,
    zahteva_istega_dobavitelja=True, cena_omejena_na_trzno=True,
    opombe="GEN-I organizira souporabo le med lastnimi odjemalci. "
           "Cena souporabe ne sme presegati tržno veljavne cene energije.",
))

_regs(StoritevSouporabe(
    id="PETROL_SOUPORABA", organizator="Petrol", ime="Storitev souporabe",
    vir="[S2]", velja_od=dt.date(2026, 6, 9),
    nadomestilo_oddajnik=4.98, nadomestilo_prejemnik=4.98,
    opombe="Enotno nadomestilo 4,98 EUR za VSAKO merilno mesto v souporabi, "
           "ne glede na vlogo in ne glede na dejansko trajanje v mesecu.",
))

# Elektro energija souporabe (še) ne organizira in zanjo nima objavljenega
# cenika [E2]; njeni odjemalci sodelujejo kot PREJEMNIKI, pri čemer je shemo
# treba registrirati samostojno ali prek organizatorja pri drugem dobavitelju
# (udeleženci so lahko pri različnih dobaviteljih) -> BREZ_ORGANIZATORJA.
# Vnosa StoritevSouporabe za Elektro energijo zato tu (še) ni.


def preveri_souporabo(storitev: StoritevSouporabe, udelezenci: Dict[str, "Gospodinjstvo"],
                      paketi: Optional[Dict[str, Paket]] = None,
                      strogo: bool = True) -> List[str]:
    """Preveri konfiguracijo sheme souporabe."""
    napake, opozorila = [], []

    oddajniki = [i for i, g in udelezenci.items()
                 if g.vloga_souporaba in (Vloga.ODDAJNIK, Vloga.OBOJE)]
    prejemniki = [i for i, g in udelezenci.items()
                  if g.vloga_souporaba in (Vloga.PREJEMNIK, Vloga.OBOJE)]

    if not oddajniki:
        napake.append("Shema souporabe nima nobenega oddajnika.")
    if not prejemniki:
        napake.append("Shema souporabe nima nobenega prejemnika.")

    for ime in prejemniki:
        if udelezenci[ime].shema_samooskrbe is Shema.NET_METERING:
            napake.append(
                f"{ime} je v letnem NET meteringu in ne more biti prejemnik "
                f"v souporabi (lahko pa je oddajnik).")

    for ime in oddajniki:
        g = udelezenci[ime]
        if g.delez_souporabe <= 0:
            opozorila.append(f"{ime} je oddajnik z deležem 0 % — nič se ne deli.")
        if g.shema_samooskrbe is Shema.NET_METERING and g.delez_souporabe > 0:
            opozorila.append(
                f"{ime} je v letnem NET meteringu: energija, oddana v souporabo, "
                f"se odšteje od letne 'zaloge' ne glede na to, ali jo prejemnik "
                f"porabi. Previsok delež ({g.delez_souporabe:.0%}) povzroči "
                f"doplačilo ob letnem obračunu.")

    if storitev.zahteva_istega_dobavitelja and paketi:
        dobavitelji = {p.dobavitelj for p in paketi.values()}
        if len(dobavitelji) > 1:
            napake.append(
                f"{storitev.organizator} organizira souporabo le med lastnimi "
                f"odjemalci, med udeleženci pa so: {', '.join(sorted(dobavitelji))}.")

    if napake and strogo:
        raise NezdruzljivPaket(" | ".join(napake))
    return napake + opozorila


# ===========================================================================
# ANALIZA PODVOJENIH / ENAKIH PONUDB
# ===========================================================================
_CENOVNA_POLJA = (
    "tip_cene", "tip_odkupa", "vt", "mt", "et",
    "soncna_ns", "soncna_vs", "osnovna", "konicna",
    "pribitek_odjem", "cap_sipx", "cap_mesecni",
    "odkup_fiksni", "odkup_soncna_ns", "odkup_soncna_vs",
    "odkup_osnovna", "odkup_konicna", "pribitek_oddaja",
)
_PREVZEM_POLJA = ("tip_cene", "vt", "mt", "et",
                  "soncna_ns", "soncna_vs", "osnovna", "konicna",
                  "pribitek_odjem", "cap_sipx", "cap_mesecni")


def _kljuc(p: Paket, polja) -> tuple:
    return tuple(getattr(p, f) for f in polja)


def najdi_enake_ponudbe(paketi: Optional[Dict[str, Paket]] = None) -> Dict[str, List]:
    """
    Poišče pakete z identičnimi cenami.
      'popolnoma_enaki' — enake vse cenovne postavke IN mesečno nadomestilo
      'enak_prevzem'    — enake cene prevzema, razlika je v odkupu/nadomestilu
    """
    paketi = paketi or PAKETI
    popolni, prevzem = {}, {}
    for p in paketi.values():
        popolni.setdefault(_kljuc(p, _CENOVNA_POLJA)
                           + (p.mesecno_nadomestilo, p.mesecno_nadomestilo_eko),
                           []).append(p)
        prevzem.setdefault(_kljuc(p, _PREVZEM_POLJA), []).append(p)
    return {
        "popolnoma_enaki": [v for v in popolni.values() if len(v) > 1],
        "enak_prevzem": [v for v in prevzem.values() if len(v) > 1],
    }
