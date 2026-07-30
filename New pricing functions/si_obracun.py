"""
si_obracun.py — mesečni obračun električne energije (SI, gospodinjstva in
energetske skupnosti).

STRUKTURA RAČUNA (vir: Agencija za energijo, Obrazložitev računa
https://www.agen-rs.si/gospodinjski/elektrika/obrazlozitev-racuna):

    energija
  + omrežnina za moč  +  omrežnina za energijo
  + prispevek OVE+SPTE (na kW) + prispevek URE + prispevek operater trga (na kWh)
  + trošarina (na kWh)
  ------------------------------------------- = neto osnova
  + 22 % DDV na neto osnovo
  ------------------------------------------- = račun z DDV
  − dobropis za odkup presežkov               (BREZ DDV)
  ------------------------------------------- = za plačilo

POPRAVEK glede odkupa: "Odkup presežka proizvedene električne energije v
napravi za samooskrbo ni predmet obdavčitve z DDV." (gen-i.si, bisol-energija.si)
Dobropis se torej odšteje ŠELE po DDV in ne zniža davčne osnove.

Intervalne funkcije imajo poenoteno signaturo:
    shema(market_price_mwh, total_consumed_kwh, utc_date, interval_minutes, ...)
Fiksne postavke doda `MesecniObracun` ob zaključku meseca — to je edina točka,
na kateri je mogoče pravilno obračunati DDV.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from si_cas import (
    aktivna_tarifa,
    bloki_v_mesecu,
    casovni_blok,
    je_visja_sezona,
    je_vt,
    razpored_za_datum,
    v_lokalni_cas,
)
from si_paketi import (
    Gospodinjstvo,
    Paket,
    TipCene,
    TipOdkupa,
    preveri_paket,
)
from si_tarife import (
    DDV,
    OMREZNINA_2026,
    OPERATER_TRGA_EUR_KWH,
    PRIVZETO_REFERENCNO_LETO,
    Omreznina,
    TROSARINA_EUR_KWH,
    URE_EUR_KWH,
    ima_tarifne_postavke,
    omreznina_za_datum,
    ove_spte_eur_kw,
)


# ===========================================================================
# 0. PRAVILA — izbira regulativnega režima, neodvisno od datuma podatkov
# ===========================================================================
@dataclass(frozen=True)
class Pravila:
    """
    Regulativni režim, po katerem se obračuna.

    Omogoča analizo starih podatkov (npr. dataset iz 2015) po DANES veljavni
    ali po prihodnji (2027) shemi:

        Pravila.veljavna()                   # režim, ki velja danes
        Pravila.od_2027()                    # režim od 1. 1. 2027
        Pravila.ob_datumu(date(2025, 6, 1))  # režim, ki je veljal takrat
        Pravila.za_leto(2026)                # režim danega referenčnega leta
        Pravila.privzeta(date(2012, 6, 30))  # režim datuma podatkov ali 2026

    `ob_datumu` / `veljavna` sta stroga: za datum brez objavljenih tarifnih
    postavk sprožita ValueError. `za_leto` in `privzeta` v tem primeru padeta
    nazaj na `PRIVZETO_REFERENCNO_LETO` (2026) — to je pot, ki jo uporabljata
    RL okolje in MILP nad starimi dataseti.

    `preslikaj_v_leto` opcijsko preslika koledar podatkov v referenčno leto,
    da se ujameta razpored praznikov in delovnih dni (npr. 2015 -> 2026).
    Privzeto None: uporabi se dejanski koledar podatkov, spremeni se le cenovni
    režim — to je običajno tisto, kar hočeš pri vprašanju "koliko bi ta poraba
    stala danes".
    """
    omreznina: Omreznina
    razpored: str                     # "2024" ali "2027"
    dajatve_datum: dt.date            # datum za OVE+SPTE / trošarino
    preslikaj_v_leto: Optional[int] = None
    oznaka: str = ""

    @classmethod
    def ob_datumu(cls, d: dt.date, **kw) -> "Pravila":
        return cls(omreznina=omreznina_za_datum(d),
                   razpored=razpored_za_datum(d),
                   dajatve_datum=d,
                   oznaka=f"režim {d.isoformat()}", **kw)

    @classmethod
    def veljavna(cls, danes: Optional[dt.date] = None, **kw) -> "Pravila":
        return cls.ob_datumu(danes or dt.date.today(), **kw)

    @classmethod
    def od_2026(cls, **kw) -> "Pravila":
        # Časovni bloki se ne spremenijo do 1. 1. 2027 (glej si_cas.razpored_za_datum),
        # zato je razpored še vedno "2024" — spremenijo se le tarifne postavke omrežnine.
        return cls(omreznina=OMREZNINA_2026, razpored="2024",
                   dajatve_datum=dt.date(2026, 1, 1),
                   oznaka="režim od 2026 (trenutno veljavni)", **kw)

    @classmethod
    def za_leto(cls, leto: int, **kw) -> "Pravila":
        """Režim za dano referenčno leto.

        Leta brez objavljenih tarifnih postavk (npr. 2012 iz Ausgrid podatkov)
        se obračunajo po privzetem letu 2026 — sicer bi `omreznina_za_datum`
        sprožila ValueError.
        """
        leto = int(leto)
        if leto >= 2027:
            return cls.od_2027(**kw)
        if leto == PRIVZETO_REFERENCNO_LETO:
            return cls.od_2026(**kw)
        d = dt.date(leto, 1, 1)
        if ima_tarifne_postavke(d):
            return cls.ob_datumu(d, **kw)
        return cls.od_2026(**kw)

    @classmethod
    def privzeta(cls, d: Optional[dt.date] = None, **kw) -> "Pravila":
        """Režim, ki je veljal na datum podatkov, s padcem nazaj na privzeto
        leto 2026, kadar za ta datum ni objavljenih tarifnih postavk."""
        if d is not None and ima_tarifne_postavke(d):
            return cls.ob_datumu(d, **kw)
        return cls.od_2026(**kw)

    @classmethod
    def od_2027(cls, **kw) -> "Pravila":
        # Tarifne postavke za 2027 še niso objavljene -> 2026 kot približek,
        # spremeni pa se razpored blokov. Ko Agencija objavi Akt za 2027,
        # dodaj OMREZNINA_2027 v si_tarife.py in ga vstavi tu.
        return cls(omreznina=OMREZNINA_2026, razpored="2027",
                   dajatve_datum=dt.date(2027, 1, 1),
                   oznaka="režim od 1. 1. 2027 (tarife 2026 kot približek)", **kw)


# ===========================================================================
# 1. INTERVALNE FUNKCIJE
# ===========================================================================
def _kontekst(utc_date: dt.datetime, interval_minutes: int,
              pravila: Pravila) -> Dict:
    lok = v_lokalni_cas(utc_date)
    if pravila.preslikaj_v_leto is not None:
        try:
            lok = lok.replace(year=pravila.preslikaj_v_leto)
        except ValueError:                       # 29. februar
            lok = lok.replace(year=pravila.preslikaj_v_leto, day=28)
    return {
        "lokalni_cas": lok,
        "blok": casovni_blok(lok, pravila.razpored),
        "vt": je_vt(lok),
        "tarifa": aktivna_tarifa(lok),
        "ure": interval_minutes / 60.0,
    }


def _cena_prevzema(paket: Paket, ctx: Dict, market_price_mwh: float,
                   meritve_15min: bool = True) -> float:
    if paket.tip_cene is TipCene.DINAMICNI:
        spot = market_price_mwh / 1000.0
        if paket.cap_sipx is not None:
            spot = min(spot, paket.cap_sipx)
        return spot + paket.pribitek_odjem

    if paket.tip_cene is TipCene.AKTIVNI:
        if not meritve_15min:
            return paket.et                      # nadomestna cena
        return {"soncna_ns": paket.soncna_ns, "soncna_vs": paket.soncna_vs,
                "osnovna": paket.osnovna, "konicna": paket.konicna}[ctx["tarifa"]]

    if paket.vt or paket.mt:
        return paket.vt if ctx["vt"] else paket.mt
    return paket.et


def _cena_oddaje(paket: Paket, ctx: Dict, market_price_mwh: float,
                 meritve_15min: bool = True) -> float:
    """EUR/kWh za oddano energijo. POZOR: ni predmet DDV."""
    if paket.tip_odkupa in (TipOdkupa.NI, TipOdkupa.NET_METERING):
        return 0.0
    if paket.tip_odkupa is TipOdkupa.DINAMICNI:
        # simetričen razmik: SIPX − pribitek (BISOL ±0,013; GEN-I ±0,01199)
        return market_price_mwh / 1000.0 - paket.pribitek_oddaja
    if paket.tip_odkupa is TipOdkupa.AKTIVNI:
        if not meritve_15min:
            return paket.odkup_fiksni            # nadomestna cena oddaje
        return {"soncna_ns": paket.odkup_soncna_ns,
                "soncna_vs": paket.odkup_soncna_vs,
                "osnovna": paket.odkup_osnovna,
                "konicna": paket.odkup_konicna}[ctx["tarifa"]]
    return paket.odkup_fiksni


def _dajatve(kwh: float) -> Dict[str, float]:
    return {"trosarina": kwh * TROSARINA_EUR_KWH,
            "prispevek_ure": kwh * URE_EUR_KWH,
            "prispevek_operater_trga": kwh * OPERATER_TRGA_EUR_KWH}


def _rezultat(ctx: Dict, prevzeto_kwh: float, obdavcljive: Dict[str, float],
              dobropis: float = 0.0, **extra) -> Dict:
    out = {"lokalni_cas": ctx["lokalni_cas"], "blok": ctx["blok"],
           "vt": ctx["vt"], "tarifa": ctx["tarifa"],
           "prevzeto_kwh": prevzeto_kwh,
           "moc_kw": prevzeto_kwh / ctx["ure"] if ctx["ure"] else 0.0,
           "obdavcljive_postavke": obdavcljive,
           "dobropis_odkup": dobropis}
    out.update(extra)
    return out


def dobava(market_price_mwh: float, total_consumed_kwh: float,
           utc_date: dt.datetime, interval_minutes: int = 15, *,
           paket: Paket, pravila: Optional[Pravila] = None,
           meritve_15min: bool = True) -> Dict:
    """
    Prevzem iz omrežja brez lastne proizvodnje. Pokriva enotarifne, dvotarifne,
    4-tarifne (aktivne) in dinamične pakete — razlika je le v paket.tip_cene.
    """
    pravila = pravila or Pravila.privzeta(v_lokalni_cas(utc_date).date())
    ctx = _kontekst(utc_date, interval_minutes, pravila)
    cena = _cena_prevzema(paket, ctx, market_price_mwh, meritve_15min)
    post = {"energija": total_consumed_kwh * cena,
            "omreznina_energija": total_consumed_kwh
                                  * pravila.omreznina.energija[ctx["blok"]]}
    post.update(_dajatve(total_consumed_kwh))
    return _rezultat(ctx, total_consumed_kwh, post, cena_energije_eur_kwh=cena)


def samooskrba(market_price_mwh: float, total_consumed_kwh: float,
               utc_date: dt.datetime, interval_minutes: int = 15, *,
               total_produced_kwh: float = 0.0, paket: Paket,
               pravila: Optional[Pravila] = None,
               meritve_15min: bool = True) -> Dict:
    """
    Nova shema (soglasje po 1. 1. 2024): netiranje ZNOTRAJ intervala.
      neto > 0 -> prevzem: energija + omrežnina + dajatve + DDV
      neto < 0 -> oddaja:  dobropis po ceni oddaje, BREZ omrežnine in BREZ DDV
    """
    pravila = pravila or Pravila.privzeta(v_lokalni_cas(utc_date).date())
    ctx = _kontekst(utc_date, interval_minutes, pravila)

    neto = total_consumed_kwh - total_produced_kwh
    prevzem, oddaja = max(neto, 0.0), max(-neto, 0.0)

    cena = _cena_prevzema(paket, ctx, market_price_mwh, meritve_15min)
    cena_odd = _cena_oddaje(paket, ctx, market_price_mwh, meritve_15min)

    post = {"energija": prevzem * cena,
            "omreznina_energija": prevzem * pravila.omreznina.energija[ctx["blok"]]}
    post.update(_dajatve(prevzem))
    return _rezultat(ctx, prevzem, post,
                     dobropis=oddaja * cena_odd,
                     oddano_kwh=oddaja,
                     lastna_raba_kwh=min(total_consumed_kwh, total_produced_kwh),
                     cena_energije_eur_kwh=cena, cena_oddaje_eur_kwh=cena_odd)


def skupnost(market_price_mwh: float, total_consumed_kwh: float,
             utc_date: dt.datetime, interval_minutes: int = 15, *,
             dodeljena_proizvodnja_kwh: float = 0.0,
             lastna_proizvodnja_kwh: float = 0.0,
             paket: Paket, pravila: Optional[Pravila] = None,
             znacilni_primer: int = 2,
             cena_skupnosti_eur_kwh: Optional[float] = None,
             meritve_15min: bool = True) -> Dict:
    """
    Član skupnostne samooskrbe. Poraba se v intervalu pokriva po vrsti:
      1. lastna proizvodnja na istem merilnem mestu (brez cene, brez omrežnine)
      2. delež proizvodnje skupnosti po ključu delitve -> cena skupnosti +
         ZNIŽANA omrežnina (prenosna postavka + skupnostna distribucijska
         postavka po `znacilni_primer` 1–10; primer 1 = 0 EUR/kWh); dajatve se
         plačajo tudi za to energijo
      3. manko iz omrežja po pogodbi z dobaviteljem (polna omrežnina)

    Če `cena_skupnosti_eur_kwh` ni podana, se uporabi "split-the-difference"
    (povprečje cene prevzema in oddaje) po Markotić et al., Energies 2026,
    19(8), 1831, https://doi.org/10.3390/en19081831
    """
    pravila = pravila or Pravila.privzeta(v_lokalni_cas(utc_date).date())
    ctx = _kontekst(utc_date, interval_minutes, pravila)
    om, blok = pravila.omreznina, ctx["blok"]

    cena_dob = _cena_prevzema(paket, ctx, market_price_mwh, meritve_15min)
    cena_odd = _cena_oddaje(paket, ctx, market_price_mwh, meritve_15min)
    if cena_skupnosti_eur_kwh is None:
        cena_skupnosti_eur_kwh = (cena_dob + cena_odd) / 2.0

    preostanek = total_consumed_kwh
    lastna = min(preostanek, lastna_proizvodnja_kwh); preostanek -= lastna
    deljena = min(preostanek, dodeljena_proizvodnja_kwh); preostanek -= deljena
    iz_omrezja = preostanek

    oddaja = (max(lastna_proizvodnja_kwh - lastna, 0.0)
              + max(dodeljena_proizvodnja_kwh - deljena, 0.0))

    om_deljena = (om.energija_prenos[blok]
                  + om.energija_skupnost[znacilni_primer][blok])

    post = {"energija": iz_omrezja * cena_dob,
            "energija_skupnost": deljena * cena_skupnosti_eur_kwh,
            "omreznina_energija": iz_omrezja * om.energija[blok],
            "omreznina_energija_skupnost": deljena * om_deljena}
    post.update(_dajatve(iz_omrezja + deljena))

    return _rezultat(ctx, iz_omrezja, post,
                     dobropis=oddaja * cena_odd,
                     deljeno_kwh=deljena, lastna_raba_kwh=lastna,
                     oddano_kwh=oddaja,
                     cena_energije_eur_kwh=cena_dob, cena_oddaje_eur_kwh=cena_odd)


# ===========================================================================
# 2. MESEČNI RAČUN
# ===========================================================================
FIKSNE_POSTAVKE = frozenset({"mesecno_nadomestilo", "omreznina_moc",
                             "omreznina_presezna_moc", "prispevek_ove_spte",
                             "nadomestilo_souporaba"})


@dataclass
class Racun:
    obdobje: str
    paket: str
    postavke: Dict[str, float]        # obdavčljive, brez DDV
    dobropis_odkup: float             # brez DDV, odšteje se po DDV
    fiksni_del: float
    spremenljivi_del: float
    prevzeto_kwh: float
    ddv_stopnja: float = DDV
    opozorila: List[str] = field(default_factory=list)
    diagnostika: Dict = field(default_factory=dict)

    @property
    def neto(self) -> float: return sum(self.postavke.values())
    @property
    def ddv(self) -> float: return self.neto * self.ddv_stopnja
    @property
    def bruto(self) -> float: return self.neto + self.ddv
    @property
    def za_placilo(self) -> float: return self.bruto - self.dobropis_odkup
    @property
    def fiksni_del_z_ddv(self) -> float:
        return self.fiksni_del * (1 + self.ddv_stopnja)

    @property
    def spremenljivi_del_z_ddv(self) -> float:
        """Spremenljivi del z DDV, zmanjšan za dobropis (ta je brez DDV)."""
        return self.spremenljivi_del * (1 + self.ddv_stopnja) - self.dobropis_odkup

    @property
    def povprecna_cena_eur_kwh(self) -> float:
        return self.za_placilo / self.prevzeto_kwh if self.prevzeto_kwh else float("nan")

    def as_dict(self) -> Dict:
        return {"obdobje": self.obdobje, "paket": self.paket,
                "postavke_brez_ddv": {k: round(v, 4) for k, v in self.postavke.items()},
                "skupaj_brez_ddv": round(self.neto, 2),
                "ddv": round(self.ddv, 2),
                "skupaj_z_ddv": round(self.bruto, 2),
                "dobropis_odkup_brez_ddv": round(self.dobropis_odkup, 2),
                "za_placilo": round(self.za_placilo, 2),
                "fiksni_del_brez_ddv": round(self.fiksni_del, 2),
                "fiksni_del_z_ddv": round(self.fiksni_del_z_ddv, 2),
                "spremenljivi_del_brez_ddv": round(self.spremenljivi_del, 2),
                "spremenljivi_del_z_ddv": round(self.spremenljivi_del_z_ddv, 2),
                "prevzeto_kwh": round(self.prevzeto_kwh, 3),
                "povprecna_cena_eur_kwh": round(self.povprecna_cena_eur_kwh, 5),
                "opozorila": self.opozorila, "diagnostika": self.diagnostika}

    def izpis(self) -> None:
        print(f"\n=== {self.obdobje} | {self.paket} ===")
        print(f"{'postavka':<40}{'brez DDV':>11}{'z DDV':>11}")
        print("-" * 62)
        for k, v in self.postavke.items():
            print(f"{k:<40}{v:>11.2f}{v * (1 + self.ddv_stopnja):>11.2f}")
        print("-" * 62)
        print(f"{'vmesni seštevek':<40}{self.neto:>11.2f}{self.bruto:>11.2f}")
        if self.dobropis_odkup:
            print(f"{'odkup presežkov (ni predmet DDV)':<40}"
                  f"{'':>11}{-self.dobropis_odkup:>11.2f}")
        print("-" * 62)
        print(f"{'FIKSNI del':<40}{self.fiksni_del:>11.2f}"
              f"{self.fiksni_del_z_ddv:>11.2f}")
        print(f"{'SPREMENLJIVI del':<40}{self.spremenljivi_del:>11.2f}"
              f"{self.spremenljivi_del_z_ddv:>11.2f}")
        print("-" * 62)
        print(f"{'ZA PLAČILO':<40}{'':>11}{self.za_placilo:>11.2f}")
        if self.prevzeto_kwh:
            print(f"prevzeto {self.prevzeto_kwh:.1f} kWh | "
                  f"{self.povprecna_cena_eur_kwh:.4f} EUR/kWh")
        for o in self.opozorila:
            print(f"  ! {o}")


class MesecniObracun:
    """Akumulira intervalne rezultate in sestavi mesečni račun."""

    def __init__(self, leto: int, mesec: int, gospodinjstvo: Gospodinjstvo,
                 paket: Paket, pravila: Optional[Pravila] = None, *,
                 ove_spte_blok: Optional[int] = None,
                 obracunaj_presezno_moc: bool = True,
                 strogo: bool = True):
        self.leto, self.mesec = leto, mesec
        self.g, self.paket = gospodinjstvo, paket
        self.pravila = pravila or Pravila.privzeta(dt.date(leto, mesec, 1))
        self.ove_spte_blok = ove_spte_blok
        self.obracunaj_presezno_moc = obracunaj_presezno_moc

        # validacija paketa proti gospodinjstvu — sproži NezdruzljivPaket
        self.opozorila = preveri_paket(paket, gospodinjstvo,
                                       dt.date(leto, mesec, 1), strogo=strogo)

        self._post: Dict[str, float] = defaultdict(float)
        self._dobropis = 0.0
        self._prevzeto = self._deljeno = self._oddano = self._lastna = 0.0
        self._neizrabljeno = 0.0
        self._po_blokih: Dict[int, float] = defaultdict(float)
        self._prekoracitve: Dict[int, float] = defaultdict(float)
        self._energija_kwh = self._energija_eur = 0.0
        self._n = 0
        self._razdelitev_moci: Dict[int, float] = {}

    def dodaj(self, interval: Dict) -> None:
        blok = interval["blok"]
        for k, v in interval["obdavcljive_postavke"].items():
            self._post[k] += v
        self._dobropis += interval.get("dobropis_odkup", 0.0)
        self._prevzeto += interval.get("prevzeto_kwh", 0.0)
        self._deljeno += (interval.get("deljeno_kwh", 0.0)
                          + interval.get("preneseno_kwh", 0.0)
                          + interval.get("izrabljeno_souporaba_kwh", 0.0))
        self._neizrabljeno += interval.get("neizrabljeno_souporaba_kwh", 0.0)
        self._oddano += interval.get("oddano_kwh", 0.0)
        self._lastna += interval.get("lastna_raba_kwh", 0.0)
        self._po_blokih[blok] += interval.get("prevzeto_kwh", 0.0)
        self._energija_kwh += interval.get("prevzeto_kwh", 0.0)
        self._energija_eur += interval["obdavcljive_postavke"].get("energija", 0.0)

        razlika = interval.get("moc_kw", 0.0) - self.g.dogovorjena_moc.get(blok, 0.0)
        if razlika > 0:
            self._prekoracitve[blok] += razlika ** 2
        self._n += 1

    def _uveljavi_mesecni_cap(self) -> None:
        """Mesečna zamejitev povprečne obračunske cene (BISOL DINAMIČNI+)."""
        cap = self.paket.cap_mesecni
        if cap is None or self._energija_kwh <= 0:
            return
        strop = (cap + self.paket.pribitek_odjem) * self._energija_kwh
        if self._energija_eur > strop:
            self.opozorila.append(
                f"Uveljavljena mesečna zamejitev cene ({cap:.5f} EUR/kWh + pribitek): "
                f"energija znižana z {self._energija_eur:.2f} na {strop:.2f} EUR.")
            self._post["energija"] = strop

    def _fiksne(self) -> None:
        prvi = dt.date(self.leto, self.mesec, 1)
        om = self.pravila.omreznina
        vs = je_visja_sezona(prvi)
        bloki = bloki_v_mesecu(self.leto, self.mesec, self.pravila.razpored)

        moc, razdelitev = 0.0, {}
        for b in sorted(bloki):
            z = self.g.dogovorjena_moc.get(b, 0.0) * om.postavka_moc(b, vs)
            razdelitev[b] = round(z, 4)
            moc += z
        self._post["omreznina_moc"] = moc
        self._razdelitev_moci = razdelitev

        if self.obracunaj_presezno_moc and self._prekoracitve:
            kazen = sum(math.sqrt(v) * om.postavka_moc(b, vs) * om.faktor_presezne_moci
                        for b, v in self._prekoracitve.items())
            if kazen:
                self._post["omreznina_presezna_moc"] = kazen

        ref = (self.ove_spte_blok if self.ove_spte_blok is not None
               else (min(bloki) if bloki else 2))
        self._post["prispevek_ove_spte"] = (
            self.g.dogovorjena_moc.get(ref, 0.0)
            * ove_spte_eur_kw(self.pravila.dajatve_datum))

        nad = self.paket.nadomestilo(self.g.eko_racun)
        if nad:
            self._post["mesecno_nadomestilo"] = nad

    def zakljuci(self) -> Racun:
        self._uveljavi_mesecni_cap()
        self._fiksne()
        post = {k: v for k, v in self._post.items() if abs(v) > 1e-12}
        fiksni = sum(v for k, v in post.items() if k in FIKSNE_POSTAVKE)
        spremenljivi = sum(v for k, v in post.items() if k not in FIKSNE_POSTAVKE)
        return Racun(
            obdobje=f"{self.mesec:02d}/{self.leto}",
            paket=f"{self.paket.dobavitelj} – {self.paket.ime}",
            postavke=dict(sorted(post.items())),
            dobropis_odkup=self._dobropis,
            fiksni_del=fiksni, spremenljivi_del=spremenljivi,
            prevzeto_kwh=self._prevzeto,
            opozorila=list(self.opozorila),
            diagnostika={
                "pravila": self.pravila.oznaka,
                "razpored_blokov": self.pravila.razpored,
                "omreznina_vir": self.pravila.omreznina.vir,
                "st_intervalov": self._n,
                "visja_sezona": je_visja_sezona(dt.date(self.leto, self.mesec, 1)),
                "prevzeto_po_blokih_kwh": {k: round(v, 2) for k, v
                                           in sorted(self._po_blokih.items())},
                "omreznina_moc_po_blokih": self._razdelitev_moci,
                "lastna_raba_kwh": round(self._lastna, 2),
                "deljeno_kwh": round(self._deljeno, 2),
                "oddano_kwh": round(self._oddano, 2),
                "neizrabljena_souporaba_kwh": round(self._neizrabljeno, 2),
                "presezna_moc_kw_po_blokih": {
                    k: round(math.sqrt(v), 3)
                    for k, v in sorted(self._prekoracitve.items())}},
        )


# ===========================================================================
# 3. ENERGETSKA SKUPNOST
# ===========================================================================
def kljuc_staticni(delezi: Dict[str, float], proizvodnja_kwh: float,
                   poraba: Dict[str, float]) -> Dict[str, float]:
    s = sum(delezi.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"vsota ključev delitve mora biti 1, je {s:.6f}")
    return {c: proizvodnja_kwh * d for c, d in delezi.items()}


def kljuc_sorazmerni(delezi: Dict[str, float], proizvodnja_kwh: float,
                     poraba: Dict[str, float]) -> Dict[str, float]:
    """Dinamični ključ: delitev sorazmerno s trenutno porabo članov."""
    skupna = sum(poraba.values())
    if skupna <= 0:
        return kljuc_staticni(delezi, proizvodnja_kwh, poraba)
    return {c: proizvodnja_kwh * (p / skupna) for c, p in poraba.items()}


def obracun_skupnosti(clani: Dict[str, Dict], podatki: Sequence[Dict],
                      leto: int, mesec: int, *,
                      kljuc: Callable = kljuc_staticni,
                      pravila: Optional[Pravila] = None,
                      cena_skupnosti_eur_kwh: Optional[float] = None,
                      strogo: bool = True) -> Dict[str, Racun]:
    """
    clani  : {ime: {"gospodinjstvo": Gospodinjstvo, "paket": Paket, "delez": float}}
    podatki: [{"utc_date": datetime, "interval_minutes": int,
               "market_price_mwh": float, "poraba": {ime: kWh},
               "proizvodnja_kwh": float,
               "lastna_proizvodnja": {ime: kWh}  # neobvezno
              }, ...]
    """
    obracuni = {ime: MesecniObracun(leto, mesec, c["gospodinjstvo"], c["paket"],
                                    pravila, strogo=strogo)
                for ime, c in clani.items()}
    delezi = {ime: c["delez"] for ime, c in clani.items()}

    for row in podatki:
        poraba = row["poraba"]
        dodeljeno = kljuc(delezi, row.get("proizvodnja_kwh", 0.0), poraba)
        for ime, kwh in poraba.items():
            g = clani[ime]["gospodinjstvo"]
            obracuni[ime].dodaj(skupnost(
                row["market_price_mwh"], kwh, row["utc_date"],
                row.get("interval_minutes", 15),
                dodeljena_proizvodnja_kwh=dodeljeno.get(ime, 0.0),
                lastna_proizvodnja_kwh=row.get("lastna_proizvodnja", {}).get(ime, 0.0),
                paket=clani[ime]["paket"], pravila=obracuni[ime].pravila,
                znacilni_primer=g.znacilni_primer,
                cena_skupnosti_eur_kwh=cena_skupnosti_eur_kwh,
                meritve_15min=g.meritve_15min))

    return {ime: o.zakljuci() for ime, o in obracuni.items()}


# ===========================================================================
# 4. SOUPORABA ELEKTRIČNE ENERGIJE (ZOEE)
# ===========================================================================
"""
Razlika proti skupnostni samooskrbi (`skupnost()`):
  * souporaba zniža SAMO obračunsko količino energije pri prejemniku;
    omrežnina, prispevki in trošarina se obračunajo od CELOTNEGA prevzema
    iz omrežja  [S1][S3],
  * deli se ODDANA energija (presežek po lastni rabi) po 15-min intervalih,
    delež se nanaša na oddano energijo, ne na proizvodnjo  [S3],
  * neizrabljena deljena energija se NE prenese naprej in ne da dobropisa —
    pripade dobavitelju prejemnika  [S1][S3].
"""
from si_paketi import StoritevSouporabe, Vloga, preveri_souporabo   # noqa: E402


def souporaba_oddajnik(
    market_price_mwh: float, total_consumed_kwh: float,
    utc_date: dt.datetime, interval_minutes: int = 15, *,
    total_produced_kwh: float = 0.0, delez_souporabe: float = 0.0,
    paket: Paket, pravila: Optional[Pravila] = None,
    cena_souporabe_eur_kwh: float = 0.0,
    placilo_za_neizrabljeno: bool = True,
    dejansko_izrabljeno_kwh: Optional[float] = None,
    meritve_15min: bool = True,
) -> Dict:
    """
    Oddajnik v souporabi.

    Presežek intervala = proizvodnja − poraba (če > 0).
      delez_souporabe * presežek  -> preneseno prejemnikom (prihodek po dogovoru)
      preostanek                  -> odkup pri lastnem dobavitelju po ceniku

    `placilo_za_neizrabljeno=True` (privzeto): oddajnik prejme plačilo za
    celotno preneseno količino, ne glede na to, ali jo prejemnik porabi —
    energija se mu odšteje v vsakem primeru. Če se pogodbeno dogovorita, da
    se plača le dejansko izrabljena količina, nastavi na False in podaj
    `dejansko_izrabljeno_kwh`.
    """
    pravila = pravila or Pravila.privzeta(v_lokalni_cas(utc_date).date())
    ctx = _kontekst(utc_date, interval_minutes, pravila)

    neto = total_consumed_kwh - total_produced_kwh
    prevzem, presezek = max(neto, 0.0), max(-neto, 0.0)

    v_souporabo = presezek * delez_souporabe
    za_odkup = presezek - v_souporabo

    cena = _cena_prevzema(paket, ctx, market_price_mwh, meritve_15min)
    cena_odd = _cena_oddaje(paket, ctx, market_price_mwh, meritve_15min)

    post = {"energija": prevzem * cena,
            "omreznina_energija": prevzem * pravila.omreznina.energija[ctx["blok"]]}
    post.update(_dajatve(prevzem))

    placano = (v_souporabo if placilo_za_neizrabljeno
               else (dejansko_izrabljeno_kwh
                     if dejansko_izrabljeno_kwh is not None else v_souporabo))

    return _rezultat(
        ctx, prevzem, post,
        dobropis=za_odkup * cena_odd + placano * cena_souporabe_eur_kwh,
        oddano_kwh=za_odkup,
        preneseno_kwh=v_souporabo,
        lastna_raba_kwh=min(total_consumed_kwh, total_produced_kwh),
        cena_energije_eur_kwh=cena, cena_oddaje_eur_kwh=cena_odd,
    )


def souporaba_prejemnik(
    market_price_mwh: float, total_consumed_kwh: float,
    utc_date: dt.datetime, interval_minutes: int = 15, *,
    prejeto_kwh: float = 0.0,
    paket: Paket, pravila: Optional[Pravila] = None,
    cena_souporabe_eur_kwh: float = 0.0,
    placilo_za_neizrabljeno: bool = True,
    lastna_proizvodnja_kwh: float = 0.0,
    meritve_15min: bool = True,
) -> Dict:
    """
    Prejemnik v souporabi.

    Prevzem iz omrežja G = poraba − lastna proizvodnja (če > 0).
    Obračunska količina ENERGIJE = max(G − prejeto, 0).
    Omrežnina, trošarina in prispevki ostanejo na CELOTNEM G.
    Neizrabljeni del prejete energije propade (ni dobropisa, ni prenosa).
    """
    pravila = pravila or Pravila.privzeta(v_lokalni_cas(utc_date).date())
    ctx = _kontekst(utc_date, interval_minutes, pravila)

    prevzem = max(total_consumed_kwh - lastna_proizvodnja_kwh, 0.0)
    izrabljeno = min(prevzem, prejeto_kwh)
    neizrabljeno = prejeto_kwh - izrabljeno
    od_dobavitelja = prevzem - izrabljeno

    cena = _cena_prevzema(paket, ctx, market_price_mwh, meritve_15min)
    placano = prejeto_kwh if placilo_za_neizrabljeno else izrabljeno

    post = {
        "energija": od_dobavitelja * cena,
        "energija_souporaba": placano * cena_souporabe_eur_kwh,
        # omrežnina in dajatve od CELOTNEGA prevzema iz omrežja
        "omreznina_energija": prevzem * pravila.omreznina.energija[ctx["blok"]],
    }
    post.update(_dajatve(prevzem))

    return _rezultat(
        ctx, prevzem, post,
        prejeto_kwh=prejeto_kwh,
        izrabljeno_souporaba_kwh=izrabljeno,
        neizrabljeno_souporaba_kwh=neizrabljeno,
        cena_energije_eur_kwh=cena,
    )


def obracun_souporabe(
    udelezenci: Dict[str, Dict],
    podatki: Sequence[Dict],
    leto: int, mesec: int, *,
    storitev: StoritevSouporabe,
    pravila: Optional[Pravila] = None,
    cena_souporabe_eur_kwh: float = 0.0,
    placilo_za_neizrabljeno: bool = True,
    strogo: bool = True,
) -> Dict[str, Racun]:
    """
    udelezenci: {ime: {"gospodinjstvo": Gospodinjstvo, "paket": Paket,
                       "delitev": {ime_prejemnika: delez_med_prejemniki}}}
                `delitev` je obvezna le pri oddajnikih; deleži se seštejejo v 1.
                Kolikšen del ODDANE energije gre v souporabo, pove
                Gospodinjstvo.delez_souporabe.
    podatki:    [{"utc_date": ..., "interval_minutes": ...,
                  "market_price_mwh": ...,
                  "poraba": {ime: kWh}, "proizvodnja": {ime: kWh}}, ...]
    """
    g_map = {i: c["gospodinjstvo"] for i, c in udelezenci.items()}
    p_map = {i: c["paket"] for i, c in udelezenci.items()}
    opozorila_sheme = preveri_souporabo(storitev, g_map, p_map, strogo=strogo)

    obracuni = {i: MesecniObracun(leto, mesec, g_map[i], p_map[i], pravila,
                                  strogo=strogo)
                for i in udelezenci}
    for i, o in obracuni.items():
        o.opozorila.extend(opozorila_sheme)
        nad = storitev.nadomestilo(g_map[i].vloga_souporaba)
        if nad:
            o._post["nadomestilo_souporaba"] = nad

    if storitev.cena_omejena_na_trzno:
        for i, o in obracuni.items():
            if cena_souporabe_eur_kwh > max(p_map[i].et, p_map[i].vt,
                                            p_map[i].osnovna, 0.0) > 0:
                o.opozorila.append(
                    f"Cena souporabe {cena_souporabe_eur_kwh:.5f} EUR/kWh presega "
                    f"tržno ceno paketa — {storitev.organizator} tega ne dovoli.")

    for row in podatki:
        poraba = row["poraba"]
        proizvodnja = row.get("proizvodnja", {})
        ts, im = row["utc_date"], row.get("interval_minutes", 15)
        cena_mwh = row["market_price_mwh"]

        # 1) koliko vsak oddajnik prenese in komu
        prejeto: Dict[str, float] = defaultdict(float)
        preneseno: Dict[str, float] = {}
        for i, g in g_map.items():
            if g.vloga_souporaba not in (Vloga.ODDAJNIK, Vloga.OBOJE):
                continue
            presezek = max(proizvodnja.get(i, 0.0) - poraba.get(i, 0.0), 0.0)
            deljeno = presezek * g.delez_souporabe
            preneseno[i] = deljeno
            delitev = udelezenci[i].get("delitev", {})
            s = sum(delitev.values())
            if s <= 0:
                continue
            for prej, w in delitev.items():
                prejeto[prej] += deljeno * w / s

        # 2) obračun po udeležencih
        for i, g in g_map.items():
            if g.vloga_souporaba in (Vloga.ODDAJNIK, Vloga.OBOJE):
                r = souporaba_oddajnik(
                    cena_mwh, poraba.get(i, 0.0), ts, im,
                    total_produced_kwh=proizvodnja.get(i, 0.0),
                    delez_souporabe=g.delez_souporabe,
                    paket=p_map[i], pravila=obracuni[i].pravila,
                    cena_souporabe_eur_kwh=cena_souporabe_eur_kwh,
                    placilo_za_neizrabljeno=placilo_za_neizrabljeno,
                    meritve_15min=g.meritve_15min)
            else:
                r = souporaba_prejemnik(
                    cena_mwh, poraba.get(i, 0.0), ts, im,
                    prejeto_kwh=prejeto.get(i, 0.0),
                    paket=p_map[i], pravila=obracuni[i].pravila,
                    cena_souporabe_eur_kwh=cena_souporabe_eur_kwh,
                    placilo_za_neizrabljeno=placilo_za_neizrabljeno,
                    lastna_proizvodnja_kwh=proizvodnja.get(i, 0.0),
                    meritve_15min=g.meritve_15min)
            obracuni[i].dodaj(r)

    return {i: o.zakljuci() for i, o in obracuni.items()}
