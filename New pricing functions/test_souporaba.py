"""Souporaba električne energije — primer za več ponudnikov + analiza podvojenih ponudb."""
import datetime as dt, math, random
from si_cas import v_lokalni_cas
from si_paketi import (PAKETI, STORITVE_SOUPORABE, Gospodinjstvo, Shema, Vloga,
                       NezdruzljivPaket, najdi_enake_ponudbe, preveri_souporabo)
from si_obracun import MesecniObracun, Pravila, obracun_souporabe, samooskrba

DOG = {1: 4.6, 2: 5.1, 3: 5.1, 4: 5.1, 5: 5.1}
INT, LETO, MESEC = 15, 2026, 7
random.seed(7)

def profil(kwp=0.0, faktor=1.0, dnevni=False):
    t = dt.datetime(LETO, MESEC, 1, tzinfo=dt.timezone.utc)
    konec = dt.datetime(LETO, MESEC + 1, 1, tzinfo=dt.timezone.utc)
    out = []
    while t < konec:
        lok = v_lokalni_cas(t); h = lok.hour + lok.minute / 60
        if dnevni:   # prejemnik s toplotno črpalko / delom od doma
            p = (0.20 + 0.60 * math.exp(-((h - 13) ** 2) / 12)) * (INT/60)
        else:
            p = (0.18 + 0.55*math.exp(-((h-7.5)**2)/2)
                 + 0.75*math.exp(-((h-19)**2)/3)) * (INT/60)
        p *= faktor * random.uniform(0.8, 1.2)
        prod = (kwp * 0.85 * math.sin(math.pi*(h-6)/12) * (INT/60)) if 6 <= h <= 18 else 0.0
        sipx = 70 + 45*math.sin(math.pi*(h-5)/14) - 25*(11 <= h <= 15)
        out.append((t, p, max(prod, 0.0), sipx))
        t += dt.timedelta(minutes=INT)
    return out

odd = profil(kwp=10.0)                 # oddajnik: 10 kWp, večerna poraba
prej = profil(faktor=1.1, dnevni=True) # prejemnik: dnevna poraba, brez PV
podatki = [{"utc_date": a[0], "interval_minutes": INT, "market_price_mwh": a[3],
            "poraba": {"Oddajnik": a[1], "Prejemnik": b[1]},
            "proizvodnja": {"Oddajnik": a[2]}}
           for a, b in zip(odd, prej)]

print("="*78); print("1. VALIDACIJA SOUPORABE"); print("="*78)
nm = Gospodinjstvo("NetMeter", DOG, ima_pv=True, shema_samooskrbe=Shema.NET_METERING,
                   vloga_souporaba=Vloga.ODDAJNIK, delez_souporabe=0.4)
print("  OK      net metering kot ODDAJNIK")
try:
    Gospodinjstvo("NetMeter", DOG, ima_pv=True, shema_samooskrbe=Shema.NET_METERING,
                  vloga_souporaba=Vloga.PREJEMNIK)
except ValueError as e:
    print(f"  NAPAKA  net metering kot PREJEMNIK\n            -> {e}")
try:
    Gospodinjstvo("BrezPV", DOG, vloga_souporaba=Vloga.ODDAJNIK, delez_souporabe=0.5)
except ValueError as e:
    print(f"  NAPAKA  oddajnik brez proizvodne naprave\n            -> {e}")
try:
    preveri_souporabo(STORITVE_SOUPORABE["GENI_SOUPORABA"],
                      {"A": Gospodinjstvo("A", DOG, ima_pv=True, shema_samooskrbe=Shema.NOVA,
                                          vloga_souporaba=Vloga.ODDAJNIK, delez_souporabe=0.4),
                       "B": Gospodinjstvo("B", DOG, vloga_souporaba=Vloga.PREJEMNIK)},
                      {"A": PAKETI["GENI_SAMO_REDNI"], "B": PAKETI["PETROL_REDNI"]})
except NezdruzljivPaket as e:
    print(f"  NAPAKA  GEN-I organizira souporabo med dvema dobaviteljema\n            -> {e}")

print("\n" + "="*78)
print("2. SOUPORABA PRI RAZLIČNIH ORGANIZATORJIH (jul 2026, delež 40 %, cena 0,05 EUR/kWh)")
print("="*78)
scenariji = [
    ("BREZ_ORGANIZATORJA", "GENI_SAMO_REDNI",  "GENI_REDNI"),
    ("GENI_SOUPORABA",     "GENI_SAMO_REDNI",  "GENI_REDNI"),
    ("GENI_SOUPORABA",     "GENI_SAMO_AKTIVNI","GENI_AKTIVNI"),
    ("PETROL_SOUPORABA",   "PETROL_SAMOOSKRBA","PETROL_REDNI"),
    ("BREZ_ORGANIZATORJA", "BISOL_SAMO_DINAMICNA", "BISOL_DINAMICNI"),
    # Elektro energija samooskrbe ne ponuja -> lahko je le PREJEMNIK, oddajnik
    # pa je pri drugem dobavitelju (souporaba med dobavitelji je dovoljena).
    ("BREZ_ORGANIZATORJA", "GENI_SAMO_REDNI",  "ELEN_ZANESLJIVA"),
    ("BREZ_ORGANIZATORJA", "BISOL_SAMO_DINAMICNA", "ELEN_DINAMICNA"),
]
print(f"  {'organizator':<20}{'paket oddajnika':<26}{'oddajnik':>10}{'prejemnik':>11}{'skupaj':>9}")
print("  " + "-"*76)
for sid, p_odd, p_prej in scenariji:
    ud = {
        "Oddajnik": {"gospodinjstvo": Gospodinjstvo("Oddajnik", DOG, ima_pv=True,
                        shema_samooskrbe=Shema.NOVA, vloga_souporaba=Vloga.ODDAJNIK,
                        delez_souporabe=0.40),
                     "paket": PAKETI[p_odd], "delitev": {"Prejemnik": 1.0}},
        "Prejemnik": {"gospodinjstvo": Gospodinjstvo("Prejemnik", DOG,
                        vloga_souporaba=Vloga.PREJEMNIK),
                      "paket": PAKETI[p_prej]},
    }
    r = obracun_souporabe(ud, podatki, LETO, MESEC,
                          storitev=STORITVE_SOUPORABE[sid],
                          cena_souporabe_eur_kwh=0.05)
    a, b = r["Oddajnik"], r["Prejemnik"]
    print(f"  {STORITVE_SOUPORABE[sid].organizator:<20}{PAKETI[p_odd].ime[:24]:<26}"
          f"{a.za_placilo:>10.2f}{b.za_placilo:>11.2f}{a.za_placilo+b.za_placilo:>9.2f}")

print("\n  Podroben račun prejemnika (GEN-I redni, organizator GEN-I):")
ud = {
    "Oddajnik": {"gospodinjstvo": Gospodinjstvo("Oddajnik", DOG, ima_pv=True,
                    shema_samooskrbe=Shema.NOVA, vloga_souporaba=Vloga.ODDAJNIK,
                    delez_souporabe=0.40),
                 "paket": PAKETI["GENI_SAMO_REDNI"], "delitev": {"Prejemnik": 1.0}},
    "Prejemnik": {"gospodinjstvo": Gospodinjstvo("Prejemnik", DOG,
                    vloga_souporaba=Vloga.PREJEMNIK), "paket": PAKETI["GENI_REDNI"]},
}
res = obracun_souporabe(ud, podatki, LETO, MESEC,
                        storitev=STORITVE_SOUPORABE["GENI_SOUPORABA"],
                        cena_souporabe_eur_kwh=0.05)
res["Prejemnik"].izpis()
d = res["Prejemnik"].diagnostika
print(f"    izrabljena souporaba {d['deljeno_kwh']:.1f} kWh | "
      f"NEIZRABLJENA (propade) {d['neizrabljena_souporaba_kwh']:.1f} kWh")

print("\n  Vpliv pogodbenega dogovora o plačilu za NEIZRABLJENO energijo:")
for placilo, opis in [(True, "placa se vsa prenesena energija"),
                      (False, "placa se le dejansko izrabljena")]:
    rr = obracun_souporabe(ud, podatki, LETO, MESEC,
                           storitev=STORITVE_SOUPORABE["GENI_SOUPORABA"],
                           cena_souporabe_eur_kwh=0.05,
                           placilo_za_neizrabljeno=placilo)
    print(f"    {opis:<34} oddajnik {rr['Oddajnik'].za_placilo:8.2f} | "
          f"prejemnik {rr['Prejemnik'].za_placilo:7.2f}")

print("\n  Vpliv deleža souporabe (placilo le za izrabljeno):")
for dz in (0.0, 0.2, 0.4, 0.7, 1.0):
    u2 = {k: dict(v) for k, v in ud.items()}
    u2["Oddajnik"]["gospodinjstvo"] = Gospodinjstvo("Oddajnik", DOG, ima_pv=True,
        shema_samooskrbe=Shema.NOVA, vloga_souporaba=Vloga.ODDAJNIK, delez_souporabe=dz)
    rr = obracun_souporabe(u2, podatki, LETO, MESEC,
                           storitev=STORITVE_SOUPORABE["GENI_SOUPORABA"],
                           cena_souporabe_eur_kwh=0.05, placilo_za_neizrabljeno=False)
    a, b = rr["Oddajnik"], rr["Prejemnik"]
    print(f"    delež {dz:>4.0%}   oddajnik {a.za_placilo:8.2f} | "
          f"prejemnik {b.za_placilo:7.2f} | skupaj {a.za_placilo+b.za_placilo:8.2f}")

print("\n  Primerjava: brez souporabe bi prejemnik plačal")
o = MesecniObracun(LETO, MESEC, Gospodinjstvo("Prejemnik", DOG), PAKETI["GENI_REDNI"])
from si_obracun import dobava
for a, b in zip(odd, prej):
    o.dodaj(dobava(b[3], b[1], b[0], INT, paket=PAKETI["GENI_REDNI"], pravila=o.pravila))
print(f"    {o.zakljuci().za_placilo:.2f} EUR")

print("\n" + "="*78); print("3. ENAKE PONUDBE"); print("="*78)
r = najdi_enake_ponudbe()
print("  Popolnoma enake (cene + nadomestilo):")
for g in r["popolnoma_enaki"] or [[]]:
    print("    " + (" == ".join(f"{p.dobavitelj}/{p.id}" for p in g) if g else "—"))
print("  Enake cene prevzema (razlika v odkupu ali nadomestilu):")
for g in r["enak_prevzem"]:
    print("    " + " == ".join(f"{p.dobavitelj}/{p.id}" for p in g))

print("\n  Ujemanja posameznih postavk med paketi:")
polja = {"pribitek_odjem": "pribitek", "et": "ET", "vt": "VT", "mt": "MT",
         "odkup_fiksni": "odkup/nadomestna oddaja", "mesecno_nadomestilo": "nadomestilo",
         "konicna": "konična", "osnovna": "osnovna", "soncna_ns": "sončna NS"}
for f, oznaka in polja.items():
    skupine = {}
    for p in PAKETI.values():
        v = getattr(p, f)
        if v: skupine.setdefault(round(v, 6), []).append(p.id)
    for v, ids in sorted(skupine.items()):
        if len(ids) > 1:
            print(f"    {oznaka:<24}{v:<10}{', '.join(ids)}")
