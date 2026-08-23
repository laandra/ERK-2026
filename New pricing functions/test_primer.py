"""Demonstracija in samotestiranje."""
import datetime as dt, math, random

from si_cas import casovni_blok, bloki_v_mesecu, aktivna_tarifa, v_lokalni_cas
from si_paketi import (PAKETI, Gospodinjstvo, Shema, NezdruzljivPaket,
                       preveri_paket, zdruzljivi_paketi)
from si_obracun import (MesecniObracun, Pravila, dobava, samooskrba,
                        obracun_skupnosti, kljuc_sorazmerni)

DOG = {1: 4.6, 2: 5.1, 3: 5.1, 4: 5.1, 5: 5.1}

# ---------------------------------------------------------------- 1. VALIDACIJA
print("=" * 70); print("1. VALIDACIJA PAKETOV"); print("=" * 70)

brez_pv = Gospodinjstvo("Novak (brez PV)", DOG)
s_pv    = Gospodinjstvo("Kovač (PV, nova shema)", DOG, ima_pv=True,
                        shema_samooskrbe=Shema.NOVA)
net_m   = Gospodinjstvo("Horvat (PV, net metering)", DOG, ima_pv=True,
                        shema_samooskrbe=Shema.NET_METERING)
skup    = Gospodinjstvo("Zupan (skupnostna)", DOG, ima_pv=True,
                        shema_samooskrbe=Shema.NOVA, skupnostna=True)

testi = [
    (s_pv,    "GENI_DINAMICNI",        "PV na nesamooskrbnem dinamičnem paketu"),
    (brez_pv, "GENI_SAMO_DINAMICNI",   "brez PV na samooskrbnem paketu"),
    (net_m,   "GENI_SAMO_REDNI",       "net metering na paketu za novo shemo"),
    (s_pv,    "GENI_NETMETERING",      "nova shema na net-metering paketu"),
    (skup,    "GENI_SAMO_DINAMICNI",   "skupnostna na individualnem ceniku"),
    (skup,    "BISOL_SAMO_DINAMICNA",  "skupnostna na dovoljenem ceniku (OK)"),
    (s_pv,    "ELEN_ZANESLJIVA",       "PV pri dobavitelju brez samooskrbe"),
    (brez_pv, "ELEN_DINAMICNA",        "brez PV na dinamičnem ELEN (OK)"),
]
for g, pid, opis in testi:
    try:
        op = preveri_paket(PAKETI[pid], g, dt.date(2026, 7, 1))
        print(f"  OK      {opis}")
        for o in op: print(f"            ! {o}")
    except NezdruzljivPaket as e:
        print(f"  NAPAKA  {opis}\n            -> {e}")

print(f"\n  Združljivi paketi za '{s_pv.ime}':")
for p in zdruzljivi_paketi(s_pv, dt.date(2026, 7, 1)):
    print(f"    - {p.dobavitelj:16} {p.ime}")

# ------------------------------------------------------------ 2. SINTETIČNI PROFIL
random.seed(42)
INT = 15

def profil(leto, mesec, kwp=0.0):
    t = dt.datetime(leto, mesec, 1, tzinfo=dt.timezone.utc)
    konec = dt.datetime(leto + (mesec == 12), mesec % 12 + 1, 1, tzinfo=dt.timezone.utc)
    out = []
    while t < konec:
        lok = v_lokalni_cas(t); h = lok.hour + lok.minute / 60
        poraba = ((0.18 + 0.55 * math.exp(-((h - 7.5) ** 2) / 2)
                   + 0.75 * math.exp(-((h - 19) ** 2) / 3))
                  * (INT / 60) * random.uniform(0.75, 1.25))
        sez = 0.30 if mesec in (11, 12, 1, 2) else 0.85
        prod = (kwp * sez * math.sin(math.pi * (h - 7) / 10) * (INT / 60)
                if 7 <= h <= 17 else 0.0)
        sipx = 70 + 45 * math.sin(math.pi * (h - 5) / 14) - 25 * (11 <= h <= 15)
        out.append((t, poraba, max(prod, 0.0), sipx))
        t += dt.timedelta(minutes=INT)
    return out

# ------------------------------------------------------- 3. PRIMERJAVA PAKETOV
print("\n" + "=" * 70); print("3. PRIMERJAVA PAKETOV — julij 2026, brez PV"); print("=" * 70)
podatki = profil(2026, 7)
for pid in ["GENI_REDNI", "GENI_AKTIVNI", "GENI_DINAMICNI",
            "BISOL_FIKSNI99", "BISOL_DINAMICNI", "BISOL_DINAMICNI_PLUS",
            "PETROL_REDNI", "PETROL_FIKS2026",
            "ELEN_ZANESLJIVA", "ELEN_FIKSNI", "ELEN_DINAMICNA"]:
    p = PAKETI[pid]
    o = MesecniObracun(2026, 7, brez_pv, p)
    for ts, kwh, _, sipx in podatki:
        o.dodaj(dobava(sipx, kwh, ts, INT, paket=p, pravila=o.pravila))
    r = o.zakljuci()
    print(f"  {p.dobavitelj:17}{p.ime[:34]:36}{r.za_placilo:8.2f} EUR "
          f"(fiks {r.fiksni_del_z_ddv:5.2f} / spr {r.spremenljivi_del_z_ddv:6.2f})")

# ------------------------------------------------------------ 4. SAMOOSKRBA
print("\n" + "=" * 70); print("4. SAMOOSKRBA — julij 2026, 6 kWp"); print("=" * 70)
podatki_pv = profil(2026, 7, kwp=6.0)
for pid in ["GENI_SAMO_REDNI", "GENI_SAMO_AKTIVNI", "GENI_SAMO_DINAMICNI",
            "BISOL_SAMO_DINAMICNA", "PETROL_SAMO_MESECNI"]:
    p = PAKETI[pid]
    o = MesecniObracun(2026, 7, s_pv, p, strogo=True)
    for ts, kwh, prod, sipx in podatki_pv:
        o.dodaj(samooskrba(sipx, kwh, ts, INT, total_produced_kwh=prod,
                           paket=p, pravila=o.pravila))
    r = o.zakljuci()
    print(f"  {p.dobavitelj:16}{p.ime[:32]:34}"
          f"za plačilo {r.za_placilo:7.2f} | dobropis {r.dobropis_odkup:6.2f}")

PAK = PAKETI["GENI_SAMO_AKTIVNI"]
o = MesecniObracun(2026, 7, s_pv, PAK)
for ts, kwh, prod, sipx in podatki_pv:
    o.dodaj(samooskrba(sipx, kwh, ts, INT, total_produced_kwh=prod,
                       paket=PAK, pravila=o.pravila))
o.zakljuci().izpis()

# ------------------------------------------ 5. STARI PODATKI PO NOVIH PRAVILIH
print("\n" + "=" * 70)
print("5. PODATKI IZ 2015, OBRAČUNANI PO RAZLIČNIH REŽIMIH")
print("=" * 70)
stari = profil(2015, 1)
P = PAKETI["GENI_REDNI"]
for oznaka, prav in [
    ("dejanski režim 2026",  Pravila.veljavna(dt.date(2026, 7, 23))),
    ("režim od 2027",        Pravila.od_2027()),
    ("2027 + koledar 2027",  Pravila.od_2027(preslikaj_v_leto=2027)),
]:
    o = MesecniObracun(2015, 1, brez_pv, P, prav)
    for ts, kwh, _, sipx in stari:
        o.dodaj(dobava(sipx, kwh, ts, INT, paket=P, pravila=prav))
    r = o.zakljuci()
    print(f"  {oznaka:24}{r.za_placilo:8.2f} EUR | bloki "
          f"{sorted(r.diagnostika['prevzeto_po_blokih_kwh'])} | "
          f"omrežnina moč {r.postavke['omreznina_moc']:.2f}")

# ------------------------------------------------------------- 6. SKUPNOST
print("\n" + "=" * 70); print("6. ENERGETSKA SKUPNOST — julij 2026"); print("=" * 70)
clani = {}
for ime, delez, fak in [("Ana", 0.40, 1.0), ("Bine", 0.35, 0.8), ("Cilka", 0.25, 1.3)]:
    clani[ime] = {
        "gospodinjstvo": Gospodinjstvo(ime, DOG, ima_pv=True,
                                       shema_samooskrbe=Shema.NOVA,
                                       skupnostna=True, znacilni_primer=3),
        "paket": PAKETI["GENI_SAMO_AKTIVNI"], "delez": delez, "_f": fak}
rows = [{"utc_date": ts, "interval_minutes": INT, "market_price_mwh": sipx,
         "poraba": {i: kwh * c["_f"] for i, c in clani.items()},
         "proizvodnja_kwh": prod * 5}
        for ts, kwh, prod, sipx in profil(2026, 7, kwp=6.0)]
for ime, r in obracun_skupnosti(clani, rows, 2026, 7, kljuc=kljuc_sorazmerni).items():
    print(f"  {ime:8} za plačilo {r.za_placilo:7.2f} | prevzem {r.prevzeto_kwh:6.1f} kWh"
          f" | deljeno {r.diagnostika['deljeno_kwh']:6.1f} | "
          f"oddano {r.diagnostika['oddano_kwh']:6.1f}")
