"""si_moc.py — dogovorjena obračunska moč (agreed billing power) per time block.

The network power charge is `sum over blocks(agreed power [kW] x block power rate)`,
and the excess-power charge is levied on whatever the realized peak does above
that agreed power. Both therefore hang off ONE per-block vector of kW, and this
module is where that vector is built and constrained.

VIRI (preverjeno 11. 8. 2026):
  [1] URO — Dogovorjena obračunska moč
      https://www.uro.si/prenova-omreznine/dogovorjena-obracunska-moc
      => minimalne moči, pravilo o naraščanju po blokih, sprememba do 8. v
         mesecu z veljavnostjo od naslednjega meseca, brezplačno.
  [2] Akt o metodologiji za obračunavanje omrežnine za elektrooperaterje
      (Ur. l. RS 146/22 … 27/25, 76/25), 12. člen in prehodne določbe
      https://pisrs.si/pregledPredpisa?id=AKT_1266
  [3] Akt o spremembah … (Ur. l. RS 27/25, v veljavi od 24. 4. 2025)
      https://pisrs.si/pregledPredpisa?id=ANJP192
      => trajna oprostitev obračuna presežne moči za gospodinjstva do 43 kW,
         ki dogovorjene obračunske moči NE spreminjajo sama.

THREE RULES THE REGULATION IMPOSES, all available through `uskladi_bloke`:

  1. **Monotone across blocks.** The agreed power in a higher block must be
     greater than or equal to the agreed power in a lower one:
     P1 <= P2 <= P3 <= P4 <= P5. This is cheap to satisfy — block 1 costs
     3.82301 EUR/kW/month in 2026 and block 5 costs 0.00245 — so the binding
     direction is always "raise the higher blocks", never "raise block 1".
     Always applied; it costs nothing and shapes the vector correctly.
  2. **Floor.** From 1. 1. 2026, a three-phase connection at or below 43 kW may
     not go below 20 % of its connection power, and never below 2.8 kW [1].
  3. **Ceiling.** It cannot exceed the connection power from the connection
     agreement.

Rules 2 and 3 both need something the profile does not carry — the connection
power from the soglasje — so both are OPTIONAL here and off by default
(`minimalna_moc_kw=0.0`, `prikljucna_moc_kw=None`). A dataset with no connection
agreement is better modelled by letting the agreed power follow the measured
peaks than by inventing a connection size and then billing the household for
exceeding it. `minimalna_dogovorjena_moc` is kept for callers that do know the
connection power and want the real floor back.

AND ONE THE CALLER HAS TO DECIDE ABOUT: a household that never touches its
agreed power is **permanently exempt** from the excess-power charge [3]; one
that sets it itself — which is exactly what a peak-shaving battery study models
— is not. So `dogovorjena_moc_iz_konic` and `obracunava_presezno_moc` belong
together: choosing the first means paying for the second.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

BLOKI = (1, 2, 3, 4, 5)

# Priključna moč we assume where the connection agreement is not in the data.
# 3 x 16 A is the standard Slovenian household connection; operators state it as
# 11 kW in the soglasje za priključitev.
PRIKLJUCNA_MOC_3X16A_KW = 11.0

# Minimalna dogovorjena obračunska moč, trifazni priključek <= 43 kW.
# Do 31. 12. 2025: 27 % priključne moči, najmanj 3,5 kW.
# Od 1. 1. 2026:   20 % priključne moči, najmanj 2,8 kW.  [1]
MIN_TRIFAZNI_KW = {2025: 3.5, 2026: 2.8}
MIN_DELEZ_TRIFAZNI = {2025: 0.27, 2026: 0.20}
MIN_ENOFAZNI_KW = {2025: 2.0, 2026: 1.8}
MIN_DELEZ_ENOFAZNI = {2025: 0.31, 2026: 0.31}


def minimalna_dogovorjena_moc(
    prikljucna_moc_kw: float, *, faze: int = 3, leto: int = 2026
) -> float:
    """Regulatory floor on the agreed power in block 1 [kW]. [1]

    Blocks 2-5 inherit it through the monotonicity rule, so this is the floor on
    every block.
    """
    tabela = 2026 if int(leto) >= 2026 else 2025
    if int(faze) == 1:
        return max(MIN_DELEZ_ENOFAZNI[tabela] * float(prikljucna_moc_kw),
                   MIN_ENOFAZNI_KW[tabela])
    return max(MIN_DELEZ_TRIFAZNI[tabela] * float(prikljucna_moc_kw),
               MIN_TRIFAZNI_KW[tabela])


def _je_omejen(meja: Optional[float]) -> bool:
    """True if `meja` is a real bound rather than "no limit" (None / inf)."""
    return meja is not None and math.isfinite(float(meja))


def uskladi_bloke(
    po_blokih: Mapping[int, float],
    *,
    minimalna_moc_kw: float = 0.0,
    prikljucna_moc_kw: Optional[float] = None,
    bloki: Iterable[int] = BLOKI,
) -> Dict[int, float]:
    """Clip a raw per-block kW vector into one the operator would accept.

    Floor first, then ceiling, then a forward running maximum so a higher block
    is never below a lower one. Raising the higher blocks (rather than lowering
    the lower ones) is both what the rule requires and what a household wants:
    the blocks that get raised are the cheap ones.

    Both bounds are optional. `minimalna_moc_kw=0.0` and `prikljucna_moc_kw=None`
    leave the vector free, which is what a study wanting the agreed power to
    follow the measured peaks and nothing else should pass -- the monotonicity
    rule then does all the shaping, and it is the one rule here that costs
    nothing to obey.
    """
    if _je_omejen(prikljucna_moc_kw) and minimalna_moc_kw > float(prikljucna_moc_kw):
        raise ValueError(
            f"minimalna_moc_kw={minimalna_moc_kw} > prikljucna_moc_kw="
            f"{prikljucna_moc_kw}; no agreed power satisfies both bounds."
        )
    out: Dict[int, float] = {}
    tekoca = 0.0
    for b in bloki:
        kw = max(float(po_blokih.get(b, 0.0)), float(minimalna_moc_kw))
        if _je_omejen(prikljucna_moc_kw):
            kw = min(kw, float(prikljucna_moc_kw))
        tekoca = max(tekoca, kw)
        out[b] = tekoca
    return out


def dogovorjena_moc_iz_konic(
    konice_kw: Mapping[int, float],
    *,
    minimalna_moc_kw: float = 0.0,
    prikljucna_moc_kw: Optional[float] = None,
) -> Dict[int, float]:
    """Agreed power a household would sign given per-block realized peaks [kW].

    "Cover exactly the peak I actually reached, no more" — the setting that pays
    the smallest power charge among those that avoid an excess charge, once the
    floor, the ceiling and the monotonicity rule are applied.
    """
    return uskladi_bloke(konice_kw, minimalna_moc_kw=minimalna_moc_kw,
                         prikljucna_moc_kw=prikljucna_moc_kw)


def mesecni_razpored(
    konice_po_mesecih: Mapping[int, Mapping[int, float]],
    *,
    minimalna_moc_kw: float = 0.0,
    prikljucna_moc_kw: Optional[float] = None,
    zamik_mesecev: int = 1,
    prenesi_manjkajoce_bloke: bool = True,
    zacetne_konice: Optional[Mapping[int, float]] = None,
) -> Dict[int, Dict[int, float]]:
    """Month-by-month agreed power, each month set from an earlier month's peaks.

    `konice_po_mesecih` maps an absolute month id (year*12 + month - 1, the same
    id `si_konica.reset_window_id(..., 1)` produces) to the per-block peak power
    realized in that month. Blocks that never occurred in a month are simply
    absent from its dict — blocks 1 and 5 are seasonal, so every month is
    missing one of them.

    `zamik_mesecev=1` is the rule this study runs: the agreed power in force in
    month M is the peak of month M-1. The regulation lets a household change the
    figure by the 8th with effect from the following month, so a strictly
    implementable version of the same idea is `zamik_mesecev=2`; at 1 the
    household is assumed to act on the previous month's meter reading in the
    same month it closes.

    `prenesi_manjkajoce_bloke=True` carries the last month in which a block did
    occur into months where it did not. Without it every November would set its
    block 1 — the single most expensive block, 3.82301 EUR/kW/month — from an
    October in which block 1 does not exist, i.e. to the bare regulatory floor,
    and would then pay an excess charge on nearly every winter peak. The real
    rule sidesteps this by looking back a full 12 months (Oct..Sep) [1].

    The first `zamik_mesecev` months have no history inside the dataset to read.
    `zacetne_konice` is the per-block peak vector they use instead -- the meter
    history the household walked in with. Passing None makes them fall back to
    their OWN month's peaks, which is the one non-causal choice available: it
    sets January's contract from January's own outcome, so the leading month can
    never pay an excess charge and the peak-shaving signal is missing from it.
    Prefer supplying a real predecessor; `Environment._bootstrap_peak_kw` builds
    one from the last complete month of the same dataset.
    """
    if int(zamik_mesecev) < 0:
        raise ValueError("zamik_mesecev must be >= 0")

    meseci = sorted(konice_po_mesecih)
    zamik = int(zamik_mesecev)
    razpored: Dict[int, Dict[int, float]] = {}
    # Seeding the carry-forward history with the bootstrap vector matters once
    # the floor is removed: a block first billed in month 2 with no observation
    # behind it would otherwise be agreed at 0 kW and charge every kW as excess.
    zgodovina: Dict[int, float] = dict(zacetne_konice or {})
    vkljuceno = 0

    for i, mesec in enumerate(meseci):
        ref = i - zamik
        if ref < 0:
            vir: Mapping[int, float] = (
                zacetne_konice if zacetne_konice is not None else konice_po_mesecih[mesec]
            )
        else:
            while vkljuceno <= ref:
                zgodovina.update(konice_po_mesecih[meseci[vkljuceno]])
                vkljuceno += 1
            vir = zgodovina if prenesi_manjkajoce_bloke else konice_po_mesecih[meseci[ref]]
        razpored[mesec] = dogovorjena_moc_iz_konic(
            vir, minimalna_moc_kw=minimalna_moc_kw, prikljucna_moc_kw=prikljucna_moc_kw
        )
    return razpored


def je_mesecni_razpored(dogovorjena) -> bool:
    """True if `dogovorjena` is a {month id: {block: kW}} schedule rather than a
    single flat {block: kW} vector.

    The two are told apart by their values, not their keys: a schedule maps to
    dicts, a flat vector maps to numbers.
    """
    if not isinstance(dogovorjena, Mapping) or not dogovorjena:
        return False
    return all(isinstance(v, Mapping) for v in dogovorjena.values())


def moc_za_mesec(dogovorjena, mesec_id: int) -> Dict[int, float]:
    """Resolve either shape to the per-block vector in force in one month.

    A flat vector is returned unchanged, so every caller can accept both a
    household that pinned one agreed power for the whole period and one that
    re-sets it monthly. Months outside a schedule clamp to its nearest end, so a
    horizon running past the data still prices.
    """
    if not dogovorjena:
        return {}
    if not je_mesecni_razpored(dogovorjena):
        return dogovorjena
    mesec = int(mesec_id)
    if mesec in dogovorjena:
        return dogovorjena[mesec]
    meseci = sorted(dogovorjena)
    return dogovorjena[min(max(mesec, meseci[0]), meseci[-1])]


def oznaka_mej(minimalna_moc_kw: float, prikljucna_moc_kw: Optional[float]) -> str:
    """Tag fragment for the bounds on the agreed power."""
    deli = []
    if float(minimalna_moc_kw) > 0.0:
        deli.append(f"min{minimalna_moc_kw:g}")
    if _je_omejen(prikljucna_moc_kw):
        deli.append(f"max{float(prikljucna_moc_kw):g}")
    return "_".join(deli) if deli else "unbounded"


def oznaka_razporeda(
    *,
    minimalna_moc_kw: float,
    prikljucna_moc_kw: Optional[float],
    zamik_mesecev: Optional[int],
    zacetek: str = "own",
) -> str:
    """Short tag identifying the agreed-power rule a result was priced under.

    Stamped onto result rows: a bill computed against a different agreed power
    is a different number, not an older one.
    """
    meje = oznaka_mej(minimalna_moc_kw, prikljucna_moc_kw)
    if zamik_mesecev is None:
        return f"fixed_{meje}"
    return f"prev{int(zamik_mesecev)}m_{zacetek}_{meje}"
