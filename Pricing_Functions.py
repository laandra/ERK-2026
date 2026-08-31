"""Import shim re-exporting `New pricing functions/` under a package-safe name."""
import importlib.util
import sys
from pathlib import Path

_IMPL_DIR = Path(__file__).resolve().parent / "New pricing functions"


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _IMPL_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    # Registering before exec is what `@dataclass` needs: it looks its own class
    # up via `sys.modules[cls.__module__]` while the module body is still
    # running, and gets None (then AttributeError) if the module is not there.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PRICING = _load("_household_pricing_impl", "Pricing_Functions.py")
_INVOICE = _load("_household_invoice_impl", "si_invoice.py")
_MOC = _load("_household_moc_impl", "si_moc.py")

calculate_interval_price = _PRICING.calculate_interval_price
compute_prorated_fixed_charge_eur = _PRICING.compute_prorated_fixed_charge_eur
resolve_block_for_datetime = _PRICING.resolve_block_for_datetime
resolve_reset_window_id = _PRICING.resolve_reset_window_id

PRIVZETO_REFERENCNO_LETO = _PRICING.PRIVZETO_REFERENCNO_LETO
SUPPORTED_SCHEMES = _PRICING.SUPPORTED_SCHEMES
SCHEME_SI_DOBAVA = _PRICING.SCHEME_SI_DOBAVA
SCHEME_SI_SAMOOSKRBA = _PRICING.SCHEME_SI_SAMOOSKRBA

PAKETI = _PRICING.PAKETI
DDV = _PRICING.DDV
TipCene = _PRICING.TipCene
TipOdkupa = _PRICING.TipOdkupa

# --- Agreed billing power (dogovorjena obracunska moc), per-block kW -------
PRIKLJUCNA_MOC_3X16A_KW = _MOC.PRIKLJUCNA_MOC_3X16A_KW
minimalna_dogovorjena_moc = _MOC.minimalna_dogovorjena_moc
uskladi_bloke = _MOC.uskladi_bloke
dogovorjena_moc_iz_konic = _MOC.dogovorjena_moc_iz_konic
mesecni_razpored_moci = _MOC.mesecni_razpored
oznaka_razporeda_moci = _MOC.oznaka_razporeda
# Both shapes are accepted: {block: kW} and {month: {block: kW}}.
je_mesecni_razpored = _MOC.je_mesecni_razpored
moc_za_mesec = _MOC.moc_za_mesec

# The operator's own once-a-year proposal (URO rule), and its no-history
# fallback -- so every path settles against the same agreed-power tables.
dogovorjena_moc_operaterja = _MOC.dogovorjena_moc_operaterja
administrativna_moc = _MOC.administrativna_moc
referencno_okno = _MOC.referencno_okno
zaokrozi_moc = _MOC.zaokrozi_moc
PRIKLJUCNE_MOCI_KW = _MOC.PRIKLJUCNE_MOCI_KW
ST_KONIC = _MOC.ST_KONIC
KONICNI_BLOKI = _MOC.KONICNI_BLOKI
povprecje_najvecjih = _MOC.povprecje_najvecjih

# --- Invoice generation ----------------------------------------------------
# `si_invoice` is the ONLY invoice generator: every path (RL environment, MILP
# benchmark, real meter profiles) accumulates intervals into an InvoiceBuilder
# and reads its views from there.
InvoiceBuilder = _INVOICE.InvoiceBuilder
InvoiceAccumulator = _INVOICE.InvoiceAccumulator
build_invoice_household = _INVOICE.build_invoice_household
racun_to_line_items = _INVOICE.racun_to_line_items
aggregate_household_invoices = _INVOICE.aggregate_household_invoices
aggregate_line_items = _INVOICE.aggregate_line_items
round_invoice_rows = _INVOICE.round_invoice_rows
block_reconciliation_gap = _INVOICE.block_reconciliation_gap
composition_frame = _INVOICE.composition_frame
invoice_frame = _INVOICE.invoice_frame
invoice_total = _INVOICE.invoice_total
sum_amount = _INVOICE.sum_amount
write_rows_csv = _INVOICE.write_rows_csv
BILL_PARTS = _INVOICE.BILL_PARTS
INVOICE_DECIMALS = _INVOICE.INVOICE_DECIMALS

# --- Bills for the real meter exports in `Input data/Poraba doma/` ---------
# Loaded on first use rather than at import: `si_poraba_doma` registers its ET
# reading of the GEN-I redni cenik into `PAKETI`, and the catalogue sweeps in
# `Horizon_Comparison` would otherwise pick it up as one more price list --
# duplicating the `GENI_REDNI@1T` option they already derive for themselves.
_HOME = None
_HOME_NAMES = frozenset({
    "Household", "HOUSEHOLDS", "RACUNI_MOCI", "household_names",
    "get_household", "unconfigured_folders", "opis_prikljucka",
    "load_profile", "verify_blocks", "konice_v_oknu",
    "agreed_power_by_year", "agreed_power_schedule", "validate_agreed_power",
    "AgreedPower", "invoice_household", "InvoiceResult",
})


def __getattr__(name):
    """Resolve the `si_poraba_doma` surface on first access (PEP 562)."""
    global _HOME
    if name in _HOME_NAMES:
        if _HOME is None:
            _HOME = _load("_household_profile_impl", "si_poraba_doma.py")
        return getattr(_HOME, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _HOME_NAMES)
