"""Root-level import shim.

`Environment.py`, `MILP_Benchmark.py` and `multi_household_tools.py` import
pricing functions from the repo root (`from Pricing_Functions import
calculate_interval_price`), but the actual implementation lives in
`New pricing functions/Pricing_Functions.py` (a directory name with a space,
which can't be imported as a normal Python package). This shim loads that file
directly via `importlib` and re-exports the surface used from outside that
folder.
"""
import importlib.util
from pathlib import Path

_IMPL_DIR = Path(__file__).resolve().parent / "New pricing functions"


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _IMPL_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PRICING = _load("_household_pricing_impl", "Pricing_Functions.py")
_INVOICE = _load("_household_invoice_impl", "si_invoice.py")

calculate_interval_price = _PRICING.calculate_interval_price
compute_prorated_fixed_charge_eur = _PRICING.compute_prorated_fixed_charge_eur
resolve_block_for_datetime = _PRICING.resolve_block_for_datetime
resolve_reset_window_id = _PRICING.resolve_reset_window_id

PRIVZETO_REFERENCNO_LETO = _PRICING.PRIVZETO_REFERENCNO_LETO
SUPPORTED_SCHEMES = _PRICING.SUPPORTED_SCHEMES
SCHEME_SI_DOBAVA = _PRICING.SCHEME_SI_DOBAVA
SCHEME_SI_SAMOOSKRBA = _PRICING.SCHEME_SI_SAMOOSKRBA

# --- Invoice generation (monthly / whole-period line-item bills) -----------
InvoiceBuilder = _INVOICE.InvoiceBuilder
aggregate_household_invoices = _INVOICE.aggregate_household_invoices
