"""Root-level import shim.

`Environment.py` / `Basic_Functions.py` import pricing functions from the
repo root (`from Pricing_Functions import calculate_interval_price`), but the
actual implementation lives in `New pricing functions/Pricing_Functions.py`
(a directory name with a space, which can't be imported as a normal Python
package). This shim loads that file directly via `importlib` and re-exports
its public surface.
"""
import importlib.util
from pathlib import Path

_IMPL_FILE = Path(__file__).resolve().parent / "New pricing functions" / "Pricing_Functions.py"
_SPEC = importlib.util.spec_from_file_location("_household_pricing_impl", _IMPL_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Aus_Base = _MODULE.Aus_Base
calculate_interval_price = _MODULE.calculate_interval_price
list_pricing_schemes = _MODULE.list_pricing_schemes
resolve_block_for_datetime = _MODULE.resolve_block_for_datetime
resolve_reset_window_id = _MODULE.resolve_reset_window_id
compute_prorated_fixed_charge_eur = _MODULE.compute_prorated_fixed_charge_eur

PRIVZETO_REFERENCNO_LETO = _MODULE.PRIVZETO_REFERENCNO_LETO

SUPPORTED_SCHEMES = _MODULE.SUPPORTED_SCHEMES
SKIPPED_MULTI_USER_SCHEMES = _MODULE.SKIPPED_MULTI_USER_SCHEMES
SCHEME_AUS_BASE = _MODULE.SCHEME_AUS_BASE
SCHEME_SI_DOBAVA = _MODULE.SCHEME_SI_DOBAVA
SCHEME_SI_SAMOOSKRBA = _MODULE.SCHEME_SI_SAMOOSKRBA

# --- Invoice generation (monthly / whole-period line-item bills) -----------
_INVOICE_FILE = Path(__file__).resolve().parent / "New pricing functions" / "si_invoice.py"
_INVOICE_SPEC = importlib.util.spec_from_file_location("_household_invoice_impl", _INVOICE_FILE)
_INVOICE_MODULE = importlib.util.module_from_spec(_INVOICE_SPEC)
_INVOICE_SPEC.loader.exec_module(_INVOICE_MODULE)

InvoiceBuilder = _INVOICE_MODULE.InvoiceBuilder
build_invoice_household = _INVOICE_MODULE.build_invoice_household
racun_to_line_items = _INVOICE_MODULE.racun_to_line_items
aggregate_line_items = _INVOICE_MODULE.aggregate_line_items
write_rows_csv = _INVOICE_MODULE.write_rows_csv
