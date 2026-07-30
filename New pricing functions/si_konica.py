"""si_konica.py — ratchet peak-power ("konica") tracking helpers (single-user).

The real Slovenian regulation bills excess power ("presezna moc") per calendar
month as sqrt(sum of squared exceedances observed that month) — a nonlinear,
monthly-resetting formula (see `si_obracun.MesecniObracun`). That formula can't
be optimized by a linear MILP solver, and gives a poor per-step RL reward
signal (all-or-nothing at month end).

This module implements a deliberately simplified, user-confirmed alternative:
track a running "peak power observed so far" per tariff time-block that only
ever increases and does not automatically reset every month (it carries
forward indefinitely by default, or every N months if configured). The excess
charge is `rate * faktor_presezne_moci * max(0, running_peak - contracted)`.

Because this is expressed as a running maximum, the *marginal* charge for a
single interval telescopes exactly to the same total as a single "peak over
the whole horizon" charge — see the proof in the project's plan document.
This lets the MILP (batch, sees the whole horizon) and the RL environment
(online, one interval at a time) charge provably consistent totals for the
same trajectory.
"""
from __future__ import annotations

from typing import Dict, Optional


def reset_window_id(year: int, month: int, peak_reset_months: Optional[int]) -> int:
    """Monotonic window id for a given (year, month).

    `peak_reset_months=None` (or <= 0) means a single global window for the
    entire dataset, i.e. the running peak never resets. A positive integer N
    groups consecutive N-month spans into the same window id; the running
    peak resets to zero at the start of each new window.
    """
    absolute_month = int(year) * 12 + (int(month) - 1)
    if peak_reset_months is None or peak_reset_months <= 0:
        return 0
    return absolute_month // int(peak_reset_months)


def update_running_peak(
    prev_peak_kw: Optional[Dict[int, float]],
    blok: int,
    power_kw: float,
) -> Dict[int, float]:
    """Return a NEW dict with block `blok`'s peak ratcheted up to
    max(prev_peak_kw.get(blok, 0.0), power_kw). Other blocks are copied
    unchanged. Never mutates the input dict."""
    new_peak = dict(prev_peak_kw or {})
    prior = float(new_peak.get(blok, 0.0))
    new_peak[blok] = max(prior, float(power_kw))
    return new_peak


def marginal_excess_charge_eur(
    prev_peak_kw: float,
    new_peak_kw: float,
    contracted_kw: float,
    rate_eur_per_kw: float,
    faktor_presezne_moci: float,
) -> float:
    """Telescoping per-step excess-power charge for one block.

    Summed over consecutive intervals with a fixed `contracted_kw`/rate, this
    equals `rate*faktor*max(0, final_peak-contracted)` exactly (telescoping
    sum) — i.e. charging the marginal peak increase each time a new peak is
    set gives the same total as charging once for the final peak.
    """
    prev_excess = max(0.0, float(prev_peak_kw) - float(contracted_kw))
    new_excess = max(0.0, float(new_peak_kw) - float(contracted_kw))
    return (new_excess - prev_excess) * float(rate_eur_per_kw) * float(faktor_presezne_moci)
