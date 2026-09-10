"""Baseline-plus-residual-AR forecasting, ported from cvxgrp/home-battery-dispatch.

The method behind Perez-Pineiro, Skogestad & Boyd, "Home battery dispatch under
a tiered peak power tariff" (Optimization and Engineering, 2026), whose code is
at https://github.com/cvxgrp/home-battery-dispatch (`hbd/forecast.py`). This
module is deliberately a near-transcription of that file so the two stay
diffable; every place it departs is marked DEPARTURE with the reason.

Two stages:

  1. A seasonal BASELINE: Fourier features at the daily, weekly and annual
     periods, fit by quantile (pinball) regression with an l2 penalty that is
     weighted by harmonic index, so the high harmonics are damped harder than
     the low ones. This is a pure function of the clock -- given `t` it needs no
     data at all, which is what lets it extrapolate to any horizon.

  2. A residual AR: one matrix `Gamma` of shape (M, L) mapping the last M
     residuals straight onto the next L. Direct multi-horizon, not recursive, so
     a horizon-20 forecast is fit against horizon-20 truth rather than
     accumulated from twenty one-step errors.

The forecast is `baseline + past_residuals @ Gamma`, clipped to the range the
training data actually occupied, and baseline-only beyond L.

Why this is worth having in a study whose roster is already eight methods: the
naive kinds read recent actuals but fit nothing, and Prophet fits but reads
nothing at prediction time -- within a refit block its answer for tomorrow 08:00
does not depend on what happened at 07:00 today. Stage 2 is exactly that missing
term, and stage 1 alone (`use_ar=False`) is the ablation that prices it.
"""

import os

import numpy as np
import cvxpy as cp

# Harmonics per period. Four is the paper's choice: enough to carry a
# double-humped weekday load curve, few enough that the l2 weighting still has
# something to damp.
N_HARMONICS = 4

# Daily, weekly, annual -- expressed in DAYS and scaled by steps_per_day at use,
# because this study is 30-minute and the reference is hourly.
PERIOD_DAYS = (1.0, 7.0, 365.0)

# Default regularisation, from the reference.
LAMBDA = 0.1

# The solver. CLARABEL ships with cvxpy and handles these QPs; the reference
# used MOSEK, which is licensed. `train_ar_model`'s column split (below) is what
# keeps the AR stage within reach of an open solver.
SOLVER = cp.CLARABEL


def featurize_baseline(t, steps_per_day: int,
                       n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """Fourier features for absolute step index `t`.

    `t` may be a scalar or an array; the return is (n_features,) or
    (len(t), n_features). Feature 0 is the constant, then sin/cos pairs ordered
    period-major, which is the order `harmonic_weights` assumes.

    DEPARTURE from the reference: vectorised over `t` and parameterised by
    `steps_per_day`. The reference builds one row at a time in a Python loop at
    a fixed hourly step, which costs minutes on a 35k-row training block.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    periods = _harmonic_periods(steps_per_day, n_harmonics)
    ang = 2.0 * np.pi * t[:, None] / periods[None, :]

    feats = np.empty((t.size, 1 + 2 * periods.size), dtype=float)
    feats[:, 0] = 1.0
    feats[:, 1::2] = np.sin(ang)
    feats[:, 2::2] = np.cos(ang)

    return feats[0] if scalar else feats


def _harmonic_periods(steps_per_day: int, n_harmonics: int) -> np.ndarray:
    """Every harmonic's period, in steps, period-major."""
    return np.array(
        [p * steps_per_day / n
         for p in PERIOD_DAYS
         for n in range(1, n_harmonics + 1)],
        dtype=float,
    )


def harmonic_weights(n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """Per-coefficient l2 weights, aligned with `featurize_baseline` minus its constant.

    The reference calls this `sqrt_mu`: harmonic n is penalised n times as hard
    as the fundamental, so the fit prefers smooth seasonal shape over wiggles it
    cannot support. Each harmonic owns a sin and a cos, hence the repeat.
    """
    per_period = np.repeat(np.arange(1, n_harmonics + 1), 2)
    return np.tile(per_period, len(PERIOD_DAYS)).astype(float)


def _pinball(r: cp.Expression, eta: float) -> cp.Expression:
    """Sum of the pinball loss, with `r` the residual PREDICTION MINUS ACTUAL.

    Read the direction carefully, because it is the reverse of the usual
    convention and the reference's choice of eta only makes sense once you have.
    Standard pinball for quantile tau is written on `y - yhat`; this is written
    on `yhat - y`, so:

        over-predict  (r > 0):  costs eta       per unit
        under-predict (r < 0):  costs 1 - eta   per unit

    Under-prediction is therefore the EXPENSIVE side when eta < 0.5, and the fit
    is pulled UP. `eta` here is `1 - tau` in standard notation: the paper's
    eta=0.2 for load fits the 80th percentile, a deliberately HIGH baseline.

    That is the conservative direction for its problem -- under a tiered peak
    power tariff, assuming more load than typical keeps the plan clear of a tier
    threshold, while assuming less walks into one. Whether the same direction
    pays here is an open question and not one to answer by guessing: the
    quantile sweep is a planned experiment of its own.

    Kept in the reference's parameterisation rather than converted to tau, so
    this file still diffs cleanly against `hbd/forecast.py` upstream.

    DEPARTURE from the reference in FORM but not in value. It writes the loss as
    `sum(maximum(eta * r, (eta - 1) * r))`, which makes cvxpy canonicalise two
    full-length expressions and an elementwise max over them -- 0.69 GB at four
    harmonics and 1.61 GB at eight, on a 35k-row training block, which is enough
    to get the process OOM-killed with no traceback. The identity

        max(eta * r, (eta - 1) * r) == (eta - 0.5) * r + 0.5 * |r|

    (check both signs of r) turns it into one linear term plus an l1 norm, which
    cvxpy canonicalises directly. Same objective, a fraction of the memory.
    `test_pinball_identity` pins the two forms together.
    """
    return (eta - 0.5) * cp.sum(r) + 0.5 * cp.norm1(r)


def train_baseline(y: np.ndarray,
                   steps_per_day: int,
                   eta: float = 0.5,
                   lambd: float = LAMBDA,
                   n_harmonics: int = N_HARMONICS,
                   t0: int = 0,
                   solver=SOLVER) -> np.ndarray:
    """Fit the seasonal baseline. Returns theta, shape (1 + 6 * n_harmonics,).

    `t0` is the absolute step index of `y[0]`. It matters: the features are a
    function of absolute time, so fitting at one origin and predicting at
    another shifts every seasonal phase. Callers pass the offset of the training
    slice within the full series.
    """
    y = np.asarray(y, dtype=float)
    X = featurize_baseline(np.arange(t0, t0 + y.size), steps_per_day, n_harmonics)

    theta = cp.Variable(X.shape[1])
    r = X @ theta - y
    weights = harmonic_weights(n_harmonics)

    objective = _pinball(r, eta) + lambd * cp.sum_squares(
        cp.multiply(weights, theta[1:]))
    cp.Problem(cp.Minimize(objective)).solve(solver=solver)

    if theta.value is None:
        raise RuntimeError("baseline fit did not converge")
    return np.asarray(theta.value, dtype=float)


def predict_baseline(t, theta: np.ndarray, steps_per_day: int,
                     n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """The baseline at absolute step index/indices `t`.

    Pure clock arithmetic, no data -- which is why this and nothing else in the
    roster can be evaluated an arbitrary distance into the future.
    """
    return featurize_baseline(t, steps_per_day, n_harmonics) @ theta


def featurize_residual(obs: np.ndarray, M: int, L: int) -> tuple:
    """Sliding windows over a residual series.

    Returns (X, y) with X (n, M) the lookback and y (n, L) the target. Built as
    strided views rather than a Python loop: at M = L = 48 over 35k rows the
    reference's loop copies ~27 MB one row at a time.
    """
    obs = np.asarray(obs, dtype=float)
    n = obs.size - M - L + 1
    if n <= 0:
        raise ValueError(
            f"need more than M + L - 1 = {M + L - 1} residuals to fit an AR "
            f"model, got {obs.size}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(obs, M + L)[:n]
    # Copies, because cvxpy and the solver want contiguous arrays anyway.
    return np.ascontiguousarray(windows[:, :M]), np.ascontiguousarray(windows[:, M:])


def train_ar_model(residuals: np.ndarray,
                   M: int,
                   L: int,
                   eta: float = 0.5,
                   lambd: float = LAMBDA,
                   solver=SOLVER,
                   max_samples: int | None = None,
                   seed: int = 0,
                   column_cache=None) -> np.ndarray:
    """Fit the residual AR. Returns Gamma, shape (M, L).

    DEPARTURE from the reference, and the one that makes this runnable: the
    reference solves for the whole (M, L) matrix in ONE problem, which at
    30-minute resolution is 35k x 48 x 48 = 80M pinball terms and needs MOSEK.

    The objective is separable across the COLUMNS of Gamma. Both terms are
    elementwise -- the pinball loss sums over entries of `X @ Gamma - y`, and
    the l2 penalty sums over entries of Gamma -- and column j of `X @ Gamma`
    depends only on column j of Gamma. So

        min_Gamma  sum_j [ pinball(X @ Gamma[:, j] - y[:, j]) + lambd * ||Gamma[:, j]||^2 ]

    decomposes into L independent problems of M variables each, whose stacked
    solution is the exact minimiser of the joint problem. Not an approximation:
    `test_hbd_forecast.py::test_ar_column_split_matches_joint_fit` holds the two
    to agree on a small case where the joint fit is still tractable.

    `max_samples` subsamples the training windows if a caller needs the fit
    cheaper still; None (the default) uses all of them.

    `column_cache` is an optional `j -> path` callable. The split above makes
    each column an independent problem, so each is also an independent unit of
    WORK: given a cache, a fit interrupted at column 30 resumes there instead of
    starting over. At a few seconds per column and 48 of them per channel, that
    is what makes the fit survivable on a machine that will not let one process
    run for ten minutes.
    """
    X, y = featurize_residual(residuals, M, L)

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(X.shape[0], size=max_samples, replace=False))
        X, y = X[keep], y[keep]

    Gamma = np.empty((M, L), dtype=float)
    for j in range(L):
        path = column_cache(j) if column_cache is not None else None
        if path is not None and os.path.exists(path):
            Gamma[:, j] = np.load(path)
            continue

        g = cp.Variable(M)
        r = X @ g - y[:, j]
        objective = _pinball(r, eta) + lambd * cp.sum_squares(g)
        cp.Problem(cp.Minimize(objective)).solve(solver=solver)
        if g.value is None:
            raise RuntimeError(f"AR fit did not converge for horizon step {j}")
        Gamma[:, j] = g.value

        if path is not None:
            # The suffix must stay .npy: np.save APPENDS it to any name
            # that lacks one, so a plain ".tmp" writes ".tmp.npy" and the
            # rename below then looks for a file that was never created.
            tmp = f"{path}.{os.getpid()}.tmp.npy"
            np.save(tmp, Gamma[:, j])
            os.replace(tmp, path)      # atomic: the sweep is multi-process
    return Gamma


def compose_forecast(past: np.ndarray,
                     past_baseline: np.ndarray,
                     future_baseline: np.ndarray,
                     Gamma: np.ndarray | None,
                     clip_min: float,
                     clip_max: float) -> np.ndarray:
    """Baseline plus the AR correction, clipped. Length is len(future_baseline).

    `past` and `past_baseline` are the M steps immediately before the forecast
    starts. Beyond Gamma's L columns the baseline stands alone, which is how the
    reference reaches horizons longer than the AR was fit for.

    DEPARTURE from the reference: it prepends the realised observation at the
    anchor (`curr_load = load_data.iloc[t]`) and forecasts only from t+1. Here
    the anchor interval is not yet over when the plan is made, so reading it is
    the leak `ReactiveController(leak_current_interval=True)` exists to measure.
    Every step is forecast from strictly-past data instead.
    """
    horizon = future_baseline.size
    out = np.array(future_baseline, dtype=float, copy=True)

    if Gamma is not None:
        residual = np.asarray(past, dtype=float) - np.asarray(past_baseline, dtype=float)
        correction = residual @ Gamma
        n = min(horizon, correction.size)
        out[:n] += correction[:n]

    return np.clip(out, clip_min, clip_max)
