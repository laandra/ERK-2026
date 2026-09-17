"""What the ported baseline-plus-AR forecaster has to be right about.

Same spirit as `test_hems_study.py`: not a unit-test suite for its own sake,
but the specific ways this port could be wrong and not look wrong.

    python test_hbd_forecast.py

Runs in well under a minute -- every case here is deliberately small enough
that the joint AR fit it checks against is still tractable.
"""

import sys

import numpy as np
import cvxpy as cp

import hbd_forecast as hbd

_passed, _failed = [], []


def check(name, ok, detail=""):
    (_passed if ok else _failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


def test_features():
    """The feature map is the seasonal shape; a wrong period is a wrong study."""
    spd = 48
    f = hbd.featurize_baseline(0, spd)
    check("baseline features: 1 + 2 x harmonics x periods",
          f.shape == (1 + 2 * hbd.N_HARMONICS * len(hbd.PERIOD_DAYS),),
          f"got {f.shape}")

    # The daily block is features 1..2*N_HARMONICS and must repeat every day.
    n = hbd.N_HARMONICS
    a = hbd.featurize_baseline(10, spd)
    b = hbd.featurize_baseline(10 + spd, spd)
    check("daily harmonics repeat after exactly steps_per_day",
          np.allclose(a[1:1 + 2 * n], b[1:1 + 2 * n]))

    # The weekly block is the next 2*N_HARMONICS and must NOT repeat daily --
    # otherwise the two periods have collapsed onto each other.
    check("weekly harmonics do not repeat after a day",
          not np.allclose(a[1 + 2 * n:1 + 4 * n], b[1 + 2 * n:1 + 4 * n]))
    check("weekly harmonics repeat after seven days",
          np.allclose(a[1 + 2 * n:1 + 4 * n],
                      hbd.featurize_baseline(10 + 7 * spd, spd)[1 + 2 * n:1 + 4 * n]))

    # Vectorised and scalar paths must agree: one is used for fitting and the
    # other reads naturally, and a mismatch would be invisible until the
    # forecast quietly used a different clock than the fit.
    ts = np.arange(5, 15)
    check("vectorised featurisation matches the scalar path",
          np.allclose(hbd.featurize_baseline(ts, spd),
                      np.array([hbd.featurize_baseline(t, spd) for t in ts])))

    check("l2 weights rise with harmonic index, one per sin/cos",
          np.array_equal(hbd.harmonic_weights(2), [1., 1., 2., 2.] * 3))


def test_baseline_recovers_a_known_season():
    """A pure seasonal signal must come back out, phase and all."""
    spd, days = 48, 400
    t = np.arange(spd * days)
    truth = 2.0 + 1.5 * np.sin(2 * np.pi * t / spd) + 0.4 * np.cos(
        2 * np.pi * t / (spd * 7))
    rng = np.random.default_rng(0)
    y = truth + rng.normal(0, 0.2, t.size)

    theta = hbd.train_baseline(y, spd, eta=0.5, lambd=0.01)
    fit = hbd.predict_baseline(t, theta, spd)
    rmse = float(np.sqrt(np.mean((fit - truth) ** 2)))
    check("baseline recovers a daily+weekly season under noise",
          rmse < 0.1, f"RMSE {rmse:.4f} against a 1.5 kW amplitude")

    # Extrapolation is the whole reason this stage exists: it is a function of
    # the clock, so a horizon past the fitted range must stay on the season.
    future = np.arange(t.size, t.size + spd * 30)
    truth_f = 2.0 + 1.5 * np.sin(2 * np.pi * future / spd) + 0.4 * np.cos(
        2 * np.pi * future / (spd * 7))
    err = float(np.abs(hbd.predict_baseline(future, theta, spd) - truth_f).max())
    check("baseline extrapolates 30 days past the training block",
          err < 0.2, f"max error {err:.4f}")


def test_quantile_asymmetry():
    """eta must bias the fit, and in the direction the reference's sign implies.

    This is the check that caught the port reading `eta` backwards. The
    reference writes its residual as `prediction - actual`, which flips the
    usual pinball convention: its `eta` is `1 - tau`, so eta=0.2 fits the 80th
    percentile and biases the baseline UP, not down. Pinned numerically because
    it is invisible in the formula and inverts the meaning of every eta in the
    study.
    """
    spd = 48
    t = np.arange(spd * 200)
    rng = np.random.default_rng(1)
    y = 3.0 + rng.normal(0, 1.0, t.size)

    def level(eta):
        return float(hbd.predict_baseline(
            t, hbd.train_baseline(y, spd, eta=eta, lambd=1e-3), spd).mean())

    low, mid, high = level(0.2), level(0.5), level(0.8)
    check("eta=0.5 recovers the median of a symmetric series",
          abs(mid - 3.0) < 0.1, f"{mid:.3f} against a true mean of 3.0")
    check("eta<0.5 biases the baseline UP (eta = 1 - tau, not tau)",
          low > mid + 0.5, f"eta=0.2 -> {low:.3f}, eta=0.5 -> {mid:.3f}")
    check("eta>0.5 biases it DOWN by a matching amount",
          high < mid - 0.5, f"eta=0.8 -> {high:.3f}")
    # N(3, 1): the 80th percentile is ~3.84 and the 20th ~2.16.
    check("the bias lands near the quantile 1 - eta of the data",
          abs(low - 3.84) < 0.2 and abs(high - 2.16) < 0.2,
          f"eta=0.2 -> {low:.2f} (expect ~3.84), eta=0.8 -> {high:.2f} (~2.16)")


def test_pinball_identity():
    """The lean form of the loss must equal the reference's literal one.

    `_pinball` does not write `sum(maximum(eta*r, (eta-1)*r))` as the reference
    does -- that canonicalises to gigabytes on a 35k-row block and gets the
    process OOM-killed without a traceback. It uses the identity
    `max(eta*r, (eta-1)*r) == (eta-0.5)*r + 0.5*|r|` instead. Same objective,
    so it is checked as one: numerically on values, and end-to-end by fitting
    the same data both ways.
    """
    rng = np.random.default_rng(4)
    r = rng.normal(0, 2.0, 500)
    worst = 0.0
    for eta in (0.1, 0.2, 0.5, 0.8, 0.95):
        literal = np.maximum(eta * r, (eta - 1.0) * r).sum()
        lean = (eta - 0.5) * r.sum() + 0.5 * np.abs(r).sum()
        worst = max(worst, abs(literal - lean))
    check("pinball: the lean form equals the reference's maximum() form",
          worst < 1e-9, f"max discrepancy {worst:.2e} over five quantiles")

    # And that the fit it produces is the same fit. Small enough to solve the
    # literal form without the memory blow-up that motivated the change.
    spd = 48
    t = np.arange(spd * 60)
    y = 1.0 + np.sin(2 * np.pi * t / spd) + rng.normal(0, 0.3, t.size)
    eta, lambd = 0.3, 0.1
    lean_theta = hbd.train_baseline(y, spd, eta=eta, lambd=lambd)

    X = hbd.featurize_baseline(t, spd)
    th = cp.Variable(X.shape[1])
    res = X @ th - y
    cp.Problem(cp.Minimize(
        cp.sum(cp.maximum(eta * res, (eta - 1) * res))
        + lambd * cp.sum_squares(cp.multiply(hbd.harmonic_weights(), th[1:]))
    )).solve(solver=hbd.SOLVER)

    err = float(np.abs(hbd.predict_baseline(t, lean_theta, spd)
                       - X @ th.value).max())
    check("pinball: both forms fit the same baseline",
          err < 1e-4, f"max |lean - literal| = {err:.2e} kW")


def test_ar_column_split_matches_joint_fit():
    """THE load-bearing claim of `train_ar_model`.

    The reference solves one problem for the whole (M, L) matrix. This port
    solves L problems, one per column, on the grounds that the objective is
    separable across columns. If that reasoning is wrong the forecaster is
    silently fitting a different model to the paper's -- so it is checked
    against the joint fit on a case small enough to run both.
    """
    rng = np.random.default_rng(2)
    M, L, n = 6, 4, 400
    e = np.zeros(n + M + L)
    for i in range(1, e.size):
        e[i] = 0.75 * e[i - 1] + rng.normal(0, 0.3)

    eta, lambd = 0.35, 0.1
    split = hbd.train_ar_model(e, M, L, eta=eta, lambd=lambd)

    X, y = hbd.featurize_residual(e, M, L)
    G = cp.Variable((M, L))
    r = X @ G - y
    cp.Problem(cp.Minimize(
        cp.sum(cp.maximum(eta * r, (eta - 1) * r)) + lambd * cp.sum_squares(G)
    )).solve(solver=hbd.SOLVER)

    err = float(np.abs(split - G.value).max())
    check("AR column split reproduces the joint fit to solver tolerance",
          err < 1e-5, f"max |split - joint| = {err:.2e}")


def test_ar_recovers_an_ar1():
    """An AR(1) must show up as geometric decay on the most recent lag."""
    rng = np.random.default_rng(3)
    M, L, phi = 6, 4, 0.75
    e = np.zeros(6000)
    for i in range(1, e.size):
        e[i] = phi * e[i - 1] + rng.normal(0, 0.3)

    # Small lambd: the l2 penalty shrinks coefficients toward zero, and this
    # check is about the SHAPE the fit finds, not its magnitude.
    G = hbd.train_ar_model(e, M, L, eta=0.5, lambd=1e-4)
    last = G[M - 1, :]
    check("AR: the most recent lag carries the weight",
          np.abs(G[:M - 1, :]).max() < np.abs(last).min(),
          f"newest lag {np.round(last, 3)} vs older max "
          f"{np.abs(G[:M - 1, :]).max():.3f}")
    check("AR: that weight decays geometrically with horizon",
          np.all(np.diff(last) < 0) and abs(last[0] - phi) < 0.1,
          f"{np.round(last, 3)} against phi^k "
          f"{np.round([phi ** (k + 1) for k in range(L)], 3)}")


def test_compose_is_baseline_plus_correction():
    """The composition, and the part of it the study's causality rests on."""
    M, L = 4, 3
    past = np.array([1.0, 2.0, 3.0, 4.0])
    past_bl = np.zeros(M)
    fut_bl = np.array([10.0, 10.0, 10.0])
    G = np.zeros((M, L))
    G[M - 1, 0] = 0.5

    out = hbd.compose_forecast(past, past_bl, fut_bl, G, -1e9, 1e9)
    check("compose: forecast = baseline + residual @ Gamma",
          np.allclose(out, [12.0, 10.0, 10.0]), f"got {out}")

    check("compose: Gamma=None is the baseline-only ablation",
          np.allclose(hbd.compose_forecast(past, past_bl, fut_bl, None, -1e9, 1e9),
                      fut_bl))

    check("compose: clipping is applied to the composed forecast",
          np.allclose(hbd.compose_forecast(past, past_bl, fut_bl, G, 0.0, 11.0),
                      [11.0, 10.0, 10.0]))

    # A horizon longer than Gamma has columns must fall back to the baseline
    # rather than raise or truncate -- that tail is what serves a long horizon.
    long_bl = np.full(10, 10.0)
    tail = hbd.compose_forecast(past, past_bl, long_bl, G, -1e9, 1e9)
    check("compose: past Gamma's L columns the baseline stands alone",
          tail.size == 10 and np.allclose(tail[1:], 10.0) and tail[0] == 12.0)


def test_residual_windows():
    """Sliding windows, because a stride bug here shifts every target by a step."""
    obs = np.arange(10.0)
    X, y = hbd.featurize_residual(obs, 3, 2)
    check("residual windows: X is the lookback, y the next L, aligned",
          np.array_equal(X[0], [0, 1, 2]) and np.array_equal(y[0], [3, 4])
          and np.array_equal(X[-1], [5, 6, 7]) and np.array_equal(y[-1], [8, 9]),
          f"{X.shape} / {y.shape}")

    raised = False
    try:
        hbd.featurize_residual(np.arange(3.0), 3, 2)
    except ValueError:
        raised = True
    check("too short a series raises instead of returning an empty fit", raised)


if __name__ == "__main__":
    print("hbd_forecast: the ported method's own invariants\n")
    test_features()
    test_baseline_recovers_a_known_season()
    test_quantile_asymmetry()
    test_pinball_identity()
    test_ar_column_split_matches_joint_fit()
    test_ar_recovers_an_ar1()
    test_compose_is_baseline_plus_correction()
    test_residual_windows()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED: " + ", ".join(_failed))
    sys.exit(1 if _failed else 0)
