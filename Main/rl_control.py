"""Learning-based battery controllers: DQN reinforcement learning and
behaviour cloning of the whole-period MILP.

The comparison the study runs is "what does a forecast buy a controller a
household could install". The rules answer it for hand-written logic and the
MPC-MILP arms for optimization; these two answer it for LEARNED logic:

    rl_dqn   a Double-DQN trained on the TRAINING period only, rewarded by the
             arm's own settlement -- the same `settle` callable every rule and
             MILP trajectory is priced through -- plus the same wear shadow
             price the MILP objective carries and the same terminal-SOC
             close-out `Cost_EUR_Closed` charges. The agent therefore optimizes
             the quantity it is later scored on, and nothing else.
    il_bc    behaviour cloning of the whole-period MILP solved over the last
             training year: the optimum demonstrates, the network imitates.
             What survives the copy is what a reactive map can express of a
             perfect-foresight plan.
    bc_dqn   the cloned network fine-tuned by the DQN: imitation supplies the
             prior, reinforcement corrects it against the actual bill.

Load-bearing constraints, matching the study's invariants:

    one evaluator   Training rewards come from the arm's `settle`; scoring goes
                    through `Rule_Based_Control.run_policy` with that same
                    `settle`, endogenous contract convergence included. The
                    agent is a `Policy`; nothing about the accounting is
                    re-implemented here.
    causality       The agent trains on data that PRECEDES the scored year, and
                    its features are causal: the meter's current interval, the
                    published day-ahead prices, and forecasts built from
                    trailing history (`median14_forecast` reads backwards only).
                    "truth" channels exist as deliberate diagnostics, exactly
                    like `price_oracle`, and are flagged the same way.
    one battery     Action -> setpoint goes through the same rule helpers
                    (`_grid_charge_room`, `_cover_load`) and the runner's
                    envelope clamps; the training loop uses
                    `Basic_Functions.max_charge_now`/`max_discharge_now`
                    directly, so training and scoring drive the same pack.

The action set is the environment's own legacy discrete set (see
`Environment.ACTION_*`): charge from anywhere, charge from PV surplus, cover
the load, discharge fully, idle. Five semantic actions rather than a
continuous setpoint because every one of them is feasible by construction --
the peak-aware charge room and the SOC envelope are applied inside the
mapping -- so the agent cannot learn its way around the physics, only around
the tariff. A continuous head was considered and rejected: it buys precision a
1.5 kW inverter on 30-minute intervals cannot express, at the price of a far
harder exploration problem on a reward dominated by rare peak charges.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from Basic_Functions import max_charge_now, max_discharge_now
import Rule_Based_Control as rbc

_EPS = 1e-12
N_ACTIONS = 5
A_CHARGE_ANY, A_CHARGE_PV, A_DISCHARGE_HOME, A_DISCHARGE_ANY, A_IDLE = range(5)
_BLOCKS = (1, 2, 3, 4, 5)

# Torch on one thread: the driver parallelises across households, and BLAS
# threads multiplying across workers is the same oversubscription the sweep
# guards against for HiGHS.
torch.set_num_threads(1)

# WHICH LEARNING RULE produced a result. Bumped whenever the update, the
# exploration or the reward changes shape in a way a stored result cannot be
# compared across -- the same job `solver=SOLVER_NAME` does in the study's own
# run config. A TrainConfig field cannot cover this: truncating the n-step
# return at exploratory continuations changed every DQN number while every
# setting kept its value, so a digest over the settings alone reported four
# superseded runs as cached and silently skipped them.
#
#   1  first screen: 1-step Double DQN, dueling head, baseline-subtracted
#      reward, unprotected warm start
#   2  n-step returns, BC-regularised fine-tuning, deep-copied BC prior
#   3  n-step returns truncated at the first exploratory continuation
#   4  episodes, validation rollouts and teacher walks inherit the ratchet
#      peak state of the window they start inside, instead of starting at 0
ALGO_VERSION = 4

# The same stamp for the SUPERVISED half, versioned apart because the two
# change on different days: a fix to the Q-update cannot move a clone, so
# sharing one number would discard every clone on disk at each RL tuning pass.
# It must still be bumped for anything the clone DOES depend on -- the teacher
# walk is the trap, since ALGO_VERSION 4 reseeded it and a BC result pinned to
# a stale version would have been served from cache under new labels.
#
#   1  first screen
#   2  teacher walk inherits the ratchet peak state of its starting window
BC_ALGO_VERSION = 2


# ---------------------------------------------------------------------------
# Forecasts the agent may see
# ---------------------------------------------------------------------------
def median14_forecast(values: np.ndarray, spd: int, window_days: int = 14) -> np.ndarray:
    """Per-interval-of-day trailing median -- the study's fit-free winner.

    For day d, interval j: the median of days d-window..d-1 at interval j.
    Strictly backward-looking, so it is causal everywhere; the first day has no
    history and falls back to its own values, which only ever touches the first
    training day, never the scored year.
    """
    values = np.asarray(values, dtype=float)
    n_days = len(values) // spd
    daily = values[: n_days * spd].reshape(n_days, spd)
    out = np.empty_like(daily)
    out[0] = daily[0]
    for d in range(1, n_days):
        lo = max(0, d - window_days)
        out[d] = np.median(daily[lo:d], axis=0)
    flat = out.reshape(-1)
    if len(values) > len(flat):                     # ragged tail: repeat last day
        flat = np.concatenate([flat, out[-1][: len(values) - len(flat)]])
    return flat


# ---------------------------------------------------------------------------
# What the agent is allowed to look at
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """One observation contract. The ablation grid is a grid over these.

    `horizon` is how far ahead the lookahead bins reach, in steps -- the RL
    analogue of the MPC arms' control horizon. `load_channel` / `pv_channel`
    pick what fills them: "none" (no forecast at all), "fc" (the causal
    median14), or "truth" (perfect foresight -- a diagnostic, like
    `price_oracle`, never a deployable claim). `price_features` carries the
    published day-ahead prices; dropping it asks whether a tariff's signal is
    in its prices at all. `peak_features` carries the SI contract state --
    block, agreed power, headroom, running peak; dropping it on SI asks
    whether the capacity charge can be learned blind.
    """

    horizon: int = 48
    load_channel: str = "fc"       # none | fc | truth
    pv_channel: str = "fc"         # none | fc | truth
    price_features: bool = True
    peak_features: bool = False
    n_bins: int = 4

    def __post_init__(self):
        for ch in (self.load_channel, self.pv_channel):
            if ch not in ("none", "fc", "truth"):
                raise ValueError(f"channel must be none|fc|truth, got {ch!r}")
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")

    @property
    def causal(self) -> bool:
        return self.load_channel != "truth" and self.pv_channel != "truth"

    def config(self) -> dict:
        return asdict(self)


# Bin edges as fractions of the horizon: finer near now, coarser far out,
# mirroring how forecast value decays. Fixed COUNT whatever the horizon, so
# every horizon variant trains the same architecture and the comparison is
# about information, not network size.
_BIN_EDGES = (0.0, 0.125, 0.25, 0.5, 1.0)


def _lookahead_bins(arr: np.ndarray, horizon: int, n_steps: int,
                    mean: bool) -> np.ndarray:
    """(n_steps, 4) sums (or means) of `arr` over [t+1+e_k*h, t+1+e_{k+1}*h).

    Past the end of the array the future is taken as zero -- the same "no tail
    to spare" behaviour `load_study_frames` documents for the final day.
    """
    arr = np.asarray(arr, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    n = len(arr)
    out = np.zeros((n_steps, len(_BIN_EDGES) - 1), dtype=np.float32)
    if horizon <= 0:
        return out
    t = np.arange(n_steps)
    for k in range(len(_BIN_EDGES) - 1):
        a = t + 1 + int(round(_BIN_EDGES[k] * horizon))
        b = t + 1 + max(int(round(_BIN_EDGES[k + 1] * horizon)),
                        int(round(_BIN_EDGES[k] * horizon)) + 1)
        a = np.clip(a, 0, n)
        b = np.clip(b, 0, n)
        width = np.maximum(b - a, 1)
        s = cs[b] - cs[a]
        out[:, k] = (s / width) if mean else s
    return out


class FeatureBuilder:
    """Observations for one household, one spec: static matrix + dynamic tail.

    The static half is everything that does not depend on the agent's own past
    (calendar, meter, prices, forecasts, contract), precomputed as one matrix.
    The dynamic tail -- SOC, and on SI the running peak relative to the agreed
    power -- is appended at act time. Built from a `Signals` bundle so the
    agent reads exactly what a rule may read, plus the forecast arrays it was
    explicitly granted.
    """

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.norm_mean = None
        self.norm_std = None

    # -- static ------------------------------------------------------------
    def build_static(self, sig, load_fc=None, pv_fc=None) -> np.ndarray:
        spec = self.spec
        n = sig.n_steps
        idx = sig.env.dataset.index[:n]

        hour_frac = sig.local_hour / 24.0
        doy = np.asarray([ts.dayofyear for ts in idx], dtype=float) / 365.25
        dow = np.asarray([ts.dayofweek for ts in idx], dtype=float) / 7.0

        cols = [
            np.sin(2 * np.pi * hour_frac), np.cos(2 * np.pi * hour_frac),
            np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy),
            np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
            sig.consumption[:n], sig.generation[:n],
        ]
        if spec.price_features:
            cols += [sig.import_rate[:n], sig.export_credit[:n]]
            bins = _lookahead_bins(sig.import_rate, spec.horizon, n, mean=True)
            cols += [bins[:, k] for k in range(bins.shape[1])]

        def _channel(kind, truth, fc):
            if kind == "none":
                return None
            if kind == "truth":
                return truth
            if fc is None:
                raise ValueError("spec asks for the 'fc' channel but no "
                                 "forecast array was provided")
            return np.asarray(fc, dtype=float)[:n]

        for series in (_channel(spec.load_channel, sig.consumption, load_fc),
                       _channel(spec.pv_channel, sig.generation, pv_fc)):
            if series is not None:
                bins = _lookahead_bins(series, spec.horizon, n, mean=False)
                cols += [bins[:, k] for k in range(bins.shape[1])]

        if spec.peak_features:
            hours = sig.hours
            net_kw = (sig.consumption[:n] - sig.generation[:n]) / hours
            cols += [
                sig.blocks[:n].astype(float) / 5.0,
                sig.agreed_kw[:n],
                sig.agreed_kw[:n] - net_kw,     # headroom before the line
            ]
        return np.column_stack(cols).astype(np.float32)

    # -- dynamic -----------------------------------------------------------
    @property
    def n_dynamic(self) -> int:
        return 2 if self.spec.peak_features else 1

    def dynamic(self, sig, idx, soc_kwh, peak_state) -> np.ndarray:
        soc_norm = soc_kwh / max(sig.capacity_kwh, _EPS)
        if not self.spec.peak_features:
            return np.array([soc_norm], dtype=np.float32)
        b = int(sig.blocks[idx])
        # Peak relative to the line: >0 means this month's charge is sunk up to
        # that level, which is exactly what `ratchet_aware` rules read.
        rel = float(peak_state.get(b, 0.0)) - float(sig.agreed_kw[idx])
        return np.array([soc_norm, rel / 5.0], dtype=np.float32)

    # -- normalisation -----------------------------------------------------
    def fit_norm(self, static: np.ndarray) -> None:
        self.norm_mean = static.mean(axis=0)
        self.norm_std = np.maximum(static.std(axis=0), 1e-6)

    def normalize(self, static: np.ndarray) -> np.ndarray:
        if self.norm_mean is None:
            raise RuntimeError("fit_norm was never called")
        return (static - self.norm_mean) / self.norm_std

    @property
    def dim(self) -> int:
        if self.norm_mean is None:
            raise RuntimeError("fit_norm was never called")
        return len(self.norm_mean) + self.n_dynamic


# ---------------------------------------------------------------------------
# Action -> setpoint: the five semantic actions, feasibility built in
# ---------------------------------------------------------------------------
def action_setpoint(action: int, sig, idx, lo: float, hi: float,
                    peak_state, respect_peak: bool) -> float:
    """Signed AC-side kWh for one semantic action. Always within [lo, hi].

    Charging from the grid goes through `_grid_charge_room`, so on SI an agent
    cannot buy arbitrage at the price of a new monthly peak unless the peak is
    already sunk -- the identical guard every grid-charging rule carries.
    """
    if action == A_CHARGE_ANY:
        return rbc._grid_charge_room(sig, idx, hi, peak_state, respect_peak)
    if action == A_CHARGE_PV:
        s = sig.surplus[idx]
        return min(s, hi) if s > _EPS else 0.0
    if action == A_DISCHARGE_HOME:
        return rbc._cover_load(sig, idx, lo)
    if action == A_DISCHARGE_ANY:
        return lo
    if action == A_IDLE:
        return 0.0
    raise ValueError(f"unknown action {action}")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    """One dueling MLP for both jobs: Q-values for the DQN, logits for the cloner.

    Sharing the architecture is what makes `bc_dqn` a warm start rather than a
    surgery: the cloned weights load into the Q-network verbatim, and the
    fine-tune only has to re-scale a ranking that is already right.

    Dueling (V + advantage) because of what this problem is: most of an
    interval's cost is the household's load, which no action changes, and the
    action-dependent slice is one or two cents against it. A single head has
    to carry both in one regression; splitting them lets the advantage stream
    learn the two-cent question on its own scale. Measured on the SI
    calibration household, the single-head net converged to near-idle
    (14 EFC/year) for exactly this reason.
    """

    def __init__(self, dim: int, hidden: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x):
        z = self.trunk(x)
        a = self.adv(z)
        return self.value(z) + a - a.mean(dim=1, keepdim=True)


class _Replay:
    """Flat numpy ring buffer. (obs, action, n-step return, bootstrap obs, done,
    discount).

    `disc` is gamma**k for the k transitions actually accumulated into `rew`,
    carried per entry rather than assumed: an episode that ends before the
    n-step window fills contributes a shorter return, and applying gamma**n to
    it would discount a bootstrap that is k steps away as though it were n.
    """

    def __init__(self, capacity: int, dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, dim), dtype=np.float32)
        self.nxt = np.zeros((capacity, dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.disc = np.ones(capacity, dtype=np.float32)
        self.n = 0
        self.i = 0

    def push(self, o, a, r, o2, d, disc):
        i = self.i
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.nxt[i] = o2
        self.done[i] = d
        self.disc[i] = disc
        self.i = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, rng, batch):
        j = rng.integers(0, self.n, size=batch)
        return (self.obs[j], self.act[j], self.rew[j], self.nxt[j],
                self.done[j], self.disc[j])


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything a training run depends on. Digested into the result file so
    a run under superseded settings is recomputed, not resumed into --
    the same rule the study's checkpoints follow."""

    total_steps: int = 500_000
    episode_days: int = 7
    batch: int = 128
    lr: float = 1e-3
    # 0.997, not the textbook 0.99, and it is a measurement: on 30-minute
    # intervals 0.99 discounts a noon-to-evening store by 13 % and an
    # overnight one by 38 %, which erases exactly the thin margins this study
    # trades in -- calibrated on Ausgrid 138, moving to 0.997 took the AU DQN
    # from 870 to 844 closed and the SI DQN from 449 to 439. Safe only
    # because rewards are baseline-subtracted; against raw bills this gamma
    # would put the Q-scale near 200 EUR.
    gamma: float = 0.997
    hidden: int = 128
    buffer: int = 200_000
    learn_start: int = 5_000
    update_every: int = 2
    target_sync: int = 2_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4     # fraction of total_steps to reach eps_end
    # Rewards are baseline-subtracted EUR: cents per interval, so scaled up
    # for gradient health. 20 puts a typical arbitrage decision near 0.4.
    reward_scale: float = 20.0
    # Convergence: greedy validation rollouts every `eval_every` steps; stop
    # when the best has not improved in `patience` POST-ANNEAL evaluations --
    # staleness during exploration says nothing about the greedy policy, and
    # counting it ended the first calibration run at exactly `decay_steps`
    # with the refinement phase never trained.
    eval_every: int = 20_000
    patience: int = 8
    validation_days: int = 60
    # The wear shadow price the MILP objective carries, EUR per EFC. The agent
    # pays it too, or it learns to cycle for gains the study then bills it for.
    wear_eur_per_efc: float = 0.0
    # BC-guided exploration: with this probability an exploratory step asks the
    # cloned network instead of the uniform die. 0 without a BC prior.
    bc_guide_prob: float = 0.5
    # How many rewards are summed before bootstrapping. The payoff for storing
    # a kWh arrives when it is spent, typically 8-24 intervals later, and
    # 1-step TD has to walk the credit back through that many bootstraps: the
    # first screen's DQN under-traded badly for exactly this reason (27-51
    # EFC/year on SI against the cloned MILP's 87-116). 8 is half a charge-to-
    # discharge cycle at this resolution.
    n_step: int = 8
    # Keeps a fine-tune near the policy it was warm-started from. Without it a
    # warm start is destroyed inside 20k steps: the clone's weights are
    # cross-entropy LOGITS and the first TD updates rescale the network to
    # Q-magnitudes, taking the ranking with them -- measured, `bc_dqn` began
    # its validation trace at 62.6 EUR, where a COLD dqn begins (63.6), rather
    # than where the clone sits. Decays to 0 so reinforcement can eventually
    # overrule the demonstration it started from.
    bc_reg: float = 1.0
    bc_reg_decay_frac: float = 0.5
    seed: int = 0

    def config(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The environment walk shared by training, validation and cloning
# ---------------------------------------------------------------------------
class _Walk:
    """One pass over a step range: SOC, peak state and settlement, nothing else.

    This is `_run_policy_once`'s loop reduced to what training needs. It reuses
    the same envelope functions and the same settle callable; it does NOT
    converge the endogenous contract (the contract in force is the one the
    signals carry), which is a stated training approximation -- scoring always
    goes through `run_policy`, which does converge it.
    """

    def __init__(self, sig, settle, env, wear_eur_per_efc: float):
        self.sig = sig
        self.settle = settle
        self.env = env
        nominal = float(getattr(env, "nominal_capacity_kwh",
                                env.battery_capacity_kwh))
        self.per_stored = (float(wear_eur_per_efc) / (2.0 * nominal)
                           if wear_eur_per_efc and nominal > 0 else 0.0)
        self.eta_ch = sig.eta_ch
        self.eta_dis = sig.eta_dis
        self.capacity = sig.capacity_kwh

    def bounds(self, soc):
        return (-max_discharge_now(soc, self.eta_dis, self.env.max_discharge_kwh),
                max_charge_now(soc, self.eta_ch, self.env.max_charge_kwh,
                               self.capacity))

    def step(self, idx, soc, peak_state, setpoint):
        """Apply one setpoint; returns (cost_eur, new_soc, new_peak_state, net)."""
        sig = self.sig
        ch = max(setpoint, 0.0)
        dis = max(-setpoint, 0.0)
        net = sig.consumption[idx] + ch - sig.generation[idx] - dis
        cost, _, _, _, peak_state = self.settle(self.env, idx, net, peak_state)
        stored = ch * self.eta_ch - dis / self.eta_dis
        cost += self.per_stored * (ch * self.eta_ch + dis / self.eta_dis)
        soc = min(max(soc + stored, 0.0), self.capacity)
        return cost, soc, peak_state, net


def _drop_on_boundary(peak_state, windows, idx):
    if idx == 0 or windows[idx] == windows[idx - 1]:
        return peak_state
    return {b: 0.0 for b in _BLOCKS}


def seed_peak_state(env, idx: int) -> dict:
    """The per-block running peak a window STARTING at `idx` inherits.

    An episode, a validation rollout and a teacher walk all begin part-way
    through a ratchet window, and the SI excess-power charge is levied only on
    the draw above the month-to-date peak. Starting them at zero therefore
    tells the controller that every peak in front of it is a new one worth
    shaving, when in the scored year most are already sunk -- a systematic
    distortion of the one feature SI's money actually turns on, and it showed:
    six of the seven non-converged runs in the screen were SI DQN, on the only
    tariff whose settlement reads this state at all.

    `HouseholdEnvironment.compute_seed_peak_kw` already answers exactly this
    question -- it is what the environment seeds a mid-dataset episode with --
    so it is asked rather than reimplemented, and it returns zeros of its own
    accord when `idx` does open a window.
    """
    seed = getattr(env, "compute_seed_peak_kw", None)
    if seed is None:
        return {b: 0.0 for b in _BLOCKS}
    return {int(b): float(v) for b, v in seed(int(idx)).items()}


def no_battery_cost_trace(sig, settle, env) -> np.ndarray:
    """Per-interval cost of doing nothing, walked with its own peak state.

    The training reward subtracts this from the settled cost, so the agent is
    rewarded on the DIFFERENCE its action made rather than on a bill dominated
    by load it cannot change. The subtraction is action-independent (a fixed
    array over t), so it shifts every Q(s, a) at a state equally and the
    argmax -- the policy -- is untouched; what it changes is the scale the
    network has to resolve. On SI the action-dependent slice of an interval is
    one or two cents against an 0.08 EUR bill, and without this the first
    calibration run learned to idle: the two-cent signal drowned in the
    fifty-EUR discounted baseline the Q-head was regressing.
    """
    peak_state = {b: 0.0 for b in _BLOCKS}
    out = np.zeros(sig.n_steps, dtype=np.float64)
    for idx in range(sig.n_steps):
        peak_state = _drop_on_boundary(peak_state, sig.windows, idx)
        net = sig.consumption[idx] - sig.generation[idx]
        out[idx], _, _, _, peak_state = settle(env, idx, net, peak_state)
    return out


def greedy_rollout(net, fb, static_norm, sig, settle, env, start: int, stop: int,
                   soc_init: float, respect_peak: bool,
                   wear_eur_per_efc: float = 0.0) -> dict:
    """Greedy policy over [start, stop): the validation measure.

    Returns the closed cost the study compares on -- bill plus terminal-SOC
    close-out at the mean delivered rate -- and the cycle count, so convergence
    is judged on the same axis the final scoring uses.
    """
    walk = _Walk(sig, settle, env, wear_eur_per_efc=0.0)  # bill, not objective
    soc = soc_init
    peak_state = seed_peak_state(env, start)
    cost = 0.0
    discharged = 0.0
    net.eval()
    with torch.no_grad():
        for idx in range(start, stop):
            peak_state = _drop_on_boundary(peak_state, sig.windows, idx)
            lo, hi = walk.bounds(soc)
            obs = np.concatenate([static_norm[idx],
                                  fb.dynamic(sig, idx, soc, peak_state)])
            a = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
            p = float(np.clip(action_setpoint(a, sig, idx, lo, hi,
                                              peak_state, respect_peak), lo, hi))
            c, soc, peak_state, _ = walk.step(idx, soc, peak_state, p)
            cost += c
            discharged += max(-p, 0.0)
    mean_rate = float(np.mean(sig.import_rate[start:stop]))
    closed = cost + (soc_init - soc) / sig.eta_ch * mean_rate
    nominal = float(getattr(env, "nominal_capacity_kwh", walk.capacity))
    return {"cost_eur": cost, "cost_eur_closed": closed,
            "efc": discharged / nominal if nominal > 0 else 0.0}


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------
def train_dqn(sig, settle, env, fb: FeatureBuilder, static_norm: np.ndarray,
              cfg: TrainConfig, respect_peak: bool,
              train_days: tuple, val_days: tuple,
              init_net: QNet | None = None, bc_net: QNet | None = None,
              soc_target: float | None = None, verbose: bool = True):
    """Double DQN on one household's training period.

    `train_days` / `val_days` are (first_day, last_day_exclusive) in local-day
    index; episodes are `cfg.episode_days` windows drawn uniformly from the
    training range, so the buffer decorrelates across seasons instead of
    replaying one January. Validation is a greedy rollout over `val_days`,
    scored as `Cost_EUR_Closed`; the weights kept are the best validation
    weights, and `converged` says whether that best had stopped moving --
    early-stop after `cfg.patience` evaluations without improvement.

    Returns `(net, history)`. The history carries everything a convergence
    figure needs: per-episode returns, the validation trace, epsilon and loss.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    spd = int(round(24.0 / sig.hours))
    walk = _Walk(sig, settle, env, cfg.wear_eur_per_efc)
    capacity = sig.capacity_kwh
    if soc_target is None:
        soc_target = 0.5 * capacity  # the study's SOC_FRACTION on the usable window
    baseline = no_battery_cost_trace(sig, settle, env)

    dim = static_norm.shape[1] + fb.n_dynamic
    # DEEP COPIES, both of them. `run_one` hands the same cloned network in as
    # the warm start and as the behaviour prior; without copying, the "frozen"
    # guide is the network being trained, so BC-guided exploration silently
    # becomes greedy exploration and the regulariser pulls the policy towards
    # itself.
    net = copy.deepcopy(init_net) if init_net is not None else QNet(dim, cfg.hidden)
    bc_frozen = copy.deepcopy(bc_net) if bc_net is not None else None
    if bc_frozen is not None:
        bc_frozen.eval()
        for p in bc_frozen.parameters():
            p.requires_grad_(False)
    target = QNet(dim, cfg.hidden)
    target.load_state_dict(net.state_dict())
    # A warm start is a prior worth protecting: full-rate exploration and a
    # full learning rate overwrite the clone with Q-fits to random-action
    # replay before it ever acts -- measured, `bc_dqn` came back WORSE than
    # its own `bc`. Fine-tune gently instead.
    warm = init_net is not None
    eps_start = min(cfg.eps_start, 0.2) if warm else cfg.eps_start
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr * (0.5 if warm else 1.0))
    buf = _Replay(cfg.buffer, dim)

    val_start, val_stop = val_days[0] * spd, val_days[1] * spd
    ep_days = cfg.episode_days
    ep_steps = ep_days * spd
    day_lo, day_hi = train_days[0], train_days[1] - ep_days
    if day_hi <= day_lo:
        raise ValueError("training range shorter than one episode")

    decay_steps = max(int(cfg.total_steps * cfg.eps_decay_frac), 1)
    history = {"episode_return_eur": [], "episode_start_day": [],
               "val_step": [], "val_cost_closed": [], "val_efc": [],
               "loss": [], "eps": [], "config": cfg.config()}
    best_val = np.inf
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best_eval_i = -1
    evals_since_best = 0
    stopped_early = False
    losses = []

    step = 0
    t0 = time.time()
    while step < cfg.total_steps and not stopped_early:
        # -- one episode ---------------------------------------------------
        d0 = int(rng.integers(day_lo, day_hi + 1))
        idx0 = d0 * spd
        # Half the episodes start where the study starts every controller;
        # half start at a random SOC, or the agent never sees a full pack in
        # January and cannot learn what to do holding one.
        soc = soc_target if rng.random() < 0.5 else float(rng.uniform(0, capacity))
        peak_state = seed_peak_state(env, idx0)
        ep_return = 0.0

        # The n-step window, held inside the episode: a return may never span
        # two episodes, which start in different seasons at different SOC.
        # Entries are (obs, action, scaled reward, was_exploratory, next_obs).
        pending = deque()

        def _flush(terminal_final):
            """Pop the oldest pending transition as an n-step return.

            TRUNCATED AT THE FIRST EXPLORATORY CONTINUATION. An n-step return
            estimates Q(s_t, a_t) only if a_{t+1}..a_{t+k-1} came from the
            policy being evaluated; a_t itself may be anything, since it is the
            action being scored. Summing through random continuations instead
            evaluates the exploration, not the policy, and biases every good
            state downward -- measured, an untruncated n=8 took the pure DQN
            from 844 to 976 EUR on AU and collapsed it to 17 EFC/year, while
            the BC-guided run, whose continuations were nearly on-policy,
            improved. Truncating makes the window adaptive for free: nearly
            1-step while epsilon is high, the full n once the policy anneals.
            """
            o0, a0 = pending[0][0], pending[0][1]
            ret, disc, k = 0.0, 1.0, 0
            for j, (_, _, r, explored, _) in enumerate(pending):
                if j > 0 and explored:
                    break
                ret += disc * r
                disc *= cfg.gamma
                k = j + 1
            # Bootstrap from the state the k-th accumulated step landed in --
            # not from the end of the window, which may be further along.
            done_flag = 1.0 if (terminal_final and k == len(pending)) else 0.0
            buf.push(o0, a0, ret, pending[k - 1][4], done_flag, disc)
            pending.popleft()

        obs = np.concatenate([static_norm[idx0],
                              fb.dynamic(sig, idx0, soc, peak_state)])
        for k in range(ep_steps):
            idx = idx0 + k
            peak_state = _drop_on_boundary(peak_state, sig.windows, idx)
            lo, hi = walk.bounds(soc)

            eps = cfg.eps_end + (eps_start - cfg.eps_end) * max(
                0.0, 1.0 - step / decay_steps)
            explored = rng.random() < eps
            if explored:
                if bc_frozen is not None and rng.random() < cfg.bc_guide_prob:
                    with torch.no_grad():
                        a = int(bc_frozen(torch.from_numpy(obs).unsqueeze(0))
                                .argmax(dim=1).item())
                else:
                    a = int(rng.integers(N_ACTIONS))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(obs).unsqueeze(0))
                            .argmax(dim=1).item())

            p = float(np.clip(action_setpoint(a, sig, idx, lo, hi,
                                              peak_state, respect_peak), lo, hi))
            cost, soc, peak_state, _ = walk.step(idx, soc, peak_state, p)
            reward = baseline[idx] - cost
            done = k == ep_steps - 1
            if done:
                # The close-out every controller pays: end the episode short
                # and the shortfall is bought back at the mean rate.
                mean_rate = float(np.mean(sig.import_rate[idx0:idx0 + ep_steps]))
                reward -= (soc_target - soc) / sig.eta_ch * mean_rate
            ep_return += reward

            nxt_idx = min(idx + 1, sig.n_steps - 1)
            nxt = np.concatenate([static_norm[nxt_idx],
                                  fb.dynamic(sig, nxt_idx, soc, peak_state)])
            pending.append((obs, a, reward * cfg.reward_scale, explored, nxt))
            if done:
                while pending:                  # every tail return ends here
                    _flush(True)
            elif len(pending) >= cfg.n_step:
                _flush(False)
            obs = nxt
            step += 1

            # -- learn -----------------------------------------------------
            if buf.n >= cfg.learn_start and step % cfg.update_every == 0:
                o, ac, r, o2, dn, disc = buf.sample(rng, cfg.batch)
                o = torch.from_numpy(o)
                o2 = torch.from_numpy(o2)
                r = torch.from_numpy(r)
                dn = torch.from_numpy(dn)
                ac = torch.from_numpy(ac)
                disc = torch.from_numpy(disc)
                with torch.no_grad():
                    # Double DQN: the online net picks, the target net prices.
                    a2 = net(o2).argmax(dim=1, keepdim=True)
                    q2 = target(o2).gather(1, a2).squeeze(1)
                    # `disc` is gamma**k for this entry's own k, not gamma**n.
                    y = r + disc * (1.0 - dn) * q2
                q_all = net(o)
                q = q_all.gather(1, ac.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                if bc_frozen is not None and cfg.bc_reg > 0:
                    lam = cfg.bc_reg * max(
                        0.0, 1.0 - step / max(cfg.bc_reg_decay_frac
                                              * cfg.total_steps, 1.0))
                    if lam > 0:
                        with torch.no_grad():
                            bc_a = bc_frozen(o).argmax(dim=1)
                        # Q-values as logits: this only constrains the RANKING
                        # of the actions, which is the part of the clone worth
                        # keeping, and leaves their magnitudes to the TD loss.
                        loss = loss + lam * nn.functional.cross_entropy(q_all, bc_a)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.item()))
            if step % cfg.target_sync == 0:
                target.load_state_dict(net.state_dict())

            # -- validate --------------------------------------------------
            if step % cfg.eval_every == 0:
                val = greedy_rollout(net, fb, static_norm, sig, settle, env,
                                     val_start, val_stop, soc_target,
                                     respect_peak)
                history["val_step"].append(step)
                history["val_cost_closed"].append(val["cost_eur_closed"])
                history["val_efc"].append(val["efc"])
                history["loss"].append(float(np.mean(losses[-500:]))
                                       if losses else float("nan"))
                history["eps"].append(eps)
                if val["cost_eur_closed"] < best_val - 1e-6:
                    best_val = val["cost_eur_closed"]
                    best_state = {k: v.clone()
                                  for k, v in net.state_dict().items()}
                    best_eval_i = len(history["val_step"]) - 1
                    evals_since_best = 0
                elif step >= decay_steps:
                    # Patience counts only refinement-phase evaluations: an
                    # eval taken at eps 0.6 measures a policy that no longer
                    # exists by the time the gate could act on it.
                    evals_since_best += 1
                if verbose:
                    print(f"    step {step:>7d}  eps {eps:.2f}  "
                          f"val {val['cost_eur_closed']:8.2f}  "
                          f"best {best_val:8.2f}  "
                          f"({evals_since_best} evals since best)", flush=True)
                if evals_since_best >= cfg.patience:
                    stopped_early = True
                    break
            if step >= cfg.total_steps:
                break

        history["episode_return_eur"].append(ep_return)
        history["episode_start_day"].append(d0)

    net.load_state_dict(best_state)
    history["runtime_s"] = time.time() - t0
    history["steps_run"] = step
    history["best_val_cost_closed"] = float(best_val)
    history["stopped_early"] = stopped_early
    # Converged = the best validation cost had stopped moving while training
    # continued: either the early stop fired, or the best sits at least
    # `patience` evaluations before the last one. A run whose best is its final
    # evaluation was still improving when the budget ran out, and saying so is
    # the whole point of tracking this.
    n_evals = len(history["val_step"])
    history["converged"] = bool(
        stopped_early or (best_eval_i >= 0
                          and n_evals - 1 - best_eval_i >= cfg.patience))
    return net, history


# ---------------------------------------------------------------------------
# Behaviour cloning of the whole-period MILP
# ---------------------------------------------------------------------------
def teacher_actions(sig, settle, env, setpoints_kwh: np.ndarray,
                    start: int, stop: int, soc_init: float,
                    respect_peak: bool):
    """Walk the MILP trajectory and label each state with the nearest action.

    The MILP hands back continuous setpoints; the agent speaks five actions.
    Each interval is labelled with the action whose OWN setpoint -- computed in
    the exact state the trajectory is in at that moment -- lands closest to
    what the MILP did. Labelling against achievable setpoints rather than raw
    magnitudes matters: a 0.1 kWh MILP discharge with the house in deficit is
    `discharge_home`, not a scaled-down `discharge_any`.

    Returns (observations' dynamic parts are the caller's job) the per-step
    (soc, peak_state_rel, label) walk: soc trace, dynamic features input and
    integer labels.
    """
    walk = _Walk(sig, settle, env, wear_eur_per_efc=0.0)
    soc = soc_init
    peak_state = seed_peak_state(env, start)
    n = stop - start
    labels = np.zeros(n, dtype=np.int64)
    soc_trace = np.zeros(n, dtype=np.float64)
    peaks = []
    for j, idx in enumerate(range(start, stop)):
        peak_state = _drop_on_boundary(peak_state, sig.windows, idx)
        lo, hi = walk.bounds(soc)
        soc_trace[j] = soc
        peaks.append(dict(peak_state))
        target = float(np.clip(setpoints_kwh[idx - start], lo, hi))
        cands = np.array([
            float(np.clip(action_setpoint(a, sig, idx, lo, hi,
                                          peak_state, respect_peak), lo, hi))
            for a in range(N_ACTIONS)])
        # Nearest achievable setpoint; ties go to the lower action index, and
        # idle is last among the four movers, so a genuine 0 still labels as
        # whichever mover is also 0 -- harmless, they execute identically.
        labels[j] = int(np.argmin(np.abs(cands - target)))
        # The walk FOLLOWS THE TEACHER, not the label: cloning learns from the
        # states the optimum visits.
        _, soc, peak_state, _ = walk.step(idx, soc, peak_state, target)
    return soc_trace, peaks, labels


def train_bc(sig, settle, env, fb: FeatureBuilder, static_norm: np.ndarray,
             setpoints_kwh: np.ndarray, start: int, stop: int,
             soc_init: float, respect_peak: bool, cfg: TrainConfig,
             verbose: bool = True):
    """Clone the MILP's action choices. Returns (net, history).

    Plain cross-entropy with inverse-frequency class weights -- idle and
    self-consumption dominate an optimal year, and an unweighted fit collapses
    onto them. 10 % of the days are held out for the agreement measure;
    convergence is early-stopping on held-out loss.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    soc_trace, peaks, labels = teacher_actions(
        sig, settle, env, setpoints_kwh, start, stop, soc_init, respect_peak)
    n = stop - start
    X = np.zeros((n, static_norm.shape[1] + fb.n_dynamic), dtype=np.float32)
    for j, idx in enumerate(range(start, stop)):
        X[j] = np.concatenate([static_norm[idx],
                               fb.dynamic(sig, idx, soc_trace[j], peaks[j])])
    y = labels

    # Hold out whole days, not random intervals: adjacent intervals are nearly
    # identical, so a random split leaks the training set into the holdout.
    spd = int(round(24.0 / sig.hours))
    days = np.arange(n // spd)
    rng.shuffle(days)
    held = set(days[: max(1, len(days) // 10)])
    mask = np.array([(j // spd) in held for j in range(n)])
    Xtr, ytr, Xva, yva = X[~mask], y[~mask], X[mask], y[mask]

    counts = np.bincount(ytr, minlength=N_ACTIONS).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    net = QNet(X.shape[1], cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)))
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

    history = {"train_loss": [], "val_loss": [], "val_agreement": [],
               "label_counts": counts.tolist(), "config": cfg.config()}
    best = np.inf
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    since_best = 0
    epochs = 200
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]
            loss = lossf(net(Xtr_t[j]), ytr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(j)
        net.eval()
        with torch.no_grad():
            out = net(Xva_t)
            vloss = float(lossf(out, yva_t).item())
            agree = float((out.argmax(dim=1) == yva_t).float().mean().item())
        history["train_loss"].append(tot / len(Xtr_t))
        history["val_loss"].append(vloss)
        history["val_agreement"].append(agree)
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        if verbose and (ep % 10 == 0 or since_best >= 10):
            print(f"    BC epoch {ep:3d}  val loss {vloss:.4f}  "
                  f"agreement {agree:.3f}", flush=True)
        if since_best >= 10:
            break
    net.load_state_dict(best_state)
    history["runtime_s"] = time.time() - t0
    history["converged"] = bool(since_best >= 10)
    history["best_val_loss"] = float(best)
    with torch.no_grad():
        out = net(Xva_t)
        history["final_val_agreement"] = float(
            (out.argmax(dim=1) == yva_t).float().mean().item())
    return net, history


# ---------------------------------------------------------------------------
# The deployable face: a trained network as a Policy
# ---------------------------------------------------------------------------
class LearnedPolicy(rbc.Policy):
    """A trained network, executed and priced exactly like every rule.

    `reset` rebuilds the static features from the signal bundle it is handed,
    which is what makes the endogenous-contract loop work unchanged: each
    `converge_agreed_power` iteration rebinds `sig.agreed_kw`, `run_policy`
    calls `reset` again, and the agent sees the contract it is currently
    billed under -- the same information flow every peak-aware rule has.
    """

    def __init__(self, net: QNet, fb: FeatureBuilder, respect_peak: bool,
                 load_fc=None, pv_fc=None, name: str = "rl_dqn",
                 label: str = "RL: DQN", causal: bool = True):
        self.net = net
        self.fb = fb
        self.respect_peak = bool(respect_peak)
        self.load_fc = load_fc
        self.pv_fc = pv_fc
        self.name = name
        self.label = label
        self.causal = bool(causal)
        self._static = None

    def reset(self, sig):
        static = self.fb.build_static(sig, load_fc=self.load_fc, pv_fc=self.pv_fc)
        self._static = self.fb.normalize(static)
        self.net.eval()

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        obs = np.concatenate([self._static[idx],
                              self.fb.dynamic(sig, idx, soc_kwh, peak_state)])
        with torch.no_grad():
            a = int(self.net(torch.from_numpy(obs).unsqueeze(0))
                    .argmax(dim=1).item())
        return action_setpoint(a, sig, idx, lo, hi, peak_state,
                               self.respect_peak)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(path: str, net: QNet, fb: FeatureBuilder, spec: FeatureSpec,
               cfg: TrainConfig, history: dict, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "dim": fb.dim,
        "hidden": cfg.hidden,
        "norm_mean": fb.norm_mean,
        "norm_std": fb.norm_std,
        "spec": spec.config(),
        "train_config": cfg.config(),
        "extra": extra or {},
    }, path)
    with open(path + ".history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh)


def load_model(path: str):
    """Returns (net, fb, spec, meta). The builder comes back normalised."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec = FeatureSpec(**blob["spec"])
    fb = FeatureBuilder(spec)
    fb.norm_mean = np.asarray(blob["norm_mean"], dtype=np.float32)
    fb.norm_std = np.asarray(blob["norm_std"], dtype=np.float32)
    net = QNet(int(blob["dim"]), int(blob["hidden"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, fb, spec, blob
