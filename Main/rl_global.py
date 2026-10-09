"""GLOBAL learned controllers: one network trained across many households.

`rl_control` trains one network per household on that household's own two
training years. That is the most a single home can learn from itself, and it is
also the least data any learned controller in the study sees: ~590 days, one
load shape, one roof. The question this module answers is the one a vendor
shipping a controller to thousands of homes faces instead -- does POOLING
households help, and how local does the pool have to be?

    global         one network over every household in the population (the
                   300 Ausgrid homes), told nothing about which home it is
                   driving. The pooled baseline.
    global_typed   the same pool, the observation extended by a one-hot of the
                   consumer TYPE (the k=30 load-shape cluster the household
                   belongs to). One network, localised to a type by its input:
                   types share every weight and differ only through what that
                   one-hot can shift.
    type           one network per type, trained on that cluster's members
                   only. Localisation by data rather than by input -- no
                   transfer between types at all.

Everything that is not "which data, which network" is imported from
`rl_control` and reused, not re-implemented: the observation (`FeatureBuilder`,
extended only by the type columns), the five semantic actions and their
feasibility (`action_setpoint`), the training walk and its settlement
(`_Walk`, the arm's own `settle`), the ratchet-peak seeding, the
baseline-subtracted reward, the validation rollout, the cloning teacher walk,
the network (`QNet`) and the deployable `LearnedPolicy`. The two training
loops below are `rl_control.train_dqn` / `train_bc` with one change each --
an episode first draws a household, and validation / the clone's holdout sum
over several -- and `test_rl_global.py` checks that a one-household pool
reproduces `rl_control.train_dqn` bit for bit, which is what licenses calling
them the same learner.

Causality and the split are unchanged: every household in a pool contributes
ONLY its training years (days 0-730, minus the held-out validation weeks);
the scored year (days 730-1095) is read once, by `run_policy`, for the study
units alone. Pooling other homes' training years cannot leak the test year --
the calendar is shared, but nothing past day 730 of ANY household is touched.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import rl_control as rl
import Rule_Based_Control as rbc

# WHICH POOLED LEARNING RULE produced a result -- stamped into every digest
# beside `rl.ALGO_VERSION` / `rl.BC_ALGO_VERSION`, for the same reason those
# exist: a change to the loop leaves every setting untouched.
#
#   1  first global/type screen: uniform household draw per episode,
#      validation summed over the validation members, pooled BC on batches of
#      2048 with patience 5 (a warm-started clone is measured against its
#      untouched prior first)
#
# Not bumped for the move of pooled BC to the local clone's 512 / 10 / 200:
# those settings are `BCConfigPool` fields and sit in every clone's digest
# (and through `bc_prior` in every bc_dqn's), so the clones and their
# fine-tunes are recomputed while the pure DQN runs -- which never read them --
# stay valid.
GLOBAL_ALGO_VERSION = 1


# ---------------------------------------------------------------------------
# The observation, extended by the consumer type
# ---------------------------------------------------------------------------
class TypedFeatureBuilder(rl.FeatureBuilder):
    """`FeatureBuilder` plus a constant one-hot of the household's type.

    The type columns are appended AFTER the study's own static features and are
    exempt from normalisation (mean 0, std 1): a one-hot standardised over a
    300-household pool would put a singleton cluster's 1 at ~17 standard
    deviations, and the network would read the rarest types loudest. `n_types`
    of 0 is the plain builder -- the `global` and `type` schemes use it, so all
    three schemes share one class and one save format.

    One builder carries the pooled normalisation; `for_type(k)` hands out a
    shallow copy bound to one type, which is what each household's
    `LearnedPolicy` is given. The normalisation arrays are shared, not copied.
    """

    def __init__(self, spec: rl.FeatureSpec, n_types: int = 0):
        super().__init__(spec)
        self.n_types = int(n_types)
        self.type_index = None

    def for_type(self, k: int | None) -> "TypedFeatureBuilder":
        out = copy.copy(self)
        out.type_index = None if k is None else int(k)
        return out

    def type_columns(self, n: int) -> np.ndarray:
        if not self.n_types:
            return np.zeros((n, 0), dtype=np.float32)
        if self.type_index is None:
            raise RuntimeError("a typed builder must be bound with for_type(k)")
        out = np.zeros((n, self.n_types), dtype=np.float32)
        out[:, self.type_index] = 1.0
        return out

    def build_static(self, sig, load_fc=None, pv_fc=None) -> np.ndarray:
        base = super().build_static(sig, load_fc=load_fc, pv_fc=pv_fc)
        return np.hstack([base, self.type_columns(len(base))]).astype(np.float32)

    # -- pooled normalisation ------------------------------------------------
    def fit_norm_pooled(self, base_blocks) -> None:
        """Mean/std of the BASE features over many households' rows at once.

        Streaming sums in float64 rather than one concatenation: 300 households
        of training rows is ~1 GB as float32, and the scaler only needs two
        moments. The rows passed must be training rows only -- the caller's job,
        exactly as `fit_norm` leaves it to `run_one`.
        """
        n, s1, s2 = 0, None, None
        for blk in base_blocks:
            b = np.asarray(blk, dtype=np.float64)
            s1 = b.sum(axis=0) if s1 is None else s1 + b.sum(axis=0)
            s2 = (b * b).sum(axis=0) if s2 is None else s2 + (b * b).sum(axis=0)
            n += len(b)
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
        self.norm_mean = np.concatenate(
            [mean, np.zeros(self.n_types)]).astype(np.float32)
        self.norm_std = np.concatenate(
            [np.maximum(std, 1e-6), np.ones(self.n_types)]).astype(np.float32)

    def normalize_base(self, base: np.ndarray) -> np.ndarray:
        """`normalize(build_static(...))` from the base matrix alone.

        The driver caches the base features per household once and binds the
        type at training time; this is the identity that makes that legal,
        checked in `test_rl_global.py`.
        """
        nb = base.shape[1]
        z = (base - self.norm_mean[:nb]) / self.norm_std[:nb]
        return np.hstack([z, self.type_columns(len(base))]).astype(np.float32)

    def normalized_rows(self, base: np.ndarray):
        """`normalize_base`, but as a row view when the builder is typed.

        The training walks only ever read `static_norm[idx]` and
        `static_norm.shape`, so a typed household need not carry its one-hot
        materialised on every row -- 30 constant columns over 35k steps is
        4 MB a household, 1.3 GB across the global pool, for no information.
        """
        if not self.n_types:
            return self.normalize_base(base)
        nb = base.shape[1]
        z = ((base - self.norm_mean[:nb]) / self.norm_std[:nb]).astype(np.float32)
        return TypedRows(z, self.type_columns(1)[0])


class TypedRows:
    """Read-only (n, base + n_types) view: normalised base rows + a constant
    type vector, concatenated on access. Integer indexing only, which is all
    `train_dqn_pool` and `rl.greedy_rollout` do."""

    def __init__(self, base_norm: np.ndarray, type_vec: np.ndarray):
        self.base = base_norm
        self.type_vec = np.asarray(type_vec, dtype=np.float32)
        self.shape = (base_norm.shape[0], base_norm.shape[1] + len(self.type_vec))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return np.concatenate([self.base[idx], self.type_vec])


# ---------------------------------------------------------------------------
# One household inside a pool
# ---------------------------------------------------------------------------
@dataclass
class Member:
    """What a pooled loop needs of one household's TRAINING bundle."""

    ident: str
    sig: object
    settle: object
    env: object
    static_norm: np.ndarray            # (n_train_steps, dim_static), type bound
    baseline: np.ndarray | None = None  # no-battery cost trace, lazily built


def _pool_validation(net, fb, val_members, val_blocks, soc_init, respect_peak,
                     val_wear, n_val_days):
    """`rl.validation_rollout` per validation member, summed.

    The lifetime-wear term is applied PER HOUSEHOLD and then summed: it is zero
    below a cycling pace and positive above it, so pooling the cycles first
    would let one household's heavy cycling hide behind nine idle ones.
    """
    closed = efc = 0.0
    for m in val_members:
        v = rl.validation_rollout(net, fb, m.static_norm, m.sig, m.settle,
                                  m.env, val_blocks, soc_init, respect_peak)
        c = v["cost_eur_closed"]
        if val_wear is not None:
            c += float(val_wear(v["efc"], n_val_days))
        closed += c
        efc += v["efc"]
    return {"cost_eur_closed": closed, "efc": efc / max(len(val_members), 1)}


# ---------------------------------------------------------------------------
# Pooled DQN: `rl.train_dqn`, an episode at a time from a random household
# ---------------------------------------------------------------------------
def train_dqn_pool(members, val_members, fb, cfg: rl.TrainConfig,
                   respect_peak: bool, train_days, val_days,
                   init_net=None, bc_net=None, soc_target=None,
                   verbose: bool = True, val_wear=None,
                   init_is_candidate: bool = False):
    """Double DQN over a POOL of households. Returns `(net, history)`.

    `init_is_candidate` makes the warm start itself the first candidate: its
    validation is scored before any update and the run keeps it unless training
    beats it. A localisation (a global network fine-tuned on a type) can then
    never hand back weights that are worse, on the type's own validation weeks,
    than the global weights it started from -- without it the best-so-far
    starts at +inf, so a fine-tune that only degraded the prior would still be
    deployed. Off by default, which keeps `rl.train_dqn` reproduced exactly.

    Line for line `rl.train_dqn` -- same exploration, n-step truncation, BC
    guide and regulariser, warm-start protection, early stop -- with two
    differences:

      * each episode first draws a household uniformly from `members`, then a
        start day from the training blocks. Uniform over households, not over
        household-days: every household has the same training calendar, so
        the two are the same distribution, and drawing the household first
        keeps the per-episode work identical to the local loop's.
      * the validation score is summed over `val_members` (each over the
        held-out weeks, from the study's starting SOC). The kept weights are
        the best on that sum.

    A one-member pool draws no household -- the RNG stream is then exactly the
    local loop's, which is what lets `test_rl_global.py` hold this function to
    `rl.train_dqn` bit for bit.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    sig0 = members[0].sig
    spd = int(round(24.0 / sig0.hours))
    walks = [rl._Walk(m.sig, m.settle, m.env, cfg.wear_eur_per_efc)
             for m in members]
    capacity = sig0.capacity_kwh
    if soc_target is None:
        soc_target = 0.5 * capacity
    for m in members:
        if m.baseline is None:
            m.baseline = rl.no_battery_cost_trace(m.sig, m.settle, m.env)

    dim = members[0].static_norm.shape[1] + fb.n_dynamic
    net = copy.deepcopy(init_net) if init_net is not None else rl.QNet(dim, cfg.hidden)
    bc_frozen = copy.deepcopy(bc_net) if bc_net is not None else None
    if bc_frozen is not None:
        bc_frozen.eval()
        for p in bc_frozen.parameters():
            p.requires_grad_(False)
    target = rl.QNet(dim, cfg.hidden)
    target.load_state_dict(net.state_dict())
    warm = init_net is not None
    eps_start = min(cfg.eps_start, 0.2) if warm else cfg.eps_start
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr * (0.5 if warm else 1.0),
                           foreach=True)
    buf = rl._Replay(cfg.buffer, dim)

    val_blocks = rl.as_blocks(val_days)
    n_val_days = sum(b - a for a, b in val_blocks)
    ep_days = cfg.episode_days
    ep_steps = ep_days * spd
    starts = np.concatenate([np.arange(a, b - ep_days + 1)
                             for a, b in rl.as_blocks(train_days)
                             if b - a >= ep_days] or [np.array([], dtype=int)])
    if len(starts) == 0:
        raise ValueError("no training block is as long as one episode")
    held = set(rl.block_days(val_blocks).tolist())
    if any(d in held for s in starts for d in range(s, s + ep_days)):
        raise ValueError("a training episode overlaps the validation days")

    decay_steps = max(int(cfg.total_steps * cfg.eps_decay_frac), 1)
    history = {"episode_return_eur": [], "episode_start_day": [],
               "episode_member": [],
               "val_step": [], "val_cost_closed": [], "val_efc": [],
               "loss": [], "eps": [], "config": cfg.config(),
               "n_members": len(members), "n_val_members": len(val_members)}
    best_val = np.inf
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best_eval_i = -1
    evals_since_best = 0
    stopped_early = False
    losses = []
    if init_is_candidate and init_net is not None:
        # Greedy rollouts draw nothing from `rng`, so scoring the prior leaves
        # the training stream exactly as it would have been.
        v0 = _pool_validation(net, fb, val_members, val_blocks, soc_target,
                              respect_peak, val_wear, n_val_days)
        best_val = v0["cost_eur_closed"]
        history["init_val_cost_closed"] = best_val

    step = 0
    t0 = time.time()
    while step < cfg.total_steps and not stopped_early:
        mi = int(rng.integers(len(members))) if len(members) > 1 else 0
        m = members[mi]
        sig, walk, static_norm, baseline = m.sig, walks[mi], m.static_norm, m.baseline
        d0 = int(starts[rng.integers(len(starts))])
        idx0 = d0 * spd
        soc = soc_target if rng.random() < 0.5 else float(rng.uniform(0, capacity))
        peak_state = rl.seed_peak_state(m.env, idx0)
        ep_return = 0.0
        pending = deque()

        def _flush(terminal_final):
            # `rl.train_dqn._flush`: n-step return truncated at the first
            # exploratory continuation; see the reasoning there.
            o0, a0 = pending[0][0], pending[0][1]
            ret, disc, k = 0.0, 1.0, 0
            for j, (_, _, r, explored, _) in enumerate(pending):
                if j > 0 and explored:
                    break
                ret += disc * r
                disc *= cfg.gamma
                k = j + 1
            done_flag = 1.0 if (terminal_final and k == len(pending)) else 0.0
            buf.push(o0, a0, ret, pending[k - 1][4], done_flag, disc)
            pending.popleft()

        obs = np.concatenate([static_norm[idx0],
                              fb.dynamic(sig, idx0, soc, peak_state)])
        for k in range(ep_steps):
            idx = idx0 + k
            peak_state = rl._drop_on_boundary(peak_state, sig.windows, idx)
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
                    a = int(rng.integers(rl.N_ACTIONS))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(obs).unsqueeze(0))
                            .argmax(dim=1).item())

            p = float(np.clip(rl.action_setpoint(a, sig, idx, lo, hi,
                                                 peak_state, respect_peak), lo, hi))
            cost, soc, peak_state, _ = walk.step(idx, soc, peak_state, p)
            reward = baseline[idx] - cost
            done = k == ep_steps - 1
            if done:
                mean_rate = float(np.mean(sig.import_rate[idx0:idx0 + ep_steps]))
                reward -= (soc_target - soc) / sig.eta_ch * mean_rate
            ep_return += reward

            nxt_idx = min(idx + 1, sig.n_steps - 1)
            nxt = np.concatenate([static_norm[nxt_idx],
                                  fb.dynamic(sig, nxt_idx, soc, peak_state)])
            pending.append((obs, a, reward * cfg.reward_scale, explored, nxt))
            if done:
                while pending:
                    _flush(True)
            elif len(pending) >= cfg.n_step:
                _flush(False)
            obs = nxt
            step += 1

            if buf.n >= cfg.learn_start and step % cfg.update_every == 0:
                o, ac, r, o2, dn, disc = buf.sample(rng, cfg.batch)
                o = torch.from_numpy(o)
                o2 = torch.from_numpy(o2)
                r = torch.from_numpy(r)
                dn = torch.from_numpy(dn)
                ac = torch.from_numpy(ac)
                disc = torch.from_numpy(disc)
                with torch.no_grad():
                    a2 = net(o2).argmax(dim=1, keepdim=True)
                    q2 = target(o2).gather(1, a2).squeeze(1)
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
                        loss = loss + lam * nn.functional.cross_entropy(q_all, bc_a)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.item()))
            if step % cfg.target_sync == 0:
                target.load_state_dict(net.state_dict())

            if step % cfg.eval_every == 0:
                val = _pool_validation(net, fb, val_members, val_blocks,
                                       soc_target, respect_peak, val_wear,
                                       n_val_days)
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
                    evals_since_best += 1
                if verbose:
                    print(f"    step {step:>8d}  eps {eps:.2f}  "
                          f"val {val['cost_eur_closed']:9.2f}  "
                          f"best {best_val:9.2f}  "
                          f"({evals_since_best} evals since best, "
                          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
                if evals_since_best >= cfg.patience:
                    stopped_early = True
                    break
            if step >= cfg.total_steps:
                break

        history["episode_return_eur"].append(ep_return)
        history["episode_start_day"].append(d0)
        history["episode_member"].append(mi)

    net.load_state_dict(best_state)
    history["runtime_s"] = time.time() - t0
    history["steps_run"] = step
    history["best_val_cost_closed"] = float(best_val)
    history["kept_init"] = bool(init_is_candidate and init_net is not None
                                and best_eval_i < 0)
    history["stopped_early"] = stopped_early
    n_evals = len(history["val_step"])
    history["converged"] = bool(
        stopped_early or (best_eval_i >= 0
                          and n_evals - 1 - best_eval_i >= cfg.patience))
    return net, history


# ---------------------------------------------------------------------------
# Pooled behaviour cloning
# ---------------------------------------------------------------------------
@dataclass
class BCConfigPool:
    """The pooled clone's optimisation settings: `rl.train_bc`'s own.

    Batch 512, patience 10, up to 200 epochs -- exactly the local clone, so a
    pooled clone differs from a local one in its DATA and nothing else. The
    first screen used 2048 / 5 / 100 on the theory that a 300-household epoch
    needs a shorter leash, and it confounded every small pool: the three
    single-household "types" -- the local clone's data, row for row -- came out
    37 EUR/yr worse than the local clone on AU, because a 28k-row pool gets 14
    updates an epoch at batch 2048 and is stopped before it has fitted. The
    global pools did not need the shorter leash either: early stopping on the
    validation weeks is what guards against overfitting, at any epoch size.
    """

    batch: int = 512
    max_epochs: int = 200
    patience: int = 10

    def config(self) -> dict:
        return {"batch": self.batch, "max_epochs": self.max_epochs,
                "patience": self.patience}


def pool_config_for(opts: "rl.BCOptions") -> BCConfigPool:
    """The pooled clone's batch / epochs / patience, taken from the local
    clone's options -- the same "differ in DATA only" rule as above, now that
    the local clone's optimiser is tuned (`run_rl_benchmark.effective_clone`)."""
    return BCConfigPool(batch=opts.batch, max_epochs=opts.max_epochs,
                        patience=opts.patience)


def train_bc_pool(X_tr, y_tr, X_va, y_va, cfg: rl.TrainConfig,
                  pool_cfg: BCConfigPool | None = None, verbose: bool = True,
                  init_net=None, lr_scale: float = 1.0,
                  opts: "rl.BCOptions | None" = None):
    """Clone the MILP across a pool. Inputs are already-built observations.

    `X_tr, y_tr` are the teacher-walk observations and labels of every member's
    TRAINING days, `X_va, y_va` those of their validation weeks -- each member's
    teacher walked over its own `TEACH_SPAN`, exactly as `rl.train_bc` walks
    one (`rl.teacher_actions`). Inverse-frequency class weights over the pooled
    labels, cross-entropy, early stop on held-out loss. Returns (net, history).

    `init_net` warm-starts the fit (deep-copied, never mutated) -- the
    `global_ft` scheme's localisation of a global clone to one type; the
    held-out loss of the UNTOUCHED prior is the first bar the fine-tune must
    clear, so a type whose own data cannot improve on the global clone keeps
    the global weights rather than a worse local refit.

    `opts` (`rl.BCOptions`) is the clone's optimiser exactly as `rl.train_bc`
    reads it -- class-weight power, label smoothing, weight decay, its own
    learning rate -- applied only where it differs from the defaults, so a
    default `opts` is the fit this function always performed.
    """
    opts = opts or rl.BCOptions()
    pool_cfg = pool_cfg or pool_config_for(opts)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    counts = np.bincount(y_tr, minlength=rl.N_ACTIONS).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if opts.class_power != 1.0:
        weights = weights ** opts.class_power
    weights = weights / weights.mean()

    net = (copy.deepcopy(init_net) if init_net is not None
           else rl.QNet(X_tr.shape[1], cfg.hidden))
    lr = cfg.lr if opts.lr is None else opts.lr
    adam_kw = {"weight_decay": opts.weight_decay} if opts.weight_decay else {}
    opt = torch.optim.Adam(net.parameters(), lr=lr * lr_scale, foreach=True,
                           **adam_kw)
    ce_kw = {"label_smoothing": opts.label_smoothing} if opts.label_smoothing else {}
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)),
                                **ce_kw)
    Xtr_t, ytr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xva_t, yva_t = torch.from_numpy(X_va), torch.from_numpy(y_va)

    def _val():
        net.eval()
        tot, agree = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(Xva_t), 65536):
                out = net(Xva_t[i:i + 65536])
                tot += float(nn.functional.cross_entropy(
                    out, yva_t[i:i + 65536], weight=lossf.weight,
                    reduction="sum", **ce_kw).item())
                agree += int((out.argmax(dim=1) == yva_t[i:i + 65536]).sum().item())
        # Weighted mean, as `CrossEntropyLoss(weight=...)` reports it.
        wsum = float(lossf.weight[yva_t].sum().item())
        return tot / max(wsum, 1e-12), agree / max(len(yva_t), 1)

    history = {"train_loss": [], "val_loss": [], "val_agreement": [],
               "label_counts": counts.tolist(), "config": cfg.config(),
               "pool_config": pool_cfg.config(), "bc_options": opts.config(),
               "n_train_rows": int(len(X_tr)), "n_val_rows": int(len(X_va))}
    best = _val()[0] if init_net is not None else np.inf
    history["init_val_loss"] = None if init_net is None else float(best)
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    since_best = 0
    t0 = time.time()
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    for ep in range(pool_cfg.max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), pool_cfg.batch):
            j = perm[i:i + pool_cfg.batch]
            loss = lossf(net(Xtr_t[j]), ytr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(j)
        vloss, agree = _val()
        history["train_loss"].append(tot / len(Xtr_t))
        history["val_loss"].append(vloss)
        history["val_agreement"].append(agree)
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        if verbose:
            print(f"    BC epoch {ep:3d}  val loss {vloss:.4f}  "
                  f"agreement {agree:.3f}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
        if since_best >= pool_cfg.patience:
            break
    net.load_state_dict(best_state)
    history["runtime_s"] = time.time() - t0
    history["converged"] = bool(since_best >= pool_cfg.patience)
    history["best_val_loss"] = float(best)
    history["final_val_agreement"] = _val()[1]
    return net, history


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(path: str, net, fb: TypedFeatureBuilder, cfg: rl.TrainConfig,
               history: dict, extra: dict | None = None):
    """`rl.save_model` plus the type width. Loads back through `load_model`."""
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "dim": fb.dim,
        "hidden": cfg.hidden,
        "norm_mean": fb.norm_mean,
        "norm_std": fb.norm_std,
        "n_types": fb.n_types,
        "spec": fb.spec.config(),
        "train_config": cfg.config(),
        "extra": extra or {},
    }, path)
    with open(path + ".history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, default=float)


def load_model(path: str):
    """Returns (net, fb, blob). `fb` is normalised and unbound to a type."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    fb = TypedFeatureBuilder(rl.FeatureSpec(**blob["spec"]),
                             n_types=int(blob.get("n_types", 0)))
    fb.norm_mean = np.asarray(blob["norm_mean"], dtype=np.float32)
    fb.norm_std = np.asarray(blob["norm_std"], dtype=np.float32)
    net = rl.QNet(int(blob["dim"]), int(blob["hidden"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, fb, blob


# ---------------------------------------------------------------------------
# Time-localised deployment: one network per period, switched by the clock
# ---------------------------------------------------------------------------
class SwitchingPolicy(rbc.Policy):
    """Several networks sharing one observation contract, one active per step.

    `period_of(index) -> int array` maps the signal bundle's timestamps to the
    network that drives each interval -- a season, or the month a rolling
    fine-tune was made for. Every network was fitted to the SAME scaler (they
    are fine-tunes of one parent), so the static features are built once and
    only the forward pass changes with the period. SOC, the ratchet peak and
    the contract carry across a switch exactly as they carry across any other
    interval: the runner owns them, not the network.
    """

    def __init__(self, nets: dict, fb, respect_peak: bool, period_of,
                 load_fc=None, pv_fc=None, name="switching", label="switching",
                 causal: bool = True):
        self.nets = nets
        self.fb = fb
        self.respect_peak = bool(respect_peak)
        self.period_of = period_of
        self.load_fc, self.pv_fc = load_fc, pv_fc
        self.name, self.label, self.causal = name, label, bool(causal)
        self._static = self._period = None

    def reset(self, sig):
        static = self.fb.build_static(sig, load_fc=self.load_fc, pv_fc=self.pv_fc)
        self._static = self.fb.normalize(static)
        self._period = np.asarray(self.period_of(sig.env.dataset.index[:sig.n_steps]))
        missing = set(np.unique(self._period)) - set(self.nets)
        if missing:
            raise KeyError(f"no network for period(s) {sorted(missing)}")
        for n in self.nets.values():
            n.eval()

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        obs = np.concatenate([self._static[idx],
                              self.fb.dynamic(sig, idx, soc_kwh, peak_state)])
        net = self.nets[int(self._period[idx])]
        with torch.no_grad():
            a = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
        return rl.action_setpoint(a, sig, idx, lo, hi, peak_state, self.respect_peak)


# ---------------------------------------------------------------------------
# A strict acceptance check for a fine-tune: day by day, on unseen days
# ---------------------------------------------------------------------------
def daily_costs(net, fb, member, start_day: int, end_day: int, soc_init: float,
                respect_peak: bool) -> np.ndarray:
    """Greedy rollout of `net` over [start_day, end_day), the bill per DAY.

    One continuous rollout -- SOC and the ratchet peak carry over midnight, as
    they do in deployment -- cut into days afterwards, with the terminal-SOC
    close-out charged to the last day (the same close-out the validation
    measure uses). Per-day costs are what a paired test needs: a single sum
    over two weeks cannot say whether a difference is more than noise.
    """
    sig, env = member.sig, member.env
    spd = int(round(24.0 / sig.hours))
    walk = rl._Walk(sig, member.settle, env, 0.0)
    soc = soc_init
    peak = rl.seed_peak_state(env, start_day * spd)
    out = np.zeros(end_day - start_day)
    net.eval()
    with torch.no_grad():
        for idx in range(start_day * spd, end_day * spd):
            peak = rl._drop_on_boundary(peak, sig.windows, idx)
            lo, hi = walk.bounds(soc)
            obs = np.concatenate([member.static_norm[idx],
                                  fb.dynamic(sig, idx, soc, peak)])
            a = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
            p = float(np.clip(rl.action_setpoint(a, sig, idx, lo, hi, peak,
                                                 respect_peak), lo, hi))
            c, soc, peak, _ = walk.step(idx, soc, peak, p)
            out[idx // spd - start_day] += c
    rate = float(np.mean(sig.import_rate[start_day * spd:end_day * spd]))
    out[-1] += (soc_init - soc) / sig.eta_ch * rate
    return out


def accept_fine_tune(candidate, parent, fb, member, start_day, end_day,
                     soc_init, respect_peak, alpha: float = 0.10) -> dict:
    """Keep `candidate` only if it beats `parent` on the check days, day by day.

    One-sided paired Wilcoxon on the daily bills, `alpha` 0.10, and a positive
    mean improvement. The check days must be days neither network was trained,
    early-stopped or labelled on -- the caller's job -- or the check is the
    same biased comparison the weak guard already makes.
    """
    from scipy.stats import wilcoxon
    dc = daily_costs(candidate, fb, member, start_day, end_day, soc_init, respect_peak)
    dp = daily_costs(parent, fb, member, start_day, end_day, soc_init, respect_peak)
    diff = dp - dc                                    # > 0: candidate cheaper
    if np.all(np.abs(diff) < 1e-12):
        p = 1.0
    else:
        p = float(wilcoxon(diff, alternative="greater", zero_method="zsplit").pvalue)
    return {"accept": bool(diff.mean() > 0 and p < alpha), "p": p,
            "gain": float(diff.sum())}
