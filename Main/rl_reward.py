"""What a learned controller is PAID for: one reward function per tariff and,
on SI, one per contract type.

`rl_control.train_dqn` used to carry a single reward for both tariffs -- the
arm's settled bill against the no-battery bill. On AU that is the whole
invoice. On SI it is not: `price_interval` returns the energy and the
excess-power charge as `variable` and the standing charge as `fixed`, and the
standing charge is the larger half of what a battery can move, because it is
levied on a dogovorjena moc the household's own peaks re-agree. The diagnosis
in FIGURES_RL (section 6) put numbers on it: of the whole-period MILP's
108 EUR/a on the 30 study households 51 came off the contract line, the peak
shaving rules took 31-36 there, and the learned controllers 1-13 -- they had
been trained, validated and tuned on a measure that could not see it.

So the reward is a property of the BILL, and there is one here per bill:

    AUEnergyReward            Ausgrid EA025. Energy only: there is no capacity
                              charge, so there is nothing else to pay for.
    SIExcessPowerReward       SI on a SIGNED contract (`fixed`). Energy plus the
                              excess-power ratchet above a line no dispatch
                              moves; the standing charge is a constant and is
                              rightly absent.
    SIMonthlyContractReward   SI on the study's contract (`monthly`): the line
                              is re-agreed every month from the five highest
                              draws of the month before. Energy, excess power,
                              AND the contract charge -- what an interval's draw
                              does to the next month's agreed power, priced at
                              the standing charge it will be billed.
    SIAnnualContractReward    SI on the operator's rolling window (`annual`):
                              the same statistic pooled over twelve months, so
                              a draw that enters the pool is paid for a year.

WHAT THE CONTRACT CHARGE IS. The agreed power of block b is the mean of the k
highest draws in the pool it is read from (`si_moc.mesecni_razpored`), made
monotone across blocks, and the standing charge is linear in it
(`MILP_Household.fixed_charge_coefficients`). So the contract cost of a pool is
a function of its per-block top-k lists, and the charge on one interval is the
DIFFERENCE that function takes when the interval's draw is added to the pool:
zero for a draw below the k-th highest, (draw - k-th highest) / k x rate above
it. Summed over a pool's lifetime the differences telescope to exactly its
contract cost. Nothing is estimated and no shaping coefficient is involved;
`test_rl_reward` checks the sum against the standing charge the environment
itself bills after re-agreeing the contract.

Three things have to hold for that to be the bill the agent is later scored
on. All three were missing, and each was arrived at by measuring the obvious
alternative first (`ContractOptions`, FIGURES_RL_CONTRACT section 2):

    the pool is the agent's        An episode is a week or two; a contract is
    to set                         decided by whole billing periods. Priced
                                   into the rest of the month as the agent
                                   last dispatched it -- the exact marginal
                                   cost of the episode -- no episode is ever
                                   paid to go below peaks the agent left
                                   standing in the others, and the LEVEL of
                                   the contract is nobody's decision. On the
                                   monthly contract the pool therefore opens
                                   empty with the episode, which is billed its
                                   time-share of the standing charge its own
                                   peaks set. On the twelve-month contract no
                                   episode can stand in for the pool, and the
                                   draws are priced into the agent's own
                                   trailing year.
    the contract is the agent's    The excess-power line the agent trains
    own                            against, and the agreed power its features
                                   read, are re-agreed from the agent's own
                                   full-period trajectory at every validation
                                   (`env.set_achieved_power_kw`), as
                                   `run_policy` re-agrees them at scoring time.
                                   Trained against the no-battery contract an
                                   agent stops shaving at a line it would never
                                   be billed under. Without this the charge is
                                   significantly worse on validation.
    validation sees it, level      The early stop and every tuning choice read
    included                       the bill on the held-out weeks of one greedy
                                   pass over the whole training period: each
                                   week's settled bill, what its draws add to
                                   the contract pool, AND its time-share of the
                                   standing charge the rest of the pool sets.
                                   Without the last term a profile shaved flat
                                   costs nothing at any level, and the MILP
                                   validated no better than a rule.

WHAT IT BOUGHT (30 households, test year, FIGURES_RL_CONTRACT). For the DQN on
the study's contract +12.6 EUR/a on the panel's seed and +7.9 averaged over
three (every seed positive), all of it off the standing charge; +9.7 on the
twelve-month rule and +6.3 on the single-tariff price list; nothing on the
active price list, where the energy line is five times the contract line; and
a LOSS of 9 EUR/a when a contract-charge model is run on a signed contract --
which is why the reward is chosen by the bill and not once for the tariff.
Nothing for the clone, which has no reward to change. A hyperparameter search
under this reward (`rl_hpo`, `reward="contract"`) found no settings better
than the panel's.

ONE EVALUATOR, still. Nothing here re-prices an invoice: the settled part of
the reward is the arm's `settle`, the rates are read off
`compute_prorated_fixed_charge_eur` by differencing (via
`fixed_charge_coefficients`), the statistic is `si_moc`'s, and SCORING is
`Rule_Based_Control.run_policy`, unchanged. The contract charge is the
training signal; the test-year number never goes through it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, asdict

import numpy as np

from Basic_Functions import max_charge_now, max_discharge_now
import MILP_Household as mh
import Rule_Based_Control as rbc
import rl_control as rl
import si_moc

# Bumped when a reward changes in a way a stored result cannot be compared
# across -- the job `rl.ALGO_VERSION` does for the update rule. The legacy
# rewards (`AUEnergyReward`, `SIExcessPowerReward`) are the loop `rl_control`
# always ran and are NOT versioned here: a result trained through them is
# identified by `rl.ALGO_VERSION` alone, exactly as before this module existed.
#
#   1  contract charge as the leave-episode-out marginal of the agent's own
#      pool, contract re-agreed at each validation, held-out contract
#      validation, contract features
REWARD_VERSION = 1

MEASURED_BLOCKS = tuple(si_moc.KONICNI_BLOKI)        # 1-4: block 5 is not read
# The threshold reported for an interval whose draw cannot enter any pool
# (block 5). Finite, because it is a network input; far above any household
# draw, because that is what "this interval is free" means.
FREE_KW = 20.0


# ---------------------------------------------------------------------------
# SI contract types
# ---------------------------------------------------------------------------
# What "the SI contract" can be, as far as the agreed power is concerned. The
# study runs ONE of them -- `monthly` -- on every arm; the other two are what
# `Environment` already models and a household can actually be on, and they
# differ in the one thing a reward has to know: whether, and for how long, a
# draw is paid for again through the standing charge.
#
#   fixed     a number the household signed and nothing moves. The operator's
#             own figure: the mean of the five highest draws per block over the
#             metering year before (`si_moc.dogovorjena_moc_operaterja`), read
#             off the no-battery profile -- the contract a household walks in
#             with when it buys a battery and does not touch its paperwork.
#             The excess-power charge is billed above it.
#   monthly   re-agreed each month from the five highest draws of the month
#             before. A shaved peak buys one month of a lower standing charge.
#             THE STUDY'S CONTRACT (`hs.REFERENCE_ARM["SI"]`).
#   annual    the same statistic over a rolling twelve months -- the window the
#             Akt reads. A draw that enters the pool is billed for a year, and
#             one that is kept out stays out for as long.
CONTRACT_TYPES = ("fixed", "monthly", "annual")
REFERENCE_CONTRACT = "monthly"

# ...and the SUPPLIER PRICE LIST, which is the other thing "the SI contract"
# means and the one a household actually chooses between. The three kinds
# `si_paketi` knows (`TipCene`), each as GEN-I's own self-supply list, all on
# the study's agreed-power rule (re-agreed monthly):
#
#   monthly   DYNAMIC: SIPX hour by hour plus a margin, both ways
#             (GENI_SAMO_DINAMICNI). THE STUDY'S PRICE LIST -- the same
#             contract as `monthly` above, which is why it keeps that key:
#             the two axes cross at the study's own contract.
#   active    "AKTIVNI": four energy prices by time of day and season --
#             solar low/high season, base, peak -- 0.04 to 0.19 EUR/kWh, and
#             a matching four-price buy-back (GENI_SAMO_AKTIVNI). The energy
#             line is two to three times the dynamic list's and has a spread
#             a clock can trade on.
#   single    ONE price for every imported kWh and one for every exported one
#             (GENI_SAMO_REDNI, "enotna tarifa"). Nothing to arbitrage: what
#             a battery can do is keep its own PV and shave the contract.
#
# GEN-I's fourth self-supply list (GENI_SAMO_FIKSNI) is the active list half a
# cent dearer under a 12-month tie-in -- the same structure, and left out.
PRICE_TYPES = ("monthly", "active", "single")
PRICE_PAKET = {"monthly": "GENI_SAMO_DINAMICNI", "active": "GENI_SAMO_AKTIVNI",
               "single": "GENI_SAMO_REDNI"}
PRICE_LABEL = {"monthly": "dynamic price", "active": "active (4-tariff) price",
               "single": "single-tariff price"}
ALL_TYPES = CONTRACT_TYPES + tuple(t for t in PRICE_TYPES
                                   if t not in CONTRACT_TYPES)

CONTRACT_WINDOW = {"fixed": None, "monthly": 1, "annual": 12,
                   "active": 1, "single": 1}
CONTRACT_LABEL = {"fixed": "signed (fixed)", "monthly": "re-agreed monthly",
                  "annual": "rolling 12 months",
                  "active": PRICE_LABEL["active"], "single": PRICE_LABEL["single"]}


def family(contract: str) -> tuple:
    """The contract types `contract` is an alternative to.

    A household chooses its price list, and separately how its agreed power is
    set; a model trained on one price list is a candidate for another price
    list, not for another agreed-power rule on a third list. The study's own
    contract sits in both families.
    """
    return tuple(f for f in (CONTRACT_TYPES, PRICE_TYPES) if contract in f)


def naive_import_kw(sig) -> np.ndarray:
    """The household's grid draw in kW with the battery idle."""
    return np.maximum(sig.consumption - sig.generation, 0.0) / sig.hours


def operator_contract(sig, start: int, stop: int, year: int = 2026) -> dict:
    """{block: kW} the operator would propose from rows [start, stop).

    The mean of the five highest no-battery draws per block, blocks 1-4,
    rounded to the 0.1 kW a bill states and made monotone -- `si_moc`'s own
    rule, with the regulatory floor off for the reason `si_moc` gives: the
    profiles carry no connection agreement to derive one from.
    """
    kw = naive_import_kw(sig)[start:stop]
    blocks = sig.blocks[start:stop]
    peaks = {b: si_moc.povprecje_najvecjih(kw[blocks == b], si_moc.ST_KONIC)
             for b in MEASURED_BLOCKS if np.any(blocks == b)}
    return si_moc.dogovorjena_moc_operaterja(peaks, leto=year,
                                             minimalna_moc_kw=0.0)


def contract_env_kwargs(contract: str, fixed_kw: dict | None = None) -> dict | None:
    """`hs.build_study_env` keywords for one contract type; None = the study's."""
    if contract == "monthly":
        return None
    if contract in PRICE_PAKET:
        return {"paket_id": PRICE_PAKET[contract]}
    if contract == "annual":
        return {"agreed_power": {"agreed_power_n_months_window": 12}}
    if contract == "fixed":
        if not fixed_kw:
            raise ValueError("a fixed contract needs the signed kW per block")
        return {"contracted_power_kw": {int(b): float(v)
                                        for b, v in fixed_kw.items()}}
    raise ValueError(f"unknown contract type {contract!r}; "
                     f"expected one of {ALL_TYPES}")


# ---------------------------------------------------------------------------
# The contract's calendar and prices, for one environment
# ---------------------------------------------------------------------------
class ContractCalendar:
    """Which pool an interval's draw enters, and what a kW in it costs.

    Built once per environment. `window` is the contract type's pooling window
    in months (1 or 12). Everything priced here is read off the shared tariff
    code: the month and block of every interval from the environment, the
    EUR-per-agreed-kW of every (block, month) from
    `MILP_Household.fixed_charge_coefficients` -- which differences the real
    prorated standing charge rather than restating a rate.

    TWO APPROXIMATIONS, both at the two season boundaries of a year and
    nowhere else, both stated because the contract charge is otherwise exact:

      * a block's draws are priced at the next month that BILLS the block. For
        ten months of twelve that is simply the next month. Block 1 in the last
        high-season month is carried to the first one of the next season
        (`prenesi_manjkajoce_bloke`), and is priced there.
      * the monotone chain is run over the blocks of the month the draw falls
        in, not of the month it is billed in.

    Scoring does not go through either; `run_policy` does its own accounting.
    """

    def __init__(self, env, sig, window: int):
        if window not in (1, 12):
            raise ValueError("the contract pool is 1 or 12 months")
        if float(env.min_agreed_power_kw) != 0.0 or env.connection_power_kw is not None:
            # A floor or a ceiling makes the contract cost flat over a range of
            # draws; the marginal charge below would price draws that are free.
            raise NotImplementedError("the contract charge assumes the study's "
                                      "unbounded agreed power")
        n = sig.n_steps
        dates = env.dataset.index[:n]
        _, _, months_sorted, month_idx_t = mh.month_calendar(dates)
        self.window = int(window)
        self.k = int(env.agreed_power_n_peaks)
        self.n_steps = n
        self.hours = sig.hours
        self.blocks = np.asarray(sig.blocks[:n], dtype=int)
        self.month = np.asarray(month_idx_t, dtype=int)
        self.n_months = len(months_sorted)
        self.first_row = np.array([int(np.argmax(self.month == m))
                                   for m in range(self.n_months)])

        order = np.arange(n)
        self._rows = {(b, m): order[(self.blocks == b) & (self.month == m)]
                      for m in range(self.n_months) for b in MEASURED_BLOCKS}
        boot = getattr(env, "agreed_power_bootstrap_kw", None) or {}
        self._bootstrap = {b: [float(v) for v in boot.get(b, ())]
                           for b in MEASURED_BLOCKS}

        opts = dict(env.pricing_options or {})
        opts.setdefault("pricing_reference_year", env.pricing_reference_year)
        coef, _ = mh.fixed_charge_coefficients(
            dates, month_idx_t, months_sorted, int(round(env.interval_minutes)),
            env.pricing_scheme, opts)
        # EUR per agreed kW by CALENDAR month: the horizon's own coefficient
        # for that month, taking the fullest instance -- a horizon that opens
        # or closes mid-month carries a prorated one, and the contract a draw
        # sets is billed for a whole month wherever the horizon happens to end.
        cal = {}
        for (b, m), c in coef.items():
            key = (int(b), months_sorted[m][1])
            cal[key] = max(cal.get(key, 0.0), float(c))
        self.rate_by_calendar_month = cal

        def billed_rate(b, month_of_year):
            """Standing charge on 1 kW of block b, for the months it buys."""
            if self.window == 12:
                return sum(cal.get((b, mo), 0.0) for mo in range(1, 13))
            for step in range(1, 13):
                mo = (month_of_year - 1 + step) % 12 + 1
                if cal.get((b, mo), 0.0) > 0.0:
                    return cal[(b, mo)]
            return 0.0

        self.rate, self.chain = [], []
        for m, (_, mo) in enumerate(months_sorted):
            r = {b: billed_rate(b, mo) for b in MEASURED_BLOCKS}
            # Block 5 has no peaks of its own; its agreed power is block 4's
            # through the monotone chain, so its (tiny) rate rides on block 4.
            r[4] += billed_rate(5, mo)
            self.rate.append(r)
            present = [b for b in MEASURED_BLOCKS
                       if self.window == 12 or len(self._rows[(b, m)])]
            self.chain.append(tuple(present))

    # -- pools ---------------------------------------------------------------
    def pool_rows(self, b: int, m: int) -> np.ndarray:
        """Rows whose draw the contract after month `m` is read from, block b."""
        if self.window == 1:
            return self._rows[(b, m)]
        return np.concatenate([self._rows[(b, j)]
                               for j in range(max(0, m - self.window + 1), m + 1)])

    def top_lists(self, profile_kw, m: int, before=None, exclude=None,
                  since=None) -> dict:
        """{block: the pool's k highest draws, ascending}.

        `before` keeps only rows earlier than that index (what a meter has seen
        so far); `since` only rows from that index on (a pool opened part-way
        through a month); `exclude=(a, b)` leaves rows [a, b) out (everything
        except one episode or one held-out week). A twelve-month pool that
        reaches back past the first month of the data also holds the meter
        history the household walked in with, exactly as
        `si_moc.mesecni_razpored` pools it -- unless the pool was opened with
        `since`, which is a pool that by construction has no history.
        """
        out = {}
        for b in MEASURED_BLOCKS:
            rows = self.pool_rows(b, m)
            if before is not None:
                rows = rows[rows < before]
            if since is not None:
                rows = rows[rows >= since]
            if exclude is not None:
                rows = rows[(rows < exclude[0]) | (rows >= exclude[1])]
            vals = profile_kw[rows]
            if (since is None and self.window > 1
                    and m - self.window + 1 < 0 and self._bootstrap[b]):
                vals = np.concatenate([vals, self._bootstrap[b]])
            if len(vals) > self.k:
                vals = np.partition(vals, len(vals) - self.k)[-self.k:]
            out[b] = sorted(float(v) for v in vals)
        return out

    def cost(self, lists: dict, m: int) -> float:
        """Standing charge the contract read from these lists will be billed.

        `si_moc`'s rule on the lists: the mean of what is there (fewer than k
        values average all of them, an empty pool is 0 kW), then a running
        maximum up the blocks.
        """
        rate = self.rate[m]
        running = total = 0.0
        for b in self.chain[m]:
            lst = lists[b]
            if lst:
                running = max(running, sum(lst) / len(lst))
            total += rate[b] * running
        return total

    def pool_cost(self, profile_kw, m: int, exclude=None) -> float:
        return self.cost(self.top_lists(profile_kw, m, exclude=exclude), m)

    def share(self, a: int, b: int, m: int) -> float:
        """The part of pool `m`'s billing period that rows [a, b) cover.

        A contract read from a pool is billed for a whole period -- a month,
        or twelve of them -- so rows covering a fraction of the period carry
        that fraction of its standing charge. Against a NOMINAL month rather
        than the calendar one: the difference is a few per cent, and a horizon
        that opens or closes mid-month would otherwise read as a full one.
        """
        nominal = 24.0 / self.hours * 30.4375 * self.window
        if self.window > 1:
            return max(b - a, 0) / nominal
        lo = int(self.first_row[m])
        hi = int(self.first_row[m + 1]) if m + 1 < self.n_months else self.n_steps
        return max(min(b, hi) - max(a, lo), 0) / nominal

    def leave_one_out(self, profile_kw) -> np.ndarray:
        """Per interval: what that ONE draw adds to its pool's contract cost.

        Non-zero only on the k highest draws of each (block, pool). Used as the
        contract half of the reward's baseline -- computed on the no-battery
        profile it is a fixed array over time, so subtracting it moves no
        argmax, and it takes the "this interval was always going to be a peak"
        part out of what the network has to regress.
        """
        out = np.zeros(self.n_steps, dtype=np.float64)
        for m in range(self.n_months):
            lists = self.top_lists(profile_kw, m)
            full = self.cost(lists, m)
            for b in MEASURED_BLOCKS:
                own = self._rows[(b, m)]
                if not len(own) or not lists[b]:
                    continue
                for r in own[profile_kw[own] >= lists[b][0]]:
                    without = dict(lists)
                    without[b] = self.top_lists(
                        profile_kw, m, exclude=(int(r), int(r) + 1))[b]
                    out[r] = full - self.cost(without, m)
        return out


    def ratchet_trace(self, profile_kw) -> np.ndarray:
        """Per interval: what its draw adds to the pool as the METER builds it.

        The pool is walked forward from the first interval of each month, so
        the entry is what a household watching its own meter would call the
        cost of that draw at that moment. The contract half of the baseline
        under `pricing="ratchet"`, for the same reason `leave_one_out` is
        under `"marginal"`.
        """
        out = np.zeros(self.n_steps, dtype=np.float64)
        tracker = ContractTracker(self, profile_kw)
        for idx in range(self.n_steps):
            out[idx] = tracker.push(idx, float(profile_kw[idx]))
        return out


class ContractTracker:
    """The top-k lists of one pool, kept up to date one metered draw at a time.

    Two uses, one class, and the seed is the whole difference between them:

      priced   (`exclude=(a, b)`) seeded from a full-period profile with one
               episode left out. `push` then returns what each of the
               episode's draws adds to the contract cost -- the reward's
               contract charge.
      causal   (`exclude=None`) seeded from the rows BEFORE the current one
               only. `threshold_at` is then what a meter knows: the k-th
               highest draw so far in this pool. This is the one the features
               read, in training, in cloning and in the field, so the agent
               never sees a threshold its deployed self could not compute.

    `own=True` makes the tracker write each pushed draw into its profile --
    the deployed agent building its own history from `observe` -- and is what
    lets a twelve-month pool roll forward on the agent's real past. A tracker
    seeded from a shared reference profile never writes to it.
    """

    def __init__(self, cal: ContractCalendar, profile_kw, exclude=None,
                 own: bool = False, span=None):
        self.cal = cal
        self.profile = profile_kw
        self.exclude = exclude
        self.own = bool(own)
        # `span=(a, b)` opens the pool EMPTY at row a: an episode treated as
        # the whole of its own billing period. `share` is then the part of
        # that period the episode covers, which is what its contract cost has
        # to be scaled by to sit beside the episode's own energy bill.
        self.span = span
        self.share = 1.0
        self.m = -1
        self.top = None
        self.value = 0.0

    def _enter(self, idx: int) -> None:
        m = int(self.cal.month[idx])
        if m == self.m:
            return
        self.m = m
        if self.span is not None:
            # Zero-filled rather than empty. `si_moc` averages what is there,
            # so the first draw into an empty list IS the agreed power and the
            # second can lower it; zero-filled, every draw adds
            # (draw - lowest) / k and the charge only ever ratchets up, to the
            # same total once k draws are in.
            self.top = {b: [0.0] * self.cal.k for b in MEASURED_BLOCKS}
            self.share = self.cal.share(self.span[0], self.span[1], m)
        elif self.exclude is None:
            self.top = self.cal.top_lists(self.profile, m, before=idx)
        else:
            self.top = self.cal.top_lists(self.profile, m, exclude=self.exclude)
        self.value = self.cal.cost(self.top, m)

    def threshold_at(self, idx: int) -> float:
        """The draw above which this interval raises the next contract, kW."""
        self._enter(idx)
        lst = self.top.get(int(self.cal.blocks[idx]))
        if lst is None:
            return FREE_KW                 # block 5: not read by the contract
        return lst[0] if len(lst) >= self.cal.k else 0.0

    def push_priced(self, idx: int, kw: float) -> float:
        """`push`, scaled by the share of the billing period the pool covers."""
        return self.push(idx, kw) * self.share

    def push(self, idx: int, kw: float) -> float:
        """Add one metered draw to the pool; return the EUR it adds."""
        self._enter(idx)
        if self.own:
            self.profile[idx] = kw
        lst = self.top.get(int(self.cal.blocks[idx]))
        if lst is None:
            return 0.0
        if len(lst) >= self.cal.k:
            if kw <= lst[0]:
                return 0.0
            lst.pop(0)
        bisect.insort(lst, float(kw))
        value = self.cal.cost(self.top, self.m)
        added, self.value = value - self.value, value
        return added


_CALENDARS: dict = {}


def calendar_for(sig, window: int) -> ContractCalendar:
    """The calendar of `sig`'s environment, built once per (environment, window).

    `LearnedPolicy.reset` asks on every pass of the contract's fixed-point
    loop; the calendar depends on the environment's index, blocks and price
    list, none of which that loop moves. Keyed on the environment object and
    holding a reference to it, so an id is never reused under a live key.
    """
    key = (id(sig.env), int(window), int(sig.n_steps))
    hit = _CALENDARS.get(key)
    if hit is None or hit[0] is not sig.env:
        # A handful of entries at most: a worker walks through thirty
        # households, and every entry pins an environment (~100 MB of signals
        # with its bundle) for as long as it stays here.
        while len(_CALENDARS) >= 6:
            _CALENDARS.pop(next(iter(_CALENDARS)))
        hit = _CALENDARS[key] = (sig.env, ContractCalendar(sig.env, sig, window))
    return hit[1]


def tracker_factory(window: int | None):
    """`LearnedPolicy(tracker_factory=...)` for a model trained on `window`.

    The window is the MODEL's -- the pool its features were defined on -- and
    the calendar is the environment's it is being run in, so a model can be
    scored under a contract type it was not trained on and still read its own
    features. None (a spec without contract features) needs no tracker.
    """
    if window is None:
        return None

    def _make(sig):
        cal = calendar_for(sig, window)
        return ContractTracker(cal, np.zeros(sig.n_steps), own=True)
    return _make


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
class RewardFunction:
    """The settled bill against the no-battery bill: the loop's own reward.

    The base class IS the reward `rl_control.train_dqn` carried before this
    module -- same baseline, no extra charge, `validation_rollout` on the
    held-out weeks -- so the two subclasses that add nothing to it
    (`AUEnergyReward`, `SIExcessPowerReward`) train a network bit for bit as
    `reward_fn=None` does, and are named for what the settlement they are
    handed bills rather than for anything they compute.
    """

    name = "settled_bill"
    tariff = None
    window = None                       # months in the contract pool, if any
    versioned = False                   # legacy rewards carry no REWARD_VERSION

    def __init__(self, bundle: dict, fb, static_norm, *, val_blocks, soc_init,
                 respect_peak: bool, load_fc=None, pv_fc=None, context=None):
        self.env = bundle["env"]
        self.sig = bundle["sig"]
        self.settle = bundle["settle"]
        self.fb = fb
        self.static_norm = static_norm
        self.val_blocks = rl.as_blocks(val_blocks)
        self.soc_init = float(soc_init)
        self.respect_peak = bool(respect_peak)
        self.load_fc, self.pv_fc = load_fc, pv_fc
        # The household's own past (`rl_features.Context`), for a spec with
        # feature groups: handed on wherever the features are rebuilt.
        self.context = context

    def config(self) -> dict:
        """What a result trained under this reward depends on (digested)."""
        return {"reward": self.name}

    # -- the hooks `train_dqn` calls ----------------------------------------
    def baseline(self) -> np.ndarray:
        return rl.no_battery_cost_trace(self.sig, self.settle, self.env)

    def start_episode(self, idx0: int, n_steps: int):
        return None

    def tau(self, state, idx: int):
        return None

    def charge(self, state, idx: int, net_kwh: float) -> float:
        return 0.0

    def validate(self, net):
        val = rl.validation_rollout(net, self.fb, self.static_norm, self.sig,
                                    self.settle, self.env, self.val_blocks,
                                    self.soc_init, self.respect_peak)
        return val, self.sig, self.static_norm

    # -- what the driver calls ----------------------------------------------
    def tracker_factory(self):
        return tracker_factory(self.window if self.fb.spec.contract_features
                               else None)

    def policy(self, net, **kw):
        return rl.LearnedPolicy(net, self.fb, self.respect_peak,
                                load_fc=self.load_fc, pv_fc=self.pv_fc,
                                context=self.context,
                                tracker_factory=self.tracker_factory(), **kw)


class AUEnergyReward(RewardFunction):
    """Ausgrid EA025: the energy bill, and nothing else.

    The tariff as modelled has a time-of-use energy rate and a standing charge
    no dispatch moves -- no capacity charge, no agreed power, no ratchet. The
    settled interval cost is therefore the whole of what an action changes, and
    a reward with a peak or contract term in it would be paying the agent for
    a line that is not on the invoice.
    """

    name = "au_energy"
    tariff = "AU"


class SIExcessPowerReward(RewardFunction):
    """SI on a contract the dispatch does NOT move: energy + excess power.

    `price_interval`'s `variable`: the delivered energy and the excess-power
    ratchet above the agreed power in force. Right for a signed contract
    (`fixed`), where the standing charge is a constant. On a contract the
    dispatch re-agrees this is the reward the panel was trained on before
    `rl_reward` -- the "contract-blind" reward -- and it is kept selectable on
    those contracts as the control the contract charge is measured against.
    """

    name = "si_excess_power"
    tariff = "SI"


@dataclass(frozen=True)
class ContractOptions:
    """The contract charge's two switches, digested with a result.

    `weight` multiplies the contract charge in the TRAINING reward. 1 is the
    bill; it is a setting only because the exact marginal overstates what a kW
    is worth to an agent that then pays excess power on the line it lowered
    (the excess charge returns 0.9 of the rate), which is an empirical
    question and is answered on validation. Validation itself always prices
    the contract at 1.

    `self_consistent` re-agrees the contract from the agent's own trajectory
    at every validation. Off, the agent trains against the no-battery
    contract throughout -- the ablation that says how much of the gain is the
    charge and how much is the line.

    `pricing` is WHICH pool an episode's draws are priced into. All three
    are the same telescoping sum over a different seed, and the seed decides
    what the agent is held responsible for:

      "marginal"  the agent's own last full-period trajectory with the
                  episode left out. The exact cost of the episode given the
                  rest of the pool -- and, measured, a coordination trap: the
                  rest of the pool is the agent's own un-shaved past, so a
                  draw is only charged above peaks the agent itself left
                  standing, and no single episode is ever paid to go below
                  them. The LEVEL of the contract is nobody's decision.
      "ratchet"   the meter's pool: what it has drawn BEFORE this interval,
                  from the agent's own last trajectory, and nothing after.
                  Every draw above the k-th highest so far is charged as it
                  happens, exactly as the excess-power ratchet charges a new
                  running peak, and that threshold is the one the features
                  read. Right where the pool is far longer than an episode
                  and its standing peaks really are sunk: the default on the
                  twelve-month contract.
      "episode"   the pool opens EMPTY with the episode, which is billed its
                  time-share of the standing charge its own peaks set -- the
                  episode as the whole of its own billing period. The level
                  is then the agent's decision in every episode, which is
                  what it is over a month the agent dispatches end to end.
                  The default on the monthly contract.

    None picks the contract type's default.
    """

    weight: float = 1.0
    self_consistent: bool = True
    pricing: str | None = None

    def __post_init__(self):
        if self.pricing not in (None, "episode", "ratchet", "marginal"):
            raise ValueError("pricing must be episode | ratchet | marginal")

    def config(self) -> dict:
        return asdict(self)


class SIContractChargeReward(RewardFunction):
    """SI on a contract the dispatch re-agrees: + the contract charge.

    See the module docstring for what the charge is and why it is the bill.
    Subclassed per contract type for the pool the charge is read from; the
    arithmetic is the same and lives here.
    """

    name = "si_contract_charge"
    tariff = "SI"
    versioned = True
    default_pricing = "ratchet"

    def __init__(self, bundle, fb, static_norm, *, options=None, **kw):
        super().__init__(bundle, fb, static_norm, **kw)
        self.options = options or ContractOptions()
        self.pricing = self.options.pricing or self.default_pricing
        if self.pricing == "episode" and self.window != 1:
            raise ValueError("an episode cannot stand in for a twelve-month "
                             "pool; price it by 'ratchet'")
        self.rates = bundle["rates"]
        self.cal = calendar_for(self.sig, self.window)
        self.naive = naive_import_kw(self.sig)
        # The contract always starts from the no-battery profile: whatever a
        # previous run left on this environment is not this agent's.
        self.env.clear_achieved_power()
        self.sig = rbc.rebind_agreed_power(self.sig, self.env)
        self.profile = self.naive
        self._load_static()
        nominal = float(getattr(self.env, "nominal_capacity_kwh",
                                self.sig.capacity_kwh))
        self._nominal = nominal
        self.last_pass = None

    def config(self) -> dict:
        return {"reward": self.name, "version": REWARD_VERSION,
                **self.options.config(), "pricing": self.pricing}

    def _load_static(self):
        # No builder: the reward is being used as a METER for a rule, which
        # reads its signals directly and has no features to rebuild.
        if self.fb is not None:
            self.static_norm = self.fb.normalize(self.fb.build_static(
                self.sig, load_fc=self.load_fc, pv_fc=self.pv_fc,
                context=self.context))

    # -- the reward ---------------------------------------------------------
    def baseline(self) -> np.ndarray:
        # Both halves on the no-battery profile under the no-battery contract:
        # fixed arrays over time, so they shift every action's value alike.
        base = rl.no_battery_cost_trace(self.sig, self.settle, self.env)
        if self.pricing == "episode":
            # The contract half of the baseline depends on where the episode
            # opened its pool, so it is walked beside the agent's
            # (`start_episode`) instead of being read off a fixed array.
            return base
        contract = (self.cal.ratchet_trace(self.naive)
                    if self.pricing == "ratchet"
                    else self.cal.leave_one_out(self.naive))
        return base + self.options.weight * contract

    def start_episode(self, idx0: int, n_steps: int):
        """(the pool the episode is priced into, the pool its features read,
        the no-battery pool walked beside it or None)."""
        if self.pricing == "episode":
            span = (idx0, idx0 + n_steps)
            pool = ContractTracker(self.cal, self.profile, span=span)
            return pool, pool, ContractTracker(self.cal, self.naive, span=span)
        seen = ContractTracker(self.cal, self.profile)
        if self.pricing == "ratchet":
            return seen, seen, None            # one pool: the meter's
        priced = ContractTracker(self.cal, self.profile,
                                 exclude=(idx0, idx0 + n_steps))
        return priced, seen, None

    def tau(self, state, idx: int) -> float:
        return state[1].threshold_at(idx)

    def charge(self, state, idx: int, net_kwh: float) -> float:
        priced, seen, idle = state
        kw = max(float(net_kwh), 0.0) / self.sig.hours
        if seen is not priced:
            seen.push(idx, kw)
        added = priced.push_priced(idx, kw)
        if idle is not None:
            # What the same interval would have added with the battery idle:
            # action-independent, so it moves no argmax, and it leaves the
            # agent paid for the DIFFERENCE it made to the pool.
            added -= idle.push_priced(idx, float(self.naive[idx]))
        return self.options.weight * added

    # -- the teacher's contract state, for the cloner -----------------------
    def teacher_thresholds(self, setpoints_kwh, start: int, stop: int,
                           soc_trace) -> np.ndarray:
        """The causal threshold at every step of the teacher's walk.

        The walk is `rl.teacher_actions`'s: the setpoint clamped to the
        envelope at the SOC the teacher is at. Its meter reading is pushed
        into a tracker that starts empty, so the clone is shown what a
        deployed controller following the teacher would have known.
        """
        sig, env = self.sig, self.env
        tracker = ContractTracker(self.cal, np.zeros(sig.n_steps), own=True)
        taus = np.zeros(stop - start, dtype=np.float64)
        for j, idx in enumerate(range(start, stop)):
            soc = float(soc_trace[j])
            lo = -max_discharge_now(soc, sig.eta_dis, env.max_discharge_kwh)
            hi = max_charge_now(soc, sig.eta_ch, env.max_charge_kwh,
                                sig.capacity_kwh)
            p = float(np.clip(setpoints_kwh[idx - start], lo, hi))
            taus[j] = tracker.threshold_at(idx)
            net = sig.consumption[idx] + max(p, 0.0) - sig.generation[idx] \
                - max(-p, 0.0)
            tracker.push(idx, max(net, 0.0) / sig.hours)
        return taus

    # -- validation, and the contract it leaves in force --------------------
    def full_pass(self, net) -> dict:
        """One greedy pass over the whole training period, contract as in force.

        `net` is a network or any `Policy` -- a rule is validated on this axis
        exactly as a learned controller is, which is what lets the held-out
        number be read against the study's own roster.
        """
        policy = net if isinstance(net, rbc.Policy) else self.policy(net)
        out = rbc._run_policy_once(
            self.env, policy, n_steps=self.sig.n_steps,
            signals=self.sig, keep_traces=True, settle=self.settle,
            soc_init_kwh=self.soc_init)
        out["_import_kw"] = np.maximum(out["_net_trace"], 0.0) / self.sig.hours
        return out

    def held_out(self, out) -> dict:
        """The bill of the held-out weeks of one full pass, contract included.

        Per week: the settled bill of its intervals, the pack's change over
        the week closed out at the week's mean rate (as an episode is), and
        the week's part of the contract, which has two halves:

            what it ADDS    the pool's cost with the week minus its cost
                            without: a held-out week whose peaks stand above
                            the weeks around it pays for the whole of what
                            they do to the contract, as the household would.
            what it SHARES  its time-share of the standing charge the rest of
                            the pool sets.

        The second half is not optional, and leaving it out was measured: on
        the first half alone the whole-period MILP scored no better than the
        peak-shaving rule and self-consumption looked as good as either,
        because a profile shaved to one flat line adds nothing week by week
        whatever the line is. The LEVEL of the contract has to be in the
        measure or a controller that lowers it is not seen to.

        Weeks the agent never trained on, inside a trajectory it produced end
        to end, under the contract its own peaks set: the nearest a held-out
        measure can come to the scored bill while the contract is something
        only whole billing periods decide.
        """
        sig, cal = self.sig, self.cal
        spd = int(round(24.0 / sig.hours))
        cum, soc, sp = out["_cost_trace"], out["_soc_trace"], out["_setpoints"]
        kw = out["_import_kw"]
        bill = closing = contract = discharged = 0.0
        for a, b in self.val_blocks:
            r0, r1 = a * spd, min(b * spd, sig.n_steps)
            bill += cum[r1 - 1] - (cum[r0 - 1] if r0 else 0.0)
            soc_end = soc[r1] if r1 < sig.n_steps else out["Final_SOC_kWh"]
            closing += ((soc[r0] - soc_end) / sig.eta_ch
                        * float(np.mean(sig.import_rate[r0:r1])))
            discharged += float(np.maximum(-sp[r0:r1], 0.0).sum())
            # A one-month pool is touched in every month the week overlaps; a
            # twelve-month pool holds the whole week once, in its last month.
            months = (sorted(set(cal.month[r0:r1].tolist())) if cal.window == 1
                      else [int(cal.month[r1 - 1])])
            for m in months:
                without = cal.pool_cost(kw, m, exclude=(r0, r1))
                contract += (cal.pool_cost(kw, m) - without
                             + cal.share(r0, r1, m) * without)
        return {"cost_eur": bill, "contract_eur": contract,
                "cost_eur_closed": bill + closing + contract,
                "efc": discharged / self._nominal if self._nominal > 0 else 0.0}

    def register(self, out) -> None:
        """Re-agree the contract from a pass's own draw, and re-read it."""
        self.profile = out["_import_kw"]
        self.env.set_achieved_power_kw(self.profile)
        self.sig = rbc.rebind_agreed_power(self.sig, self.env)
        self._load_static()

    def validate(self, net):
        out = self.full_pass(net)
        val = self.held_out(out)
        self.last_pass = {k: float(out[k]) for k in
                          ("Cost_EUR", "Fixed_EUR", "Energy_EUR", "Power_EUR")}
        if self.options.self_consistent:
            self.register(out)
        return val, self.sig, self.static_norm

    def settle_contract(self, net, passes: int = 3) -> dict:
        """Run `net` until the contract in force is (nearly) its own.

        What `run_policy` does at scoring time, kept open-ended on purpose:
        `passes` passes, each billed under the contract the one before set.
        Used to start a fine-tune from the clone's own contract rather than
        the no-battery one, and for the final validation number.
        """
        val = None
        for _ in range(max(int(passes), 1)):
            out = self.full_pass(net)
            val = self.held_out(out)
            self.register(out)
        return val


class SIMonthlyContractReward(SIContractChargeReward):
    """The study's SI contract: a draw is billed again for the month after.

    Priced by "episode": the agent dispatches whole months, so the level of a
    month's contract is its own doing, and an episode is billed as the month
    it is a part of.
    """

    name = "si_monthly_contract"
    window = 1
    default_pricing = "episode"


class SIAnnualContractReward(SIContractChargeReward):
    """The rolling twelve-month contract: a draw is billed again for a year.

    The pool is twelve months deep, so the threshold a draw has to clear is the
    fifth highest of a YEAR -- far fewer intervals matter than under the
    monthly contract -- and each one that does is priced at twelve months of
    standing charge instead of one.

    Priced by "ratchet", seeded from the agent's own trailing year: against a
    pool that long an episode's peaks either clear a line the year has already
    set or they do not, and below it a draw really is free.
    """

    name = "si_annual_contract"
    window = 12
    default_pricing = "ratchet"


class SIActivePriceReward(SIMonthlyContractReward):
    """The active (four-tariff) price list, agreed power re-agreed monthly.

    The bill is the same three lines as the study's contract and the reward
    the same sum -- energy, excess power, contract charge -- but what the
    ENERGY line pays for is different in kind: four posted prices by time of
    day and season instead of an hourly market price, a spread of 15 cents
    that is the same every working day. On this list the energy line is most
    of what a battery can win, and the contract charge the smaller part.
    """

    name = "si_active_price"


class SISingleTariffReward(SIMonthlyContractReward):
    """The single-tariff price list, agreed power re-agreed monthly.

    One import price and one export price: the energy line pays for keeping
    the household's own PV and for nothing else, so no action that buys from
    the grid can earn anything on it. What is left to decide is the contract
    -- the charge this reward was written for is here the ONLY thing that
    makes grid charging or holding a reserve worth a cent.
    """

    name = "si_single_tariff"


# The reward each bill is trained with. One entry per (tariff, contract type);
# AU has no contract, so its key carries None. The settled part of every SI
# reward is the arm's own `settle`, which bills the price list the environment
# is on -- so the three price lists differ in what `settle` charges, and the
# three agreed-power rules in what is added to it.
REWARDS = {
    ("AU", None): AUEnergyReward,
    ("SI", "fixed"): SIExcessPowerReward,
    ("SI", "monthly"): SIMonthlyContractReward,
    ("SI", "annual"): SIAnnualContractReward,
    ("SI", "active"): SIActivePriceReward,
    ("SI", "single"): SISingleTariffReward,
}


def reward_class(tariff: str, contract: str | None = None, blind: bool = False):
    """The reward function for one bill.

    `blind=True` asks for the contract-blind reward on an SI contract that the
    dispatch does move -- the reward the panel was trained with, as a control.
    """
    if tariff == "AU":
        return AUEnergyReward
    if blind:
        return SIExcessPowerReward
    return REWARDS[(tariff, contract or REFERENCE_CONTRACT)]
