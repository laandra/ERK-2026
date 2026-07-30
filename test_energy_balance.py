"""
test_energy_balance.py
──────────────────────
Energy balance diagnostics for ContinuousHouseholdWrapper discharge path.

Three invariants checked per step
  1. Bus balance  – electrical node: gen + grid_in + bat_delivered == con + grid_out + bat_charged
  2. Battery conservation – actual Δbat matches computed sprememba_baterije
  3. Total system energy  – gen + grid_bought + bat_draw == con + grid_sold + bat_stored + heat

Phantom energy leak hypothesis:
  When the battery discharges P_dis kWh, only P_dis*eta kWh reaches the
  electrical bus (the rest is heat). If the grid accounting incorrectly
  uses P_dis instead of P_dis*eta, P_dis*(1-eta) kWh disappears without
  being credited anywhere — a phantom leak.

Run: python test_energy_balance.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

TOL = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Minimal stub of HouseholdEnvironment (must be a real gym.Env subclass)
# ──────────────────────────────────────────────────────────────────────────────

import gymnasium as gym


class _StubEnv(gym.Env):
    """
    Minimal stand-in for HouseholdEnvironment that satisfies
    gymnasium.Wrapper.__init__'s isinstance check while exposing only
    the attributes accessed by ContinuousHouseholdWrapper.step.
    """

    def __init__(self, bat, gen, con, eta, max_dis, max_ch, cap, price):
        super().__init__()
        from Environment import _StateDQN

        self._StateDQN = _StateDQN
        self._bat0 = bat          # initial battery (reset on each scenario)
        self._gen  = gen
        self._con  = con

        # Hardware parameters
        self.bat_ucinkovitost    = eta
        self.bat_max_polnjenje   = max_ch
        self.bat_max_praznjenje  = max_dis
        self.bat_kapaciteta      = cap
        self._price              = price

        # Mutable env state
        self._battery            = bat
        self._current_step       = 0
        self._cumulative_payment = 0.0
        self._episode_steps      = 0
        self.data_length         = 50
        self._episode_end_exclusive = 50

        # Night-time UTC → off-peak ToU tariff
        self._idx = pd.date_range("2024-06-15 02:00", periods=50, freq="15min", tz="UTC")
        self.dataset             = type("DS", (), {"index": self._idx})()
        self.arr_MedianPrice     = np.full(50, price)
        self.korakov_na_dan      = 96

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)   # overridden by wrapper

    def _get_state_object(self, step_i, battery, payment):
        cap = self.bat_kapaciteta
        return self._StateDQN(
            Korak=step_i,
            CenaEl=self._price,
            Baterija=battery,
            Baterija_norm=battery / cap,
            Generiranje=self._gen,
            Generiranje_norm=self._gen / 5.0,
            Poraba=self._con,
            Poraba_norm=self._con / 5.0,
            Placilo=payment,
            CenaEl_norm=0.5,
            CenaElRel=1.0,
        )

    def _nagrada_skupno(self, s, delta, placilo, cena_med): return 0.0
    def _nagrada_1(self, s):                                return 0.0
    def _nagrada_2(self, s, delta, cena_med):               return 0.0
    def _nagrada_3(self, placilo):                          return 0.0

    def _build_observation(self, step_i, bat_norm):
        return np.zeros(10, dtype=np.float32)

    def _build_info(self, s, action_int=None, energy_flows=None, reward_components=None):
        d = {"battery": s.Baterija, "cumulative_payment": s.Placilo}
        if energy_flows:
            d.update(energy_flows)
        if reward_components:
            d.update(reward_components)
        return d

    def reset(self, **kwargs):
        self._battery = self._bat0
        self._current_step = 0
        self._cumulative_payment = 0.0
        self._episode_steps = 0
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        raise NotImplementedError("Use ContinuousHouseholdWrapper.step")

    def render(self): pass


def _make_env(bat, gen, con, eta=0.95, max_dis=1.5, max_ch=1.5, cap=10.0, price=50.0):
    return _StubEnv(bat=bat, gen=gen, con=con, eta=eta,
                    max_dis=max_dis, max_ch=max_ch, cap=cap, price=price)


# ──────────────────────────────────────────────────────────────────────────────
# Core checker
# ──────────────────────────────────────────────────────────────────────────────

def check_step(label, bat, gen, con, action, eta=0.95, max_dis=1.5, max_ch=1.5, cap=10.0):
    """
    Run one wrapper step with known inputs and verify three energy balance invariants.
    Returns True on pass, False on any failure.
    """
    from SAC_Agent import ContinuousHouseholdWrapper

    mock = _make_env(bat, gen, con, eta=eta, max_dis=max_dis, max_ch=max_ch, cap=cap)
    wrapper = ContinuousHouseholdWrapper(mock)
    _, _, _, _, info = wrapper.step(np.array([action], dtype=np.float32))

    bat_new       = mock._battery
    solar_to_bat  = info["paneli_baterija"]
    grid_to_bat   = info["omrezje_baterija"]
    home_from_bat = info["baterija_dom"]
    grid_from_bat = info["baterija_omrezje"]
    kupljena      = info["kupljena_elektrika"]
    P_dis         = info["P_dis_kWh"]
    P_ch          = info["P_ch_kWh"]

    errors = []

    # ── 1. Electrical bus balance ──────────────────────────────────────────────
    # (charging taken from bus) + con - gen - grid_net - (delivery to bus) == 0
    bus_residual = (
        (solar_to_bat + grid_to_bat) + con - gen - kupljena
        - (home_from_bat + grid_from_bat)
    )
    if abs(bus_residual) > TOL:
        errors.append(f"BUS BALANCE RESIDUAL={bus_residual:.3e}  (should be 0)")

    # ── 2. Battery conservation ────────────────────────────────────────────────
    expected_delta = eta * (solar_to_bat + grid_to_bat) - (1.0/eta) * (home_from_bat + grid_from_bat)
    actual_delta   = bat_new - bat
    if abs(actual_delta - expected_delta) > TOL:
        errors.append(
            f"BATTERY DELTA mismatch: actual={actual_delta:.6f}  expected={expected_delta:.6f}"
        )

    # ── 3. Discharge-specific checks ───────────────────────────────────────────
    if P_dis > TOL:
        delivered        = home_from_bat + grid_from_bat
        expected_deliver = P_dis * eta
        if abs(delivered - expected_deliver) > TOL:
            errors.append(
                f"DISCHARGE DELIVERY: delivered={delivered:.6f}  P_dis*eta={expected_deliver:.6f}"
                f"  phantom leak={P_dis - delivered:.6f} kWh"
            )
        bat_lost = bat - bat_new
        if abs(bat_lost - P_dis) > TOL:
            errors.append(
                f"BATTERY DRAW: bat_lost={bat_lost:.6f}  P_dis={P_dis:.6f}"
            )

    # ── 4. Total system energy (includes heat losses) ──────────────────────────
    grid_bought = max(0.0, kupljena)
    grid_sold   = max(0.0, -kupljena)
    bat_draw    = max(0.0, -(bat_new - bat))   # kWh leaving battery raw
    bat_stored  = max(0.0,   bat_new - bat)    # kWh added to battery raw
    heat_ch     = (solar_to_bat + grid_to_bat) * (1.0 - eta)
    heat_dis    = bat_draw * (1.0 - eta)

    sys_lhs = gen + grid_bought + bat_draw
    sys_rhs = con + grid_sold + bat_stored + heat_ch + heat_dis
    if abs(sys_lhs - sys_rhs) > TOL:
        errors.append(
            f"SYSTEM ENERGY: LHS={sys_lhs:.6f}  RHS={sys_rhs:.6f}  diff={sys_lhs-sys_rhs:.3e}"
        )

    status = "PASS" if not errors else "FAIL"
    print(f"[{status}] {label}")
    for msg in errors:
        print(f"       !! {msg}")
    if not errors:
        delivered = home_from_bat + grid_from_bat
        heat = P_dis * (1 - eta) if P_dis > 0 else 0.0
        print(
            f"       bat {bat:.3f}→{bat_new:.3f}  P_dis={P_dis:.3f}  "
            f"P_ch={P_ch:.3f}  delivered={delivered:.3f}  "
            f"heat={heat:.4f}  grid_net={kupljena:.4f}"
        )
    return len(errors) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    print("=" * 65)
    print("Discharge scenarios")
    print("=" * 65)

    results.append(check_step(
        "Full discharge, home deficit (bat covers partial)",
        bat=5.0, gen=0.0, con=2.0, action=-1.0,
    ))
    results.append(check_step(
        "Full discharge, small load → surplus exported to grid",
        bat=5.0, gen=0.0, con=0.5, action=-1.0,
    ))
    results.append(check_step(
        "Full discharge, solar already covers home → all to grid",
        bat=5.0, gen=2.0, con=1.0, action=-1.0,
    ))
    results.append(check_step(
        "Partial discharge (action=-0.5)",
        bat=5.0, gen=0.0, con=2.0, action=-0.5,
    ))
    results.append(check_step(
        "Discharge clipped by SOC (nearly empty battery)",
        bat=0.3, gen=0.0, con=2.0, action=-1.0,
    ))
    results.append(check_step(
        "Discharge with eta=1.0 (lossless) — heat should be 0",
        bat=5.0, gen=0.0, con=2.0, action=-1.0, eta=1.0,
    ))
    results.append(check_step(
        "Discharge with eta=0.80 (large losses)",
        bat=5.0, gen=0.0, con=2.0, action=-1.0, eta=0.80,
    ))
    results.append(check_step(
        "Discharge with eta=0.99 (near-lossless)",
        bat=5.0, gen=0.0, con=2.0, action=-1.0, eta=0.99,
    ))

    print()
    print("=" * 65)
    print("Charge scenarios")
    print("=" * 65)

    results.append(check_step(
        "Full charge from grid (no solar)",
        bat=2.0, gen=0.0, con=1.0, action=1.0,
    ))
    results.append(check_step(
        "Full charge, solar surplus fills battery",
        bat=2.0, gen=3.0, con=1.0, action=1.0,
    ))
    results.append(check_step(
        "Charge clipped by capacity (near-full battery)",
        bat=9.9, gen=0.0, con=0.0, action=1.0,
    ))

    print()
    print("=" * 65)
    print("Idle / edge scenarios")
    print("=" * 65)

    results.append(check_step(
        "Idle (action=0) — no battery movement",
        bat=5.0, gen=1.0, con=2.0, action=0.0,
    ))
    results.append(check_step(
        "Zero generation and consumption, discharge",
        bat=5.0, gen=0.0, con=0.0, action=-1.0,
    ))

    print()
    passed = sum(results)
    total  = len(results)
    print("=" * 65)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("PHANTOM ENERGY LEAK DETECTED — see FAIL entries above.")
        sys.exit(1)
    else:
        print("All energy balance checks passed — no phantom leak found.")
