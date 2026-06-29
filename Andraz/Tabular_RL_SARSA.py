import random

import numpy as np
import pandas as pd

from Basic_Functions import (
    Action,
    BatMaxPolTrenutno,
    BatMaxPraTrenutno,
    BaterijaSprememba,
    calculate_interval_price,
    PaneliOdvec,
    State,
)


class LinearFunctionSarsaAgent:
    """Linear SARSA agent that trains directly on the shared gym environment."""

    def __init__(self, environment, discount_factor=0.8, _lambda=0.8, steviloLastnosti=20):
        self._validate_environment(environment)

        self.default_env = environment
        self.env = environment
        self.epsilon = 0.05
        self.steviloLastnosti = int(steviloLastnosti)
        self.disc_factor = float(discount_factor)
        self._lambda = float(_lambda)
        self.alfa = 0.01
        self.actions_count = int(environment.action_space.n)
        self.number_of_parameters = self.steviloLastnosti**3 * self.actions_count

        self.theta = np.zeros(self.number_of_parameters, dtype=float)
        self.E = self.get_clear_tensor().flatten()

        self._build_feature_ranges(environment)
        self._reset_tracking()

    def _validate_environment(self, environment):
        required_attributes = [
            "action_space",
            "arr_SMP",
            "arr_Gen",
            "arr_Con",
            "bat_kapaciteta",
            "bat_max_polnjenje",
            "bat_max_praznjenje",
            "reset",
            "step",
            "dataset",
        ]
        missing = [name for name in required_attributes if not hasattr(environment, name)]
        if missing:
            raise TypeError(
                "environment must be compatible with HouseholdEnvironment; missing: "
                + ", ".join(missing)
            )

    def _build_feature_ranges(self, environment):
        bat_kapaciteta = float(environment.bat_kapaciteta)
        ostala_poraba_max = float(np.max(environment.arr_Con) + environment.bat_max_polnjenje)
        ostala_poraba_min = float(-(np.max(environment.arr_Gen) + environment.bat_max_praznjenje))
        cena_el_max = float(np.max(environment.arr_SMP))
        cena_el_min = float(np.min(environment.arr_SMP))

        self.baterija_edges = self._make_edges(0.0, bat_kapaciteta)
        self.ostala_poraba_edges = self._make_edges(ostala_poraba_min, ostala_poraba_max)
        self.cena_edges = self._make_edges(cena_el_min, cena_el_max)

    def _make_edges(self, minimum, maximum):
        if not np.isfinite(minimum):
            minimum = 0.0
        if not np.isfinite(maximum):
            maximum = minimum
        if maximum <= minimum:
            maximum = minimum + 1e-9
        return np.linspace(minimum, maximum, self.steviloLastnosti + 1, dtype=float)

    def _bucket_index(self, value, edges):
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        return int(np.clip(idx, 0, self.steviloLastnosti - 1))

    def _action_to_int(self, action):
        if isinstance(action, Action):
            action = action.value
        elif hasattr(action, "value"):
            action = action.value
        action_int = int(action)
        if action_int < 0 or action_int >= self.actions_count:
            raise ValueError(f"Unsupported action: {action}")
        return action_int

    def _reset_tracking(self):
        self.Price = []
        self.Date = []
        self.Nagrada = []
        self.NapolnjenostBaterije = []
        self.Cena = []
        self.NagradaSkupno = [0]
        self.NagradaKapaciteta = [0]
        self.NagradaSprememba = [0]
        self.NagradaPlacilo = [0]

    def _extract_features(self, info):
        return (
            float(info.get("battery", 0.0)),
            float(info.get("generation", 0.0) - info.get("consumption", 0.0)),
            float(info.get("price", 0.0)),
        )

    def _date_from_step(self, environment, step_idx):
        if not hasattr(environment.dataset, "index") or len(environment.dataset.index) == 0:
            return pd.NaT
        safe_idx = int(np.clip(step_idx, 0, len(environment.dataset.index) - 1))
        return environment.dataset.index[safe_idx]

    def _calculate_interval_price(self, environment, current_info, next_info):
        step_idx = int(current_info.get("step_idx", 0))
        energy_flows = next_info.get("energy_flows", {})
        total_consumed_kwh = float(
            energy_flows.get(
                "kupljena_elektrika",
                float(next_info.get("consumption", 0.0)) - float(next_info.get("generation", 0.0)),
            )
        )

        return calculate_interval_price(
            float(current_info.get("price", 0.0)),
            total_consumed_kwh,
            self._date_from_step(environment, step_idx),
            interval_minutes=1440.0 / environment.korakov_na_dan,
        )

    def get_clear_tensor(self):
        return np.zeros(
            [self.steviloLastnosti, self.steviloLastnosti, self.steviloLastnosti, self.actions_count],
            dtype=float,
        )

    def get_q(self, info, action):
        return float(np.dot(self.phi(info, action), self.theta))

    def phi(self, info, action):
        baterija, ostala_poraba, cena_el = self._extract_features(info)
        action_int = self._action_to_int(action)

        i_idx = self._bucket_index(baterija, self.baterija_edges)
        j_idx = self._bucket_index(ostala_poraba, self.ostala_poraba_edges)
        k_idx = self._bucket_index(cena_el, self.cena_edges)

        features = np.zeros(self.number_of_parameters, dtype=float)
        flat_index = np.ravel_multi_index(
            (i_idx, j_idx, k_idx, action_int),
            (self.steviloLastnosti, self.steviloLastnosti, self.steviloLastnosti, self.actions_count),
        )
        features[flat_index] = 1.0
        return features

    def get_alpha(self, _info, _action):
        return self.alfa

    def choose_random_action(self):
        return random.randrange(self.actions_count)

    def try_all_actions(self, info):
        return [self.get_q(info, action) for action in range(self.actions_count)]

    def choose_best_action(self, info):
        return int(np.argmax(self.try_all_actions(info)))

    def chose_alternating_actions(self, akcija):
        return int(akcija % self.actions_count)

    def policy(self, info):
        if random.random() <= self.epsilon:
            return self.choose_random_action()
        return self.choose_best_action(info)

    def get_value_function(self, info, action):
        return self.get_q(info, action)

    def set_discount_factor(self, discount_factor):
        self.disc_factor = float(discount_factor)

    def set_alfa(self, alfa):
        self.alfa = float(alfa)

    def train(
        self,
        ponovitev=1,
        theta=None,
        epsilon=0.05,
        epsilon_decay=0,
        epsilon_min=0,
        leto=1,
        alternating_actions=False,
        *,
        env=None,
        reset_options=None,
    ):
        if env is None and hasattr(self.default_env, "get_env"):
            active_env = self.default_env.get_env(leto)
        else:
            active_env = self.default_env if env is None else env
        self._validate_environment(active_env)

        if int(active_env.action_space.n) != self.actions_count:
            raise ValueError("Environment action space does not match agent configuration.")

        self._reset_tracking()
        self.epsilon = float(epsilon)

        if theta is None:
            self.theta = np.zeros(self.number_of_parameters, dtype=float)
        else:
            self.theta = np.array(theta, dtype=float, copy=True)

        for episode in range(int(ponovitev)):
            self.E = self.get_clear_tensor().flatten()
            _, info = active_env.reset(options=reset_options)
            current_info = info
            done = False

            if alternating_actions:
                action = self.chose_alternating_actions(current_info.get("step_idx", 0))
            else:
                action = self.policy(current_info)

            while not done:
                _, reward, terminated, truncated, next_info = active_env.step(action)
                done = bool(terminated or truncated)

                phi = self.phi(current_info, action)
                q = self.get_q(current_info, action)

                if not done:
                    if alternating_actions:
                        next_action = self.chose_alternating_actions(next_info.get("step_idx", 0))
                    else:
                        next_action = self.policy(next_info)
                    delta = (reward + self.disc_factor * self.get_q(next_info, next_action)) - q
                else:
                    next_action = None
                    delta = reward - q

                self.E *= self.disc_factor * self._lambda
                self.E += phi
                self.theta += self.get_alpha(current_info, action) * delta * self.E

                reward_components = next_info.get("reward_components", {})
                self.Price.append(float(next_info.get("cumulative_payment", 0.0)))
                self.Date.append(self._date_from_step(active_env, next_info.get("step_idx", 0)))
                self.Nagrada.append(float(delta))
                self.NapolnjenostBaterije.append(float(next_info.get("battery", 0.0)))
                interval_price = self._calculate_interval_price(active_env, current_info, next_info)
                self.Cena.append(
                    float(interval_price["constant_price_aud"]) + float(interval_price["variable_price_aud"])
                )
                self.NagradaSkupno.append(self.NagradaSkupno[-1] + float(reward))
                self.NagradaKapaciteta.append(
                    self.NagradaKapaciteta[-1] + float(reward_components.get("r_kapaciteta", 0.0))
                )
                self.NagradaSprememba.append(
                    self.NagradaSprememba[-1] + float(reward_components.get("r_sprememba", 0.0))
                )
                self.NagradaPlacilo.append(
                    self.NagradaPlacilo[-1] + float(reward_components.get("r_placilo", 0.0))
                )

                current_info = next_info
                if next_action is not None:
                    action = next_action

            if episode % 4 == 0 and episode != 0 and self.Price:
                print("Episode: %d, score: %f" % (episode, float(self.Price[-1])))

            self.epsilon = max(float(epsilon_min), self.epsilon - float(epsilon_decay))

        return self.theta


def QL_trening_krajsi(agent_train, *, env=None):
    """Shorter training schedule."""
    theta_trening = agent_train.train(1, None, 1.0, 0.01, 0.01, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.6, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.2, env=env)
    theta_trening = agent_train.train(1, theta_trening, 1.0, 0.02, 0.01, env=env)
    return theta_trening, agent_train


def QL_trening_daljsi(agent_train, *, env=None):
    """Longer training schedule."""
    theta_trening = agent_train.train(1, None, 1.0, 0.01, 0.01, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.8, 0.02, 0.01, env=env, alternating_actions=True)
    theta_trening = agent_train.train(1, theta_trening, 0.8, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.6, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.4, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.2, env=env)
    theta_trening = agent_train.train(1, theta_trening, 0.05, env=env)
    theta_trening = agent_train.train(1, theta_trening, 1.0, 0.02, 0.01, env=env)
    return theta_trening, agent_train