from typing import Optional

import gymnasium as gym
import numpy as np

from Basic_Functions import (
    BatMaxPolTrenutno,
    BatMaxPraTrenutno,
    BaterijaSprememba,
    PaneliOdvec,
    calculate_interval_price,
)


class _StateDQN:
    """Internal state container mirroring notebook fields used by reward/step logic."""

    __slots__ = [
        "Korak",
        "CenaEl",
        "Baterija",
        "Generiranje",
        "Poraba",
        "Mesec",
        "DanVTednu",
        "Ura",
        "Minuta",
        "CenaEl_norm",
        "Generiranje_norm",
        "Baterija_norm",
        "Poraba_norm",
        "Placilo",
        "CenaElRel",
    ]

    def __init__(
        self,
        Korak=0,
        CenaEl=0.0,
        Baterija=0.0,
        Generiranje=0.0,
        Poraba=0.0,
        Mesec=1,
        DanVTednu=0,
        Ura=0,
        Minuta=0,
        CenaEl_norm=0.0,
        Generiranje_norm=0.0,
        Baterija_norm=0.0,
        Poraba_norm=0.0,
        Placilo=0.0,
        CenaElRel=0.0,
    ):
        self.Korak = int(Korak)
        self.CenaEl = float(CenaEl)
        self.Baterija = float(Baterija)
        self.Generiranje = float(Generiranje)
        self.Poraba = float(Poraba)
        self.Mesec = int(Mesec)
        self.DanVTednu = int(DanVTednu)
        self.Ura = int(Ura)
        self.Minuta = int(Minuta)
        self.CenaEl_norm = float(CenaEl_norm)
        self.Generiranje_norm = float(Generiranje_norm)
        self.Baterija_norm = float(Baterija_norm)
        self.Poraba_norm = float(Poraba_norm)
        self.Placilo = float(Placilo)
        self.CenaElRel = float(CenaElRel)


class HouseholdEnvironment(gym.Env):
    """Custom Gymnasium environment based on the notebook DQN household model."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset,
        dataset_norm=None,
        observation_mode="sliding_window",
        reset_mode="deterministic",
        episode_length=None,
        korakov_na_dan=96,
        bat_kapaciteta=20.0,
        bat_ucinkovitost=0.95,
        bat_max_polnjenje=1.5,
        bat_max_praznjenje=1.5,
        faktor_n1=0.0,
        faktor_n2=1.0,
        faktor_n3=1.0,
        median_window_days=30,
    ):
        self.dataset = dataset
        self.dataset_norm = dataset_norm if dataset_norm is not None else dataset

        required_cols = ["SMP", "Energy_Generation", "Energy_Consumption"]
        for col in required_cols:
            if col not in self.dataset.columns:
                raise ValueError(f"Missing required column in dataset: {col}")
            if col not in self.dataset_norm.columns:
                raise ValueError(f"Missing required column in dataset_norm: {col}")

        self.observation_mode = str(observation_mode)
        if self.observation_mode not in {"sliding_window", "compact"}:
            raise ValueError("observation_mode must be 'sliding_window' or 'compact'")

        self.reset_mode = str(reset_mode)
        if self.reset_mode not in {"deterministic", "random", "sequential"}:
            raise ValueError("reset_mode must be 'deterministic', 'random', or 'sequential'")

        self.korakov_na_dan = int(korakov_na_dan)
        if self.korakov_na_dan <= 0:
            raise ValueError("korakov_na_dan must be > 0")

        self.bat_kapaciteta = float(bat_kapaciteta)
        self.bat_ucinkovitost = float(bat_ucinkovitost)
        self.bat_max_polnjenje = float(bat_max_polnjenje)
        self.bat_max_praznjenje = float(bat_max_praznjenje)
        self.faktor_n1 = float(faktor_n1)
        self.faktor_n2 = float(faktor_n2)
        self.faktor_n3 = float(faktor_n3)

        self.data_length = len(self.dataset)
        if self.data_length == 0:
            raise ValueError("Dataset is empty. Please provide a valid dataset.")

        self.arr_SMP = self.dataset["SMP"].to_numpy(dtype=np.float64)
        self.arr_Gen = self.dataset["Energy_Generation"].to_numpy(dtype=np.float64)
        self.arr_Con = self.dataset["Energy_Consumption"].to_numpy(dtype=np.float64)

        self.arr_SMP_norm = self.dataset_norm["SMP"].to_numpy(dtype=np.float64)
        self.arr_Gen_norm = self.dataset_norm["Energy_Generation"].to_numpy(dtype=np.float64)
        self.arr_Con_norm = self.dataset_norm["Energy_Consumption"].to_numpy(dtype=np.float64)

        if hasattr(self.dataset.index, "month"):
            self.arr_Month = self.dataset.index.month.to_numpy()
            self.arr_DayOfWeek = self.dataset.index.dayofweek.to_numpy()
            self.arr_Hour = self.dataset.index.hour.to_numpy()
            self.arr_Minute = self.dataset.index.minute.to_numpy()
        else:
            self.arr_Month = np.zeros(self.data_length, dtype=np.int32)
            self.arr_DayOfWeek = np.zeros(self.data_length, dtype=np.int32)
            self.arr_Hour = np.zeros(self.data_length, dtype=np.int32)
            self.arr_Minute = np.zeros(self.data_length, dtype=np.int32)

        window_size = max(1, int(median_window_days * self.korakov_na_dan))
        self.arr_MedianPrice = (
            self.dataset["SMP"].rolling(window=window_size, min_periods=1).median().to_numpy(dtype=np.float64)
        )
        epsilon = 1e-8
        self.arr_RelativePrice = (
            (self.arr_SMP - self.arr_MedianPrice) / (self.arr_MedianPrice + epsilon)
        )

        self.episode_length = (
            int(episode_length)
            if episode_length is not None
            else int(self.data_length - 1)
        )
        self.episode_length = max(1, self.episode_length)

        self._sequential_counter = 0
        self._current_step = 0
        self._episode_steps = 0
        self._episode_start = 0
        self._episode_end_exclusive = min(self.data_length, self._episode_start + self.episode_length)
        self._battery = max(0.0, self.bat_kapaciteta / 2.0)
        self._cumulative_payment = 0.0

        self.window_past = self.korakov_na_dan * 0
        self.window_future = 11 * (self.korakov_na_dan // 24)

        self.action_space = gym.spaces.Discrete(5)
        state_dim = self._state_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

    def _state_dim(self):
        if self.observation_mode == "compact":
            return 6
        return (
            1 + 2
            + self.window_past
            + self.window_past
            + (self.window_past + self.window_future)
        )

    def _get_state_object(self, idx, baterija, placilo):
        baterija = float(np.clip(baterija, 0.0, self.bat_kapaciteta))
        return _StateDQN(
            Korak=idx,
            CenaEl=self.arr_SMP[idx],
            Baterija=baterija,
            Generiranje=self.arr_Gen[idx],
            Poraba=self.arr_Con[idx],
            Mesec=self.arr_Month[idx],
            DanVTednu=self.arr_DayOfWeek[idx],
            Ura=self.arr_Hour[idx],
            Minuta=self.arr_Minute[idx],
            CenaEl_norm=self.arr_SMP_norm[idx],
            CenaElRel=self.arr_RelativePrice[idx],
            Generiranje_norm=self.arr_Gen_norm[idx],
            Poraba_norm=self.arr_Con_norm[idx],
            Baterija_norm=(baterija / self.bat_kapaciteta) if self.bat_kapaciteta > 0 else 0.0,
            Placilo=placilo,
        )

    def _action_to_int(self, action):
        if hasattr(action, "value"):
            action = action.value
        a = int(action)
        if a < 0 or a >= self.action_space.n:
            raise ValueError(f"Unsupported action: {action}")
        return a

    def action_masks(self):
        """Return a boolean action mask for MaskablePPO.

        The original environment can technically execute every action, but some are
        effectively degenerate when the battery is full/empty or when no excess solar
        is available. Exposing these simple feasibility constraints gives
        MaskablePPO a valid mask interface without changing the base transition logic.
        """
        s = self._get_state_object(self._current_step, self._battery, self._cumulative_payment)
        excess_solar = PaneliOdvec(s.Generiranje, s.Poraba)
        can_charge = s.Baterija < (self.bat_kapaciteta - 1e-8)
        can_discharge = s.Baterija > 1e-8

        return np.array(
            [
                can_charge,
                can_charge and (excess_solar > 1e-8),
                can_discharge,
                can_discharge,
                True,
            ],
            dtype=bool,
        )

    def _build_observation(self, idx, baterija_norm):
        # Create cyclical time features
        hour_fraction = (self.arr_Hour[idx] + self.arr_Minute[idx]/60.0) / 24.0
        sin_time = np.sin(2 * np.pi * hour_fraction)
        cos_time = np.cos(2 * np.pi * hour_fraction)

        if self.observation_mode == "compact":
            return np.array(
                [
                    baterija_norm,
                    self.arr_Gen_norm[idx],
                    self.arr_Con_norm[idx],
                    self.arr_SMP_norm[idx],
                    sin_time,
                    cos_time
                ],
                dtype=np.float32,
            )

        start_past = max(0, idx - self.window_past + 1)
        pad_left = self.window_past - (idx - start_past + 1)

        gen_slice = self.arr_Gen_norm[start_past : idx + 1].astype(np.float32)
        gen_window = np.concatenate([np.zeros(pad_left, dtype=np.float32), gen_slice])

        con_slice = self.arr_Con_norm[start_past : idx + 1].astype(np.float32)
        con_window = np.concatenate([np.zeros(pad_left, dtype=np.float32), con_slice])

        end_price = min(self.data_length, idx + self.window_future + 1)
        price_slice = self.arr_SMP_norm[start_past:end_price].astype(np.float32)
        pad_right = (self.window_past + self.window_future) - pad_left - len(price_slice)
        price_window = np.concatenate(
            [
                np.zeros(pad_left, dtype=np.float32),
                price_slice,
                np.zeros(max(0, pad_right), dtype=np.float32),
            ]
        )

        return np.concatenate(
            [
                np.array([baterija_norm], dtype=np.float32),
                gen_window,
                con_window,
                price_window,
                np.array([sin_time, cos_time], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _nagrada_1(self, s):
        alfa = 0.1
        beta = 0.8
        soc = s.Baterija_norm
        if soc < alfa:
            return -5.0 * (alfa - soc) * self.faktor_n1
        if soc > beta:
            return -5.0 * (soc - beta) * self.faktor_n1
        return 0.0

    def _nagrada_2(self, s, sprememba_baterije, _cena_el_med):
        if self.bat_kapaciteta <= 0:
            return 0.0
        gamma2 = 3.0
        norm_change = sprememba_baterije / self.bat_kapaciteta
        return -5.0 * gamma2 * norm_change * s.CenaElRel * self.faktor_n2

    def _nagrada_3(self, placilo_zdaj):
        delta = 5.0 / 8.0
        return -delta * placilo_zdaj * self.faktor_n3

    def _nagrada_skupno(self, s, sprememba_baterije, placilo_zdaj, cena_el_med):
        if self.bat_kapaciteta <= 0:
            return self._nagrada_3(placilo_zdaj)
        denominator = self.faktor_n1 + self.faktor_n2 + self.faktor_n3
        if denominator <= 0:
            return self._nagrada_3(placilo_zdaj)
        return (
            self._nagrada_1(s)
            + self._nagrada_2(s, sprememba_baterije, cena_el_med)
            + self._nagrada_3(placilo_zdaj)
        ) / denominator

    def _resolve_start_index(self, options):
        mode = self.reset_mode
        if options is not None:
            mode = str(options.get("reset_mode", mode))

        max_valid_start = max(0, self.data_length - self.episode_length)
        if mode == "deterministic":
            return 0
        if mode == "random":
            return int(self.np_random.integers(0, max_valid_start + 1))
        if mode == "sequential":
            n = int(options.get("sequential_n", 10)) if options else 10
            n = max(1, n)
            idx = self._sequential_counter % n
            self._sequential_counter += 1
            return int(max_valid_start * idx / n)
        raise ValueError("reset_mode must be 'deterministic', 'random', or 'sequential'")

    def _build_info(self, s, action_int=None, energy_flows=None, reward_components=None):
        info = {
            "step_idx": int(s.Korak),
            "battery": float(s.Baterija),
            "battery_norm": float(s.Baterija_norm),
            "cumulative_payment": float(s.Placilo),
            "price": float(s.CenaEl),
            "price_norm": float(s.CenaEl_norm),
            "relative_price": float(s.CenaElRel),
            "generation": float(s.Generiranje),
            "consumption": float(s.Poraba),
        }
        if action_int is not None:
            info["action"] = int(action_int)
        if energy_flows is not None:
            info["energy_flows"] = energy_flows
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._episode_start = self._resolve_start_index(options)
        self._episode_end_exclusive = min(
            self.data_length,
            self._episode_start + self.episode_length,
        )
        if self._episode_end_exclusive <= self._episode_start:
            self._episode_end_exclusive = min(self.data_length, self._episode_start + 1)

        self._current_step = self._episode_start
        self._episode_steps = 0
        self._battery = max(0.0, self.bat_kapaciteta / 2.0)
        self._cumulative_payment = 0.0

        s0 = self._get_state_object(self._current_step, self._battery, self._cumulative_payment)
        obs = self._build_observation(s0.Korak, s0.Baterija_norm)
        return obs, self._build_info(s0)

    def step(self, action):
        action_int = self._action_to_int(action)

        s = self._get_state_object(self._current_step, self._battery, self._cumulative_payment)
        next_idx = int(s.Korak + 1)

        if next_idx >= self.data_length:
            obs = self._build_observation(s.Korak, s.Baterija_norm)
            return obs, 0.0, True, False, self._build_info(s, action_int=action_int)

        ostala_energija = PaneliOdvec(s.Generiranje, s.Poraba)
        paneli_baterija = 0.0
        omrezje_baterija = 0.0
        baterija_dom = 0.0
        baterija_omrezje = 0.0

        bat_max_pol_trenutno = BatMaxPolTrenutno(
            s,
            self.bat_ucinkovitost,
            self.bat_max_polnjenje,
            self.bat_kapaciteta,
        )
        bat_max_pra_trenutno = BatMaxPraTrenutno(
            s,
            self.bat_ucinkovitost,
            self.bat_max_praznjenje,
        )

        if action_int == 0:
            if ostala_energija >= bat_max_pol_trenutno:
                paneli_baterija = bat_max_pol_trenutno
            else:
                paneli_baterija = ostala_energija
                omrezje_baterija = bat_max_pol_trenutno - ostala_energija
            kupljena_elektrika = s.Poraba + bat_max_pol_trenutno - s.Generiranje

        elif action_int == 1:
            if ostala_energija > bat_max_pol_trenutno:
                paneli_baterija = bat_max_pol_trenutno
                kupljena_elektrika = s.Poraba + bat_max_pol_trenutno - s.Generiranje
            else:
                paneli_baterija = ostala_energija
                kupljena_elektrika = s.Poraba + ostala_energija - s.Generiranje

        elif action_int == 2:
            if (s.Poraba > s.Generiranje) and (bat_max_pra_trenutno > (s.Poraba - s.Generiranje)):
                baterija_dom = s.Poraba - s.Generiranje
                kupljena_elektrika = 0.0
            elif s.Poraba <= s.Generiranje:
                kupljena_elektrika = -ostala_energija
            else:
                baterija_dom = bat_max_pra_trenutno
                kupljena_elektrika = (s.Poraba - s.Generiranje) - bat_max_pra_trenutno

        elif action_int == 3:
            if (s.Poraba > s.Generiranje) and (bat_max_pra_trenutno > (s.Poraba - s.Generiranje)):
                baterija_dom = s.Poraba - s.Generiranje
                baterija_omrezje = bat_max_pra_trenutno - (s.Poraba - s.Generiranje)
                kupljena_elektrika = -baterija_omrezje
            elif s.Poraba <= s.Generiranje:
                baterija_omrezje = bat_max_pra_trenutno
                kupljena_elektrika = -ostala_energija - baterija_omrezje
            else:
                baterija_dom = bat_max_pra_trenutno
                kupljena_elektrika = (s.Poraba - s.Generiranje) - bat_max_pra_trenutno

        elif action_int == 4:
            kupljena_elektrika = s.Poraba - s.Generiranje

        sprememba_baterije = BaterijaSprememba(
            paneli_baterija,
            omrezje_baterija,
            baterija_dom,
            baterija_omrezje,
            self.bat_ucinkovitost,
        )
        
        if abs((paneli_baterija + omrezje_baterija ) + s.Poraba - s.Generiranje - kupljena_elektrika - (baterija_dom + baterija_omrezje)) > 1e-8:
            raise ValueError(
                f"Energy balance error: "
                f"(paneli_baterija + omrezje_baterija)={paneli_baterija + omrezje_baterija}, "
                f"Poraba={s.Poraba}, Generiranje={s.Generiranje}, "
                f"kupljena_elektrika={kupljena_elektrika}, "
                f"(baterija_dom + baterija_omrezje)={baterija_dom + baterija_omrezje}"
            )
        
        #if kupljena_elektrika > 0:
        #    placilo_zdaj = s.CenaEl * kupljena_elektrika
        #else:
        #    placilo_zdaj = s.CenaEl * kupljena_elektrika * self.faktor_cenitve
            
        _price_result = calculate_interval_price(
            s.CenaEl,
            kupljena_elektrika,
            utc_date = self.dataset.index[s.Korak],
            interval_minutes = 1440.0 / self.korakov_na_dan,
        )
        konstantno_placilo = float(_price_result["constant_price_aud"])
        placilo_zdaj = float(_price_result["variable_price_aud"])

        new_battery = float(np.clip(s.Baterija + sprememba_baterije, 0.0, self.bat_kapaciteta))
        if s.Baterija + sprememba_baterije < -1e-8 or s.Baterija + sprememba_baterije > self.bat_kapaciteta + 1e-8:
            raise ValueError(
                f"Battery state out of bounds: "
                f"current={s.Baterija}, change={sprememba_baterije}, "
                f"new={new_battery}, capacity={self.bat_kapaciteta}"
            )
        if abs(new_battery - s.Baterija - sprememba_baterije) > 1e-8:
            raise ValueError(
                f"Battery state mismatch: "
                f"current={s.Baterija}, change={sprememba_baterije}, "
                f"new={new_battery}"
            )
        new_payment = s.Placilo + placilo_zdaj + konstantno_placilo
        next_s = self._get_state_object(next_idx, new_battery, new_payment)

        cena_el_med = self.arr_MedianPrice[s.Korak]
        
        # Calculate illegal action penalty
        penalty = 0.0
        if action_int in [2, 3] and s.Baterija <= 1e-8:
            penalty = -0.5 # Tried to discharge an empty battery
        elif action_int in [0, 1] and s.Baterija >= (self.bat_kapaciteta - 1e-8) and ostala_energija > 0:
            penalty = -0.5 # Tried to charge a full battery
                    
        reward = self._nagrada_skupno(s, sprememba_baterije, placilo_zdaj, cena_el_med) + penalty
        reward = float(np.clip(reward, -10.0, 5.0))
        
        r_kapaciteta = self._nagrada_1(s)
        r_sprememba = self._nagrada_2(s, sprememba_baterije, cena_el_med) if self.bat_kapaciteta > 0 else 0.0
        r_placilo = self._nagrada_3(placilo_zdaj)

        self._current_step = next_s.Korak
        self._battery = next_s.Baterija
        self._cumulative_payment = next_s.Placilo
        self._episode_steps += 1

        terminated = self._current_step >= (self.data_length - 1)
        truncated = self._current_step >= (self._episode_end_exclusive - 1)
        obs = self._build_observation(next_s.Korak, next_s.Baterija_norm)

        info = self._build_info(
            next_s,
            action_int=action_int,
            energy_flows={
                "paneli_baterija": float(paneli_baterija),
                "omrezje_baterija": float(omrezje_baterija),
                "baterija_dom": float(baterija_dom),
                "baterija_omrezje": float(baterija_omrezje),
                "kupljena_elektrika": float(kupljena_elektrika),
                "sprememba_baterije": float(sprememba_baterije),
            },
            reward_components={
                "total": float(reward),
                "r_kapaciteta": float(r_kapaciteta),
                "r_sprememba": float(r_sprememba),
                "r_placilo": float(r_placilo),
                "placilo_zdaj": float(placilo_zdaj),
            },
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return None

    def close(self):
        return None
