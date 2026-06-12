"""
rail_env_theia.py — Mediul RL pentru statia Theia

Interfata: compatibila cu gymnasium (gym) API.
Poate fi folosit cu Stable-Baselines3 (PPO, DQN).

Design:
  - Un episod = un scenariu generat de ScenarioGenerator
  - Un episod = o singura decizie a agentului
  - Spatiu de actiuni: Discrete(7)
  - Spatiu de observatii: Box(42,) float32

Spatiu de actiuni: Discrete(7)
  0 = asteapta        (nu poate fi directionat acum)
  1 = L1              (incearca parcurs pe Linia 1)
  2 = LII             (incearca parcurs pe Linia II - directa)
  3 = L3              (incearca parcurs pe Linia 3)
  4 = L4              (incearca parcurs pe Linia 4)
  5 = tranzit_capat   (spre capatul opus X sau Y)
  6 = tranzit_semnal  (spre semnalul corespondent de pe aceeasi linie)

Spatiu de observatii: Box(42,) float32
  [0..11]  Starea macazurilor (6 macazuri x 2 atribute)
           pozitie: 0=plus, 1=minus
           stare:   0=operational, 1=blocat, 2=defect

  [12..37] Starea segmentelor (26 segmente x 1 atribut)
           0=liber, 1=rezervat, 2=ocupat, 3=indisponibil

  [38]     origine_cerere:       0..9 (index in ORIGINI, include semnale)
  [39]     destinatie_cerere:    0..9 (index in DESTINATII)
  [40]     ruta_activa_exista:   0 sau 1
  [41]     dest_activ_idx:       0..9 (valida doar daca [40]=1)

Reward:
  +10  decizie corecta: parcurs realizabil ales
  -10  decizie gresita: parcurs imposibil sau incompatibil ales
  +3   asteapta corect: nu exista nicio ruta valida
  -10  asteapta gresit: exista ruta dar agentul a ales sa astepte
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    GYM_VERSION = "gym"

from simulation.scenario_generator import (
    generate_scenario,
    sunt_compatibile,
    ScenarioState,
)

from graph.theia import get_rute_valide


MACAZURI_ORDINE = ["M1", "M3", "M5", "M2", "M8", "DJ46"]

SEGMENTE_ORDINE = [
    "SEG_X_M1",    "SEG_M1_M3",   "SEG_M3_M5",
    "SEG_M1_L1",   "SEG_M3_LII",  "SEG_M5_L3",   "SEG_M5_L4",
    "SEG_Y1_L1",   "SEG_YII_LII", "SEG_Y3_L3",   "SEG_Y4_L4",
    "SEG_L1",      "SEG_LII",     "SEG_L3",      "SEG_L4",
    "SEG_L1_X1",   "SEG_LII_XII", "SEG_L3_X3",   "SEG_L4_X4",
    "SEG_M8_L1",   "SEG_M8_LII",  "SEG_DJ46_L3", "SEG_DJ46_L4",
    "SEG_M2_M8",   "SEG_M2_DJ46", "SEG_Y_M2",
]

DESTINATII = [
    "X1", "XII", "X3", "X4",
    "Y1", "YII", "Y3", "Y4",
    "X", "Y",
]

DESTINATIE_IDX = {
    destinatie: idx
    for idx, destinatie in enumerate(DESTINATII)
}

ORIGINI = [
    "X", "Y",
    "Y1", "YII", "Y3", "Y4",
    "X1", "XII", "X3", "X4",
]

ORIGINE_IDX = {
    origine: idx
    for idx, origine in enumerate(ORIGINI)
}

POZITIE_ENC = {
    "plus": 0,
    "minus": 1,
}

STARE_ENC = {
    "operational": 0,
    "blocat": 1,
    "defect": 2,
}

SEG_ENC = {
    "liber": 0,
    "rezervat": 1,
    "ocupat": 2,
    "indisponibil": 3,
}

ACTIUNE_LINIE = {
    0: None,
    1: "L1",
    2: "LII",
    3: "L3",
    4: "L4",
    5: "tranzit_capat",
    6: "tranzit_semnal",
}

PARCURS_LINIE = {
    ("X", "X1"): "L1",
    ("X", "XII"): "LII",
    ("X", "X3"): "L3",
    ("X", "X4"): "L4",

    ("Y", "Y1"): "L1",
    ("Y", "YII"): "LII",
    ("Y", "Y3"): "L3",
    ("Y", "Y4"): "L4",

    ("X", "Y"): "tranzit_capat",
    ("Y", "X"): "tranzit_capat",

    ("Y1", "X"): "tranzit_capat",
    ("YII", "X"): "tranzit_capat",
    ("Y3", "X"): "tranzit_capat",
    ("Y4", "X"): "tranzit_capat",

    ("X1", "Y"): "tranzit_capat",
    ("XII", "Y"): "tranzit_capat",
    ("X3", "Y"): "tranzit_capat",
    ("X4", "Y"): "tranzit_capat",

    ("Y1", "X1"): "tranzit_semnal",
    ("YII", "XII"): "tranzit_semnal",
    ("Y3", "X3"): "tranzit_semnal",
    ("Y4", "X4"): "tranzit_semnal",

    ("X1", "Y1"): "tranzit_semnal",
    ("XII", "YII"): "tranzit_semnal",
    ("X3", "Y3"): "tranzit_semnal",
    ("X4", "Y4"): "tranzit_semnal",
}

OBS_DIM = len(MACAZURI_ORDINE) * 2 + len(SEGMENTE_ORDINE) + 4


class TheiaCFEnv(gym.Env):
    """
    Mediu RL pentru managementul conflictelor in statia Theia.

    Un episod:
      reset() -> genereaza scenariu -> returneaza observatie
      step(actiune) -> calculeaza reward -> terminated=True

    Compatibil cu Stable-Baselines3 (PPO, DQN).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        difficulty_level: int = 1,
        mode: str = "antrenare",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.difficulty_level = difficulty_level
        self.mode = mode
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=0.0,
            high=9.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self._scenariu: Optional[ScenarioState] = None
        self._step_count: int = 0

        self._stats = {
            "total_episoade": 0,
            "decizii_corecte": 0,
            "decizii_gresite": 0,
            "asteapta_corect": 0,
            "asteapta_gresit": 0,
        }

    def _encode_obs(self, scenariu: ScenarioState) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        G = scenariu.graf
        idx = 0

        # [0..11] Macazuri
        for macaz_id in MACAZURI_ORDINE:
            nod = G.nodes.get(macaz_id, {})
            obs[idx] = POZITIE_ENC.get(nod.get("pozitie", "plus"), 0)
            obs[idx + 1] = STARE_ENC.get(nod.get("stare", "operational"), 0)
            idx += 2

        # [12..37] Segmente
        for seg_id in SEGMENTE_ORDINE:
            nod = G.nodes.get(seg_id, {})
            obs[idx] = SEG_ENC.get(nod.get("stare", "liber"), 0)
            idx += 1

        # [38] origine cerere
        # [39] destinatie cerere
        tren = scenariu.trenuri_noi[0]
        obs[idx] = float(ORIGINE_IDX.get(tren.origine, 0))
        obs[idx + 1] = float(DESTINATIE_IDX.get(tren.destinatie, 0))
        idx += 2

        # [40] ruta activa exista
        # [41] destinatie ruta activa
        if scenariu.rute_active:
            ruta_activa = list(scenariu.rute_active.values())[0]
            dest_activ = ruta_activa[-1]
            obs[idx] = 1.0
            obs[idx + 1] = float(DESTINATIE_IDX.get(dest_activ, 0))
        else:
            obs[idx] = 0.0
            obs[idx + 1] = 0.0

        return obs

    def _interpreteaza_actiune(
        self,
        actiune: int,
        scenariu: ScenarioState,
    ) -> tuple[bool, str, float]:
        tren = scenariu.trenuri_noi[0]
        orig = tren.origine
        dest = tren.destinatie
        G = scenariu.graf
        rute_active = scenariu.rute_active

        linie_ceruta = PARCURS_LINIE.get((orig, dest))
        linie_actiune = ACTIUNE_LINIE.get(actiune)

        if actiune == 0:
            rute_valide = get_rute_valide(
                G,
                orig,
                dest,
                rute_active,
            )

            if rute_valide:
                return False, "asteapta_gresit", -10.0

            return True, "asteapta_corect", 3.0

        if linie_actiune != linie_ceruta:
            return (
                False,
                f"actiune_nepotrivita: cerut {linie_ceruta}, ales {linie_actiune}",
                -10.0,
            )

        rute_valide = get_rute_valide(
            G,
            orig,
            dest,
            rute_active,
        )

        if not rute_valide:
            return False, "ruta_imposibila", -10.0

        return True, "parcurs_realizabil", 10.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._scenariu = generate_scenario(
            difficulty_level=self.difficulty_level,
            mode=self.mode,
            seed=seed,
        )

        self._step_count = 0
        self._stats["total_episoade"] += 1

        obs = self._encode_obs(self._scenariu)
        info = self._build_info()

        return obs, info

    def step(self, actiune: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._scenariu is not None, "Apeleaza reset() inainte de step()."
        assert self.action_space.contains(actiune), f"Actiune invalida: {actiune}"

        self._step_count += 1

        succes, motiv, reward = self._interpreteaza_actiune(
            actiune,
            self._scenariu,
        )

        if motiv == "parcurs_realizabil":
            self._stats["decizii_corecte"] += 1
        elif motiv == "asteapta_corect":
            self._stats["asteapta_corect"] += 1
        elif motiv == "asteapta_gresit":
            self._stats["asteapta_gresit"] += 1
        else:
            self._stats["decizii_gresite"] += 1

        obs = self._encode_obs(self._scenariu)

        terminated = True
        truncated = False

        info = self._build_info()
        info.update({
            "succes": succes,
            "motiv": motiv,
            "reward": reward,
            "actiune": actiune,
        })

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._scenariu is None:
            print("Mediu neinitializat. Apeleaza reset().")
            return

        tren = self._scenariu.trenuri_noi[0]

        print(f"\n[TheiaCFEnv] Episod {self._stats['total_episoade']}")
        print(
            f"  Cerere   : {tren.origine} -> {tren.destinatie} "
            f"({PARCURS_LINIE.get((tren.origine, tren.destinatie), '?')})"
        )
        print(f"  Solvabil : {self._scenariu.solvabilitate_tip}")
        print(f"  Dificult.: {self._scenariu.difficulty_score:.2f}")

        if self._scenariu.macazuri_cu_probleme:
            print("  Macazuri :", [
                f"{m.macaz}({m.stare})"
                for m in self._scenariu.macazuri_cu_probleme
            ])

        if self._scenariu.rute_active:
            ruta = list(self._scenariu.rute_active.values())[0]
            print(f"  Trafic   : {ruta[0]}->{ruta[-1]}")

    def get_stats(self) -> dict:
        total = self._stats["total_episoade"]

        if total == 0:
            return self._stats

        return {
            **self._stats,
            "rata_succes": round(
                (
                    self._stats["decizii_corecte"]
                    + self._stats["asteapta_corect"]
                ) / max(total, 1),
                3,
            ),
        }

    def _build_info(self) -> dict:
        if self._scenariu is None:
            return {}

        tren = self._scenariu.trenuri_noi[0]

        return {
            "difficulty_score": self._scenariu.difficulty_score,
            "solvabilitate_tip": self._scenariu.solvabilitate_tip,
            "cerere_origine": tren.origine,
            "cerere_destinatie": tren.destinatie,
            "linie_ceruta": PARCURS_LINIE.get(
                (tren.origine, tren.destinatie),
                "?",
            ),
            "nr_rute_valide": len(
                self._scenariu.rute_valide_per_tren.get(tren.id, [])
            ),
        }