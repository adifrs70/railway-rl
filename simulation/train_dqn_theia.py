"""
train_dqn_theia.py — Antrenare DQN pe statia Theia

Foloseste Stable-Baselines3 2.8.0 cu gymnasium.

Structura antrenament:
  Faza 1: difficulty_level=1 (scenarii simple, fara trafic activ)
  Faza 2: difficulty_level=2 (scenarii medii, cu trafic activ posibil)

Modele salvate:
  models/dqn_level1.zip
  models/dqn_level2.zip
  models/dqn_level1_best/best_model.zip   <- cel mai bun model pe eval
  models/dqn_level2_best/best_model.zip

Loguri TensorBoard:
  logs/dqn_level1/
  logs/dqn_level2/

Evaluare finala salvata:
  logs/dqn_level1_eval_summary.json
  logs/dqn_level2_eval_summary.json

Rulare:
  python train_dqn_theia.py                         # ambele niveluri, 100k steps
  python train_dqn_theia.py --level 1               # doar nivel 1
  python train_dqn_theia.py --level 1 --timesteps 20000  # recomandat la prima testare
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import compatibil cu rularea locala din folderul de lucru curent.
# Daca proiectul este rulat ca pachet din radacina proiectului,
# importul poate trebui schimbat in: from simulation.rail_env_theia import TheiaCFEnv
from rail_env_theia import TheiaCFEnv


# ---------------------------------------------------------------------------
# Directoare
# ---------------------------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)
os.makedirs("plots",  exist_ok=True)


# ---------------------------------------------------------------------------
# Callback pentru logging episodic
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    """
    Inregistreaza reward-ul mediu la fiecare N episoade.
    Util pentru grafice de convergenta.
    """

    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq        = log_freq
        self.episode_rewards : list[float] = []
        self.episode_count   = 0
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones", [False])[0]
        self._current_reward += reward

        if done:
            self.episode_count  += 1
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

            if self.episode_count % self.log_freq == 0:
                recent = self.episode_rewards[-self.log_freq:]
                mean_r = np.mean(recent)
                self.logger.record("train/mean_reward_recent", mean_r)
                if self.verbose >= 1:
                    print(
                        f"  Episod {self.episode_count:>6} | "
                        f"Reward mediu (ultimele {self.log_freq}): {mean_r:.3f}"
                    )

        return True


# ---------------------------------------------------------------------------
# Antrenare
# ---------------------------------------------------------------------------

def antreneaza_dqn(
    difficulty_level : int   = 1,
    total_timesteps  : int   = 100_000,
    # Nota: pentru primele teste se recomanda --timesteps 20000 sau 30000.
    # Valoarea de 100_000 este potrivita pentru antrenament complet.
    eval_freq        : int   = 5_000,
    n_eval_episodes  : int   = 100,
    reward_threshold : float = 6.5,
    # Pragul este 6.5, nu 10, deoarece politica optima nu maximizeaza
    # intotdeauna reward-ul la +10: exista episoade in care actiunea
    # corecta este "asteapta" (reward +3). Un amestec realist de episoade
    # solvabile (~50%) si nesolvabile (~50%) conduce la un reward mediu
    # optim in jur de 6.5. Pragul poate fi ajustat in sus dupa validare.
    verbose          : int   = 1,
) -> DQN:
    """
    Antreneaza un agent DQN pe statia Theia.

    Parametri:
      difficulty_level  : 1 (simplu) sau 2 (mediu)
      total_timesteps   : numar maxim de pasi de antrenament
                          (recomandat 20_000-30_000 la prima testare)
      eval_freq         : frecventa evaluarii (in timesteps)
      n_eval_episodes   : episoade per evaluare
      reward_threshold  : opreste antrenamentul daca reward mediu >= threshold
                          (implicit 6.5 — vezi comentariul de mai sus)
      verbose           : 0=silentios, 1=progres, 2=detaliat

    Returneaza modelul antrenat.
    """
    run_name = f"dqn_level{difficulty_level}"

    print("=" * 60)
    print(f"ANTRENARE DQN — Nivel {difficulty_level}")
    print("=" * 60)
    print(f"  Timesteps maxim : {total_timesteps:,}")
    print(f"  Eval freq       : {eval_freq:,}")
    print(f"  Reward threshold: {reward_threshold}")
    print()

    # --- Verificare compatibilitate mediu cu SB3 ---
    # check_env se apeleaza pe un obiect brut, inainte de wrapperele Monitor,
    # pentru a evita conflicte cu interfata interna a Monitor.
    if verbose >= 1:
        print("Verificare compatibilitate mediu cu SB3...")
    env_check = TheiaCFEnv(difficulty_level=difficulty_level, mode="antrenare")
    check_env(env_check, warn=True)
    env_check.close()
    if verbose >= 1:
        print("  OK\n")

    # --- Medii de antrenament si evaluare (wrapped cu Monitor) ---
    env_train = Monitor(
        TheiaCFEnv(difficulty_level=difficulty_level, mode="antrenare"),
        filename=f"logs/{run_name}_train",
    )
    env_eval = Monitor(
        TheiaCFEnv(difficulty_level=difficulty_level, mode="evaluare"),
        filename=f"logs/{run_name}_eval",
    )

    # --- Callbacks ---
    reward_logger = RewardLoggerCallback(
        log_freq=max(eval_freq // 10, 100),
        verbose=verbose,
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=verbose,
    )

    # SB3 salveaza cel mai bun model in: models/dqn_levelN_best/best_model.zip
    eval_callback = EvalCallback(
        eval_env             = env_eval,
        callback_on_new_best = stop_callback,
        eval_freq            = eval_freq,
        n_eval_episodes      = n_eval_episodes,
        best_model_save_path = f"models/{run_name}_best",
        log_path             = f"logs/{run_name}_eval",
        deterministic        = True,
        verbose              = verbose,
    )

    # --- Hiperparametri DQN ---
    # Calibrati pentru:
    #   - spatiu de observatii mic: 42 valori float32
    #   - actiuni discrete: Discrete(7)
    #   - episoade de 1 pas (terminated=True dupa fiecare step)
    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env_train,
        learning_rate          = 1e-3,       # potrivit pentru problema simpla
        buffer_size            = 50_000,     # suficient pentru episoade scurte
        learning_starts        = 1_000,      # explorare pura inainte de primul update
        batch_size             = 64,
        tau                    = 1.0,
        gamma                  = 0.99,
        train_freq             = 4,
        gradient_steps         = 1,
        target_update_interval = 1_000,
        exploration_fraction   = 0.3,        # 30% din timesteps = explorare
        exploration_initial_eps= 1.0,
        exploration_final_eps  = 0.05,
        policy_kwargs          = dict(net_arch=[64, 64]),  # retea mica, suficienta
        tensorboard_log        = "logs/",
        verbose                = 0,          # suprimam output-ul SB3, folosim callback-ul nostru
        device                 = "auto",
    )

    if verbose >= 1:
        print(f"Arhitectura retea: MLP [64, 64]")
        print(f"Learning rate    : {model.learning_rate}")
        print(f"Buffer size      : {model.buffer_size:,}")
        print(f"Exploration      : 1.0 -> 0.05 (in {int(total_timesteps * 0.3):,} steps)")
        print()
        print("Inceput antrenament...")
        print()

    # --- Antrenament ---
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [eval_callback, reward_logger],
        tb_log_name         = run_name,
        reset_num_timesteps = True,
        progress_bar        = verbose >= 1,
    )

    # --- Salvare model final ---
    model_path = f"models/{run_name}"
    model.save(model_path)
    if verbose >= 1:
        print(f"\nModel final salvat: {model_path}.zip")

    # --- Evaluare finala rapida ---
    if verbose >= 1:
        print("\nEvaluare finala rapida...")
    mean_reward, std_reward = _evalueaza_model(
        model, difficulty_level, n_episodes=200
    )
    if verbose >= 1:
        print(f"  Reward mediu final: {mean_reward:.3f} ± {std_reward:.3f}")

    # --- Grafic convergenta ---
    _salveaza_grafic(
        reward_logger.episode_rewards,
        title   = f"DQN — Nivel {difficulty_level} — Convergenta reward",
        path    = f"plots/{run_name}_convergenta.png",
        verbose = verbose,
    )

    env_train.close()
    env_eval.close()

    return model


# ---------------------------------------------------------------------------
# Evaluare
# ---------------------------------------------------------------------------

def _evalueaza_model(
    model            : DQN,
    difficulty_level : int,
    n_episodes       : int = 200,
    mode             : str = "evaluare",
) -> tuple[float, float]:
    """Evalueaza modelul pe N episoade si returneaza (mean, std) reward."""
    env = TheiaCFEnv(difficulty_level=difficulty_level, mode=mode)
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, _ = env.step(int(action))
        rewards.append(reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def evalueaza_complet(
    model            : DQN,
    difficulty_level : int,
    n_episodes       : int = 500,
) -> dict:
    """
    Evaluare detaliata: breakdown pe tipuri de decizie.
    Returneaza un dict cu statistici complete, gata de salvat in JSON.
    """
    env = TheiaCFEnv(difficulty_level=difficulty_level, mode="evaluare")
    rezultate = {
        "parcurs_realizabil" : 0,
        "asteapta_corect"    : 0,
        "asteapta_gresit"    : 0,
        "actiune_nepotrivita": 0,
        "ruta_imposibila"    : 0,
        "rewards"            : [],
    }

    for _ in range(n_episodes):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, step_info = env.step(int(action))

        motiv = step_info.get("motiv", "necunoscut")
        if motiv in rezultate:
            rezultate[motiv] += 1
        elif "actiune_nepotrivita" in motiv:
            rezultate["actiune_nepotrivita"] += 1

        rezultate["rewards"].append(reward)

    env.close()

    rewards = rezultate.pop("rewards")
    total   = n_episodes
    return {
        **{k: {"count": v, "procent": round(v / total * 100, 1)}
           for k, v in rezultate.items()},
        "mean_reward"      : round(float(np.mean(rewards)), 3),
        "std_reward"       : round(float(np.std(rewards)),  3),
        "n_episodes"       : total,
        "difficulty_level" : difficulty_level,
    }


def afiseaza_evaluare(rezultate: dict) -> None:
    print(f"\nEvaluare detaliata ({rezultate['n_episodes']} episoade):")
    print(f"  Reward mediu: {rezultate['mean_reward']:.3f} ± {rezultate['std_reward']:.3f}")
    print()
    for cheie in ["parcurs_realizabil", "asteapta_corect",
                  "asteapta_gresit", "actiune_nepotrivita", "ruta_imposibila"]:
        d   = rezultate.get(cheie, {"count": 0, "procent": 0.0})
        bar = "█" * int(d["procent"] / 2)
        print(f"  {cheie:<25}: {d['count']:>4}  ({d['procent']:>5.1f}%)  {bar}")


# ---------------------------------------------------------------------------
# Salvare evaluare
# ---------------------------------------------------------------------------

def _salveaza_evaluare_json(
    rezultate : dict,
    path      : str,
    verbose   : int = 1,
) -> None:
    """
    Salveaza rezultatele evaluarii finale intr-un fisier JSON.
    Structura: mean_reward, std_reward, n_episodes, breakdown pe decizii.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rezultate, f, indent=2, ensure_ascii=False)
    if verbose >= 1:
        print(f"Evaluare salvata: {path}")


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------

def _salveaza_grafic(
    rewards : list[float],
    title   : str,
    path    : str,
    window  : int = 200,
    verbose : int = 1,
) -> None:
    """Salveaza graficul de convergenta al reward-ului."""
    if len(rewards) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.8,
            label="Reward per episod")

    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            color="steelblue", linewidth=2,
            label=f"Medie mobila ({window} ep.)",
        )

    ax.axhline(y=0,   color="gray",  linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=10,  color="green", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Reward maxim (+10)")
    ax.axhline(y=-10, color="red",   linestyle="--", linewidth=0.8, alpha=0.5,
               label="Reward minim (-10)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episod")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose >= 1:
        print(f"Grafic salvat: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Antrenare DQN pe statia Theia"
    )
    parser.add_argument(
        "--level", type=int, default=0,
        help="Nivel dificultate: 1, 2 sau 0 pentru ambele (default: 0)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help=(
            "Numar de timesteps per nivel (default: 100_000). "
            "La prima testare se recomanda 20_000 sau 30_000."
        ),
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=500,
        help="Episoade pentru evaluarea finala detaliata (default: 500)"
    )
    args = parser.parse_args()

    levels = [1, 2] if args.level == 0 else [args.level]

    for level in levels:
        model = antreneaza_dqn(
            difficulty_level = level,
            total_timesteps  = args.timesteps,
            verbose          = 1,
        )

        print(f"\n{'='*60}")
        print(f"EVALUARE FINALA — DQN Nivel {level}")
        print(f"{'='*60}")

        rezultate = evalueaza_complet(
            model, level, n_episodes=args.eval_episodes
        )
        afiseaza_evaluare(rezultate)

        # Salvare JSON evaluare finala
        json_path = f"logs/dqn_level{level}_eval_summary.json"
        _salveaza_evaluare_json(rezultate, path=json_path, verbose=1)
        print()


if __name__ == "__main__":
    main()