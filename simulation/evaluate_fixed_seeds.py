"""
evaluate_fixed_seeds.py — Validare cantitativa pe seed-uri fixe

Evalueaza modelul antrenat (DQN si/sau PPO) pe 1000 de scenarii
generate cu un interval fix de seed-uri rezervat exclusiv evaluarii
finale (implicit: 100000-100999).

In antrenare, scenariile sunt generate aleatoriu fara seed fix,
deci nu exista o lista explicita de seed-uri utilizate. Intervalul
de evaluare este ales conventional pentru a fi separat si reproductibil.

Scopul:
  Demonstreaza ca agentul a invatat o politica generala de decizie,
  evaluata pe scenarii generate independent de procesul de antrenare.

Metrici raportate:
  - reward mediu si deviatie standard
  - rata deciziilor corecte = (parcurs_realizabil + asteapta_corect) / total
  - rata de eroare = (asteapta_gresit + actiune_nepotrivita + ruta_imposibila) / total
  - breakdown complet pe tipuri de decizie
  - comparatie DQN vs PPO (daca ambele modele sunt disponibile)

Salvare rezultate:
  logs/eval_fixed_seeds_dqn.json
  logs/eval_fixed_seeds_ppo.json
  logs/eval_fixed_seeds_comparison.json  (daca ambele disponibile)

Rulare:
  python evaluate_fixed_seeds.py
  python evaluate_fixed_seeds.py --model ppo
  python evaluate_fixed_seeds.py --model both
  python evaluate_fixed_seeds.py --n 500
  python evaluate_fixed_seeds.py --seed-start 200000
"""

import os
import sys
import json
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

for _p in [_SCRIPT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulation.rail_env_theia import TheiaCFEnv, ACTIUNE_LINIE

os.makedirs("logs", exist_ok=True)


def incarca_model(tip_model: str):
    try:
        if tip_model == "dqn":
            from stable_baselines3 import DQN as CLS
            cai = [
                "models/dqn_level2_best/best_model.zip",
                "models/dqn_level1_best/best_model.zip",
                "models/dqn_level2.zip",
                "models/dqn_level1.zip",
            ]
        else:
            from stable_baselines3 import PPO as CLS
            cai = [
                "models/ppo_level2_best/best_model.zip",
                "models/ppo_level1_best/best_model.zip",
                "models/ppo_level2.zip",
                "models/ppo_level1.zip",
            ]

        _dirs = [
            _SCRIPT_DIR,
            _ROOT_DIR,
            os.getcwd(),
        ]

        for d in _dirs:
            for cale in cai:
                full = os.path.join(d, cale)
                if os.path.exists(full):
                    model = CLS.load(full)
                    print(f"  Model {tip_model.upper()} incarcat: {full}")
                    return model

        print(f"  [!] Niciun model {tip_model.upper()} gasit.")
        return None

    except ImportError:
        print("  [!] stable-baselines3 nu este instalat.")
        return None

    except Exception as e:
        print(f"  [!] Eroare la incarcarea modelului: {e}")
        return None


def evalueaza_pe_seed_uri_fixe(
    model,
    tip_model: str,
    difficulty_level: int = 2,
    n_episoade: int = 1000,
    seed_start: int = 100_000,
    verbose: bool = True,
) -> dict:
    env = TheiaCFEnv(
        difficulty_level=difficulty_level,
        mode="evaluare",
    )

    rezultate = {
        "parcurs_realizabil": 0,
        "asteapta_corect": 0,
        "asteapta_gresit": 0,
        "actiune_nepotrivita": 0,
        "ruta_imposibila": 0,
    }

    rewards = []
    actiuni_alese = []

    if verbose:
        print(
            f"\n  Evaluare {tip_model.upper()} pe {n_episoade} scenarii "
            f"(seed-uri {seed_start}..{seed_start + n_episoade - 1})..."
        )

    for i in range(n_episoade):
        seed = seed_start + i

        obs, info = env.reset(seed=seed)

        actiune, _ = model.predict(
            obs,
            deterministic=True,
        )

        actiune = int(actiune)

        _, reward, _, _, step_info = env.step(actiune)

        motiv = step_info.get("motiv", "necunoscut")

        if "actiune_nepotrivita" in motiv:
            rezultate["actiune_nepotrivita"] += 1
        elif motiv in rezultate:
            rezultate[motiv] += 1

        rewards.append(float(reward))
        actiuni_alese.append(actiune)

        if verbose and (i + 1) % 200 == 0:
            pct = (i + 1) / n_episoade * 100
            mean_r = np.mean(rewards)

            print(
                f"    {i + 1:>5}/{n_episoade} ({pct:.0f}%) "
                f"- reward mediu pana acum: {mean_r:.3f}"
            )

    total = n_episoade

    rata_corecte = (
        rezultate["parcurs_realizabil"]
        + rezultate["asteapta_corect"]
    ) / total

    rata_eroare = (
        rezultate["asteapta_gresit"]
        + rezultate["actiune_nepotrivita"]
        + rezultate["ruta_imposibila"]
    ) / total

    dist_actiuni = {}
    n_actiuni = env.action_space.n

    for a in range(n_actiuni):
        cnt = actiuni_alese.count(a)

        dist_actiuni[str(a)] = {
            "linie": ACTIUNE_LINIE.get(a, "?"),
            "count": cnt,
            "procent": round(cnt / total * 100, 2),
        }

    env.close()

    rezultat_final = {
        "model": tip_model.upper(),
        "difficulty_level": difficulty_level,
        "n_episoade": total,
        "seed_start": seed_start,
        "seed_end": seed_start + n_episoade - 1,

        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "min_reward": round(float(np.min(rewards)), 4),
        "max_reward": round(float(np.max(rewards)), 4),

        "rata_decizii_corecte": round(rata_corecte * 100, 2),
        "rata_eroare": round(rata_eroare * 100, 2),

        "breakdown": {
            k: {
                "count": v,
                "procent": round(v / total * 100, 2),
            }
            for k, v in rezultate.items()
        },

        "distributie_actiuni": dist_actiuni,
    }

    return rezultat_final


def afiseaza_rezultate(rezultat: dict) -> None:
    print()
    print("=" * 60)
    print(f"  REZULTATE EVALUARE - {rezultat['model']}")
    print(
        f"  {rezultat['n_episoade']} scenarii | "
        f"seed-uri {rezultat['seed_start']}..{rezultat['seed_end']}"
    )
    print("=" * 60)

    print("\n  Reward:")
    print(f"    Mediu          : {rezultat['mean_reward']:.4f}")
    print(f"    Dev. standard  : {rezultat['std_reward']:.4f}")
    print(
        f"    Min / Max      : "
        f"{rezultat['min_reward']:.1f} / {rezultat['max_reward']:.1f}"
    )

    print("\n  Metrici principale:")
    print(
        f"    Rata decizii corecte : "
        f"{rezultat['rata_decizii_corecte']:.2f}%"
    )
    print(
        f"    Rata eroare          : "
        f"{rezultat['rata_eroare']:.2f}%"
    )

    print("\n  Breakdown decizii:")

    etichete = {
        "parcurs_realizabil": "Parcurs realizabil ales   (+10)",
        "asteapta_corect": "Asteapta corect           (+3) ",
        "asteapta_gresit": "Asteapta gresit           (-10)",
        "actiune_nepotrivita": "Actiune nepotrivita       (-10)",
        "ruta_imposibila": "Ruta imposibila aleasa    (-10)",
    }

    for cheie, eticheta in etichete.items():
        d = rezultat["breakdown"].get(
            cheie,
            {
                "count": 0,
                "procent": 0,
            },
        )

        bar = "#" * int(d["procent"] / 2)

        print(
            f"    {eticheta}: "
            f"{d['count']:>5} "
            f"({d['procent']:>5.2f}%)  {bar}"
        )

    print("\n  Distributie actiuni alese de model:")

    for a_str, info in rezultat["distributie_actiuni"].items():
        if info["count"] == 0:
            continue

        bar = "#" * int(info["procent"] / 2)

        print(
            f"    {int(a_str)}={str(info['linie']):<16}: "
            f"{info['count']:>5} "
            f"({info['procent']:>5.2f}%)  {bar}"
        )

    print("=" * 60)


def afiseaza_comparatie(rez_dqn: dict, rez_ppo: dict) -> None:
    print()
    print("=" * 60)
    print("  COMPARATIE DQN vs PPO")
    print(f"  {rez_dqn['n_episoade']} scenarii identice per model")
    print("=" * 60)

    metrici = [
        ("Reward mediu", "mean_reward", ".4f"),
        ("Dev. standard reward", "std_reward", ".4f"),
        ("Rata decizii corecte", "rata_decizii_corecte", ".2f"),
        ("Rata eroare", "rata_eroare", ".2f"),
    ]

    print(f"\n  {'Metrica':<28} {'DQN':>10} {'PPO':>10} {'Delta':>10}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}")

    for eticheta, cheie, fmt in metrici:
        v_dqn = rez_dqn[cheie]
        v_ppo = rez_ppo[cheie]
        delta = v_ppo - v_dqn
        semn = "+" if delta >= 0 else ""

        print(
            f"  {eticheta:<28} "
            f"{v_dqn:>10{fmt}} "
            f"{v_ppo:>10{fmt}} "
            f"{semn}{delta:>9{fmt}}"
        )

    print("\n  Breakdown comparativ:")

    breakdown_keys = [
        ("parcurs_realizabil", "Parcurs realizabil"),
        ("asteapta_corect", "Asteapta corect"),
        ("asteapta_gresit", "Asteapta gresit"),
        ("actiune_nepotrivita", "Actiune nepotrivita"),
        ("ruta_imposibila", "Ruta imposibila"),
    ]

    print(f"  {'Tip decizie':<22} {'DQN%':>8} {'PPO%':>8}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8}")

    for cheie, eticheta in breakdown_keys:
        p_dqn = rez_dqn["breakdown"].get(cheie, {}).get("procent", 0)
        p_ppo = rez_ppo["breakdown"].get(cheie, {}).get("procent", 0)

        print(
            f"  {eticheta:<22} "
            f"{p_dqn:>7.2f}% "
            f"{p_ppo:>7.2f}%"
        )

    if rez_dqn["rata_decizii_corecte"] > rez_ppo["rata_decizii_corecte"]:
        castigator = "DQN"
    elif rez_ppo["rata_decizii_corecte"] > rez_dqn["rata_decizii_corecte"]:
        castigator = "PPO"
    else:
        castigator = "Egal"

    print(f"\n  Model cu rata de succes mai mare: {castigator}")
    print("=" * 60)


def salveaza_rezultate(rezultat: dict, path: str) -> None:
    os.makedirs(
        os.path.dirname(path) if os.path.dirname(path) else ".",
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            rezultat,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"  Salvat: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validare cantitativa model RL pe seed-uri fixe"
    )

    parser.add_argument(
        "--model",
        choices=["dqn", "ppo", "both"],
        default="dqn",
        help="Modelul de evaluat: dqn, ppo sau both",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Numar de scenarii de evaluare",
    )

    parser.add_argument(
        "--difficulty",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Nivel dificultate scenarii",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=100_000,
        help="Primul seed fix",
    )

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   VALIDARE CANTITATIVA - Statia Theia               ║")
    print("║   Evaluare pe seed-uri fixe, separate de antrenare  ║")
    print("╚══════════════════════════════════════════════════════╝")

    print(f"\n  Scenarii   : {args.n}")
    print(f"  Dificultate: {args.difficulty}")
    print(f"  Seed-uri   : {args.seed_start} .. {args.seed_start + args.n - 1}")
    print(f"  Model(e)   : {args.model.upper()}")

    modele_de_evaluat = (
        ["dqn", "ppo"]
        if args.model == "both"
        else [args.model]
    )

    rezultate = {}

    for tip in modele_de_evaluat:
        print(f"\n  Incarcare model {tip.upper()}...")

        model = incarca_model(tip)

        if model is None:
            print(f"  [!] Model {tip.upper()} indisponibil, sarit.")
            continue

        rez = evalueaza_pe_seed_uri_fixe(
            model=model,
            tip_model=tip,
            difficulty_level=args.difficulty,
            n_episoade=args.n,
            seed_start=args.seed_start,
            verbose=True,
        )

        afiseaza_rezultate(rez)

        path = os.path.join(
            "logs",
            f"eval_fixed_seeds_{tip}.json",
        )

        salveaza_rezultate(
            rez,
            path,
        )

        rezultate[tip] = rez

    if "dqn" in rezultate and "ppo" in rezultate:
        afiseaza_comparatie(
            rezultate["dqn"],
            rezultate["ppo"],
        )

        comp = {
            "dqn": rezultate["dqn"],
            "ppo": rezultate["ppo"],
        }

        path_comp = os.path.join(
            "logs",
            "eval_fixed_seeds_comparison.json",
        )

        salveaza_rezultate(
            comp,
            path_comp,
        )


if __name__ == "__main__":
    main()