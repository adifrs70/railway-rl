import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_monitor_csv(model_name, level):
    patterns = [
        os.path.join("logs", f"{model_name}_level{level}_train.monitor.csv"),
        os.path.join("logs", f"{model_name}_level{level}_train*.monitor.csv"),
        os.path.join("logs", "**", f"{model_name}_level{level}_train*.monitor.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    return sorted(set(files))


def read_monitor_csv(path):
    return pd.read_csv(path, comment="#")


def moving_average(values, window=500):
    return pd.Series(values).rolling(window=window, min_periods=1).mean()


def generate_plot(model_name, level, window=500):
    files = find_monitor_csv(model_name, level)

    if not files:
        print(f"[LIPSA] Nu am gasit log de antrenare pentru {model_name.upper()} level {level}")
        return

    dataframes = []

    for file in files:
        df = read_monitor_csv(file)
        if "r" in df.columns:
            dataframes.append(df)

    if not dataframes:
        print(f"[EROARE] Logurile gasite pentru {model_name.upper()} level {level} nu contin coloana r")
        return

    data = pd.concat(dataframes, ignore_index=True)
    rewards = data["r"].astype(float).reset_index(drop=True)
    avg = moving_average(rewards, window=window)

    os.makedirs("plots", exist_ok=True)

    output_path = os.path.join(
        "plots",
        f"{model_name}_level{level}_convergenta_final.png"
    )

    plt.figure(figsize=(10, 4))

    plt.plot(
        avg,
        linewidth=2,
        label=f"Medie mobila ({window} episoade)"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="Reward neutru (0)"
    )

    plt.axhline(
        y=10,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="Reward maxim (+10)"
    )

    plt.axhline(
        y=-10,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="Reward minim (-10)"
    )

    plt.title(f"{model_name.upper()} - Nivel {level} - Evolutia recompensei la antrenare")
    plt.xlabel("Episod")
    plt.ylabel("Reward mediu mobil")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Grafic salvat: {output_path}")


def main():
    for model_name in ["dqn", "ppo"]:
        for level in [1, 2]:
            generate_plot(model_name, level)


if __name__ == "__main__":
    main()