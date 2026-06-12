import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from stable_baselines3 import DQN

from simulation.rail_env_theia import (
    TheiaCFEnv,
    ACTIUNE_LINIE,
    PARCURS_LINIE,
)

from simulation.scenario_generator import (
    TrenCerere,
    PERECHI_VALIDE,
    sunt_compatibile,
)

try:
    from graph.theia import get_rute_valide, elibereaza_ruta
except ImportError:
    from graph.theia import get_rute_valide
    elibereaza_ruta = None


MODEL_PATH = "models/dqn_level2.zip"
OUTPUT_PATH = "plots/demo_dqn_rerouting.png"


POS = {
    "X": (0, 0),
    "SEG_X_M1": (1, 0),
    "M1": (2, 0),

    "SEG_M1_L1": (3, -2),
    "Y1": (4, -2),
    "SEG_Y1_L1": (5, -2),
    "SEG_L1": (6, -2),
    "SEG_L1_X1": (7, -2),
    "X1": (8, -2),
    "SEG_M8_L1": (9, -2),

    "SEG_M1_M3": (3, 0),
    "M3": (4, 0),
    "SEG_M3_LII": (5, 0),
    "YII": (6, 0),
    "SEG_YII_LII": (7, 0),
    "SEG_LII": (8, 0),
    "SEG_LII_XII": (9, 0),
    "XII": (10, 0),
    "SEG_M8_LII": (11, 0),

    "SEG_M3_M5": (5, 1.5),
    "M5": (6, 1.5),

    "SEG_M5_L3": (7, 1),
    "Y3": (8, 1),
    "SEG_Y3_L3": (9, 1),
    "SEG_L3": (10, 1),
    "SEG_L3_X3": (11, 1),
    "X3": (12, 1),
    "SEG_DJ46_L3": (13, 1),

    "SEG_M5_L4": (7, 2.5),
    "Y4": (8, 2.5),
    "SEG_Y4_L4": (9, 2.5),
    "SEG_L4": (10, 2.5),
    "SEG_L4_X4": (11, 2.5),
    "X4": (12, 2.5),
    "SEG_DJ46_L4": (13, 2.5),

    "DJ46": (14, 1.75),
    "SEG_M2_DJ46": (15, 1.75),

    "M8": (12, -1),
    "SEG_M2_M8": (15, -1),

    "M2": (16, 0.4),
    "SEG_Y_M2": (17, 0.4),
    "Y": (18, 0.4),
}


def completeaza_pozitii(G):
    pos = dict(POS)
    lipsa = [n for n in G.nodes() if n not in pos]

    if lipsa:
        extra = nx.spring_layout(G.subgraph(lipsa), seed=42)
        for n, p in extra.items():
            pos[n] = (p[0] * 3 + 9, p[1] * 3 - 4)

    return pos


def muchii_din_ruta(ruta):
    if not ruta:
        return []
    return list(zip(ruta[:-1], ruta[1:]))


def copie_fara_rute_active(scenariu):
    G2 = scenariu.graf.copy()
    rute_active_copy = dict(scenariu.rute_active)

    if elibereaza_ruta is not None:
        for tren_id in list(rute_active_copy.keys()):
            try:
                elibereaza_ruta(G2, tren_id, rute_active_copy)
            except Exception:
                pass

    return G2


def ruta_fara_activ(scenariu, origine, destinatie):
    G2 = copie_fara_rute_active(scenariu)

    rute = get_rute_valide(
        G2,
        origine,
        destinatie,
        {},
    )

    return rute[0] if rute else None


def ruta_cu_activ(scenariu, origine, destinatie):
    rute = get_rute_valide(
        scenariu.graf,
        origine,
        destinatie,
        scenariu.rute_active,
    )

    return rute[0] if rute else None


def propune_rerutare(scenariu, origine, destinatie_initiala):
    if origine == "X":
        alternative = ["X1", "XII", "X3", "X4", "Y"]
    else:
        alternative = ["Y1", "YII", "Y3", "Y4", "X"]

    for destinatie in alternative:
        if destinatie == destinatie_initiala:
            continue

        ruta = ruta_cu_activ(scenariu, origine, destinatie)

        if ruta:
            return destinatie, ruta

    return None, None


def modifica_cererea_scenariului(scenariu, origine, destinatie):
    scenariu.trenuri_noi = [
        TrenCerere(
            id="T_demo",
            origine=origine,
            destinatie=destinatie,
        )
    ]

    ruta = ruta_cu_activ(scenariu, origine, destinatie)

    scenariu.rute_valide_per_tren = {
        "T_demo": [ruta] if ruta else []
    }

    scenariu.solvabil = bool(ruta)
    scenariu.solvabilitate_tip = (
        "solvabil_total" if ruta else "nesolvabil_interesant"
    )


def evalueaza_decizie_dqn(scenariu, actiune):
    tren = scenariu.trenuri_noi[0]
    origine = tren.origine
    destinatie = tren.destinatie

    linie_ceruta = PARCURS_LINIE.get((origine, destinatie))
    linie_aleasa = ACTIUNE_LINIE.get(int(actiune))

    ruta_valida = ruta_cu_activ(scenariu, origine, destinatie)

    if int(actiune) == 0:
        if ruta_valida:
            return False, "DQN a decis ASTEAPTA, dar ruta ceruta era realizabila"
        return True, "DQN a decis ASTEAPTA, iar ruta ceruta nu era realizabila"

    if linie_aleasa != linie_ceruta:
        return False, f"DQN a ales {linie_aleasa}, dar cererea era {linie_ceruta}"

    if not ruta_valida:
        return False, "DQN a ales ruta ceruta, dar aceasta nu era realizabila"

    return True, "DQN a acceptat corect ruta ceruta"


def gaseste_scenariu_demo(model, max_incercari=5000):
    for seed in range(max_incercari):
        env = TheiaCFEnv(difficulty_level=2, mode="antrenare")
        obs, info = env.reset(seed=seed)
        scenariu = env._scenariu

        if not scenariu.rute_active:
            continue

        ruta_activa = list(scenariu.rute_active.values())[0]
        origine_activa = ruta_activa[0]
        destinatie_activa = ruta_activa[-1]

        for origine_ceruta, destinatie_ceruta in PERECHI_VALIDE:
            if sunt_compatibile(
                origine_activa,
                destinatie_activa,
                origine_ceruta,
                destinatie_ceruta,
            ):
                continue

            ruta_initiala = ruta_fara_activ(
                scenariu,
                origine_ceruta,
                destinatie_ceruta,
            )

            if not ruta_initiala:
                continue

            ruta_acceptata = ruta_cu_activ(
                scenariu,
                origine_ceruta,
                destinatie_ceruta,
            )

            if ruta_acceptata:
                continue

            destinatie_alt, ruta_alt = propune_rerutare(
                scenariu,
                origine_ceruta,
                destinatie_ceruta,
            )

            if not ruta_alt:
                continue

            modifica_cererea_scenariului(
                scenariu,
                origine_ceruta,
                destinatie_ceruta,
            )

            obs_demo = env._encode_obs(scenariu)

            actiune, _ = model.predict(obs_demo, deterministic=True)
            actiune = int(actiune)

            dqn_corect, mesaj_dqn = evalueaza_decizie_dqn(
                scenariu,
                actiune,
            )

            return (
                seed,
                scenariu,
                actiune,
                dqn_corect,
                mesaj_dqn,
                ruta_initiala,
                None,
                destinatie_alt,
                ruta_alt,
            )

    return None


def deseneaza_demo(
    seed,
    scenariu,
    actiune,
    dqn_corect,
    mesaj_dqn,
    ruta_initiala,
    ruta_acceptata,
    destinatie_alt,
    ruta_alt,
):
    G = scenariu.graf
    pos = completeaza_pozitii(G)

    tren = scenariu.trenuri_noi[0]
    ruta_activa = (
        list(scenariu.rute_active.values())[0]
        if scenariu.rute_active
        else None
    )

    tip_culori = {
        "capat": "lightcoral",
        "macaz": "lightblue",
        "segment": "lightgray",
        "semnal": "lightgreen",
    }

    node_colors = [
        tip_culori.get(G.nodes[n].get("tip"), "white")
        for n in G.nodes()
    ]

    plt.figure(figsize=(23, 10))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=850,
        edgecolors="black",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=9,
        width=1.0,
        alpha=0.25,
        edge_color="gray",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
    )

    if ruta_activa:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=muchii_din_ruta(ruta_activa),
            arrows=True,
            arrowsize=13,
            width=4.0,
            alpha=0.95,
            edge_color="red",
            connectionstyle="arc3,rad=0.0",
        )

    if ruta_initiala:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=muchii_din_ruta(ruta_initiala),
            arrows=True,
            arrowsize=14,
            width=4.0,
            alpha=0.90,
            edge_color="green",
            style="dashed",
            connectionstyle="arc3,rad=-0.08",
        )

    if ruta_acceptata:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=muchii_din_ruta(ruta_acceptata),
            arrows=True,
            arrowsize=15,
            width=5.0,
            alpha=1.0,
            edge_color="green",
            style="solid",
            connectionstyle="arc3,rad=-0.08",
        )

    if ruta_alt:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=muchii_din_ruta(ruta_alt),
            arrows=True,
            arrowsize=16,
            width=5.5,
            alpha=1.0,
            edge_color="orange",
            style="dashdot",
            connectionstyle="arc3,rad=0.08",
        )

    activ_txt = "fara ruta activa"
    if ruta_activa:
        activ_txt = f"{ruta_activa[0]} -> {ruta_activa[-1]}"

    cerere_txt = f"{tren.origine} -> {tren.destinatie}"
    actiune_txt = ACTIUNE_LINIE.get(actiune)

    if ruta_acceptata:
        rezultat_txt = "Ruta ceruta este compatibila si poate fi acceptata"
    elif ruta_alt:
        rezultat_txt = (
            f"Ruta ceruta este incompatibila cu ruta activa; "
            f"alternativa propusa: {tren.origine} -> {destinatie_alt}"
        )
    else:
        rezultat_txt = "Ruta ceruta este incompatibila; nu exista alternativa valida"

    titlu = (
        f"Demo DQN rerutare | seed={seed}\n"
        f"Ruta activa: {activ_txt} | Ruta ceruta: {cerere_txt} | "
        f"Actiune DQN: {actiune} ({actiune_txt})\n"
        f"{mesaj_dqn} | {rezultat_txt}"
    )

    plt.title(titlu, fontsize=13)
    plt.axis("off")

    legenda = [
        Line2D([0], [0], color="red", lw=4, label="Ruta activa existenta"),
        Line2D(
            [0],
            [0],
            color="green",
            lw=4,
            linestyle="--",
            label="Ruta ceruta initial",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            lw=5,
            linestyle="dashdot",
            label="Ruta alternativa propusa",
        ),
    ]

    plt.legend(handles=legenda, loc="lower center", ncol=3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=250)
    plt.show()


def main():
    model = DQN.load(MODEL_PATH)

    rezultat = gaseste_scenariu_demo(model)

    if rezultat is None:
        print("Nu s-a gasit scenariu potrivit pentru demo.")
        print("Incearca sa maresti max_incercari sau sa rulezi pe nivel 3.")
        return

    (
        seed,
        scenariu,
        actiune,
        dqn_corect,
        mesaj_dqn,
        ruta_initiala,
        ruta_acceptata,
        destinatie_alt,
        ruta_alt,
    ) = rezultat

    tren = scenariu.trenuri_noi[0]
    ruta_activa = (
        list(scenariu.rute_active.values())[0]
        if scenariu.rute_active
        else None
    )

    print("=" * 70)
    print("DEMO DQN + RERUTARE")
    print("=" * 70)
    print("Seed:", seed)
    print("Ruta activa:", " -> ".join(ruta_activa) if ruta_activa else "niciuna")
    print("Ruta ceruta:", tren.origine, "->", tren.destinatie)
    print("Ruta ceruta initiala:", " -> ".join(ruta_initiala))
    print("Ruta ceruta cu ruta activa: nerealizabila")
    print("Actiune DQN:", actiune, ACTIUNE_LINIE.get(actiune))
    print("Decizie DQN:", mesaj_dqn)
    print("DQN corect:", dqn_corect)
    print("Rerutare propusa:", tren.origine, "->", destinatie_alt)
    print("Ruta alternativa:", " -> ".join(ruta_alt))
    print("Muchii alternativa:", muchii_din_ruta(ruta_alt))
    print("Imagine salvata in:", OUTPUT_PATH)
    print("=" * 70)

    deseneaza_demo(
        seed,
        scenariu,
        actiune,
        dqn_corect,
        mesaj_dqn,
        ruta_initiala,
        ruta_acceptata,
        destinatie_alt,
        ruta_alt,
    )


if __name__ == "__main__":
    main()