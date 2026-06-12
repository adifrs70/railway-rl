"""
demo_theia.py — Validare calitativa a modelului RL pe statia Theia

ROLUL ACESTUI SCRIPT
====================
Acest script realizeaza VALIDAREA CALITATIVA a agentului antrenat.
Nu este destinat calculului statistic al performantei — aceasta se
face in evaluate_fixed_seeds.py pe 1000 de scenarii cu seed-uri fixe.

Scopul validarii calitative este:
  - demonstrarea comportamentului agentului in scenarii concrete,
    usor de interpretat operational
  - verificarea ca deciziile sunt corecte din punct de vedere feroviar
  - ilustrarea cazurilor reprezentative pentru prezentare si lucrare

SCENARII RECOMANDATE PENTRU VALIDARE CALITATIVA
================================================
  1. Macaz defect pe ruta ceruta
     ex: M8 defect + cerere Y->YII → INDISPONIBIL (M8 e pe ruta)

  2. Macaz defect pe alta linie, cerere posibila
     ex: M5 defect + cerere Y->LII → DISPONIBIL (M5 nu e pe ruta)

  3. Tranzit complet cu infrastructura normala
     ex: X->Y, fara defecte → DISPONIBIL

  4. Parcurs de iesire de pe linie
     ex: X1->Y → DISPONIBIL (tranzit_capat)

  5. Transfer pe aceeasi linie
     ex: Y1->X1 → DISPONIBIL (tranzit_semnal)

  6. Conflict cu parcurs activ generat de sistem
     ex: cerere Y->L3 + activ X->L3 → INDISPONIBIL (conflict)

  7. Situatie fara parcurs disponibil
     ex: M2 defect + cerere Y->L1 → INDISPONIBIL

DESPRE RERURARE
===============
In acest sistem, rerutarea inseamna gasirea unei cai alternative
catre ACEEASI destinatie, nu schimbarea destinatiei.
Exemplu: cerere X->Y cu M8 defect → sistemul cauta alta ruta X->Y
fara M8. Daca nu exista, afiseaza INDISPONIBIL.
Agentul NU propune o alta destinatie (ex: X->X3 in loc de X->Y).

Flux:
  1. Utilizatorul introduce defectele de infrastructura
  2. Utilizatorul introduce cererea trenului
  3. Sistemul decide random daca exista un parcurs activ
  4. Modelul DQN/PPO analizeaza starea si decide
  5. Vizualizare grafica matplotlib

Rulare:
  python demo_theia.py                     # DQN (default)
  python demo_theia.py --model ppo         # PPO
  python demo_theia.py --model dqn         # explicit DQN
  python demo_theia.py --no-model          # fara model (logica directa)
"""

import os
import sys
import random
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---------------------------------------------------------------------------
# Path setup — compatibil cu rularea din simulation/ sau din radacina
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
for _p in [_SCRIPT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from graph.theia import (
    build_station,
    get_rute_valide,
    set_stare_macaz,
    set_stare_segment,
    rezerva_ruta,
    inferenta_configuratie,
    build_planning_graph,
)
from simulation.scenario_generator import sunt_compatibile, parcurs_la_cheie
from simulation.rail_env_theia import (
    TheiaCFEnv,
    PARCURS_LINIE,
    ACTIUNE_LINIE,
    MACAZURI_ORDINE,
    SEGMENTE_ORDINE,
    DESTINATII,
    DESTINATIE_IDX,
    ORIGINE_IDX,
    POZITIE_ENC,
    STARE_ENC,
    SEG_ENC,
    OBS_DIM,
)


# ---------------------------------------------------------------------------
# Constante
# ---------------------------------------------------------------------------

MACAZURI_VALIDE = ["M1", "M3", "M5", "M2", "M8", "DJ46"]

# Cereri acceptate de modelul RL (origine strict X sau Y)
# Modelul a fost antrenat pe aceste cereri — ORIGINE_IDX = {"X":0, "Y":1}
# Daca se introduc cereri cu origine semnal (Y1, X1 etc.) in modul cu model,
# encoding-ul ar fi incorect (default 0 = X), deci sunt restrictionate.
CERERI_MODEL_RL = {
    "X->L1" : ("X","X1"),   "X->LII": ("X","XII"),
    "X->L3" : ("X","X3"),   "X->L4" : ("X","X4"),
    "Y->L1" : ("Y","Y1"),   "Y->LII": ("Y","YII"),
    "Y->L3" : ("Y","Y3"),   "Y->L4" : ("Y","Y4"),
    "X->Y"  : ("X","Y"),    "Y->X"  : ("Y","X"),
}

# Cereri extinse — disponibile doar in modul --no-model
# (validare cu logica directa din graf, fara model RL)
# Includ iesiri de pe linie si transferuri semnal->semnal.
CERERI_EXTINSE_GRAF = {
    "Y1->X" : ("Y1","X"),   "YII->X": ("YII","X"),
    "Y3->X" : ("Y3","X"),   "Y4->X" : ("Y4","X"),
    "X1->Y" : ("X1","Y"),   "XII->Y": ("XII","Y"),
    "X3->Y" : ("X3","Y"),   "X4->Y" : ("X4","Y"),
    "Y1->X1" : ("Y1","X1"), "YII->XII":("YII","XII"),
    "Y3->X3" : ("Y3","X3"), "Y4->X4" :("Y4","X4"),
    "X1->Y1" : ("X1","Y1"), "XII->YII":("XII","YII"),
    "X3->Y3" : ("X3","Y3"), "X4->Y4" :("X4","Y4"),
}

# CERERE_MAP este setat dinamic in main() in functie de modul de rulare
CERERE_MAP: dict = {}

# Alias-uri pentru normalizare input utilizator
# Acopera toate formatele posibile: "x->l1", "x -> l1", "x->l2" etc.
CERERE_ALIAS_RL = {
    "x->l1":"X->L1",   "x->lii":"X->LII",  "x->l2":"X->LII",
    "x->l3":"X->L3",   "x->l4":"X->L4",
    "y->l1":"Y->L1",   "y->lii":"Y->LII",  "y->l2":"Y->LII",
    "y->l3":"Y->L3",   "y->l4":"Y->L4",
    "x -> l1":"X->L1", "x -> lii":"X->LII","x -> l2":"X->LII",
    "x -> l3":"X->L3", "x -> l4":"X->L4",
    "y -> l1":"Y->L1", "y -> lii":"Y->LII","y -> l2":"Y->LII",
    "y -> l3":"Y->L3", "y -> l4":"Y->L4",
    "x->y":"X->Y",     "y->x":"Y->X",
    "x -> y":"X->Y",   "y -> x":"Y->X",
}

CERERE_ALIAS_GRAF = {
    **CERERE_ALIAS_RL,
    "y1->x":"Y1->X",   "yii->x":"YII->X",
    "y3->x":"Y3->X",   "y4->x":"Y4->X",
    "x1->y":"X1->Y",   "xii->y":"XII->Y",
    "x3->y":"X3->Y",   "x4->y":"X4->Y",
    "y1 -> x":"Y1->X", "yii -> x":"YII->X",
    "y3 -> x":"Y3->X", "y4 -> x":"Y4->X",
    "x1 -> y":"X1->Y", "xii -> y":"XII->Y",
    "x3 -> y":"X3->Y", "x4 -> y":"X4->Y",
    "y1->x1":"Y1->X1",  "yii->xii":"YII->XII",
    "y3->x3":"Y3->X3",  "y4->x4":"Y4->X4",
    "x1->y1":"X1->Y1",  "xii->yii":"XII->YII",
    "x3->y3":"X3->Y3",  "x4->y4":"X4->Y4",
    "y1 -> x1":"Y1->X1","yii -> xii":"YII->XII",
    "y3 -> x3":"Y3->X3","y4 -> x4":"Y4->X4",
    "x1 -> y1":"X1->Y1","xii -> yii":"XII->YII",
    "x3 -> y3":"X3->Y3","x4 -> y4":"X4->Y4",
}

# CERERE_ALIAS este setat dinamic in main()
CERERE_ALIAS: dict = {}

# Probabilitate ca sistemul sa genereze un parcurs activ
PROB_PARCURS_ACTIV = 0.5

# Pozitii fixe pentru vizualizare (x, y) per nod
# Layout pe 4 linii orizontale: L1=y1.0, LII=y2.0, L3=y3.0, L4=y4.0
# Axa X: 0.0 (capatul X) -> 14.0 (capatul Y)
POZITII_VIZUALIZARE = {
    # ── Capete ──────────────────────────────────────
    "X"           : (0.0,  2.0),
    "Y"           : (14.0, 2.0),
    # ── Segment exterior cap X ──────────────────────
    "SEG_X_M1"    : (1.0,  2.0),
    "M1"          : (2.0,  2.0),
    # ── Zona ramificare cap X ───────────────────────
    "SEG_M1_M3"   : (3.0,  2.5),
    "SEG_M1_L1"   : (3.0,  1.0),
    "M3"          : (4.0,  3.0),
    "SEG_M3_LII"  : (5.0,  2.0),
    "SEG_M3_M5"   : (5.0,  3.5),
    "M5"          : (6.0,  3.5),
    "SEG_M5_L3"   : (6.5,  3.0),
    "SEG_M5_L4"   : (6.5,  4.0),
    # ── Semnale cap X ───────────────────────────────
    "Y1"          : (4.0,  1.0),
    "YII"         : (6.5,  2.0),
    "Y3"          : (7.5,  3.0),
    "Y4"          : (7.5,  4.0),
    # ── Segmente semnal → corp linie ────────────────
    "SEG_Y1_L1"   : (5.0,  1.0),
    "SEG_YII_LII" : (7.5,  2.0),
    "SEG_Y3_L3"   : (8.0,  3.0),
    "SEG_Y4_L4"   : (8.0,  4.0),
    # ── Corpuri linii ────────────────────────────────
    "SEG_L1"      : (6.5,  1.0),
    "SEG_LII"     : (8.5,  2.0),
    "SEG_L3"      : (9.0,  3.0),
    "SEG_L4"      : (9.0,  4.0),
    # ── Segmente corp → semnal cap Y ─────────────────
    "SEG_L1_X1"   : (8.0,  1.0),
    "SEG_LII_XII" : (9.5,  2.0),
    "SEG_L3_X3"   : (10.0, 3.0),
    "SEG_L4_X4"   : (10.0, 4.0),
    # ── Semnale cap Y ────────────────────────────────
    "X1"          : (9.0,  1.0),
    "XII"         : (10.5, 2.0),
    "X3"          : (11.0, 3.0),
    "X4"          : (11.0, 4.0),
    # ── Segmente semnal → macazuri cap Y ─────────────
    "SEG_M8_L1"   : (10.0, 1.0),
    "SEG_M8_LII"  : (11.0, 2.0),
    "SEG_DJ46_L3" : (11.5, 3.0),
    "SEG_DJ46_L4" : (11.5, 4.0),
    # ── Macazuri cap Y ───────────────────────────────
    "M8"          : (11.5, 1.5),
    "DJ46"        : (12.0, 3.5),
    # ── Zona convergenta cap Y ───────────────────────
    "SEG_M2_M8"   : (12.0, 1.8),
    "SEG_M2_DJ46" : (12.5, 3.0),
    "M2"          : (13.0, 2.5),
    "SEG_Y_M2"    : (13.5, 2.5),
}


# ---------------------------------------------------------------------------
# Parsare input utilizator
# ---------------------------------------------------------------------------

def parseaza_defecte(text: str) -> list[dict]:
    """
    Parseaza sirul de defecte introdus de utilizator.
    Formate acceptate:
      "M8 defect"
      "M3 blocat minus"
      "M3 blocat plus"
      "nimic" / "" / "none"
    Multiple defecte separate prin virgula sau punct si virgula.
    """
    text = text.strip().lower()
    if not text or text in ("nimic", "none", "-", "nu"):
        return []

    defecte = []
    separatori = "," if "," in text else ";"
    parti = [p.strip() for p in text.split(separatori)] if separatori in text else [text]

    for parte in parti:
        cuvinte = parte.strip().split()
        if len(cuvinte) < 2:
            continue

        macaz = cuvinte[0].upper()
        if macaz not in MACAZURI_VALIDE:
            print(f"  [!] Macaz necunoscut ignorat: {macaz}")
            continue

        stare = cuvinte[1].lower()
        if stare == "defect":
            defecte.append({"macaz": macaz, "stare": "defect", "pozitie": "plus"})
        elif stare == "blocat":
            pozitie = cuvinte[2].lower() if len(cuvinte) >= 3 else "plus"
            if pozitie not in ("plus", "minus"):
                pozitie = "plus"
            defecte.append({"macaz": macaz, "stare": "blocat", "pozitie": pozitie})
        else:
            print(f"  [!] Stare necunoscuta ignorata: {stare}")

    return defecte


def parseaza_cerere(text: str) -> tuple[str, str] | None:
    """
    Parseaza cererea trenului.
    Formate acceptate: "Y -> LII", "X->L3", "y -> l1" etc.
    """
    text = text.strip()
    cheie = CERERE_ALIAS.get(text.lower())
    if cheie:
        return CERERE_MAP[cheie]

    # Incercam parsare manuala
    text_norm = text.replace(" ", "").upper()
    for alias, cheie in CERERE_ALIAS.items():
        if alias.replace(" ", "").upper() == text_norm:
            return CERERE_MAP[cheie]

    return None


# ---------------------------------------------------------------------------
# Generare stare statie
# ---------------------------------------------------------------------------

def aplica_defecte(G, defecte: list[dict]) -> None:
    """Aplica defectele introduse de utilizator pe graful G."""
    for d in defecte:
        set_stare_macaz(G, d["macaz"], pozitie=d["pozitie"], stare=d["stare"])


# Perechi valide pentru generarea parcursului activ.
# Include doar tipurile operationale reale:
#   Tip 1: capăt->semnal (intrare->iesire)
#   Tip 2: capăt->capăt  (tranzit)
#   Tip 3: semnal->capăt (iesire->intrare)
# Exclus: semnal->semnal (neoperational)
_PERECHI_ACTIVE = (
    [("X", d) for d in ["X1","XII","X3","X4"]]
    + [("Y", d) for d in ["Y1","YII","Y3","Y4"]]
    + [("X","Y"), ("Y","X")]
    + [("Y1","X"), ("YII","X"), ("Y3","X"), ("Y4","X")]
    + [("X1","Y"), ("XII","Y"), ("X3","Y"), ("X4","Y")]
)


def genereaza_parcurs_activ(G, origine_cerere: str, dest_cerere: str) -> tuple[str, list] | None:
    """
    Genereaza aleatoriu un parcurs activ valid infrastructural.
    Nu impune compatibilitate cu cererea — modelul va gestiona asta.
    Foloseste doar tipurile de parcurs operationale (exclus semnal->semnal).
    Returneaza (tren_id, ruta) sau None daca nu exista rute disponibile.
    """
    perechi = list(_PERECHI_ACTIVE)
    random.shuffle(perechi)

    for orig, dest in perechi:
        rute = get_rute_valide(G, orig, dest, {})
        if rute:
            return "T_activ_1", rute[0]

    return None


# ---------------------------------------------------------------------------
# Encoding observatie (identic cu rail_env_theia.py)
# ---------------------------------------------------------------------------

def encode_obs(G, tren_origine, tren_dest, rute_active) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    idx = 0

    for macaz_id in MACAZURI_ORDINE:
        nod = G.nodes.get(macaz_id, {})
        obs[idx]     = POZITIE_ENC.get(nod.get("pozitie", "plus"), 0)
        obs[idx + 1] = STARE_ENC.get(nod.get("stare", "operational"), 0)
        idx += 2

    for seg_id in SEGMENTE_ORDINE:
        nod = G.nodes.get(seg_id, {})
        obs[idx] = SEG_ENC.get(nod.get("stare", "liber"), 0)
        idx += 1

    obs[idx]     = float(ORIGINE_IDX.get(tren_origine, 0))
    obs[idx + 1] = float(DESTINATIE_IDX.get(tren_dest, 0))
    idx += 2

    if rute_active:
        ruta_activa = list(rute_active.values())[0]
        dest_activ  = ruta_activa[-1]
        obs[idx]     = 1.0
        obs[idx + 1] = float(DESTINATIE_IDX.get(dest_activ, 0))
    else:
        obs[idx]     = 0.0
        obs[idx + 1] = 0.0

    return obs


# ---------------------------------------------------------------------------
# Decizia modelului
# ---------------------------------------------------------------------------

def decide_model(model, obs: np.ndarray) -> tuple[int, str]:
    """Obtine actiunea modelului si linia corespunzatoare."""
    actiune, _ = model.predict(obs, deterministic=True)
    actiune    = int(actiune)
    linie      = ACTIUNE_LINIE.get(actiune, "?")
    return actiune, linie


def decide_fara_model(G, origine: str, dest: str, rute_active: dict) -> tuple[int, str]:
    """
    Fallback fara model: cauta ruta optima direct din graf.
    Util pentru testare fara model antrenat.
    """
    linie_ceruta = PARCURS_LINIE.get((origine, dest))
    rute = get_rute_valide(G, origine, dest, rute_active)
    if rute:
        actiune = next(
            (k for k, v in ACTIUNE_LINIE.items() if v == linie_ceruta), 0
        )
        return actiune, linie_ceruta
    return 0, None  # asteapta


def verifica_decizie(G, actiune: int, origine: str, dest: str,
                     rute_active: dict) -> tuple[bool, str, list | None]:
    """
    Verifica daca decizia modelului este realizabila.
    Returneaza (succes, motiv, ruta_aleasa).

    NOTA despre rerutare:
    Modelul alege TIPUL de parcurs (L1, LII, L3, L4, tranzit),
    nu ruta concreta. Daca actiunea corespunde cererii, sistemul
    apeleaza get_rute_valide() care cauta ORICE cale fizica valida
    catre aceeasi destinatie, ocolind elementele defecte.

    Aceasta inseamna ca rerutarea este implicita si corecta:
      - destinatia nu se schimba niciodata
      - daca exista o cale alternativa, este propusa automat
      - daca nu exista nicio cale, se afiseaza INDISPONIBIL
    """
    linie_ceruta  = PARCURS_LINIE.get((origine, dest))
    linie_actiune = ACTIUNE_LINIE.get(actiune)

    if actiune == 0:
        rute = get_rute_valide(G, origine, dest, rute_active)
        if rute:
            return False, "asteapta_gresit", None
        return True, "asteapta_corect", None

    if linie_actiune != linie_ceruta:
        return False, f"actiune_nepotrivita (cerut {linie_ceruta}, ales {linie_actiune})", None

    rute = get_rute_valide(G, origine, dest, rute_active)
    if not rute:
        return False, "ruta_imposibila", None

    return True, "parcurs_realizabil", rute[0]


# ---------------------------------------------------------------------------
# Vizualizare grafica
# ---------------------------------------------------------------------------

def vizualizeaza(
    G,
    ruta_propusa  : list | None,
    ruta_activa   : list | None,
    defecte       : list[dict],
    origine       : str,
    dest          : str,
    titlu         : str = "",
) -> None:
    """
    Deseneaza graful statiei Theia pe 4 linii orizontale clare.

    ROLUL VIZUALIZARII (validare calitativa)
    =========================================
    Aceasta vizualizare este destinata interpretarii deciziilor
    agentului si ilustrarii cazurilor reprezentative operational.
    Nu reprezinta dovada statistica a performantei modelului.
    Dovada statistica se obtine din evaluate_fixed_seeds.py.

    Culori si semnificatia lor operationala:
      - Verde  (#22c55e) : ruta propusa/validata de model
                           → traseul pe care sistemul recomanda
                             sa circule trenul
      - Portocaliu (#f97316) : parcurs activ deja rezervat in statie
                               → tren care ocupa deja o linie,
                                 creand potential conflict
      - Rosu   (#ef4444) : element defect (macaz sau segment)
                           → infrastructura fizic indisponibila
      - Galben (#eab308) : element blocat (macaz in pozitie fixa)
                           → macaz functional dar necomandat
      - Violet (#a855f7) : capete de linie X si Y
      - Albastru (#38bdf8): semnale de intrare/iesire
      - Turcoaz (#34d399): macazuri operationale
      - Gri    (#475569) : segmente de cale neutre
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    pos = {n: POZITII_VIZUALIZARE.get(n, (7.0, 2.0)) for n in G.nodes()}

    # Seturi pentru colorare rapida
    defecte_set = {d["macaz"] for d in defecte if d["stare"] == "defect"}
    blocate_set = {d["macaz"] for d in defecte if d["stare"] == "blocat"}
    ruta_set    = set(ruta_propusa) if ruta_propusa else set()
    activa_set  = set(ruta_activa)  if ruta_activa  else set()

    # Perechi de muchii pentru colorare
    ruta_perechi   = (set(zip(ruta_propusa, ruta_propusa[1:]))
                      | set(zip(ruta_propusa[1:], ruta_propusa))
                      if ruta_propusa else set())
    activa_perechi = (set(zip(ruta_activa, ruta_activa[1:]))
                      | set(zip(ruta_activa[1:], ruta_activa))
                      if ruta_activa else set())

    # ── Linii de referinta orizontale (fundalul liniilor) ─────────────────
    for y, label in [(1.0, "L1"), (2.0, "LII"), (3.0, "L3"), (4.0, "L4")]:
        ax.axhline(y=y, color="#1e293b", linewidth=8, zorder=0, alpha=0.6)
        ax.text(-0.3, y, label, fontsize=8, color="#64748b",
                ha="right", va="center", fontstyle="italic")

    # ── Muchii ────────────────────────────────────────────────────────────
    for u, v in G.edges():
        x0, y0 = pos.get(u, (7, 2))
        x1, y1 = pos.get(v, (7, 2))

        if (u, v) in ruta_perechi or (v, u) in ruta_perechi:
            culoare, lw, alpha, zorder = "#22c55e", 3.0, 1.0, 3
        elif (u, v) in activa_perechi or (v, u) in activa_perechi:
            culoare, lw, alpha, zorder = "#f97316", 2.5, 1.0, 2
        else:
            culoare, lw, alpha, zorder = "#334155", 1.0, 0.8, 1

        ax.plot([x0, x1], [y0, y1], color=culoare, linewidth=lw,
                alpha=alpha, zorder=zorder, solid_capstyle="round")

    # ── Noduri ────────────────────────────────────────────────────────────
    for nod in G.nodes():
        if nod not in pos:
            continue
        x, y = pos[nod]
        tip  = G.nodes[nod].get("tip", "segment")

        # Prioritate culori: defect > blocat > in ruta > in activa > tip
        if nod in defecte_set:
            c, s, ew = "#ef4444", 220, 1.5
        elif nod in blocate_set:
            c, s, ew = "#eab308", 220, 1.5
        elif nod in ruta_set and tip != "segment":
            c, s, ew = "#22c55e", 200, 1.0
        elif nod in activa_set and tip != "segment":
            c, s, ew = "#f97316", 180, 1.0
        elif tip == "capat":
            c, s, ew = "#a855f7", 280, 1.5
        elif tip == "semnal":
            c, s, ew = "#38bdf8", 160, 1.0
        elif tip == "macaz":
            c, s, ew = "#34d399", 180, 1.0
        else:
            # Segmente: colorate daca in ruta, altrfel gri mic
            if nod in ruta_set:
                c, s, ew = "#22c55e", 60, 0.5
            elif nod in activa_set:
                c, s, ew = "#f97316", 50, 0.5
            else:
                c, s, ew = "#475569", 40, 0.3

        ax.scatter(x, y, c=c, s=s, zorder=4,
                   edgecolors="white", linewidths=ew)

    # ── Labels ────────────────────────────────────────────────────────────
    for nod in G.nodes():
        if nod not in pos:
            continue
        x, y = pos[nod]
        tip  = G.nodes[nod].get("tip", "segment")

        if tip == "capat":
            ax.text(x, y + 0.32, nod, fontsize=10, color="#e2e8f0",
                    ha="center", va="bottom", zorder=5, fontweight="bold")
        elif tip == "macaz":
            # Label sub nod pentru macazuri, cu background semi-transparent
            ax.text(x, y - 0.30, nod, fontsize=7, color="#94a3b8",
                    ha="center", va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#0f172a",
                              ec="none", alpha=0.7))
        elif tip == "semnal":
            # Label alternativ sus/jos per linie pentru a evita suprapunerea
            offset = 0.28 if y >= 2.0 else -0.28
            ax.text(x, y + offset, nod, fontsize=6.5, color="#7dd3fc",
                    ha="center", va="center", zorder=5)

    # ── Legenda ───────────────────────────────────────────────────────────
    legende = [
        mpatches.Patch(color="#22c55e", label="Ruta propusa de model"),
        mpatches.Patch(color="#f97316", label="Parcurs activ in statie"),
        mpatches.Patch(color="#ef4444", label="Element defect"),
        mpatches.Patch(color="#eab308", label="Element blocat"),
        mpatches.Patch(color="#a855f7", label="Capat linie (X / Y)"),
        mpatches.Patch(color="#38bdf8", label="Semnal"),
        mpatches.Patch(color="#34d399", label="Macaz"),
    ]
    legend = ax.legend(
        handles=legende, loc="upper left", fontsize=8.5,
        facecolor="#1e293b", labelcolor="#e2e8f0",
        edgecolor="#334155", framealpha=0.95,
        handlelength=1.2, handleheight=1.0,
    )

    ax.set_title(titlu, color="#f1f5f9", fontsize=11, pad=12,
                 fontweight="normal", linespacing=1.6)

    # ── Info parcurs activ — text box dreapta jos ─────────────────────
    if ruta_activa:
        orig_activ = ruta_activa[0]
        dest_activ = ruta_activa[-1]
        linie_activ = PARCURS_LINIE.get((orig_activ, dest_activ), "?")
        info_activ = f"Parcurs activ: {orig_activ} → {dest_activ} ({linie_activ})"
    else:
        info_activ = "Niciun parcurs activ"

    ax.text(
        14.6, 0.55, info_activ,
        fontsize=8, color="#f97316" if ruta_activa else "#64748b",
        ha="right", va="bottom", zorder=6,
        bbox=dict(boxstyle="round,pad=0.4", fc="#1e293b",
                  ec="#f97316" if ruta_activa else "#334155",
                  alpha=0.9, linewidth=1.0),
    )

    ax.set_xlim(-0.8, 14.8)
    ax.set_ylim(0.4, 4.7)
    ax.axis("off")
    plt.tight_layout(pad=1.5)
    plt.show()


# ---------------------------------------------------------------------------
# Afisare stare statie
# ---------------------------------------------------------------------------

def afiseaza_stare(G, defecte, origine, dest, rute_active, parcurs_activ_info):
    print()
    print("=" * 60)
    print("  STARE STATIE THEIA")
    print("=" * 60)

    if defecte:
        print("\n  Defecte infrastructura:")
        for d in defecte:
            if d["stare"] == "defect":
                print(f"    {d['macaz']}: DEFECT")
            else:
                print(f"    {d['macaz']}: BLOCAT pe {d['pozitie'].upper()}")
    else:
        print("\n  Infrastructura: fara defecte")

    print(f"\n  Cerere tren nou: {origine} -> {dest}")
    print(f"  Linie ceruta   : {PARCURS_LINIE.get((origine, dest), '?')}")

    if rute_active and parcurs_activ_info:
        tren_id, ruta = parcurs_activ_info
        orig_activ = ruta[0]
        dest_activ = ruta[-1]
        linie_activ = PARCURS_LINIE.get((orig_activ, dest_activ), "?")
        compat = sunt_compatibile(origine, dest, orig_activ, dest_activ)
        print(f"\n  Parcurs activ  : {orig_activ} -> {dest_activ} ({linie_activ})")
        print(f"  Compatibilitate: {'COMPATIBIL' if compat else 'INCOMPATIBIL'}")
    else:
        print("\n  Parcurs activ  : niciun tren in statie")

    print()


def afiseaza_decizie(succes, motiv, ruta_propusa, origine, dest):
    # Afisare rezultat decizie — parte din validarea calitativa.
    # Interpreteaza decizia agentului in termeni operationali clari.
    print("=" * 60)
    print("  DECIZIE MODEL")
    print("=" * 60)

    if succes and motiv == "parcurs_realizabil":
        linie = PARCURS_LINIE.get((origine, dest), "?")
        print(f"\n  ✓ PARCURS DISPONIBIL — Linia {linie}")
        # Nota metodologica:
        # DQN decide tipul de parcurs (L1/LII/L3/L4/tranzit).
        # Ruta fizica concreta este determinata de validatorul grafic
        # (get_rute_valide), nu de DQN direct.
        # Sistemul propune rerutarea: DQN pentru decizie,
        # validatorul grafic pentru alegerea parcursului fizic.
        # Destinatia nu se schimba niciodata.
        if ruta_propusa:
            noduri_cheie = [n for n in ruta_propusa
                           if G_global.nodes[n].get("tip") in
                           ("capat", "semnal", "macaz")]
            print(f"  Ruta: {' → '.join(noduri_cheie)}")
            config = inferenta_configuratie(G_global, ruta_propusa)
            comutari = [(m, i["pozitie_necesara"])
                        for m, i in config.items() if i["necesita_comutare"]]
            if comutari:
                print(f"  Comutari macazuri: "
                      f"{', '.join(f'{m}→{p}' for m,p in comutari)}")
            else:
                print("  Comutari macazuri: niciuna necesara")

    elif succes and motiv == "asteapta_corect":
        print("\n  ✗ NU EXISTA PARCURS DISPONIBIL")
        print("  Modelul a decis corect sa astepte.")
        print("  Cauza: infrastructura defecta sau conflict cu parcursul activ.")

    elif motiv == "asteapta_gresit":
        print("\n  ⚠ DECIZIE SUBOPTIMA: model a ales sa astepte")
        print("  Exista totusi rute valide disponibile.")

    else:
        print(f"\n  ✗ DECIZIE INCORECTA: {motiv}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Incarcare model
# ---------------------------------------------------------------------------

def incarca_model(tip_model: str):
    """Incearca sa incarce modelul DQN sau PPO."""
    try:
        if tip_model == "dqn":
            from stable_baselines3 import DQN
            cai = [
                "models/dqn_level2_best/best_model.zip",
                "models/dqn_level1_best/best_model.zip",
                "models/dqn_level2.zip",
                "models/dqn_level1.zip",
            ]
            cls = DQN
        else:
            from stable_baselines3 import PPO
            cai = [
                "models/ppo_level2_best/best_model.zip",
                "models/ppo_level1_best/best_model.zip",
                "models/ppo_level2.zip",
                "models/ppo_level1.zip",
            ]
            cls = PPO

        # Cautam modelul in mai multe locatii
        _dirs = [_SCRIPT_DIR, _ROOT_DIR,
                 os.path.join(_SCRIPT_DIR, ".."),
                 os.getcwd()]
        for d in _dirs:
            for cale in cai:
                full = os.path.join(d, cale)
                if os.path.exists(full):
                    model = cls.load(full)
                    print(f"  Model {tip_model.upper()} incarcat: {full}")
                    return model

        print(f"  [!] Niciun model {tip_model.upper()} gasit.")
        print(f"      Ruleaza mai intai: python train_{tip_model}_theia.py --level 1")
        return None

    except ImportError:
        print("  [!] stable-baselines3 nu este instalat.")
        return None
    except Exception as e:
        print(f"  [!] Eroare la incarcarea modelului: {e}")
        return None


# ---------------------------------------------------------------------------
# Variabila globala pentru grafic (necesara in afiseaza_decizie)
# ---------------------------------------------------------------------------
G_global = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global G_global

    parser = argparse.ArgumentParser(
        description="Demo interactiv model RL — Statia Theia"
    )
    parser.add_argument(
        "--model", choices=["dqn", "ppo"], default="dqn",
        help="Modelul RL folosit (default: dqn)"
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Ruleaza fara model antrenat (logica directa din graf)"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   DEMO — Model RL pentru Statia Theia           ║")
    print("║   Validare calitativa a deciziilor agentului    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Incarcare model
    model = None
    if not args.no_model:
        print(f"  Incarcare model {args.model.upper()}...")
        model = incarca_model(args.model)
        if model is None:
            print("  Continuam fara model (mod fallback).\n")

    # Setam cereri si alias-uri in functie de modul de rulare:
    #   cu model RL  → doar cereri cu origine X sau Y (ce a vazut modelul)
    #   fara model   → toate cererile, inclusiv semnal->semnal
    global CERERE_MAP, CERERE_ALIAS
    if args.no_model or model is None:
        CERERE_MAP   = {**CERERI_MODEL_RL, **CERERI_EXTINSE_GRAF}
        CERERE_ALIAS = CERERE_ALIAS_GRAF
        if args.no_model:
            print("  Mod: logica directa din graf "
                  "(toate cererile disponibile)")
    else:
        CERERE_MAP   = CERERI_MODEL_RL
        CERERE_ALIAS = CERERE_ALIAS_RL
        print(f"  Mod: model {args.model.upper()} "
              f"(cereri cu origine X sau Y)")

    # -----------------------------------------------------------------------
    # Pasul 1: Defecte infrastructura
    # -----------------------------------------------------------------------
    print("─" * 52)
    print("  Pas 1 — Defecte infrastructura")
    print("  Macazuri disponibile: M1, M3, M5, M2, M8, DJ46")
    print("  Format: 'M8 defect' | 'M3 blocat minus' | 'nimic'")
    print("─" * 52)
    text_defecte = input("  Defecte: ").strip()
    defecte = parseaza_defecte(text_defecte)

    if defecte:
        print("  Defecte inregistrate:")
        for d in defecte:
            if d["stare"] == "defect":
                print(f"    {d['macaz']}: DEFECT")
            else:
                print(f"    {d['macaz']}: BLOCAT pe {d['pozitie'].upper()}")
    else:
        print("  Infrastructura: fara defecte.")

    # -----------------------------------------------------------------------
    # Pasul 2: Cererea trenului
    # -----------------------------------------------------------------------
    print()
    print("─" * 52)
    print("  Pas 2 — Cerere tren nou")
    if args.no_model or model is None:
        print("  Tip 1 intrare->iesire : X->L1, X->LII, X->L3, X->L4")
        print("                          Y->L1, Y->LII, Y->L3, Y->L4")
        print("  Tip 2 intrare->intrare: X->Y, Y->X  (tranzit)")
        print("  Tip 3 iesire->intrare : Y1->X, YII->X, Y3->X, Y4->X")
        print("                          X1->Y, XII->Y, X3->Y, X4->Y")
    else:
        print("  Tip 1 intrare->iesire : X->L1, X->LII, X->L3, X->L4")
        print("                          Y->L1, Y->LII, Y->L3, Y->L4")
        print("  Tip 2 intrare->intrare: X->Y, Y->X  (tranzit)")
        print("  (Tip 3 iesire->intrare disponibil doar cu --no-model)")
    print("─" * 52)

    cerere = None
    while cerere is None:
        text_cerere = input("  Cerere (ex: Y -> LII): ").strip()
        cerere = parseaza_cerere(text_cerere)
        if cerere is None:
            # Verificam daca utilizatorul a introdus un parcurs Tip 3
            # (iesire->intrare) in modul cu model RL
            if not (args.no_model or model is None):
                # Parsam manual ca sa vedem daca e un parcurs Tip 3 valid
                text_norm = text_cerere.replace(" ", "").upper().replace("->L2", "->LII")
                if text_norm in CERERI_EXTINSE_GRAF:
                    print(f"  [!] Parcursul '{text_cerere}' (Tip 3: iesire->intrare) "
                          f"nu este disponibil cu modelul RL.")
                    print(f"      Modelul a fost antrenat doar pe parcursuri cu origine "
                          f"X sau Y (Tip 1 si Tip 2).")
                    print(f"      Pentru a testa acest tip de parcurs foloseste: "
                          f"python demo_theia.py --no-model")
                else:
                    print("  [!] Format necunoscut. Incearca din nou.")
            else:
                print("  [!] Format necunoscut. Incearca din nou.")

    origine, dest = cerere
    print(f"  Cerere inregistrata: {origine} -> {dest} "
          f"({PARCURS_LINIE.get((origine, dest), '?')})")

    # -----------------------------------------------------------------------
    # Pasul 3: Construire stare statie
    # -----------------------------------------------------------------------
    print()
    print("─" * 52)
    print("  Pas 3 — Generare stare statie...")

    G = build_station()
    G_global = G
    aplica_defecte(G, defecte)

    # Sistemul decide random daca exista parcurs activ
    rute_active      = {}
    parcurs_activ_info = None

    if random.random() < PROB_PARCURS_ACTIV:
        rezultat = genereaza_parcurs_activ(G, origine, dest)
        if rezultat:
            tren_id, ruta_activa = rezultat
            # Rezerva ruta (marcheaza segmentele ca rezervate)
            rezerva_ruta(G, tren_id, ruta_activa, rute_active)
            parcurs_activ_info = (tren_id, ruta_activa)
            orig_activ = ruta_activa[0]
            dest_activ = ruta_activa[-1]
            linie_activ = PARCURS_LINIE.get((orig_activ, dest_activ), "?")
            print(f"  Parcurs activ generat: {orig_activ} -> {dest_activ} "
                  f"({linie_activ})")
        else:
            print("  Niciun parcurs activ posibil cu infrastructura curenta.")
    else:
        print("  Statie libera — niciun tren activ.")

    # -----------------------------------------------------------------------
    # Pasul 4: Decizia modelului
    # -----------------------------------------------------------------------
    print()
    print("─" * 52)
    print("  Pas 4 — Analiza model RL...")

    obs = encode_obs(G, origine, dest, rute_active)

    if model is not None:
        actiune, linie_actiune = decide_model(model, obs)
        print(f"  Actiune aleasa: {actiune} ({ACTIUNE_LINIE.get(actiune, '?')})")
    else:
        actiune, linie_actiune = decide_fara_model(G, origine, dest, rute_active)
        print(f"  Actiune (fallback graf): {actiune} "
              f"({ACTIUNE_LINIE.get(actiune, '?')})")

    succes, motiv, ruta_propusa = verifica_decizie(
        G, actiune, origine, dest, rute_active
    )

    # -----------------------------------------------------------------------
    # Afisare rezultate
    # -----------------------------------------------------------------------
    ruta_activa_lista = (
        list(rute_active.values())[0] if rute_active else None
    )

    afiseaza_stare(G, defecte, origine, dest, rute_active, parcurs_activ_info)
    afiseaza_decizie(succes, motiv, ruta_propusa, origine, dest)

    # -----------------------------------------------------------------------
    # Pasul 5: Vizualizare grafica
    # -----------------------------------------------------------------------
    print()
    print("  Generare vizualizare grafica...")

    linie_ceruta = PARCURS_LINIE.get((origine, dest), "?")
    if ruta_propusa:
        titlu = (f"Statia Theia — {origine}→{dest} ({linie_ceruta}) — "
                 f"PARCURS DISPONIBIL")
    else:
        titlu = (f"Statia Theia — {origine}→{dest} ({linie_ceruta}) — "
                 f"INDISPONIBIL")

    if defecte:
        detalii = ", ".join(
            f"{d['macaz']} {d['stare']}"
            + (f" {d['pozitie']}" if d['stare'] == 'blocat' else "")
            for d in defecte
        )
        titlu += f"\nDefecte: {detalii}"

    vizualizeaza(
        G,
        ruta_propusa   = ruta_propusa,
        ruta_activa    = ruta_activa_lista,
        defecte        = defecte,
        origine        = origine,
        dest           = dest,
        titlu          = titlu,
    )


if __name__ == "__main__":
    main()