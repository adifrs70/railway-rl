"""
scenario_generator.py — Generator de scenarii pentru antrenare si evaluare RL

Pipeline de generare:
  1. Genereaza starea infrastructurii (macazuri + segmente)
  2. Genereaza rute active si le aplica in graf
  3. Genereaza cereri noi de trenuri
  4. Evalueaza scenariul cu graful (validator)
  5. Calculeaza difficulty_score
  6. Clasifica solvabilitatea
  7. Accepta / respinge / regenereaza

Clasificare solvabilitate:
  solvabil_total      — toate cererile au cel putin o ruta valida
  solvabil_partial    — unele cereri au ruta, altele nu
  nesolvabil_interesant — conflict real de resurse, exista partial solutii
  nesolvabil_trivial  — infrastructura blocata, nicio cerere nu poate fi servita

Difficulty score (formula):
  score = a*nr_trenuri + b*nr_macazuri_probleme + c*nr_segmente_probleme
        + d*nr_rute_active + e*grad_conflict + f*deficit_rute_valide
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from graph.theia import (
    build_station,
    get_rute_valide,
    set_stare_macaz,
    set_stare_segment,
    rezerva_ruta,
    inferenta_configuratie,
)


# ---------------------------------------------------------------------------
# Constante si probabilitati
# ---------------------------------------------------------------------------

# Noduri terminale valide ca origine/destinatie
CAPETE      = ["X", "Y"]
SEMNALE_X   = ["Y1", "YII", "Y3", "Y4"]   # semnale cap X (intrare din X)
SEMNALE_Y   = ["X1", "XII", "X3", "X4"]   # semnale cap Y (intrare din Y)

# Corespondenta linie: semnal capX <-> semnal capY
SEMNAL_PERECHE = {
    "Y1": "X1", "YII": "XII", "Y3": "X3", "Y4": "X4",
    "X1": "Y1", "XII": "YII", "X3": "Y3", "X4": "Y4",
}

# Perechi valide (origine, destinatie)
# Tipuri de parcursuri operationale valide:
#   Tip 1: intrare->iesire  — capăt -> semnal      ex: X->X1, Y->YII
#   Tip 2: intrare->intrare — capăt -> capăt        ex: X->Y, Y->X (tranzit)
#   Tip 3: iesire->intrare  — semnal -> capăt       ex: X1->Y, YII->X
#
# EXCLUS: semnal->semnal (ex: XII->YII, Y1->X1) — neoperational:
# un tren nu poate face parcurs intre doua semnale de iesire
# fara sa iasa din statie.
PERECHI_VALIDE = (
    # Tip 1: intrare->iesire (capăt -> semnal)
    [("X", d) for d in SEMNALE_Y]
    + [("Y", d) for d in SEMNALE_X]
    # Tip 2: intrare->intrare (tranzit complet)
    + [("X", "Y"), ("Y", "X")]
    # Tip 3: iesire->intrare (semnal -> capăt)
    + [("Y1","X"), ("YII","X"), ("Y3","X"), ("Y4","X")]
    + [("X1","Y"), ("XII","Y"), ("X3","Y"), ("X4","Y")]
)

# ---------------------------------------------------------------------------
# Tabel de incompatibilitate parcursuri — statia Theia
# Include incompatibilitati de grad 0, precum si incompatibilitati
# rezultate din prelungirea parcursurilor, drumuri de alunecare
# si conditii de protectie operationala.
#
# Parcursuri compatibile prin separare fizica sau protectie (nu apar in set):
#   X->L1  cu  Y->L3, Y->L4
#   X->L3  cu  Y->L1, Y->L4  (Y->L4 compatibil prin linia de evitare DJ46)
#   X->L4  cu  Y->L1, Y->L3  (Y->L3 compatibil prin linia de evitare DJ46)
#   X->Y   cu  Y->X  (doar acelasi tren - tranzit fara oprire)
# ---------------------------------------------------------------------------
PARCURSURI_INCOMPATIBILE = {
    frozenset({"X->L1",  "X->LII"}),
    frozenset({"X->L1",  "X->L3"}),
    frozenset({"X->L1",  "X->L4"}),
    frozenset({"X->L1",  "X->Y"}),
    frozenset({"X->LII", "X->L3"}),
    frozenset({"X->LII", "X->L4"}),
    frozenset({"X->LII", "X->Y"}),
    frozenset({"X->LII", "Y->LII"}),
    frozenset({"X->L3",  "X->L4"}),
    frozenset({"X->L3",  "X->Y"}),
    frozenset({"X->L3",  "Y->L3"}),
    # X->L3 + Y->L4: compatibile prin linia de evitare de la DJ46
    # (L3 si L4 sunt pe ramuri separate ale DJ46, fara segment comun)
    frozenset({"X->L4",  "X->Y"}),
    # X->L4 + Y->L3: compatibile prin linia de evitare de la DJ46
    frozenset({"X->L4",  "Y->L4"}),
    frozenset({"Y->L1",  "Y->LII"}),
    frozenset({"Y->L1",  "Y->L3"}),
    frozenset({"Y->L1",  "Y->L4"}),
    frozenset({"Y->L1",  "Y->X"}),
    frozenset({"Y->LII", "Y->L3"}),
    frozenset({"Y->LII", "Y->L4"}),
    frozenset({"Y->LII", "Y->X"}),
    frozenset({"Y->L3",  "Y->L4"}),
    frozenset({"Y->L3",  "Y->X"}),
    frozenset({"Y->L4",  "Y->X"}),
    frozenset({"X->Y",   "Y->L1"}),
    frozenset({"X->Y",   "Y->L3"}),
    frozenset({"X->Y",   "Y->L4"}),
    frozenset({"Y->X",   "Y->L1"}),
    frozenset({"Y->X",   "Y->L3"}),
    frozenset({"Y->X",   "Y->L4"}),
    # X->Y si Y->X sunt incompatibile daca sunt trenuri DIFERITE
    # (trece prin aceleasi segmente in sens opus)
    frozenset({"X->Y",   "Y->X"}),
    # -----------
    # Perechi adaugate dupa verificarea sistematica a grafului.
    # Incompatibile grad 0 (segment fizic comun) sau prin
    # prelungire / drum de alunecare / tranzit complet.
    # -----------
    # X->L1 + Y->L1: impart SEG_L1, SEG_Y1_L1, SEG_L1_X1 (toata linia 1)
    frozenset({"X->L1",  "Y->L1"}),
    # X->Y tranzit + Y->LII: impart SEG_LII, SEG_M8_LII, SEG_M2_M8, M2, M8
    frozenset({"X->Y",   "Y->LII"}),
    # Y->X tranzit impart SEG_X_M1 si M1 cu toate parcursurile din capatul X
    frozenset({"X->L1",  "Y->X"}),
    frozenset({"X->LII", "Y->X"}),
    frozenset({"X->L3",  "Y->X"}),
    frozenset({"X->L4",  "Y->X"}),
    # -----------
    # Perechi incompatibile pentru parcursuri de iesire si transfer
    # (semnal->capat, semnal->semnal) — verificate sistematic din graf
    # -----------
    frozenset({"X->L1",   "Y1->X"}),    frozenset({"X->L1",   "Y1->X1"}),
    frozenset({"X->L1",   "YII->X"}),   frozenset({"X->L3",   "Y1->X"}),
    frozenset({"X->L3",   "Y3->X"}),    frozenset({"X->L3",   "Y3->X3"}),
    frozenset({"X->L3",   "Y4->X"}),    frozenset({"X->L3",   "YII->X"}),
    frozenset({"X->L4",   "Y1->X"}),    frozenset({"X->L4",   "Y3->X"}),
    frozenset({"X->L4",   "Y4->X"}),    frozenset({"X->L4",   "Y4->X4"}),
    frozenset({"X->L4",   "YII->X"}),   frozenset({"X->LII",  "XII->YII"}),
    frozenset({"X->LII",  "Y1->X"}),    frozenset({"X->LII",  "Y3->X"}),
    frozenset({"X->LII",  "Y4->X"}),    frozenset({"X->LII",  "YII->X"}),
    frozenset({"X->LII",  "YII->XII"}),
    frozenset({"X->Y",    "X1->Y"}),    frozenset({"X->Y",    "X3->Y"}),
    frozenset({"X->Y",    "X4->Y"}),    frozenset({"X->Y",    "XII->Y"}),
    frozenset({"X->Y",    "XII->YII"}), frozenset({"X->Y",    "Y1->X"}),
    frozenset({"X->Y",    "Y3->X"}),    frozenset({"X->Y",    "Y4->X"}),
    frozenset({"X->Y",    "YII->X"}),   frozenset({"X->Y",    "YII->XII"}),
    frozenset({"X1->Y",   "X3->Y"}),    frozenset({"X1->Y",   "X4->Y"}),
    frozenset({"X1->Y",   "XII->Y"}),   frozenset({"X1->Y",   "Y->L1"}),
    frozenset({"X1->Y",   "Y->L3"}),    frozenset({"X1->Y",   "Y->L4"}),
    frozenset({"X1->Y",   "Y->LII"}),   frozenset({"X1->Y",   "Y->X"}),
    frozenset({"X->L1",   "X1->Y1"}),   frozenset({"X1->Y1",  "Y->L1"}),
    frozenset({"X1->Y1",  "Y1->X1"}),
    frozenset({"X->L3",   "X3->Y3"}),   frozenset({"X3->Y3",  "Y->L3"}),
    frozenset({"X3->Y3",  "Y3->X3"}),
    frozenset({"X->L4",   "X4->Y4"}),   frozenset({"X4->Y4",  "Y->L4"}),
    frozenset({"X4->Y4",  "Y4->X4"}),
    frozenset({"X->LII",  "YII->XII"}),
    frozenset({"XII->YII","YII->XII"}), frozenset({"Y->LII",  "YII->XII"}),
    frozenset({"Y->X",    "YII->XII"}),
    frozenset({"X3->Y",   "X4->Y"}),    frozenset({"X3->Y",   "XII->Y"}),
    frozenset({"X4->Y",   "XII->Y"}),   frozenset({"X4->Y",   "Y->L4"}),
    frozenset({"X4->Y",   "Y->LII"}),   frozenset({"X4->Y",   "Y->L1"}),
    frozenset({"X4->Y",   "Y->L3"}),    frozenset({"X4->Y",   "Y->X"}),
    frozenset({"XII->Y",  "Y->LII"}),   frozenset({"XII->Y",  "Y->L1"}),
    frozenset({"XII->Y",  "Y->L3"}),    frozenset({"XII->Y",  "Y->L4"}),
    frozenset({"XII->Y",  "Y->X"}),     frozenset({"XII->Y",  "YII->X"}),
    frozenset({"XII->Y",  "Y1->X"}),
    frozenset({"X3->Y",   "Y->L1"}),    frozenset({"X3->Y",   "Y->L3"}),
    frozenset({"X3->Y",   "Y->L4"}),    frozenset({"X3->Y",   "Y->LII"}),
    frozenset({"X3->Y",   "Y->X"}),     frozenset({"X3->Y",   "Y1->X"}),
    frozenset({"X3->Y",   "Y3->X"}),    frozenset({"X3->Y3",  "Y->LII"}),
    frozenset({"X3->Y3",  "Y->L4"}),
    frozenset({"XII->YII","Y->LII"}),   frozenset({"XII->YII","Y->X"}),
    frozenset({"XII->YII","Y->L3"}),    frozenset({"XII->YII","Y->L4"}),
    frozenset({"XII->YII","Y->L1"}),
    frozenset({"Y->X",    "Y1->X"}),    frozenset({"Y->X",    "Y3->X"}),
    frozenset({"Y->X",    "Y4->X"}),    frozenset({"Y->X",    "YII->X"}),
    frozenset({"Y1->X",   "Y3->X"}),    frozenset({"Y1->X",   "Y4->X"}),
    frozenset({"Y1->X",   "YII->X"}),   frozenset({"Y3->X",   "Y4->X"}),
    frozenset({"Y3->X",   "YII->X"}),   frozenset({"Y4->X",   "YII->X"}),
    frozenset({"Y->L1",   "Y1->X1"}),   frozenset({"Y->L3",   "Y3->X3"}),
    frozenset({"Y->L4",   "Y4->X4"}),
    frozenset({"Y->L1",   "Y1->X"}),    frozenset({"Y->L3",   "Y1->X"}),
    frozenset({"Y->L4",   "Y1->X"}),    frozenset({"Y->LII",  "Y1->X"}),
    frozenset({"Y->L4",   "Y4->X"}),    frozenset({"Y->L3",   "Y3->X"}),
}

_CHEIE_PARCURS = {
    # Intrari din capete
    ("X", "X1") : "X->L1",  ("X", "XII"): "X->LII",
    ("X", "X3") : "X->L3",  ("X", "X4") : "X->L4",
    ("Y", "Y1") : "Y->L1",  ("Y", "YII"): "Y->LII",
    ("Y", "Y3") : "Y->L3",  ("Y", "Y4") : "Y->L4",
    # Tranzit complet
    ("X", "Y")  : "X->Y",   ("Y", "X")  : "Y->X",
    # Iesiri spre capatul opus
    ("Y1","X")  : "Y1->X",  ("YII","X") : "YII->X",
    ("Y3","X")  : "Y3->X",  ("Y4","X")  : "Y4->X",
    ("X1","Y")  : "X1->Y",  ("XII","Y") : "XII->Y",
    ("X3","Y")  : "X3->Y",  ("X4","Y")  : "X4->Y",
    # Tip 3: iesire->intrare (semnal -> capăt)
}


def parcurs_la_cheie(origine, destinatie):
    return _CHEIE_PARCURS.get((origine, destinatie), f"{origine}->{destinatie}")


def sunt_compatibile(origine_a, destinatie_a, origine_b, destinatie_b,
                     acelasi_tren=False):
    cheie_a = parcurs_la_cheie(origine_a, destinatie_a)
    cheie_b = parcurs_la_cheie(origine_b, destinatie_b)
    # Doua trenuri diferite nu pot primi simultan acelasi parcurs.
    if cheie_a == cheie_b:
        return False
    pereche = frozenset({cheie_a, cheie_b})
    if pereche == frozenset({"X->Y", "Y->X"}) and acelasi_tren:
        return True
    return pereche not in PARCURSURI_INCOMPATIBILE

# Probabilitati stare macaz
PROB_MACAZ = {
    "operational": 0.80,
    "blocat":      0.15,
    "defect":      0.05,
}

# Probabilitati stare segment
PROB_SEGMENT = {
    "liber":        0.75,
    "rezervat":     0.15,
    "ocupat":       0.08,
    "indisponibil": 0.02,
}

# Pondere difficulty score
WEIGHT_TRENURI          = 1.5
WEIGHT_MACAZURI_PROB    = 1.0
WEIGHT_SEGMENTE_PROB    = 0.8
WEIGHT_RUTE_ACTIVE      = 1.2
WEIGHT_CONFLICT         = 2.0
WEIGHT_DEFICIT_RUTE     = 1.8

# Limite regenerare
MAX_REGENERARI = 20

# Profile per nivel de dificultate
# Constrangere globala: maxim 2 parcursuri simultane in statie
#   (maxim 1 ruta activa + maxim 1 cerere noua)
# Aceasta reflecta regula de incompatibilitate feroviara:
# doua parcursuri simultane trebuie verificate explicit.
PROFILE_DIFICULTATE = {
    1: {
        "nr_trenuri_noi"       : (1, 1),
        "nr_macazuri_probleme" : (0, 1),
        "nr_segmente_probleme" : (0, 1),
        "nr_rute_active"       : (0, 0),  # nivel 1: fara trafic activ
        "prob_solvabil_target" : 0.90,
    },
    2: {
        "nr_trenuri_noi"       : (1, 1),  # maxim 1 cerere noua
        "nr_macazuri_probleme" : (0, 2),
        "nr_segmente_probleme" : (0, 2),
        "nr_rute_active"       : (0, 1),  # maxim 1 ruta activa
        "prob_solvabil_target" : 0.75,
    },
    3: {
        "nr_trenuri_noi"       : (1, 1),  # maxim 1 cerere noua
        "nr_macazuri_probleme" : (0, 2),
        "nr_segmente_probleme" : (0, 2),
        "nr_rute_active"       : (1, 1),  # exact 1 ruta activa la nivel 3
        "prob_solvabil_target" : 0.50,
    },
}


# ---------------------------------------------------------------------------
# Structuri de date
# ---------------------------------------------------------------------------

@dataclass
class TrenCerere:
    id: str
    origine: str
    destinatie: str


@dataclass
class MacazProblema:
    macaz: str
    stare: str           # "blocat" sau "defect"
    pozitie: str = "plus"  # relevant doar pentru "blocat"


@dataclass
class SegmentProblema:
    segment: str
    stare: str           # "rezervat", "ocupat", "indisponibil"
    tren: Optional[str] = None


@dataclass
class ScenarioState:
    """
    Starea completa a unui scenariu generat.
    Toate campurile calculate sunt derivate din graf dupa generare,
    nu presupuse din parametrii de intrare.
    """
    trenuri_noi           : list[TrenCerere]
    macazuri_cu_probleme  : list[MacazProblema]
    segmente_cu_probleme  : list[SegmentProblema]
    rute_active           : dict[str, list[str]]

    # Campuri calculate post-generare
    difficulty_level_target : int
    difficulty_score         : float
    solvabil                 : bool
    solvabilitate_tip        : str   # solvabil_total / solvabil_partial /
                                     # nesolvabil_interesant / nesolvabil_trivial
    rute_valide_per_tren     : dict[str, list[list[str]]]

    # Graful cu starea aplicata (pentru mediul RL)
    graf                     : nx.DiGraph = field(repr=False)


# ---------------------------------------------------------------------------
# Generare infrastructura
# ---------------------------------------------------------------------------

def _alege_ponderat(optiuni: dict) -> str:
    """Alege o cheie din dict cu valorile ca probabilitati."""
    r = random.random()
    cumul = 0.0
    for cheie, prob in optiuni.items():
        cumul += prob
        if r <= cumul:
            return cheie
    return list(optiuni.keys())[-1]


def _genereaza_stare_infrastructura(
    G: nx.DiGraph,
    nr_macazuri_probleme: int,
    nr_segmente_probleme: int,
) -> tuple[list[MacazProblema], list[SegmentProblema]]:
    """
    Aplica probleme de infrastructura pe graful G.
    Garanteaza ca cel putin un macaz per capat ramane operational.
    """
    macazuri_probleme  = []
    segmente_probleme  = []

    # Macazuri disponibile pentru probleme
    # Nu permitem ambii macazuri de la acelasi capat sa fie defecti simultan
    macazuri_cap_x = ["M1", "M3", "M5"]
    macazuri_cap_y = ["M2", "M8", "DJ46"]
    macazuri_candidati = macazuri_cap_x + macazuri_cap_y
    random.shuffle(macazuri_candidati)

    defecte_cap_x = 0
    defecte_cap_y = 0

    for macaz_id in macazuri_candidati[:nr_macazuri_probleme]:
        # Evitam sa defectam M1 si M3 simultan (blocare totala cap X)
        # sau M2 si M8 simultan (blocare totala cap Y)
        stare = "defect" if random.random() < 0.30 else "blocat"

        if stare == "defect":
            if macaz_id in macazuri_cap_x:
                if defecte_cap_x >= 1:
                    stare = "blocat"  # degradam la blocat in loc de defect
                else:
                    defecte_cap_x += 1
            else:
                if defecte_cap_y >= 1:
                    stare = "blocat"
                else:
                    defecte_cap_y += 1

        pozitie = random.choice(["plus", "minus"]) if stare == "blocat" else "plus"
        set_stare_macaz(G, macaz_id, pozitie=pozitie, stare=stare)
        macazuri_probleme.append(MacazProblema(macaz_id, stare, pozitie))

    # Segmente disponibile pentru probleme
    # Excludem segmentele critice de intrare/iesire
    segmente_excluse = {"SEG_X_M1", "SEG_Y_M2"}
    segmente_candidati = [
        nod for nod, date in G.nodes(data=True)
        if date.get("tip") == "segment"
        and nod not in segmente_excluse
    ]
    random.shuffle(segmente_candidati)

    for seg_id in segmente_candidati[:nr_segmente_probleme]:
        stare = _alege_ponderat({
            "rezervat":     0.55,
            "ocupat":       0.35,
            "indisponibil": 0.10,
        })
        tren = f"T_extern_{random.randint(0, 99)}" if stare != "indisponibil" else None
        set_stare_segment(G, seg_id, stare=stare, tren=tren)
        segmente_probleme.append(SegmentProblema(seg_id, stare, tren))

    return macazuri_probleme, segmente_probleme


# ---------------------------------------------------------------------------
# Generare rute active
# ---------------------------------------------------------------------------

def _genereaza_rute_active(
    G: nx.DiGraph,
    nr_rute: int,
    rute_active: dict,
) -> None:
    """
    Genereaza rute deja active in statie si le aplica in graf.
    Rutele active sunt generate DUPA starea infrastructurii,
    garantand consistenta cu graful curent.
    Limita: maxim 1 ruta activa (regula: maxim 2 parcursuri simultane).
    Compatibilitatea cu cererea noua este verificata separat,
    in _genereaza_trenuri_noi_compatibili().
    """
    if nr_rute == 0:
        return

    perechi = list(PERECHI_VALIDE)
    random.shuffle(perechi)

    tren_idx = 0
    for orig, dest in perechi:
        if len(rute_active) >= min(nr_rute, 1):  # maxim 1 ruta activa
            break

        rute = get_rute_valide(G, orig, dest, rute_active)
        if not rute:
            continue

        tren_id = f"T_activ_{tren_idx}"
        rezerva_ruta(G, tren_id, rute[0], rute_active)
        tren_idx += 1


# ---------------------------------------------------------------------------
# Generare cereri noi
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluare scenariu
# ---------------------------------------------------------------------------

def _este_compatibil_cu_rutele_active(
    origine: str,
    destinatie: str,
    rute_active: dict,
) -> bool:
    """
    Verifica daca un parcurs nou este compatibil cu toate rutele active.
    Un parcurs e incompatibil daca formeaza o pereche interzisa
    cu oricare dintre rutele deja rezervate in statie.
    """
    for ruta_activa in rute_active.values():
        orig_activ = ruta_activa[0]
        dest_activ = ruta_activa[-1]
        if not sunt_compatibile(origine, destinatie, orig_activ, dest_activ):
            return False
    return True


# Grupuri de perechi per tip de actiune — pentru sampling balansat.
# Garanteaza ca agentul vede fiecare tip de actiune cu frecventa egala
# in antrenament, evitand dominanta actiunilor tranzit_capat/tranzit_semnal.
_PERECHI_PER_ACTIUNE = {
    "L1"             : [("X","X1"),  ("Y","Y1")],
    "LII"            : [("X","XII"), ("Y","YII")],
    "L3"             : [("X","X3"),  ("Y","Y3")],
    "L4"             : [("X","X4"),  ("Y","Y4")],
    "tranzit_capat"  : [("X","Y"),   ("Y","X"),
                        ("Y1","X"),  ("YII","X"), ("Y3","X"),  ("Y4","X"),
                        ("X1","Y"),  ("XII","Y"), ("X3","Y"),  ("X4","Y")],
    # tranzit_semnal (semnal->semnal) eliminat — neoperational
}
_TIPURI_ACTIUNE = list(_PERECHI_PER_ACTIUNE.keys())


def _genereaza_trenuri_noi_compatibili(
    nr_trenuri: int,
    rute_active: dict,
) -> list:
    """
    Genereaza cereri noi compatibile cu rutele active.
    Foloseste sampling balansat per tip de actiune: alege aleatoriu
    tipul de actiune (L1/LII/L3/L4/tranzit_capat/tranzit_semnal) si
    apoi o pereche din grupul respectiv. Aceasta garanteaza ca agentul
    vede fiecare tip de actiune cu frecventa aproximativ egala in
    antrenament, evitand dominanta actiunilor tranzit.

    Returneaza lista goala daca nicio cerere compatibila nu exista
    — generate_scenario va regenera intregul scenariu.
    """
    trenuri = []
    tipuri_incercate = set()

    # Incercam max 3x numarul de tipuri pentru a gasi o cerere compatibila
    for _ in range(len(_TIPURI_ACTIUNE) * 3):
        if len(trenuri) >= nr_trenuri:
            break

        # Alegem aleatoriu un tip de actiune
        tip = random.choice(_TIPURI_ACTIUNE)
        perechi_tip = _PERECHI_PER_ACTIUNE[tip]

        # Alegem aleatoriu o pereche din grupul tipului
        random.shuffle(perechi_tip)
        for orig, dest in perechi_tip:
            if not _este_compatibil_cu_rutele_active(orig, dest, rute_active):
                continue
            trenuri.append(TrenCerere(
                id         = f"T{len(trenuri)+1}",
                origine    = orig,
                destinatie = dest,
            ))
            break

    # Fallback: daca sampling-ul balansat nu a gasit nimic,
    # incercam toate perechile in ordine aleatorie
    if not trenuri:
        perechi = list(PERECHI_VALIDE)
        random.shuffle(perechi)
        for orig, dest in perechi:
            if len(trenuri) >= nr_trenuri:
                break
            if not _este_compatibil_cu_rutele_active(orig, dest, rute_active):
                continue
            trenuri.append(TrenCerere(
                id         = f"T{len(trenuri)+1}",
                origine    = orig,
                destinatie = dest,
            ))

    return trenuri


def _evalueaza_scenariu(
    G: nx.DiGraph,
    trenuri_noi: list[TrenCerere],
    rute_active: dict,
) -> dict[str, list[list[str]]]:
    """
    Calculeaza rutele valide pentru fiecare tren nou,
    tinand cont de starea curenta a grafului si rutele active.
    """
    rute_valide_per_tren = {}
    rute_simulate = dict(rute_active)  # copie pentru simulare

    for tren in trenuri_noi:
        rute = get_rute_valide(G, tren.origine, tren.destinatie, rute_simulate)
        rute_valide_per_tren[tren.id] = rute

    return rute_valide_per_tren


def _calculeaza_grad_conflict(
    rute_valide_per_tren: dict[str, list[list[str]]],
    trenuri_noi: list[TrenCerere],
) -> float:
    """
    Calculeaza gradul mediu de conflict intre trenuri.
    0.0 = niciun conflict, 1.0 = toate cererile in conflict total.
    """
    if len(trenuri_noi) < 2:
        return 0.0

    total_perechi  = 0
    perechi_conflict = 0

    rute_list = list(rute_valide_per_tren.values())
    for i in range(len(rute_list)):
        for j in range(i + 1, len(rute_list)):
            total_perechi += 1
            # Conflict = niciunul din cele doua nu are ruta simultana
            if not rute_list[i] or not rute_list[j]:
                perechi_conflict += 1

    return perechi_conflict / total_perechi if total_perechi > 0 else 0.0


def _calculeaza_difficulty_score(
    trenuri_noi           : list[TrenCerere],
    macazuri_cu_probleme  : list[MacazProblema],
    segmente_cu_probleme  : list[SegmentProblema],
    rute_active           : dict,
    rute_valide_per_tren  : dict[str, list[list[str]]],
) -> float:
    """
    Calculeaza scorul de dificultate real al scenariului.
    Bazat pe caracteristici masurate, nu pe parametri de intrare.
    """
    nr_trenuri         = len(trenuri_noi)
    nr_macazuri_prob   = len(macazuri_cu_probleme)
    nr_segmente_prob   = len(segmente_cu_probleme)
    nr_rute_active     = len(rute_active)
    grad_conflict      = _calculeaza_grad_conflict(
        rute_valide_per_tren, trenuri_noi
    )

    # Deficit de rute: procentul de trenuri fara nicio ruta valida
    nr_fara_ruta  = sum(
        1 for rute in rute_valide_per_tren.values() if not rute
    )
    deficit_rute = nr_fara_ruta / nr_trenuri if nr_trenuri > 0 else 0.0

    score = (
        WEIGHT_TRENURI       * nr_trenuri
        + WEIGHT_MACAZURI_PROB * nr_macazuri_prob
        + WEIGHT_SEGMENTE_PROB * nr_segmente_prob
        + WEIGHT_RUTE_ACTIVE   * nr_rute_active
        + WEIGHT_CONFLICT      * grad_conflict
        + WEIGHT_DEFICIT_RUTE  * deficit_rute * nr_trenuri
    )

    return round(score, 2)


def _clasifica_solvabilitate(
    trenuri_noi          : list[TrenCerere],
    rute_valide_per_tren : dict[str, list[list[str]]],
    macazuri_cu_probleme : list[MacazProblema],
    rute_active          : dict,
) -> tuple[bool, str]:
    """
    Clasifica scenariul in una din categoriile:
      solvabil_total        — cererea are cel putin o ruta valida
      nesolvabil_interesant — blocaj din conflict cu trafic activ
                              sau segment ocupat/rezervat
      nesolvabil_trivial    — macaz defect la capatul de intrare
                              al trenului (infrastructura complet blocata)

    In configuratia curenta (maxim un tren nou), solvabil_partial
    nu apare practic. Categoria este pastrata pentru extensii viitoare.
    Returneaza (solvabil: bool, tip: str).
    """
    nr_cu_ruta = sum(1 for r in rute_valide_per_tren.values() if r)

    if nr_cu_ruta == len(trenuri_noi):
        return True, "solvabil_total"

    # Nicio ruta — trivial sau interesant?
    # Trivial: macaz defect chiar la capatul de intrare al trenului,
    # fara nicio posibilitate fizica de acces — agentul nu poate face nimic.
    defecte = {m.macaz for m in macazuri_cu_probleme if m.stare == "defect"}
    origini = {t.origine for t in trenuri_noi}

    blocat_trivial = (
        ("X" in origini and "M1" in defecte) or
        ("Y" in origini and "M2" in defecte)
    )

    # Daca infrastructura e blocata SI nu exista trafic activ care sa
    # fi cauzat problema, e trivial — agentul nu invata nimic util.
    if blocat_trivial and not rute_active:
        return False, "nesolvabil_trivial"

    # Altfel: blocaj interesant — conflict cu trafic activ, segment ocupat,
    # sau macaz blocat care poate fi circumscris prin logica agentului.
    return False, "nesolvabil_interesant"


# ---------------------------------------------------------------------------
# Generatorul principal
# ---------------------------------------------------------------------------

def generate_scenario(
    difficulty_level : int = 1,
    mode             : str = "antrenare",
    seed             : int = None,
) -> ScenarioState:
    """
    Genereaza un scenariu complet pentru antrenare sau evaluare RL.

    Parametri:
      difficulty_level : 1 (simplu), 2 (mediu), 3 (dificil)
      mode             : "antrenare" sau "evaluare"
                         antrenare  -> filtreaza nesolvabil_trivial
                         evaluare   -> accepta orice scenariu
      seed             : seed pentru reproductibilitate (None = aleatoriu)

    Returneaza ScenarioState cu toate campurile calculate.
    Regenereaza automat daca scenariul nu respecta criteriile modului.
    """
    if seed is not None:
        random.seed(seed)

    profil = PROFILE_DIFICULTATE.get(difficulty_level, PROFILE_DIFICULTATE[1])

    for tentativa in range(MAX_REGENERARI):
        G            = build_station()
        rute_active  = {}

        # Pasul 1: infrastructura
        nr_mac_prob  = random.randint(*profil["nr_macazuri_probleme"])
        nr_seg_prob  = random.randint(*profil["nr_segmente_probleme"])
        macazuri_prob, segmente_prob = _genereaza_stare_infrastructura(
            G, nr_mac_prob, nr_seg_prob
        )

        # Pasul 2: rute active (dupa infrastructura, nu inainte)
        nr_rute_active = random.randint(*profil["nr_rute_active"])
        _genereaza_rute_active(G, nr_rute_active, rute_active)

        # Pasul 3: cereri noi — compatibile cu rutele active
        nr_trenuri = random.randint(*profil["nr_trenuri_noi"])
        trenuri_noi = _genereaza_trenuri_noi_compatibili(nr_trenuri, rute_active)

        # Daca nicio cerere compatibila nu exista, regeneram scenariul
        if not trenuri_noi:
            continue

        # Pasul 4: evaluare cu graful
        rute_valide = _evalueaza_scenariu(G, trenuri_noi, rute_active)

        # Pasul 5: difficulty score
        diff_score = _calculeaza_difficulty_score(
            trenuri_noi, macazuri_prob, segmente_prob,
            rute_active, rute_valide
        )

        # Pasul 6: clasificare solvabilitate
        solvabil, tip_solv = _clasifica_solvabilitate(
            trenuri_noi, rute_valide, macazuri_prob, rute_active
        )

        # Pasul 7: accept / respinge
        if mode == "antrenare" and tip_solv == "nesolvabil_trivial":
            continue  # regenereaza — nu e util pentru invatare

        # Acceptat
        return ScenarioState(
            trenuri_noi            = trenuri_noi,
            macazuri_cu_probleme   = macazuri_prob,
            segmente_cu_probleme   = segmente_prob,
            rute_active            = rute_active,
            difficulty_level_target= difficulty_level,
            difficulty_score       = diff_score,
            solvabil               = solvabil,
            solvabilitate_tip      = tip_solv,
            rute_valide_per_tren   = rute_valide,
            graf                   = G,
        )

    # Dupa MAX_REGENERARI tentative, returnam ce avem
    # (poate fi nesolvabil_trivial, dar e mai bine decat loop infinit)
    return ScenarioState(
        trenuri_noi            = trenuri_noi,
        macazuri_cu_probleme   = macazuri_prob,
        segmente_cu_probleme   = segmente_prob,
        rute_active            = rute_active,
        difficulty_level_target= difficulty_level,
        difficulty_score       = diff_score,
        solvabil               = solvabil,
        solvabilitate_tip      = tip_solv,
        rute_valide_per_tren   = rute_valide,
        graf                   = G,
    )


# ---------------------------------------------------------------------------
# Vizualizare textuala
# ---------------------------------------------------------------------------

def display_scenario(scenariu: ScenarioState) -> None:
    """
    Afiseaza scenariul intr-un format clar pentru debugging si prezentare.
    """
    print("=" * 65)
    print(f"SCENARIU GENERAT")
    print(f"  Nivel tinta    : {scenariu.difficulty_level_target}")
    print(f"  Difficulty score: {scenariu.difficulty_score:.2f}")
    print(f"  Solvabilitate  : {scenariu.solvabilitate_tip.upper()}")
    print("=" * 65)

    # Macazuri cu probleme
    if scenariu.macazuri_cu_probleme:
        print("\nMacazuri cu probleme:")
        for m in scenariu.macazuri_cu_probleme:
            if m.stare == "defect":
                print(f"  {m.macaz}: DEFECT")
            else:
                print(f"  {m.macaz}: BLOCAT pe {m.pozitie.upper()}")
    else:
        print("\nMacazuri: toate operationale")

    # Segmente cu probleme
    if scenariu.segmente_cu_probleme:
        print("\nSegmente cu restrictii:")
        for s in scenariu.segmente_cu_probleme:
            tren_str = f" ({s.tren})" if s.tren else ""
            print(f"  {s.segment}: {s.stare.upper()}{tren_str}")
    else:
        print("\nSegmente: toate libere")

    # Rute active
    if scenariu.rute_active:
        print("\nTrafic activ (rute rezervate):")
        for tren_id, ruta in scenariu.rute_active.items():
            print(f"  {tren_id}: {' -> '.join(ruta)}")
    else:
        print("\nTrafic activ: niciun tren")

    # Cereri noi
    print(f"\nCereri noi ({len(scenariu.trenuri_noi)} trenuri):")
    for tren in scenariu.trenuri_noi:
        rute = scenariu.rute_valide_per_tren.get(tren.id, [])
        nr_rute = len(rute)
        status = f"{nr_rute} rute disponibile" if nr_rute > 0 else "NICIO RUTA"
        print(f"  {tren.id}: {tren.origine} -> {tren.destinatie}  [{status}]")
        if rute:
            ruta_best = rute[0]
            config = inferenta_configuratie(scenariu.graf, ruta_best)
            comutari = [
                f"{m}({info['pozitie_necesara']})"
                for m, info in config.items()
                if info["necesita_comutare"]
            ]
            if comutari:
                print(f"    Ruta optima: {' -> '.join(ruta_best)}")
                print(f"    Comutari necesare: {', '.join(comutari)}")
            else:
                print(f"    Ruta optima: {' -> '.join(ruta_best)}")
                print(f"    Comutari necesare: niciuna")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Statistici agregate (pentru validarea generatorului)
# ---------------------------------------------------------------------------

def generate_statistics(
    n_scenarii       : int = 1000,
    difficulty_level : int = 1,
    mode             : str = "antrenare",
) -> dict:
    """
    Genereaza N scenarii si returneaza statistici de distributie.
    Util pentru validarea generatorului si calibrarea probabilitatilor.
    """
    # Frecvente distributie origine/destinatie
    _toate_originile  = ["X", "Y"]
    _toate_dest_x     = ["X1", "XII", "X3", "X4", "Y"]
    _toate_dest_y     = ["Y1", "YII", "Y3", "Y4", "X"]
    _toate_dest       = list(set(_toate_dest_x + _toate_dest_y))
    _toate_macazuri   = ["M1", "M3", "M5", "M2", "M8", "DJ46"]
    _toate_segmente   = [
        "SEG_X_M1", "SEG_M1_M3", "SEG_M3_M5", "SEG_M1_L1",
        "SEG_M3_LII", "SEG_M5_L3", "SEG_M5_L4",
        "SEG_Y1_L1", "SEG_YII_LII", "SEG_Y3_L3", "SEG_Y4_L4",
        "SEG_L1", "SEG_LII", "SEG_L3", "SEG_L4",
        "SEG_L1_X1", "SEG_LII_XII", "SEG_L3_X3", "SEG_L4_X4",
        "SEG_M8_L1", "SEG_M8_LII", "SEG_DJ46_L3", "SEG_DJ46_L4",
        "SEG_M2_M8", "SEG_M2_DJ46", "SEG_Y_M2",
    ]

    stats = {
        "total"                  : n_scenarii,
        "difficulty_level"       : difficulty_level,
        "mode"                   : mode,
        "solvabilitate"          : {
            "solvabil_total"       : 0,
            "solvabil_partial"     : 0,
            "nesolvabil_interesant": 0,
            "nesolvabil_trivial"   : 0,
        },
        "difficulty_score"       : [],
        "nr_trenuri"             : [],
        "nr_macazuri_probleme"   : [],
        "nr_segmente_probleme"   : [],
        "nr_rute_active"         : [],
        # Cele 3 statistici anti-bias
        "dist_origini"           : {o: 0 for o in _toate_originile},
        "dist_destinatii"        : {d: 0 for d in sorted(_toate_dest)},
        "freq_macazuri_afectate" : {m: 0 for m in _toate_macazuri},
        "freq_segmente_afectate" : {s: 0 for s in _toate_segmente},
        # Contor total cereri (pentru procente corecte)
        "_total_cereri"          : 0,
    }

    for _ in range(n_scenarii):
        s = generate_scenario(difficulty_level=difficulty_level, mode=mode)
        stats["solvabilitate"][s.solvabilitate_tip] += 1
        stats["difficulty_score"].append(s.difficulty_score)
        stats["nr_trenuri"].append(len(s.trenuri_noi))
        stats["nr_macazuri_probleme"].append(len(s.macazuri_cu_probleme))
        stats["nr_segmente_probleme"].append(len(s.segmente_cu_probleme))
        stats["nr_rute_active"].append(len(s.rute_active))

        # Distributie origini/destinatii
        for tren in s.trenuri_noi:
            stats["_total_cereri"] += 1
            if tren.origine in stats["dist_origini"]:
                stats["dist_origini"][tren.origine] += 1
            if tren.destinatie in stats["dist_destinatii"]:
                stats["dist_destinatii"][tren.destinatie] += 1

        # Frecventa macazuri afectate
        for m in s.macazuri_cu_probleme:
            if m.macaz in stats["freq_macazuri_afectate"]:
                stats["freq_macazuri_afectate"][m.macaz] += 1

        # Frecventa segmente afectate
        for seg in s.segmente_cu_probleme:
            if seg.segment in stats["freq_segmente_afectate"]:
                stats["freq_segmente_afectate"][seg.segment] += 1

    # Calculeaza medii
    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    stats["avg_difficulty_score"]     = avg(stats["difficulty_score"])
    stats["avg_nr_trenuri"]           = avg(stats["nr_trenuri"])
    stats["avg_nr_macazuri_probleme"] = avg(stats["nr_macazuri_probleme"])
    stats["avg_nr_segmente_probleme"] = avg(stats["nr_segmente_probleme"])
    stats["avg_nr_rute_active"]       = avg(stats["nr_rute_active"])

    # Procentaje solvabilitate
    for tip, count in stats["solvabilitate"].items():
        stats["solvabilitate"][tip] = {
            "count"  : count,
            "procent": round(count / n_scenarii * 100, 1),
        }

    # Procentaje origini/destinatii (din total cereri)
    total_cereri = max(stats["_total_cereri"], 1)
    stats["dist_origini"] = {
        o: {"count": c, "procent": round(c / total_cereri * 100, 1)}
        for o, c in stats["dist_origini"].items()
    }
    stats["dist_destinatii"] = {
        d: {"count": c, "procent": round(c / total_cereri * 100, 1)}
        for d, c in stats["dist_destinatii"].items()
    }

    # Procentaje macazuri afectate (din total scenarii)
    stats["freq_macazuri_afectate"] = {
        m: {"count": c, "procent": round(c / n_scenarii * 100, 1)}
        for m, c in stats["freq_macazuri_afectate"].items()
    }

    # Procentaje segmente afectate (din total scenarii)
    stats["freq_segmente_afectate"] = {
        s: {"count": c, "procent": round(c / n_scenarii * 100, 1)}
        for s, c in stats["freq_segmente_afectate"].items()
    }

    return stats


def display_statistics(stats: dict) -> None:
    """Afiseaza statisticile agregate intr-un format clar."""
    print("=" * 65)
    print(f"STATISTICI GENERATOR")
    print(f"  {stats['total']} scenarii | nivel {stats['difficulty_level']} | mod {stats['mode']}")
    print("=" * 65)

    print("\nSolvabilitate:")
    for tip, data in stats["solvabilitate"].items():
        bar = "█" * int(data["procent"] / 2)
        print(f"  {tip:<28}: {data['count']:>4}  ({data['procent']:>5.1f}%)  {bar}")

    print(f"\nValori medii:")
    print(f"  Difficulty score    : {stats['avg_difficulty_score']}")
    print(f"  Trenuri noi         : {stats['avg_nr_trenuri']}")
    print(f"  Macazuri cu probleme: {stats['avg_nr_macazuri_probleme']}")
    print(f"  Segmente cu probleme: {stats['avg_nr_segmente_probleme']}")
    print(f"  Rute active         : {stats['avg_nr_rute_active']}")

    # Distributie origini
    print("\nDistributie origini (din total cereri):")
    for origine, data in stats["dist_origini"].items():
        bar = "█" * int(data["procent"] / 2)
        print(f"  {origine:<6}: {data['count']:>4}  ({data['procent']:>5.1f}%)  {bar}")

    # Distributie destinatii
    print("\nDistributie destinatii (din total cereri):")
    for dest, data in sorted(stats["dist_destinatii"].items(),
                              key=lambda x: -x[1]["count"]):
        bar = "█" * int(data["procent"] / 2)
        print(f"  {dest:<6}: {data['count']:>4}  ({data['procent']:>5.1f}%)  {bar}")

    # Frecventa macazuri afectate
    print("\nFrecventa macazuri afectate (din total scenarii):")
    for macaz, data in stats["freq_macazuri_afectate"].items():
        bar = "█" * int(data["procent"] / 2)
        print(f"  {macaz:<8}: {data['count']:>4}  ({data['procent']:>5.1f}%)  {bar}")

    # Frecventa segmente afectate (doar cele cu count > 0)
    print("\nFrecventa segmente afectate (din total scenarii):")
    segmente_afectate = {
        s: d for s, d in stats["freq_segmente_afectate"].items()
        if d["count"] > 0
    }
    if segmente_afectate:
        for seg, data in sorted(segmente_afectate.items(),
                                 key=lambda x: -x[1]["count"]):
            bar = "█" * int(data["procent"] / 2)
            print(f"  {seg:<22}: {data['count']:>4}  ({data['procent']:>5.1f}%)  {bar}")
    else:
        print("  Niciun segment afectat in aceasta rulare.")

    print("=" * 65)