"""
station.py — Graful statiei Theia cu semnale ca noduri terminale

Topologie (de jos in sus):
  Linia 1  — abatere, cea mai de jos
  Linia II — directa (linie principala)
  Linia 3  — abatere
  Linia 4  — abatere, cea mai de sus

Macazuri:
  Capatul X (impare): M1, M3, M5
  Capatul Y (pare)  : M2, M8, DJ46 (dubla jonctiune 4/6)

Semnale de iesire — noduri terminale:
  Capatul X (prefix Y = semnal de iesire spre X):
    Y1, YII, Y3, Y4
  Capatul Y (prefix X = semnal de iesire spre Y):
    X1, XII, X3, X4

Tipuri de parcurs:
  Tranzit complet  : X -> ... -> Y  sau  Y -> ... -> X
  Intrare+oprire   : X -> ... -> X1 / X -> ... -> X3 etc.
                     Y -> ... -> Y1 / Y -> ... -> YII etc.

Structura unei linii (ex. linia 1):
  X -> SEG_X_M1 -> M1 -> SEG_M1_L1 -> Y1 -> SEG_L1_X -> SEG_L1 -> SEG_L1_Y -> X1 -> SEG_M8_L1 -> M8 -> ... -> Y
                                        ^                                         ^
                                  terminal capX                            terminal capY

Parcursuri (sursa: tabelul de parcursuri oficial):
  X -> Linia 1  : M1(-)
  X -> Linia II : M1(+), M3(+)
  X -> Linia 3  : M1(+), M3(-), M5(-)
  X -> Linia 4  : M1(+), M3(-), M5(+)

  Y -> Linia 1  : M2(+), M8(-)
  Y -> Linia II : M2(+), M8(+)
  Y -> Linia 3  : M2(-), DJ46(+)
  Y -> Linia 4  : M2(-), DJ46(-)

Nota DJ46:
  Dubla jonctiune formata din macazurile 4 si 6, actionate simultan.
  Modelata ca nod unic — abstractizare a starii combinate a celor doua
  macazuri fizice.

Nota semnale:
  Semnalele sunt modelate ca noduri terminale, nu ca elemente de control.
  Ele marcheaza limita fizica pana unde poate ajunge un tren care
  ramane in statie, separand astfel parcursurile de tranzit de cele
  de intrare/oprire.

Stari macaz:
  operational : poate fi comutat liber
  blocat      : pozitia nu poate fi schimbata, macazul e utilizabil
  defect      : scos complet din graf

Stari segment:
  liber        : disponibil
  rezervat     : alocat unui tren planificat
  ocupat       : tren prezent fizic
  indisponibil : scos din serviciu
"""

import networkx as nx


# ---------------------------------------------------------------------------
# Traversari permise per macaz
# Fiecare pereche (seg_in, seg_out) este o traversare fizic valida
# in pozitia data, acoperind ambele sensuri de mers.
# ---------------------------------------------------------------------------
MACAZ_TRAVERSARI = {
    # ------------------------------------------------------------------
    # M1 — capatul X, primul macaz
    # plus  = directa -> interior (spre M3, liniile II/3/4)
    # minus = abatere -> linia 1
    # ------------------------------------------------------------------
    "M1": {
        "plus": [
            ("SEG_X_M1",  "SEG_M1_M3"),
            ("SEG_M1_M3", "SEG_X_M1"),
        ],
        "minus": [
            ("SEG_X_M1",  "SEG_M1_L1"),
            ("SEG_M1_L1", "SEG_X_M1"),
        ],
    },

    # ------------------------------------------------------------------
    # M3 — capatul X, al doilea macaz
    # plus  = directa -> linia II
    # minus = abatere -> spre M5 (liniile 3/4)
    # ------------------------------------------------------------------
    "M3": {
        "plus": [
            ("SEG_M1_M3", "SEG_M3_LII"),
            ("SEG_M3_LII","SEG_M1_M3"),
        ],
        "minus": [
            ("SEG_M1_M3", "SEG_M3_M5"),
            ("SEG_M3_M5", "SEG_M1_M3"),
        ],
    },

    # ------------------------------------------------------------------
    # M5 — capatul X, acces linii 3/4
    # plus  = linia 4  (X->L4: M5 pe +)
    # minus = linia 3  (X->L3: M5 pe -)
    # ------------------------------------------------------------------
    "M5": {
        "plus": [
            ("SEG_M3_M5", "SEG_M5_L4"),
            ("SEG_M5_L4", "SEG_M3_M5"),
        ],
        "minus": [
            ("SEG_M3_M5", "SEG_M5_L3"),
            ("SEG_M5_L3", "SEG_M3_M5"),
        ],
    },

    # ------------------------------------------------------------------
    # M2 — capatul Y, primul macaz
    # plus  = directa -> spre M8 (liniile II/1)
    # minus = abatere -> spre DJ46 (liniile 3/4)
    # ------------------------------------------------------------------
    "M2": {
        "plus": [
            ("SEG_Y_M2",   "SEG_M2_M8"),
            ("SEG_M2_M8",  "SEG_Y_M2"),
        ],
        "minus": [
            ("SEG_Y_M2",   "SEG_M2_DJ46"),
            ("SEG_M2_DJ46","SEG_Y_M2"),
        ],
    },

    # ------------------------------------------------------------------
    # M8 — capatul Y, acces linii II/1
    # plus  = directa -> linia II
    # minus = abatere -> linia 1
    # ------------------------------------------------------------------
    "M8": {
        "plus": [
            ("SEG_M2_M8", "SEG_M8_LII"),
            ("SEG_M8_LII","SEG_M2_M8"),
        ],
        "minus": [
            ("SEG_M2_M8", "SEG_M8_L1"),
            ("SEG_M8_L1", "SEG_M2_M8"),
        ],
    },

    # ------------------------------------------------------------------
    # DJ46 — dubla jonctiune 4/6, capatul Y
    # Nod unic: macazurile 4 si 6 actionate simultan de un singur
    # electromecanism. Pozitiile plus/minus reprezinta starea combinata.
    # plus  = linia 3 conectata la Y
    # minus = linia 4 conectata la Y
    # ------------------------------------------------------------------
    "DJ46": {
        "plus": [
            ("SEG_M2_DJ46", "SEG_DJ46_L3"),
            ("SEG_DJ46_L3", "SEG_M2_DJ46"),
        ],
        "minus": [
            ("SEG_M2_DJ46", "SEG_DJ46_L4"),
            ("SEG_DJ46_L4", "SEG_M2_DJ46"),
        ],
    },
}

# Lookup derivat automat: { macaz_id: { (seg_in, seg_out): pozitie } }
_TRAVERSARE_LA_POZITIE: dict = {}
for _m, _pozitii in MACAZ_TRAVERSARI.items():
    _TRAVERSARE_LA_POZITIE[_m] = {}
    for _poz, _lista in _pozitii.items():
        for _pereche in _lista:
            _TRAVERSARE_LA_POZITIE[_m][_pereche] = _poz

# Noduri terminale explicite ale statiei
CAPETE_LINIE   = {"X", "Y"}
SEMNALE_CAP_X  = {"Y1", "YII", "Y3", "Y4"}   # semnale iesire spre capatul X
SEMNALE_CAP_Y  = {"X1", "XII", "X3", "X4"}   # semnale iesire spre capatul Y
NODURI_TERMINALE = CAPETE_LINIE | SEMNALE_CAP_X | SEMNALE_CAP_Y

# _NODURI_VALIDE_SET este construit dupa definirea build_station()
# la sfarsitul fisierului, pentru a include toate nodurile grafului.


# ---------------------------------------------------------------------------
# Constructia grafului de baza
# ---------------------------------------------------------------------------

def build_station() -> nx.DiGraph:
    """
    Graful complet al statiei Theia.
    Include semnalele de iesire ca noduri terminale, permitand
    reprezentarea atat a tranzitelor complete (X->Y) cat si a
    parcursurilor de intrare+oprire (X->X1, Y->Y3 etc.)
    """
    G = nx.DiGraph()

    # --- Capete de linie ---
    G.add_node("X", tip="capat", descriere="Capatul X (stanga)")
    G.add_node("Y", tip="capat", descriere="Capatul Y (dreapta)")

    # --- Semnale de iesire — noduri terminale ---
    # Capatul X: semnale Y_linie (tren intrat din X, ramas in statie)
    for semnal, linie in [
        ("Y1",  "L1"),
        ("YII", "LII"),
        ("Y3",  "L3"),
        ("Y4",  "L4"),
    ]:
        G.add_node(semnal, tip="semnal", linie=linie, capat="X",
                   descriere=f"Semnal iesire linia {linie}, capatul X")

    # Capatul Y: semnale X_linie (tren intrat din Y, ramas in statie)
    for semnal, linie in [
        ("X1",  "L1"),
        ("XII", "LII"),
        ("X3",  "L3"),
        ("X4",  "L4"),
    ]:
        G.add_node(semnal, tip="semnal", linie=linie, capat="Y",
                   descriere=f"Semnal iesire linia {linie}, capatul Y")

    # --- Macazuri ---
    for nod_id, desc in [
        ("M1",   "Macaz 1 — capatul X, primul macaz"),
        ("M3",   "Macaz 3 — capatul X, al doilea macaz"),
        ("M5",   "Macaz 5 — capatul X, acces linii 3/4"),
        ("M2",   "Macaz 2 — capatul Y, primul macaz"),
        ("M8",   "Macaz 8 — capatul Y, acces linii II/1"),
        ("DJ46", "Dubla jonctiune 4/6 — capatul Y, acces linii 3/4"),
    ]:
        G.add_node(nod_id, tip="macaz", pozitie="plus",
                   stare="operational", descriere=desc)

    # --- Segmente ---
    # Structura per linie:
    #   SEG_M?_L?  = segment intre macaz si semnal capX
    #   SEG_L?_X   = segment intre semnal capX si corpul liniei
    #   SEG_L?     = corpul liniei (intre cele doua semnale)
    #   SEG_L?_Y   = segment intre corpul liniei si semnal capY
    #   SEG_M?_L?  = segment intre semnal capY si macaz
    for seg_id, linie in [
        # Capatul X — exterior
        ("SEG_X_M1",    "intrare_X"),
        # Capatul X — interior (intre macazuri)
        ("SEG_M1_M3",   "interior_X_LII"),
        ("SEG_M3_M5",   "interior_X_L34"),
        # Segmente macaz->semnal (capatul X al fiecarei linii)
        ("SEG_M1_L1",   "L1_capX"),
        ("SEG_M3_LII",  "LII_capX"),
        ("SEG_M5_L3",   "L3_capX"),
        ("SEG_M5_L4",   "L4_capX"),
        # Segmente semnal->corp (capatul X)
        ("SEG_Y1_L1",   "L1"),
        ("SEG_YII_LII", "LII"),
        ("SEG_Y3_L3",   "L3"),
        ("SEG_Y4_L4",   "L4"),
        # Corpuri linii
        ("SEG_L1",      "L1"),
        ("SEG_LII",     "LII"),
        ("SEG_L3",      "L3"),
        ("SEG_L4",      "L4"),
        # Segmente corp->semnal (capatul Y)
        ("SEG_L1_X1",   "L1"),
        ("SEG_LII_XII", "LII"),
        ("SEG_L3_X3",   "L3"),
        ("SEG_L4_X4",   "L4"),
        # Segmente semnal->macaz (capatul Y)
        ("SEG_M8_L1",   "L1_capY"),
        ("SEG_M8_LII",  "LII_capY"),
        ("SEG_DJ46_L3", "L3_capY"),
        ("SEG_DJ46_L4", "L4_capY"),
        # Capatul Y — interior
        ("SEG_M2_M8",   "interior_Y_L12"),
        ("SEG_M2_DJ46", "interior_Y_L34"),
        # Capatul Y — exterior
        ("SEG_Y_M2",    "intrare_Y"),
    ]:
        G.add_node(seg_id, tip="segment", stare="liber",
                   tren=None, liber_dupa=None, linie=linie)

    # --- Muchii ---
    for u, v in [
        # ── Capatul X ──────────────────────────────────────────────────
        ("X",          "SEG_X_M1"),
        ("SEG_X_M1",   "X"),
        ("SEG_X_M1",   "M1"),
        ("M1",         "SEG_X_M1"),

        # M1 plus -> M3
        ("M1",         "SEG_M1_M3"),
        ("SEG_M1_M3",  "M1"),
        ("SEG_M1_M3",  "M3"),
        ("M3",         "SEG_M1_M3"),

        # M1 minus -> Y1 (semnal capX linia 1)
        ("M1",         "SEG_M1_L1"),
        ("SEG_M1_L1",  "M1"),
        ("SEG_M1_L1",  "Y1"),
        ("Y1",         "SEG_M1_L1"),

        # M3 plus -> YII (semnal capX linia II)
        ("M3",         "SEG_M3_LII"),
        ("SEG_M3_LII", "M3"),
        ("SEG_M3_LII", "YII"),
        ("YII",        "SEG_M3_LII"),

        # M3 minus -> M5
        ("M3",         "SEG_M3_M5"),
        ("SEG_M3_M5",  "M3"),
        ("SEG_M3_M5",  "M5"),
        ("M5",         "SEG_M3_M5"),

        # M5 minus -> Y3 (semnal capX linia 3)
        ("M5",         "SEG_M5_L3"),
        ("SEG_M5_L3",  "M5"),
        ("SEG_M5_L3",  "Y3"),
        ("Y3",         "SEG_M5_L3"),

        # M5 plus -> Y4 (semnal capX linia 4)
        ("M5",         "SEG_M5_L4"),
        ("SEG_M5_L4",  "M5"),
        ("SEG_M5_L4",  "Y4"),
        ("Y4",         "SEG_M5_L4"),

        # ── Semnale capX -> corp linie ──────────────────────────────────
        ("Y1",         "SEG_Y1_L1"),
        ("SEG_Y1_L1",  "Y1"),
        ("SEG_Y1_L1",  "SEG_L1"),
        ("SEG_L1",     "SEG_Y1_L1"),

        ("YII",        "SEG_YII_LII"),
        ("SEG_YII_LII","YII"),
        ("SEG_YII_LII","SEG_LII"),
        ("SEG_LII",    "SEG_YII_LII"),

        ("Y3",         "SEG_Y3_L3"),
        ("SEG_Y3_L3",  "Y3"),
        ("SEG_Y3_L3",  "SEG_L3"),
        ("SEG_L3",     "SEG_Y3_L3"),

        ("Y4",         "SEG_Y4_L4"),
        ("SEG_Y4_L4",  "Y4"),
        ("SEG_Y4_L4",  "SEG_L4"),
        ("SEG_L4",     "SEG_Y4_L4"),

        # ── Corp linie -> semnale capY ──────────────────────────────────
        ("SEG_L1",     "SEG_L1_X1"),
        ("SEG_L1_X1",  "SEG_L1"),
        ("SEG_L1_X1",  "X1"),
        ("X1",         "SEG_L1_X1"),

        ("SEG_LII",    "SEG_LII_XII"),
        ("SEG_LII_XII","SEG_LII"),
        ("SEG_LII_XII","XII"),
        ("XII",        "SEG_LII_XII"),

        ("SEG_L3",     "SEG_L3_X3"),
        ("SEG_L3_X3",  "SEG_L3"),
        ("SEG_L3_X3",  "X3"),
        ("X3",         "SEG_L3_X3"),

        ("SEG_L4",     "SEG_L4_X4"),
        ("SEG_L4_X4",  "SEG_L4"),
        ("SEG_L4_X4",  "X4"),
        ("X4",         "SEG_L4_X4"),

        # ── Semnale capY -> macazuri ────────────────────────────────────
        ("X1",         "SEG_M8_L1"),
        ("SEG_M8_L1",  "X1"),
        ("SEG_M8_L1",  "M8"),
        ("M8",         "SEG_M8_L1"),

        ("XII",        "SEG_M8_LII"),
        ("SEG_M8_LII", "XII"),
        ("SEG_M8_LII", "M8"),
        ("M8",         "SEG_M8_LII"),

        ("X3",         "SEG_DJ46_L3"),
        ("SEG_DJ46_L3","X3"),
        ("SEG_DJ46_L3","DJ46"),
        ("DJ46",       "SEG_DJ46_L3"),

        ("X4",         "SEG_DJ46_L4"),
        ("SEG_DJ46_L4","X4"),
        ("SEG_DJ46_L4","DJ46"),
        ("DJ46",       "SEG_DJ46_L4"),

        # ── Capatul Y ──────────────────────────────────────────────────
        ("M8",         "SEG_M2_M8"),
        ("SEG_M2_M8",  "M8"),
        ("SEG_M2_M8",  "M2"),
        ("M2",         "SEG_M2_M8"),

        ("DJ46",       "SEG_M2_DJ46"),
        ("SEG_M2_DJ46","DJ46"),
        ("SEG_M2_DJ46","M2"),
        ("M2",         "SEG_M2_DJ46"),

        ("M2",         "SEG_Y_M2"),
        ("SEG_Y_M2",   "M2"),
        ("SEG_Y_M2",   "Y"),
        ("Y",          "SEG_Y_M2"),
    ]:
        G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------------
# Graf operational
# ---------------------------------------------------------------------------

def build_operational_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Subgraful valid la momentul t, bazat pe starea curenta.

    Reguli:
      1. Macaz defect      -> nod eliminat complet
      2. Macaz operational/blocat -> pastram doar traversarile
         permise de pozitia curenta
      3. Segment ocupat/rezervat/indisponibil -> muchii adiacente eliminate
    """
    G_op = G.copy()

    for macaz_id, traversari in MACAZ_TRAVERSARI.items():
        if macaz_id not in G_op.nodes:
            continue
        nod   = G_op.nodes[macaz_id]
        stare = nod["stare"]

        if stare == "defect":
            G_op.remove_node(macaz_id)
            continue

        traversari_permise = traversari[nod["pozitie"]]
        muchii_permise = set()
        for (seg_in, seg_out) in traversari_permise:
            muchii_permise.add((seg_in,   macaz_id))
            muchii_permise.add((macaz_id, seg_out))

        de_sters = [
            (u, v) for u, v in (
                list(G_op.in_edges(macaz_id)) +
                list(G_op.out_edges(macaz_id))
            )
            if (u, v) not in muchii_permise
        ]
        G_op.remove_edges_from(de_sters)

    for nod_id, date in list(G_op.nodes(data=True)):
        if date.get("tip") == "segment" and \
           date.get("stare") in ("ocupat", "rezervat", "indisponibil"):
            G_op.remove_edges_from(list(G_op.in_edges(nod_id)))
            G_op.remove_edges_from(list(G_op.out_edges(nod_id)))

    return G_op


# ---------------------------------------------------------------------------
# Graf de planificare
# ---------------------------------------------------------------------------

def build_planning_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Graf de planificare cu trei tratamente distincte per stare macaz:
      operational -> toate traversarile posibile (plus + minus)
                     macazul poate fi comutat, orice pozitie e accesibila
      blocat      -> doar traversarile pozitiei curente
                     macazul nu poate fi comutat, ramane restrictionat
      defect      -> nod eliminat complet din graf

    Segmentele ocupate/rezervate/indisponibile raman blocate.
    """
    G_plan = G.copy()

    for macaz_id, traversari in MACAZ_TRAVERSARI.items():
        if macaz_id not in G_plan.nodes:
            continue
        nod   = G_plan.nodes[macaz_id]
        stare = nod["stare"]

        if stare == "defect":
            G_plan.remove_node(macaz_id)
            continue

        if stare == "blocat":
            traversari_permise = traversari[nod["pozitie"]]
        else:
            traversari_permise = traversari["plus"] + traversari["minus"]

        muchii_permise = set()
        for (seg_in, seg_out) in traversari_permise:
            muchii_permise.add((seg_in,   macaz_id))
            muchii_permise.add((macaz_id, seg_out))

        de_sters = [
            (u, v) for u, v in (
                list(G_plan.in_edges(macaz_id)) +
                list(G_plan.out_edges(macaz_id))
            )
            if (u, v) not in muchii_permise
        ]
        G_plan.remove_edges_from(de_sters)

    for nod_id, date in list(G_plan.nodes(data=True)):
        if date.get("tip") == "segment" and \
           date.get("stare") in ("ocupat", "rezervat", "indisponibil"):
            G_plan.remove_edges_from(list(G_plan.in_edges(nod_id)))
            G_plan.remove_edges_from(list(G_plan.out_edges(nod_id)))

    return G_plan


# ---------------------------------------------------------------------------
# Inferenta configuratiei
# ---------------------------------------------------------------------------

def inferenta_configuratie(G: nx.DiGraph, ruta: list) -> dict:
    """
    Determina pozitia necesara pentru fiecare macaz traversat in ruta.
    Returneaza dict cu info completa per macaz.
    """
    configuratie = {}

    for i, nod_id in enumerate(ruta):
        if nod_id not in G.nodes:
            continue
        if G.nodes[nod_id].get("tip") != "macaz":
            continue

        seg_in  = ruta[i - 1] if i > 0 else None
        seg_out = ruta[i + 1] if i < len(ruta) - 1 else None

        if seg_in is None or seg_out is None:
            continue

        pozitie_necesara = _TRAVERSARE_LA_POZITIE.get(nod_id, {}).get(
            (seg_in, seg_out)
        )
        if pozitie_necesara is None:
            continue

        nod               = G.nodes[nod_id]
        pozitie_curenta   = nod["pozitie"]
        stare             = nod["stare"]
        necesita_comutare = pozitie_curenta != pozitie_necesara
        poate_fi_comutat  = stare == "operational"

        configuratie[nod_id] = {
            "pozitie_curenta"  : pozitie_curenta,
            "pozitie_necesara" : pozitie_necesara,
            "necesita_comutare": necesita_comutare,
            "poate_fi_comutat" : poate_fi_comutat,
            "stare"            : stare,
            "blocat"           : stare == "blocat",
            "conflict"         : necesita_comutare and not poate_fi_comutat,
        }

    return configuratie


def ruta_este_executabila(configuratie: dict) -> tuple[bool, list]:
    conflicte = [
        (macaz_id, info)
        for macaz_id, info in configuratie.items()
        if info["conflict"]
    ]
    return (len(conflicte) == 0), conflicte


def validare_structurala_ruta(G: nx.DiGraph, ruta: list) -> tuple[bool, list]:
    """
    Verifica ca fiecare traversare prin macaz din ruta
    corespunde unei traversari definite in MACAZ_TRAVERSARI.
    """
    probleme = []

    for i, nod_id in enumerate(ruta):
        if nod_id not in G.nodes:
            continue
        if G.nodes[nod_id].get("tip") != "macaz":
            continue

        seg_in  = ruta[i - 1] if i > 0 else None
        seg_out = ruta[i + 1] if i < len(ruta) - 1 else None

        if seg_in is None or seg_out is None:
            probleme.append({
                "macaz": nod_id,
                "motiv": "macaz la capatul rutei fara segment adiacent",
                "seg_in": seg_in, "seg_out": seg_out,
            })
            continue

        if _TRAVERSARE_LA_POZITIE.get(nod_id, {}).get((seg_in, seg_out)) is None:
            probleme.append({
                "macaz": nod_id,
                "motiv": (
                    f"traversarea ({seg_in} -> {nod_id} -> {seg_out}) "
                    f"nu exista in MACAZ_TRAVERSARI"
                ),
                "seg_in": seg_in, "seg_out": seg_out,
            })

    return (len(probleme) == 0), probleme


# ---------------------------------------------------------------------------
# Scorare si cautare rute
# ---------------------------------------------------------------------------

def scor_ruta(G: nx.DiGraph, ruta: list) -> int:
    configuratie = inferenta_configuratie(G, ruta)
    scor = 0
    for nod_id in ruta:
        if nod_id not in G.nodes:
            continue
        date = G.nodes[nod_id]
        if date.get("tip") == "segment":
            scor += 1
        elif date.get("tip") == "macaz":
            info = configuratie.get(nod_id)
            if info and info["necesita_comutare"]:
                scor += 2
    return scor


def verifica_conflict_rute(
    ruta_noua: list,
    rute_active: dict,
    G: nx.DiGraph = None,
    config_noua: dict = None,
    config_active: dict = None,
) -> list:
    """
    Detecteaza conflicte intre ruta noua si rutele active.

    Conflict segment : doua trenuri pe acelasi segment — blocat intotdeauna.
    Conflict macaz   : doua rute cer macazul in pozitii diferite — blocat.
    Doua rute in aceeasi pozitie pe macaz comun — compatibile, OK.

    LIMITARE CUNOSCUTA: nu detecteaza conflicte de zona de protectie
    (doua parcursuri fara segment comun dar cu zone de protectie suprapuse).
    """
    noduri_noi = set(ruta_noua)
    conflicte  = []

    _config_noua = config_noua if config_noua is not None else (
        inferenta_configuratie(G, ruta_noua) if G else {}
    )

    for tren_id, ruta_existenta in rute_active.items():
        overlap = noduri_noi & set(ruta_existenta)

        _config_existenta = (
            (config_active.get(tren_id) or
             inferenta_configuratie(G, ruta_existenta))
            if config_active is not None
            else (inferenta_configuratie(G, ruta_existenta) if G else {})
        )

        for nod_id in overlap:
            tip_nod = G.nodes[nod_id].get("tip") if G else None

            if tip_nod in ("capat", "semnal"):
                # Capetele si semnalele pot fi comune fara conflict
                continue

            elif tip_nod == "segment":
                conflicte.append({
                    "tren"   : tren_id,
                    "tip"    : "segment",
                    "resursa": nod_id,
                    "detaliu": (
                        f"Segmentul {nod_id} este deja rezervat "
                        f"de trenul {tren_id}"
                    ),
                })

            elif tip_nod == "macaz":
                poz_noua = _config_noua.get(nod_id, {}).get("pozitie_necesara")
                poz_ext  = _config_existenta.get(nod_id, {}).get("pozitie_necesara")
                if poz_noua and poz_ext and poz_noua != poz_ext:
                    conflicte.append({
                        "tren"   : tren_id,
                        "tip"    : "macaz",
                        "resursa": nod_id,
                        "detaliu": (
                            f"Macazul {nod_id} cerut pe {poz_noua} "
                            f"de ruta noua, dar pe {poz_ext} "
                            f"de trenul {tren_id}"
                        ),
                    })
            else:
                conflicte.append({
                    "tren"   : tren_id,
                    "tip"    : "necunoscut",
                    "resursa": nod_id,
                    "detaliu": f"Nod comun cu ruta trenului {tren_id}",
                })

    return conflicte


def get_rute_valide(
    G: nx.DiGraph,
    origine: str,
    destinatie: str,
    rute_active: dict,
) -> list:
    """
    Returneaza rutele valide intre origine si destinatie, sortate dupa scor.

    Origine si destinatie pot fi orice nod terminal:
      X, Y                          — capete de linie (tranzit complet)
      Y1, YII, Y3, Y4               — semnale capX (tren ramas din X)
      X1, XII, X3, X4               — semnale capY (tren ramas din Y)
    """
    G_plan = build_planning_graph(G)

    # Cutoff de lungime: ruta cea mai lunga valida in Theia are 19 noduri.
    # Rutele mai lungi sunt intotdeauna ocoluri invalide generate de
    # bidiectionalitatea grafului. Cutoff-ul elimina explozia combinatoriala
    # fara a filtra nicio ruta legitima.
    CUTOFF = 21

    try:
        toate_rutele = list(nx.all_simple_paths(
            G_plan, origine, destinatie, cutoff=CUTOFF
        ))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    config_active = {
        tren_id: inferenta_configuratie(G, ruta)
        for tren_id, ruta in rute_active.items()
    }

    rute_valide = []
    for ruta in toate_rutele:
        structurala_ok, _ = validare_structurala_ruta(G, ruta)
        if not structurala_ok:
            continue

        config = inferenta_configuratie(G, ruta)
        executabila, _ = ruta_este_executabila(config)
        if not executabila:
            continue

        conflicte = verifica_conflict_rute(
            ruta, rute_active, G, config, config_active
        )
        if conflicte:
            continue

        rute_valide.append(ruta)

    return sorted(rute_valide, key=lambda r: scor_ruta(G, r))


# ---------------------------------------------------------------------------
# Gestionarea starii
# ---------------------------------------------------------------------------

def set_stare_macaz(G, macaz_id, pozitie=None, stare=None):
    if pozitie is not None:
        assert pozitie in ("plus", "minus"), f"Pozitie invalida: {pozitie}"
        G.nodes[macaz_id]["pozitie"] = pozitie
    if stare is not None:
        assert stare in ("operational", "blocat", "defect"), \
            f"Stare invalida: {stare}"
        G.nodes[macaz_id]["stare"] = stare


def set_stare_segment(G, segment_id, stare, tren=None, liber_dupa=None):
    assert stare in ("liber", "rezervat", "ocupat", "indisponibil"), \
        f"Stare invalida: {stare}"
    G.nodes[segment_id]["stare"]      = stare
    G.nodes[segment_id]["tren"]       = tren
    G.nodes[segment_id]["liber_dupa"] = liber_dupa


def rezerva_ruta(G, tren_id, ruta, rute_active):
    for nod_id in ruta:
        if G.nodes[nod_id].get("tip") == "segment":
            set_stare_segment(G, nod_id, stare="rezervat", tren=tren_id)
    rute_active[tren_id] = ruta


def elibereaza_ruta(G, tren_id, rute_active):
    ruta = rute_active.pop(tren_id, [])
    for nod_id in ruta:
        if nod_id in G.nodes and G.nodes[nod_id].get("tip") == "segment":
            if G.nodes[nod_id].get("tren") == tren_id:
                set_stare_segment(G, nod_id, stare="liber")


def aplica_configuratie_macazuri(G, configuratie):
    actiuni = []
    for macaz_id, info in configuratie.items():
        if not info["necesita_comutare"]:
            actiuni.append({
                "macaz": macaz_id, "din": info["pozitie_curenta"],
                "in": info["pozitie_necesara"], "rezultat": "deja_corect",
            })
        elif info["conflict"]:
            actiuni.append({
                "macaz": macaz_id, "din": info["pozitie_curenta"],
                "in": info["pozitie_necesara"], "rezultat": "eroare_conflict",
            })
        else:
            set_stare_macaz(G, macaz_id, pozitie=info["pozitie_necesara"])
            actiuni.append({
                "macaz": macaz_id, "din": info["pozitie_curenta"],
                "in": info["pozitie_necesara"], "rezultat": "comutat",
            })
    return actiuni


def get_stare_completa(G):
    macazuri, segmente = {}, {}
    for nod_id, date in G.nodes(data=True):
        if date.get("tip") == "macaz":
            macazuri[nod_id] = {
                "pozitie": date["pozitie"],
                "stare"  : date["stare"],
            }
        elif date.get("tip") == "segment":
            segmente[nod_id] = {
                "stare"     : date["stare"],
                "tren"      : date["tren"],
                "linie"     : date["linie"],
                "liber_dupa": date.get("liber_dupa"),
            }
    return {"macazuri": macazuri, "segmente": segmente}


# ---------------------------------------------------------------------------
# _NODURI_VALIDE_SET — derivat din graful complet al statiei
# Construit dupa build_station() pentru a include TOATE nodurile,
# inclusiv segmentele de corp si segmentele semnal->corp care nu
# apar direct in MACAZ_TRAVERSARI.
# ---------------------------------------------------------------------------
_NODURI_VALIDE_SET: set = set(build_station().nodes())