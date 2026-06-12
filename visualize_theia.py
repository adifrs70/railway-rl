import os
import matplotlib.pyplot as plt
import networkx as nx

from graph.theia import build_station


G = build_station()

pos = {
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

missing = [n for n in G.nodes() if n not in pos]
if missing:
    extra_pos = nx.spring_layout(G.subgraph(missing), seed=42)
    for n, p in extra_pos.items():
        pos[n] = (p[0] * 3 + 9, p[1] * 3 - 4)

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

plt.figure(figsize=(22, 10))

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
    arrowsize=10,
    width=1.1,
    alpha=0.55,
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7,
)

plt.title("Graful orientat al stației Theia", fontsize=16)
plt.axis("off")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/theia_graph_manual.png", dpi=250)
plt.show()