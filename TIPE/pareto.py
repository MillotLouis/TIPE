import matplotlib.pyplot as plt

# # --- Données des points adaptées aux nouveaux axes ---
# voitures = {
#     'A': (90, 366.7, 'gray'),  
#     'B': (85, 233.3, 'gray'),
#     'C': (95, 286.7, 'gray'),
#     'D':(98,450,'red')
# }

# # Configuration des limites des axes
# plt.xlim(80, 100)
# plt.ylim(100, 500)

# # Tracé des points et de leurs labels
# for nom, (x, y, couleur) in voitures.items():
#     plt.scatter(x, y, marker='x', s=120, color=couleur, linewidths=2)
#     plt.text(x + 0.3, y, f' {nom}', fontsize=12, color='#333333', va='center')

# # Définition des graduations demandées
# plt.xticks(range(80, 101, 5), fontsize=13)
# plt.yticks(range(100, 501, 100), fontsize=13)

# # Légendes des axes
# plt.xlabel("Delivery ratio", fontsize=12, labelpad=8)
# plt.ylabel("Mort 10%", fontsize=12, labelpad=8)

# # Style de la grille en pointillés fins
# plt.grid(True, linestyle='--', color='#d3d3d3', linewidth=0.7)

# # Suppression des bordures supérieure et droite (Style épuré)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(1.2)
# ax.spines['bottom'].set_linewidth(1.2)

plt.show()



import numpy as np
import matplotlib.pyplot as plt

F = np.array([
    [-1927.09947937,   -94.69466423,     3.41964873],
    [-1925.68364571,   -94.74699284,     3.33107855],
    [-1922.59483189,   -94.79113731,     3.44252256],
    [-1916.30757394,   -94.68869123,     3.31931499],
    [-1932.66194289,   -95.02022245,     3.6494753 ]
])

life = -F[:, 0]
delivery = -F[:, 1]
std_bat = F[:, 2]

plt.figure(figsize=(8, 5.5))

sc = plt.scatter(
    life,
    std_bat,
    c=delivery,
    s=120,
    edgecolors="black"
)

# for i, (x, y) in enumerate(zip(life, std_bat)):
#     plt.annotate(
#         "A" if i==4 else "",
#         (x, y),
#         textcoords="offset points",
#         xytext=(6, 6),
#         fontsize=10
#     )

plt.xlabel("Temps avant 20 % de nœuds morts (s)")
plt.ylabel("Énergie par message délivré")
plt.title("Front de Pareto obtenu par NSGA-II")
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(sc)
cbar.set_label("Taux de livraison (%)")

plt.tight_layout()
plt.savefig("front_pareto_tipe.pdf", bbox_inches="tight")
plt.show()