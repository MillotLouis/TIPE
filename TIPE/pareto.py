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
    [-1986.69898805,   -95.23773547,     4.65709874],
    [-1973.42458479,   -95.36039348,     3.77841574],
    [-1975.76774903,   -95.32528046,     3.7481591 ],
    [-1966.27150927,   -95.42054648,     4.16402849],
    [-1998.34124283,   -95.07823257,     4.09160981],
    [-1988.34248814,   -95.1391335 ,     3.24292111],
    [-1967.96666591,   -95.40358459,     4.17553836],
    [-1979.35972877,   -95.28599955,     4.63394595],
    [-1978.92270459,   -95.31996663,     4.09056918],
    [-1974.22479051,   -95.16026859,     3.36273509],
    [-1984.66668371,   -95.18935001,     3.37564542],
    [-1971.74106695,   -95.33355325,     3.74182297],
    [-1989.70309919,   -95.17917767,     3.46042918],
    [-1993.20783748,   -95.13841829,     3.96052988],
    [-1987.7808685 ,   -95.14809875,     3.19312118],
    [-1982.96789937,   -95.26368912,     4.01855993],
    [-1995.97411822,   -95.13808832,     3.26726231],
    [-1975.26855534,   -95.38488817,     3.7791077 ],
    [-1984.51529907,   -95.23713472,     3.98853574],
    [-1990.87887417,   -95.09234828,     3.22643755],
    [-1988.73980695,   -95.17435717,     3.44522907],
    [-1992.84153943,   -95.23606335,     3.4939967 ],
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

plt.xlabel("Temps avant 10 % de nœuds morts (s)")
plt.ylabel("Écart-type final de batterie")
plt.title("Front de Pareto obtenu par NSGA-II")
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(sc)
cbar.set_label("Taux de livraison (%)")

plt.tight_layout()
plt.savefig("front_pareto_tipe.pdf", bbox_inches="tight")
plt.show()