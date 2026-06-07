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
    [-778.04682612,  -92.92871371    ,8.87595764],
    [-805.97087259,  -92.20805564,    8.18417442],
    [-779.63786322,  -92.85585042,    8.95610728],
    [-769.08822488,  -92.87541103,    8.83345056],
    [-817.26589878,  -92.57240691,    8.59968   ],
    [-772.32181059,  -92.89971416,    8.85438817],
    [-801.0360337 ,  -92.83786163,    8.77650784],
    [-806.9337376 ,  -92.29898732,    8.30901076],
    [-806.11711834,  -92.36811594,    8.26012382],
    [-806.1803659 ,  -92.21404139,    8.22378725],
    [-791.60749306,  -92.54943817,    8.35212127],
    [-809.73680084,  -92.42955197,    8.44182901],
    [-782.08710648,  -92.80465493,    8.58185337],
    [-812.68846448,  -92.64372435,    8.71012366],
    [-804.47357759,  -92.72244278,    8.59599592],
    [-809.26598019,  -92.50079625,    8.44175156],
    [-786.96635037,  -92.6875    ,    8.48284463],
    [-807.87739781,  -92.51860898,    8.3796077 ],
    [-788.65255343,  -92.67827899,    8.41604263],
    [-776.99783853,  -92.87397293,    8.84489633],
    [-793.81102785,  -92.66499336,    8.51105608],
    [-815.55691963,  -92.33409611,    8.45152383],
    [-795.98131467,  -92.74050931,    8.61452658],
    [-814.16751668,  -92.33105978,    8.33021268],
    [-817.19602948,  -92.5922759 ,    8.72765254],
    [-798.02405361,  -92.62372164,    8.55098168],
    [-792.72628142,  -92.64275364,    8.41853845],
    [-805.84832655,  -92.31996288,    8.22962269],
    [-783.84886259,  -92.77079593,    8.60371537],
    [-800.17786596,  -92.77650379,    8.67182248],
    [-810.81462953,  -92.51645452,    8.49015526],
    [-816.83590893,  -92.40241702,    8.45821088],
    [-787.78179087,  -92.76384613,    8.56084232],
    [-798.96707972,  -92.85271638,    8.75401285],
    [-809.66751502,  -92.40276053,    8.40293605],
    [-805.37568572,  -92.55620025,    8.52980611],
    [-777.45897517,  -92.91317203,    8.8723339 ],
    [-787.80933936,  -92.70201119,    8.51909201],
    [-809.1094179 ,  -92.62434197,    8.68691328],
    [-795.50263582,  -92.57284271,    8.46229661],
    [-797.86292337,  -92.68586896,    8.57853665],
    [-807.12846264,  -92.73583808,    8.63299595],
    [-800.47045455,  -92.53565258,    8.50364224],
    [-795.75604009,  -92.85250162,    8.6697371 ],
    [-795.79413353,  -92.82042191,    8.62837447],
    [-806.20654724,  -92.21223543,    8.2946839 ],
    [-783.832332  ,  -92.73118537,    8.52722573],
    [-808.85960934,  -92.50159171,    8.46973748]
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
plt.ylabel("Écart type énergie finale")
plt.title("Front de Pareto obtenu par NSGA-II")
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(sc)
cbar.set_label("Taux de livraison (%)")

plt.tight_layout()
plt.savefig("front_pareto_tipe.pdf", bbox_inches="tight")
plt.show()