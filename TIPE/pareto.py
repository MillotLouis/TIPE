import matplotlib.pyplot as plt

# --- Données des points adaptées aux nouveaux axes ---
voitures = {
    'A': (90, 366.7, 'gray'),  
    'B': (85, 233.3, 'gray'),
    'C': (95, 286.7, 'gray'),
    'D':(98,450,'red')
}

# Configuration des limites des axes
plt.xlim(80, 100)
plt.ylim(100, 500)

# Tracé des points et de leurs labels
for nom, (x, y, couleur) in voitures.items():
    plt.scatter(x, y, marker='x', s=120, color=couleur, linewidths=2)
    plt.text(x + 0.3, y, f' {nom}', fontsize=12, color='#333333', va='center')

# Définition des graduations demandées
plt.xticks(range(80, 101, 5), fontsize=13)
plt.yticks(range(100, 501, 100), fontsize=13)

# Légendes des axes
plt.xlabel("Delivery ratio", fontsize=12, labelpad=8)
plt.ylabel("Mort 10%", fontsize=12, labelpad=8)

# Style de la grille en pointillés fins
plt.grid(True, linestyle='--', color='#d3d3d3', linewidth=0.7)

# Suppression des bordures supérieure et droite (Style épuré)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.show()