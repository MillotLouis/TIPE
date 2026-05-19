# from matplotlib import pyplot as plt
# import numpy as np


# b = np.arange(0,100,0.1)
# # fx = 1/((x-90)**2)+1/((x)**2)

# # plt.plot(x,fx,label="f")

# g = []
# bi = 100
# # s = 0.05 
# # S = 10
# # for i in range(len(b)):
# #     if b[i] <= bi*s:
# #         g.append(1)
# #     else :
# #         g.append(bi*s/(b[i]))

# for i in range(len(b)):
#     g.append((bi+0.01)/(0.01+b[i]))

# plt.plot(b,g,label="g(b)")
# plt.xlabel("batterie %")
# plt.ylabel("g(b)")
# # plt.axvline(x=5,color="orange",label="x=5")
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres donnés ---
d_min = 0.2
d_max = 0.8
seuil = 0.15
P = 2

# --- Définition des fonctions ---
def p_dist(d):
    """Calcule p_dist en fonction de d_norm"""
    mid = (d_min + d_max)/2
        
    poids_distance = ((d - mid)/mid)**2
    return poids_distance

def p_bat(b):
    """Calcule p_bat en fonction de b_norm"""
    if b > seuil:
        return (1 - b) ** 2
    else:
        return (1 - b) ** 2 + P

# Vectorisation des fonctions pour les appliquer facilement sur des tableaux numpy
v_p_dist = np.vectorize(p_dist)
v_p_bat = np.vectorize(p_bat)

# --- Génération des données ---
# Les variables normalisées évoluent entre 0 et 1
x = np.linspace(0, 1, 500)
y_dist = v_p_dist(x)
y_bat = v_p_bat(x)

# --- Tracé des courbes ---
plt.figure(figsize=(12, 5))

# Subplot 1 : Courbe de p_dist
plt.subplot(1, 2, 1)
plt.plot(x, y_dist, label=r'$p_{\mathrm{dist}}$', color='royalblue', linewidth=2)
plt.axvline(d_min, color='gray', linestyle='--', alpha=0.7, label=r'$d_{\min} = 0.2$')
plt.axvline(d_max, color='gray', linestyle='--', alpha=0.7, label=r'$d_{\max} = 0.8$')
plt.xlabel(r'$d_{\mathrm{norm}}$')
plt.ylabel(r'$p_{\mathrm{dist}}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Subplot 2 : Courbe de p_bat
plt.subplot(1, 2, 2)
plt.plot(x, y_bat, label=r'$p_{\mathrm{bat}}$', color='crimson', linewidth=2)
plt.axvline(seuil, color='gray', linestyle='--', alpha=0.7, label=r'$\mathrm{seuil} = 0.15$')
plt.xlabel(r'$b_{\mathrm{norm}}$')
plt.ylabel(r'$p_{\mathrm{bat}}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Affichage
plt.tight_layout()
plt.show()
