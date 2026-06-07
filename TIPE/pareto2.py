f1 = [
    805.97087259,
    806.1803659,
    806.20654724,
    806.9337376,
    812.16751668,
    815.55691963,
    816.83590893,
    817.26589878
]

f2 = [
    8.18417442,
    8.22378725,
    8.2946839,
    8.30901076,
    8.39021268,
    8.45152383,
    8.45821088,
    8.59968
]

# Exemple de tracé :
import matplotlib.pyplot as plt
plt.plot(f1, f2, marker='o')
plt.xlabel("Temps avant 10 % de nœuds morts (s)")
plt.ylabel("Écart type énergie finale")
plt.title("Front de Pareto obtenu par NSGA-II")
plt.grid(True, alpha=0.3)
plt.show()