---
share: true
---
Je vais probablement utiliser NetworkX et simPy
[SimPy in 10 Minutes — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html)
[Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html)
##### À faire :
- [x] Utilisation SimPy
- [x] Utilisation NetworkX
- [ ] Écrire bon algorithme pseudo code avant d'implémenter

##### Idées features algorithme : 
- Pénaliser fortement les routes comprenant des appareils avec peu de batterie (pondération des arcs prend en compte batterie)
- Système de mise en veille et réveil pour des messages urgents (pas compliqué avec SimPy surement)
- voir : [DeepSeek - Into the Unknown](https://chat.deepseek.com/a/chat/s/4e2a9815-fd19-4653-b3c2-a087f881b637)
- Réduction puissance d'émition en fonction de la distance du voisin à qui on envoie des donées

[panisson/pymobility: python implementation of mobility models](https://github.com/panisson/pymobility)

##### Pseudo-Code :
- Permet de rafraichir la table de routage (toutes les `x` secondes) 
	- Pour chaque noeud, Exécuter algorithme de Dijkstra modifié :
		- Rendre tous les sommets bleus
		- Créer file verte de priorité, tableau des distances min et des prédécesseurs
		- Tant que file verte non vide :
			- Choisir noeud vert avec distance à la source min
			- Le sortir de la file, le rendre rouge
			- Pour tous les voisins de ce noeud : 
				- Si noeud bleu devient vert 
				- Actualiser distance et predécesseur dans tableau : min entre poids actuelle et poids en passant par le noeud
					- Si le noeud a moins de `p` pourcentage de batterie : mettre un poids très grand afin de ne choisir ce noeud que si il n'y a pas le choix
					- Sinon poids = $\alpha*\text{distance} + \beta*1/\text{batterie}$  à ajuster pour voir meilleur résultat
