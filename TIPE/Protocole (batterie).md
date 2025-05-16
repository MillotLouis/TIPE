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
- [ ] Documentation recherches sur le sujet

##### Idées features algorithme : 
- Pénaliser fortement les routes comprenant des appareils avec peu de batterie (pondération des arcs prend en compte batterie)
	- Si batterie inférieure à un seuil `p` : pénaliser très fortement afin que le nœud ne soit utilisé que si il n'y a pas d'autres alternatives
	- Sinon poids = $\alpha*\text{distance} + \beta*1/\text{batterie}$
- Système de mise en veille et réveil pour des messages urgents (pas compliqué avec SimPy surement)
- Voir : [DeepSeek](https://chat.deepseek.com/a/chat/s/4e2a9815-fd19-4653-b3c2-a087f881b637)
- Réduction puissance d'émition en fonction de la distance du voisin à qui on envoie des données (pas sûr)
- Adaptation d'AODV : au lieu de choisir une route avec le nombre de sauts minimal on ajoute la batterie et la distance en métriques à prendre en compte 

___
#### Implémentation :
##### Classe Node :
###### Attributs 
- environnement SimPy
- id
- position
- batterie
- table de routage : dictionnaire { destination : (next_hop, weight, seq_num, lifetime) }
- numéro de séquence 
- queue des messages à traiter : Simpy 'store'
- Distance d'émisssion maximum

##### Classe network : 
###### Attributs 
- graphe NetworkX
- environnement simpy
- consomation : (x,y) où x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message
- seuil de batterie en dessous duquel on pénalise fortement les noeuds
- a,b : paramètres de pondération des arcs : weight = a*distance + b*(1/batterie)
###### Méthodes
- add_node
- add_link
- 
---
***Obsolète*** : **Dijkstra**
##### Pseudo-Code :
- Pour chaque nœud, Exécuter algorithme de Dijkstra modifié :
	- Créer file de priorité, tableau des distances min et des prédécesseurs
	- Tant que file non vide :
		- Choisir nœud de la file avec distance à la source min
		- Le sortir de la file
		- Si la distance associée à ce noeud dans la file est supérieure à celle stockée dans le tableau des distance min, continuer car ça signifie que cette entrée dans la file était obsolète car on a trouvé mieux avant (car on ne peut pas modifier les elements de la file en utilisant le module python)
		- Pour tous les voisins de ce noeud : 
			- Si la batterie est inférieure à un seuil `p` : 
				- Poids = valeur très grande devant poids possibles
			- Sinon :
				- Poids = $\alpha*\text{distance} + \beta*1/\text{batterie}$ 
			- Actualiser distance et prédécesseur dans tableau : min entre poids actuel et poids en passant par le nœud qui vient d'être pop


