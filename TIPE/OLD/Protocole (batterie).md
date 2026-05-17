---
share: true
---
___
##### Idées features algorithme :
Protocole de routage se basant sur AODV (norme RFC 3561)
- Pénaliser fortement les routes comprenant des appareils avec peu de batterie (pondération des arcs prend en compte batterie)
	- Si batterie inférieure à un seuil `p` : pénaliser très fortement afin que le nœud ne soit utilisé que si il n'y a pas d'autres alternatives
	- Sinon poids = $\alpha*\text{distance} + \beta*1/\text{batterie}$
- Système de mise en veille et réveil pour des messages urgents (pas compliqué avec SimPy surement)
- Voir : [DeepSeek](https://chat.deepseek.com/a/chat/s/4e2a9815-fd19-4653-b3c2-a087f881b637)
- Réduction puissance d'émition en fonction de la distance du voisin à qui on envoie des données (pas sûr)
- Adaptation d'AODV : au lieu de choisir une route avec le nombre de sauts minimal on ajoute la batterie et la distance en métriques à prendre en compte 

___
#### Paramètres de simulation :
[Energy Consumption Evaluation of AODV and AOMDV Routing Protocols](https://thesai.org/Downloads/Volume9No8/Paper_35-Energy_Consumption_Evaluation_of_AODV.pdf?utm_source=chatgpt.com)
![400](../../Z%20-%20Pasted/Z%20-%20MP2I/Protocole%20_batterie_-1750688194203.png)
![400](../../Z%20-%20Pasted/Z%20-%20MP2I/Protocole%20_batterie_-1750688204264.png)

![Protocole _batterie_-1756486405604.png](../../Pasted/Protocole%20_batterie_-1756486405604.png) depuis [LANOMS 2003 Proceedings - document](https://citeseerx.ist.psu.edu/document?doi=75cc4e7b0705eb71a85662dfeae59cd2eef94796&repid=rep1&type=pdf&utm_source=chatgpt.com)

[DeepSeek - Into the Unknown](https://chat.deepseek.com/a/chat/s/0cfdbebc-cdb9-4579-8c2a-71802b342e13) $\implies$ 250m pour la portée

[Source de chatgpt](https://chatgpt.com/c/68595b82-9e04-8011-81ab-587a8c4fb44a) si besoin
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
- file des messages à traiter : Simpy 'store'
- Distance d'émisssion maximum
- alive : indique si il reste de la batterie au noeud
- network : classe `Network` dans lequel le noeud se situe

###### Méthodes
- Process_messages : boucle ``while True`` récupérant les messages dans la file des messages
- 

##### Classe network : 
###### Attributs 
- graphe NetworkX
- environnement simpy
- consomation : (x,y) où x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message (en fonction de la longueur du message peut-être plus tard)
- seuil de batterie en dessous duquel on pénalise fortement les noeuds
- a,b : paramètres de pondération des arcs : weight = `a*distance + b*(1/batterie)`
###### Méthodes
- update_battery : met à jour la batterie du noeud en fonction du type de message : rrep/rreq ou données, si le noeud n'a plus de batterie on supprime toutes les liaisons à ce dernier et renvoie un booléen indiquant si il reste de la batterie au noeud
- 
___
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


