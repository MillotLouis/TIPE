---
share: true
---
Je vais probablement utiliser NetworkX et simPy
[SimPy in 10 Minutes — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html)
[Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html)
##### À faire :
- [x] Utilisation SimPy
- [x] Utilisation NetworkX
- [x] Écrire bon algorithme pseudo code avant d'implémenter
- [x] Documentation recherches sur le sujet
- [ ] Voir si je discard toutes les RREQ après en avoir vu une venant de la même source ou bien si je discard uniquement celles venant de la même source et d'un voisin déja vu car sinon je risque de discard un meilleur chemin potentiel si il a un noeud qui a plusieurs voisins lui envoyant la RREQ
- [ ] Ajouter TTL aux routes (à peu près fréquence "HELLO messages" afin de coller à la réalité)
- [ ] Ajouter consommation de batterie selon la distance
- [ ] Ajouter limite de distance afin de ne pouvoir transmettre qu'à ses voisins les plus proches
- [ ] Ajouter réponse des noeuds intermédiaires
- [ ] Ajouter modèle de mobilité
- [ ] Pour compter nombre de paquets perdus : à chaque fois qu'un nœud mort reçoit un message (Pas une RREQ) qu'il est censé forwarder alors on incrémente et on compare aux nombre de paquets perdus avec AODV classique qui empruntera possiblement des noeuds avec peu de batterie qui peuvent donc mourir avant la fin du TTL sur une route enregistrée
___
Dans la réalité, AODV est utilisé dans des réseaux de noeuds mobiles donc des connexions se font et se défont donc le protocole AODV est utilisé uniquement afin de recréer une route ou en créer une si elle n'existe pas.
Ici comme pas de déplacements pour l'instant je définis un TTL pour les routes afin de simuler ces créations de routes quand il y a effectivement des déplacements. cf : [AODV](./Technos%20acutelles/AODV.md)

Dans la norme **RFC 3561** décrivant le fonctionnement d'AODV il est écrit : "_The destination node... MUST send a RREP back to the source_" et ce RREP est le premier valide à être reçu par le noeud de destination. à ce moment là le nombre de saut de la route est calculé sur le chemin "retour".
Cependant ici si on fait cela alors la pondération selon la batterie ne pourra pas être prise en compte : il faut donc [ajouter un petit délai d'attente](https://chat.deepseek.com/a/chat/s/101cc16b-010d-48c9-ba9e-eee1aaacbcbc) quand la destination reçoit le premier RREQ émanant de la source afin de tous les collecter et renvoyer uniquement le meilleur. Ici le poids de la route sera calculé à "l'aller" afin de pouvoir comparer à l'arrivée mais aussi au retour (indispensable afin de pouvoir ajouter le bon poids aux noeuds intermédiaires)

De même dans **RFC 3561** si un noeud intermédiaire a un chemin plus frais et valide il renvoie un RREP mais alors le noeud source peut recevoir plusieurs RREP (le noeud de destination renvoie **toujours** un RREP). Dans un premier temps je ne vais pas faire cela : seulement la destination répond
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


