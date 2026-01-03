---
share: true
---
** Je vais probablement utiliser NetworkX et simPy
[SimPy in 10 Minutes — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html)
[Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html)
##### À faire :
- [x] Utilisation SimPy
- [x] Utilisation NetworkX
- [x] Écrire bon algorithme pseudo code avant d'implémenter
- [x] Documentation recherches sur le sujet
- [x] Voir si je discard toutes les RREQ après en avoir vu une venant de la même source ou bien si je discard uniquement celles venant de la même source et d'un voisin déjà vues car sinon je risque de discard un meilleur chemin potentiel si il a un noeud qui a plusieurs voisins lui envoyant la RREQ $\implies$ ok au final je les regarde toutes cf 4. 
- [x] Ajouter consommation de batterie selon la distance
- [x] Ajouter limite de distance afin de ne pouvoir transmettre qu'à ses voisins les plus proches
- [x] modifier pour pénaliser fortement au lieu de ne pas transmettre du tout le rreq si batterie en dessous du seuil : ![50](../Z%20-%20Pasted/Z%20-%20MP2I/Protocole%20_batterie_-1749980583851.png)
- [x] Ajouter TTL aux routes (à peu près fréquence "HELLO messages" afin de coller à la réalité) 
- [x] Arrêter la simu si la moitié des noeuds sont morts i.e. `network.stop = True`
- [x] **Important** : Actuellement je prends tout les RREQs avec le même (src_id,rsc_seq) venant de tous les voisins => Problème un voisin peut recevoir plusieurs fois un RREQ avec un chemin différent du même voisin donc il faudrait peut-être ajouter weight
- [x] ==Modeler puissance émission si distance + basse en envoyant des paquets data== pour mod aodv
- [ ] Ajouter une consommation de batterie à la reception ? Me paraît négligeable comparé à emission [Consommation radios tactiques](https://chatgpt.com/c/684e8756-d540-8011-a4ac-28612f8f2609) à voir
	- Serait potentiellement plus que émission ? [Optimisation AODV Réseau Ad-Hoc](https://chatgpt.com/c/68595b82-9e04-8011-81ab-587a8c4fb44a)
	- **Non** voir plus bas, paramètres de simulation
- [x] Ajouter modèle de mobilité
- [x] Pour compter nombre de paquets perdus : à chaque fois qu'un nœud mort reçoit un message (Pas une RREQ) qu'il est censé forwarder alors on incrémente et on compare aux nombre de paquets perdus avec AODV classique qui empruntera possiblement des noeuds avec peu de batterie qui peuvent donc mourir avant la fin du TTL sur une route enregistrée
- [ ] **Rajouter TTL ?**
- [ ] Ajouter réponse des noeuds intermédiaires
	https://chatgpt.com/g/g-p-68ae01179a9c819188df4800374ab039-louis/c/68fa54f8-4f28-832c-9d4f-c56ed61d3cfe
	- [ ] Il faudrait ajouter un TTL alors pour pas que des routes pas optimales soient gardées trop longtemps
___
1. Dans la réalité, AODV est utilisé dans des réseaux de noeuds mobiles donc des connexions se font et se défont donc le protocole AODV est utilisé uniquement afin de recréer une route ou en créer une si elle n'existe pas.
	Ici comme pas de déplacements pour l'instant je définis un TTL pour les routes afin de simuler ces créations de routes quand il y a effectivement des déplacements. cf : [AODV](./Technos%20acutelles/AODV.md)

2. Dans la norme **RFC 3561** décrivant le fonctionnement d'AODV il est écrit : "_The destination node... MUST send a RREP back to the source_" et ce RREP est le premier valide à être reçu par le noeud de destination. à ce moment là le nombre de saut de la route est calculé sur le chemin "retour".
	Cependant ici si on fait cela alors la pondération selon la batterie ne pourra pas être prise en compte : il faut donc [ajouter un petit délai d'attente](https://chat.deepseek.com/a/chat/s/101cc16b-010d-48c9-ba9e-eee1aaacbcbc) quand la destination reçoit le premier RREQ émanant de la source afin de tous les collecter et renvoyer uniquement le meilleur. Ici le poids de la route sera calculé à "l'aller" afin de pouvoir comparer à l'arrivée mais aussi au retour (indispensable afin de pouvoir ajouter le bon poids aux noeuds intermédiaires)

3. De même dans **RFC 3561** si un noeud intermédiaire a un chemin plus frais et valide il renvoie un RREP mais alors le noeud source peut recevoir plusieurs RREP (le noeud de destination renvoie **toujours** un RREP). Dans un premier temps je ne vais pas faire cela, seulement la destination répond

4. De plus dans AODV quand un noeud reçoit une RREQ il discard toutes les autres requêtes venant de ce noeud avec le même seq_num ie toutes celles lancées en même temps qui se sont séparées à des embranchements. Ici si je fait ca je vais perdre la comparaison avec la batterie car le noeud empruntant le chemin le plus rapide atteindra le noeud A en premier et tous les suivants seront discard même si ils ont un meilleur poids en pondérant par la batterie donc il faut que je regarde tous les RREQ venant d'un même noeud source avec un même seq_num : (source,seq_num,voisin) au lieu de juste (source,seq_num) dans le dico des RREQ vues
	1. **Update**: Si on fait ça on peut perdre des chemins meilleurs passant par le même prev_hop => pas bien. **ET** transmet des RREQs inutiles car ne seront pas choisis car poids plus grand qu'un qui est déjà passe
	2. Donc il faut comparer avec un dico `seen = (rreq.src_id, rreq.src_seq, rreq.prev_hop) : meilleur poids`
	3. Si je fais ça ça réduit de plus de 3x le nb de transmissions => noice

5. Normalement quand un noeud reçoit un RREQ de la source il ajoute un reverse path à sa table de routage pointant vers la source utilisant le chemin emprunté par le RREQ jusqu'ici pour pouvoir forward le RREP si ce chemin est choisi. Cependant cette entrée est marquée comme invalide, réservée à ce RREP, ici pour simplifier je n'implémente pas ça : les chemins vers la source sont mis à jour tout au long du chemin du RREQ si ils sont plus récent ou mieux et peuvent être utilisés pour n'importe quelle transmission

6. J'avais un problème : les délais n'était pas randomisés donc tous les RREQs était scheduled pour t = 0.0 et cela quasi à l'infini donc le temps de la simulation  n'avancait pas => RREPs pas collectés donc pas envoyés

7. Galère pour représenter les données sachant que je ne peux pas forcément faire varier un seul paramètre en fixant les autres cf [Analyse](./Analyse.md)

8. Au début je comprenais pas pk les résultats étaient mauvais en comparant la mort de 50% des noeuds mais c'est sûrement que comme il n'y a plus bcp de noeuds vivants avec reg_aodv ou que les noeuds "critiques"/"centraux" sont morts les messages ne peuvent pas être envoyés correctement. **à verifier**

9. à faire : tracer le delivery ratio et d'autres métriques de transfer de data afin de montrer ↑ vrai
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
![400](../Z%20-%20Pasted/Z%20-%20MP2I/Protocole%20_batterie_-1750688194203.png)
![400](../Z%20-%20Pasted/Z%20-%20MP2I/Protocole%20_batterie_-1750688204264.png)

![Protocole _batterie_-1756486405604.png](../Pasted/Protocole%20_batterie_-1756486405604.png) depuis [LANOMS 2003 Proceedings - document](https://citeseerx.ist.psu.edu/document?doi=75cc4e7b0705eb71a85662dfeae59cd2eef94796&repid=rep1&type=pdf&utm_source=chatgpt.com)

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


