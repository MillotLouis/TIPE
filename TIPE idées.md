---
share: true
---
[TIPE](file:///C:%5CUsers%5Cmillo%5CDesktop%5CTIPE)

Si sinistre ou catastrophe naturel ou opération secours milieu isolé $\Rightarrow$ réseaux traditionnels pas dispos donc faut un P2P ou alors une nœud central pour faire communiquer secouristes, drones, équipements….
$\Rightarrow$ ad hoc network, infrastructure-less network

Crucial avoir faible latence (ne pas retarder opération) **et surtout** fiabilité (ne pas perdre données importantes : coordonnées, images...)
Couverture réseau ou capacité à capter peut ê modélisée par des graphes en fct de la position : Random Waypoint ? mais aussi batterie ou bien traditionellement qualité connexion
$\Rightarrow$ faire fonction qui prend cela en compte pour recalculer meilleurs chemins toutes les $x$ minutes par exemple




https://chatgpt.com/c/67caefc2-2b58-8011-8077-41a73283d05c




___
___
___ 
Boucles réseau, routage infini ? 

**Protocole Spanning Tree (STP)** : Graphes, lien avec maths : parcours pour modéliser trafic
à quelle échelle devient intéressant ? 

optimiser réseau pour empêcher cycles/boucles 
Plus court chemin graphe pour lien réseau
à l'échelle mondiale gain avec optimisation ?

Relier tous les nœuds réseau : Graphe connexe (tous les sommets peuvent reliés) et besoin graphe couvrant, graphe simple :
![100](./Pasted/Pasted%20image%2020250207120104.png)   
éviter boucles (spanning tree)

arbre couvrant/ graphe connexe / graphe acyclique / 

Shortest path problem

Complexité 

P2P ?
 
VPN Tree Routing Conjecture

[Méthodes Algorithmiques, Simulation et Combinatoire pour l’Optimisation des Télécommunications](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6cbe53f408237a0348a2aeb4019343aecc91ee87)

[A Short Introduction to Graph Theory Modélisation et Performance des Réseaux](https://marceaucoupechoux.wp.imt.fr/files/2018/02/graphtheory.pdf)

[Optimisation robuste des réseaux de télécommunications - TEL - Thèses en ligne](https://theses.hal.science/tel-00321868/)

[Algorithms for Detecting Cycles in Graphs: A Comprehensive Guide - AlgoCademy Blog](https://algocademy.com/blog/algorithms-for-detecting-cycles-in-graphs-a-comprehensive-guide/)

[Spanning Tree Protocol : optimiser le réseau LAN - IONOS](https://www.ionos.fr/digitalguide/serveur/know-how/spanning-tree-protocol/)

[These_MargueriteFaycal_280510_v1 - These_MargueriteFaycal_280510_v2.pdf](https://pastel.hal.science/pastel-00521935v1/document)


Il faut une propriété originale sur le reseau pour rendre le sujet intéressant et "recherchable"


Probabilité de fiabilité d'un nœud variable en fonction de la position (modélisé par un graphe) ex : opérations de sauvetage, couverture réseau variable.
Protocoles P2P classique -> génèrent boucles ou rupture chemins -> ajustement dynamique nécessaire
$\implies$ algorithmes pour privilégier nœuds avec plus haute proba de fiabilité optimiser latence et réduire perte paquets

