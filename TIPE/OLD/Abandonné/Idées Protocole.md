---
share: true
---
Crucial avoir faible latence (ne pas retarder opération) **et surtout** fiabilité (ne pas perdre données importantes : coordonnées, images...). 
Couverture réseau ou capacité à capter peut ê modélisée par des fonctions en fct de la position : Random Waypoint pour prédiction sommaire ? 

$\Rightarrow$ faire fonction qui prend cela en compte pour recalculer meilleurs chemins toutes les $x$ minutes en prenant en compte :
- batterie restante^[si batterie pas trop faible aucun impact mais si batterie très faible, décroissance exponentielle score ?]
- Position
- Historique de fiabilité
- Qualité de connexion (couverture réseau)
Tout en ayant un **faible coût** (ou moindre par rapport aux protocoles existants), une **faible latence** un faible **temps de convergence** et étant **fiable**
>[!info] Temps de convergence = temps pour recréer les routes en cas de déconnexion

___
Type glouton : chaque noeud stocke ses propres métriques et envoie un message de type hello à ses voisins toutes les $x$ secondes 

Garde table de métriques des noeuds les plus proches et calcule leur score quand il veut transmettre un paquet il l'envoie à celui avec le plus grand score (les deux plus grand peut-être pour limiter perte potentielle : faire en premier sans dire que pas ouf et proposer cette modification avec d'autres peut être)

___ 
**[Implementation possible](../../../Implementation%20possible.md)**
