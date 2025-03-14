---
share: true
---
[TIPE](file:///C:%5CUsers%5Cmillo%5CDesktop%5CTIPE)

Si sinistre ou catastrophe naturel ou opération secours milieu isolé $\Rightarrow$ réseaux traditionnels pas dispos donc faut un P2P ou alors une nœud central pour faire communiquer secouristes, drones, équipements….
$\Rightarrow$ ad hoc network, infrastructure-less network

Crucial avoir faible latence (ne pas retarder opération) **et surtout** fiabilité (ne pas perdre données importantes : coordonnées, images...)
Couverture réseau ou capacité à capter peut ê modélisée par des graphes en fct de la position : Random Waypoint ? mais aussi batterie ou bien traditionnellement qualité connexion
$\Rightarrow$ faire fonction qui prend cela en compte pour recalculer meilleurs chemins toutes les $x$ minutes par exemple

Donc créer une fonction qui calcule la couverture réseau simplement en fonction de la position et faire se déplacer les noeuds aléatoirement


https://chatgpt.com/c/67caefc2-2b58-8011-8077-41a73283d05c
[Routage ad hoc — Wikipédia](https://fr.wikipedia.org/wiki/Routage_ad_hoc)
___

Il faut une propriété originale sur le reseau pour rendre le sujet intéressant et "recherchable"


Probabilité de fiabilité d'un nœud variable en fonction de la position (modélisé par un graphe) ex : opérations de sauvetage, couverture réseau variable.
Protocoles P2P classique -> génèrent boucles ou rupture chemins -> ajustement dynamique nécessaire 
$\implies$ algorithmes pour privilégier nœuds avec plus haute proba de fiabilité optimiser latence et réduire perte paquets

**Cycles et boucles** :
- *count to infinity*
- [Rapport_Lohier, page 6](./TIPE/Rapport_Lohier.pdf.md#page=6&selection=1,0,5,7)

![Pasted image 20250314133605.png](./Pasted/Pasted%20image%2020250314133605.png)