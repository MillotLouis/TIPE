---
share: true
---
[TIPE](file:///C:%5CUsers%5Cmillo%5CDesktop%5CTIPE)

Si sinistre ou catastrophe naturelle ou opération secours milieu isolé ou opération militaire $\Rightarrow$ réseaux traditionnels pas dispos donc faut un P2P pour faire communiquer secouristes, drones, équipements….
$\Rightarrow$ ad hoc network, infrastructure-less network

> Un **réseau ad hoc** (généralement appelé MANet (*mobile ad hoc network*)) est un réseau décentralisé où les appareils (nœuds) communiquent directement entre eux, sans dépendre d'une infrastructure fixe (comme un routeur ou un point d'accès Wi-Fi). Il est souvent utilisé dans des situations où une configuration rapide et flexible est nécessaire, comme en milieu militaire, dans des zones sinistrées, ou pour des connexions temporaires entre appareils mobiles.

Crucial avoir faible latence (ne pas retarder opération) **et surtout** fiabilité (ne pas perdre données importantes : coordonnées, images...) Mais pas grosse puissance de calcul
Couverture réseau ou capacité à capter peut ê modélisée par des fonctions en fct de la position : Random Waypoint pour prédiction sommaire ? mais aussi batterie^[si batterie pas trop faible aucun impact mais si batterie très faible, décroissance exponentielle score maybe] ou bien traditionnellement qualité connexion
$\Rightarrow$ faire fonction qui prend cela en compte pour recalculer meilleurs chemins toutes les $x$ minutes par exemple

Donc créer une fonction qui calcule la couverture réseau simplement en fonction de la position et faire se déplacer les noeuds aléatoirement

**État actuel de la technologie** :

[État actuel technologie](./%C3%89tat%20actuel%20technologie.md)

> [!info]
> Aujourd'hui aucun programme ne combine vraiment tout cela : batterie restante, position, historique de fiabilité (trajectoire prédite : random waypoint mais probablement trop compliqué)


___
___
Il faut une propriété originale sur le reseau pour rendre le sujet intéressant et "recherchable"


Probabilité de fiabilité d'un nœud variable en fonction de la position (modélisé par un graphe) ex : opérations de sauvetage, couverture réseau variable.
Protocoles P2P classique -> génèrent boucles ou rupture chemins -> ajustement dynamique nécessaire 
$\implies$ algorithmes pour privilégier nœuds avec plus haute proba de fiabilité optimiser latence et réduire perte paquets

**Cycles et boucles** :
- *count to infinity*
- [Rapport_Lohier, page 6](./TIPE/Rapport_Lohier.pdf.md#page=6&selection=1,0,5,7) AODV
- 

**Sources et liens** : 
https://chatgpt.com/c/67caefc2-2b58-8011-8077-41a73283d05c
[Routage ad hoc — Wikipédia](https://fr.wikipedia.org/wiki/Routage_ad_hoc)
https://chat.deepseek.com/a/chat/s/47e9b763-6d1d-443a-8c26-63b40a7f9914
[Chapitre 3 - Chapitre3.pdf](http://opera.inrialpes.fr/people/Tayeb.Lemlouma/Papers/MasterThesis/Chapitre3.pdf)
[INTRODUCTION - AdHoc_Presentation.pdf](http://opera.inrialpes.fr/people/Tayeb.Lemlouma/Papers/AdHoc_Presentation.pdf)
[BonnMotion - A mobility scenario generation and analysis tool](https://sys.cs.uos.de/bonnmotion/)
**Sources/thèses/papers sur le sujet :**
[A Composite Mobility Model for Ad Hoc Networks in Disaster Areas - CentraleSupélec](https://centralesupelec.hal.science/hal-00589846v1)