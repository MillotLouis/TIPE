---
share: true
---
### Positionnements thématiques et mots-clés
---
**Positionnements thématiques**:
- Informatique pratique
- Technologies informatiques

**Mots clés** :

| Français                     | Anglais                       |
| ---------------------------- | ----------------------------- |
| 1. Réseaux ad hoc (MANET)    | Ad-hoc networks (MANETs)      |
| 2. Routage AODV              | AODV routing                  |
| 3. Économie d'énergie        | Energy-aware/Energy-efficient |
| 4. Optimisation paramétrique | Parametric Optimization       |
| 5. Heuristique               | Heuristic                     |

 
### Bibliographie commentée
---
Parfois les infrastructures nécessaires pour faire fonctionner un réseau d'appareil en passant par des points relais afin de faire communiquer les appareils indirectement sont absentes. C'est le cas lors de situations d'urgence (des catastrophes naturelles par exemple). La solution prépondérante pour palier à ce manque est la mise en place d'un réseau *ad-hoc*. Ceci consiste à faire communiquer les appareils directement entre eux au lieu de passer par un ou plusieurs point relais centraux. 
Parmi les deux principaux types de protocoles permettant de mettre en place de tels réseaux, je m'intéresserai ici aux protocoles dits *réactifs* ou *à la demande*, qui établissent une route (une suite de noeuds qui vont transmettre le message) entre deux noeuds seulement quand cette dernière est requise. Plus particulièrement au protocole de routage AODV (Ad-hoc On-demand Distance Vector) dans le cas de petits réseaux d'appareils mobiles tels que des radios ou des caméras piétons. 
Ce protocole, décrit dans les années 2000 par C. Perkins \[1\] est un des plus utilisés aujourd'hui dans cette catégorie et un des plus efficaces en termes de consommation énergétique \[2\], ce qui a motivé le choix de ce protocole.
Cependant, pour choisir une route, il ne prend en compte que le nombre de sauts c'est à dire de noeuds empruntés sur le chemin, qu'il cherche à minimiser. Cela tend à surcharger certains noeuds ce qui vide plus rapidement leur batterie, diminuant potentiellement la durée de vie du réseau et/ou sa connectivité. \[3\] 
De nombreuses études ont eu pour objectif d'améliorer la durée de vie du réseau,
Certaines tentent de réduire la consommation de batterie en réduisant le traffic passant par les noeuds ayant peu de batterie \[4\] ou bien en réduisant le nombre de messages requis pour faire fonctionner le réseau. \[5\]
D'autres (\*) s'attaquent à la racine du problème en réduisant la charge sur les noeuds surexploités. 
>(Note : Je ne sais pas si je devrais citer / détailler ces méthodes (\*) étant donné que je ne les utilise pas vraiment mais je présente juste les limites qu'elles ont afin d'essayer de dépasser ces dernières.)

Cependant, ces améliorations bien qu'ayant montré des résultats positifs, exposant au passage une réelle marge d'amélioration, présentent des limites. 
Soit elles sont bien trop complexes à implémenter ou bien trop lourdes à calculer, les rendant incompatibles avec les réseaux de systèmes embarqués ayant des ressources limitées. 
Soit les calculs sont plus légers mais reposent souvent sur des pondérations fixées de manière empirique ou générique \[3\], laissant une autre marge d'amélioration dans ce contexte.
Pour palier à cela, des méthodes permettant de calculer dynamiquement les meilleurs pondérations ont été mises au point \[3\]. Mais ces calculs sont eux aussi lourds et complexes pour des systèmes embarqués tels que ceux présentés ici.

La difficulté ici réside donc dans l'équilibre entre complexité de calculs et pondérations optimales afin de ne pas être contre-productif en menant des calculs trop lourds qui consomment de la batterie, tout en optimisant au mieux la consommation de cette dernière à travers des améliorations du protocole AODV.

### Problématique retenue
---
Comment l'intégration d'une métrique de routage dépendant de l'énergie résiduelle dans le protocole AODV permet-elle d'équilibrer la charge du réseau et d'en augmenter la durée de vie ?

### Objectifs
---
1. Implémenter le protocole AODV en python ainsi que la consommation énergétique des noeuds.
2. Déterminer des améliorations possibles dans le protocole prenant en compte la batterie et les implémenter.
3. Déterminer à l'aide de simulations les paramètres optimaux dans ce contexte.
4. Comparer enfin les gains et/ou pertes sur plusieurs critères par rapport au protocole AODV.

### Références bibliographiques
---
\[1\] *Ad hoc On-Demand Distance Vector (AODV) Routing*, C. Perkins (Norme RFC 356), Juillet 2003 (http://www.rfc.fr/rfc/en/rfc3561.pdf)

\[2\] *Network Lifetime Analysis of AODV, DSR and ZRP at Different Network Parameters* Juillet 2012(https://arxiv.org/abs/1207.2584)

\[3\] *AODV-EOCW: An Energy-Optimized Combined Weighting AODV Protocol for Mobile Ad Hoc Networks*(https://www.mdpi.com/1424-8220/23/15/6759#B8-sensors-23-06759)

\[4\] *Implementation and Performance Evaluation of an Energy Constraint Routing Protocol for Mobile Ad Hoc Networks*, (https://ieeexplore.ieee.org/abstract/document/4215234)

\[5\] *New routing protocol “Dicho-AODV” for energy optimization in MANETS* (https://ieeexplore.ieee.org/abstract/document/6320126)
