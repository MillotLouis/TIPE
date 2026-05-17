---
share: true
---
## Optimisation paramétrique d'une métrique de routage inspirée du protocole AODV dans le cadre de petits réseaux d'appareils mobiles.
### Motivation
---
Étant personnellement intéressé par le domaine des télécommunications et étant dérangé par la faible autonomie des drones grand public, j'ai concilié ces deux points et me suis demandé comment ce problème affectait plus généralement les petits réseaux d'appareils mobiles. Plus particulièrement en termes de consommation due à l'échange de messages.

Étant personnellement intéressé par le domaine des télécommunications et dérangé par la faible autonomie des drones grand public. J'ai donc concilié ces deux points en me demandant comment ce problème affectait plus généralement les petits réseaux d'appareils mobiles. du point de vue de la consommation énergétique de l'échange de messages.

### Ancrage au thème de l'année
---
Le principal danger auquel sont exposés les protocoles de routage tels que AODV qui inondent le réseau avec des requêtes est la présence de boucles de routage, fatales pour l'efficacité énergétique. 

Les modifications apportées ici au protocole AODV sont susceptible de générer des boucles de routage, fatales pour la performance mais surtout l'efficacité énergétique. J'ai donc intégré des mécanismes de contrôle pour éviter cela.

### Positionnements thématiques et mots-clés
---
**Positionnements thématiques**:
- Informatique pratique
- Technologies informatiques
- Mathématiques appliquées

**Mots clés** :

| Français                     | Anglais                  |
| ---------------------------- | ------------------------ |
| 1. Réseaux ad hoc (MANET)    | Ad-hoc networks (MANETs) |
| 2. Routage AODV              | AODV routing             |
| 3. Économie d'énergie        | Energy-aware             |
| 4. Optimisation paramétrique | Parametric Optimization  |

 
### Bibliographie commentée
---
La plupart des réseaux de communication sans fil aujourd'hui fonctionnent en se basant sur des points relais : une box pour le wifi, des antennes relai pour les radios, ...

Mais parfois les infrastructures nécessaires sont absentes. C'est le cas par exemple lors de catastrophes naturelles comme des tremblement de terre ayant détruit ces infrastructures ou bien pendant des opérations en zone reculée.

La solution prépondérante pour pallier ce manque est la mise en place d'un réseau dit _ad-hoc_. Ceci consiste à faire communiquer les appareils directement entre eux afin de se transmettre les messages de proche en proche à la façon d'une chaine (communication dite peer-to-peer) au lieu de passer par un ou plusieurs points relais centraux. Une route est donc définie comme une suite de nœuds qui vont se transmettre le message d'un émetteur à un récepteur.

Il existe deux grand types de protocoles permettant de mettre en place de tels réseaux : ceux dits proactifs dans lesquels les nœuds gardent en mémoire les différentes routes pour communiquer avec les autre nœuds et ceux dits _réactifs_ ou _à la demande_ qui établissent une route entre deux nœuds seulement quand cette dernière est requise.

Dans les années 2000, C. Perkins [1] décrit le protocole AODV (Ad-hoc On-demand Distance Vector), un protocole réactif qui établit des routes en propageant une requête à travers tout le réseau. Il est aujourd'hui l'un des plus utilisé dans cette catégorie et un des plus efficaces en termes de consommation énergétique [2].

Cependant, pour choisir une route, il cherche à minimiser seulement le nombre de sauts c'est à dire de nœuds empruntés sur cette route, ainsi que la latence. Cela tend à surcharger les nœuds centraux qui sont plus sollicités. Pour des appareils fonctionnant sur batterie cela vide plus rapidement cette dernière, diminuant la durée de vie du réseau et sa connectivité [3]. Ce problème se pose particulièrement dans le cas de petits réseaux d'appareils mobiles tels que des radios, des capteurs ou des caméras piétons.

De nombreuses études ont eu pour objectif d'améliorer la durée de vie du réseau. Une première approche est l'introduction de seuils de batterie en dessous desquels un nœud ne relaie pas les messages qu'il reçoit [4] ou la réduction du nombre de messages requis pour faire fonctionner le réseau [5].

Une autre approche proposée est d'améliorer les métriques de routage : ne plus seulement choisir le chemin le plus court mais prendre en compte par exemple la batterie restante des nœuds, la congestion à certains points du réseau grâce à la taille de la file d'attente des requêtes à traiter. [6]

Le défi majeur est alors la pondération de ces différents critères dans le choix d'une route. Des méthodes permettant de calculer de façon dynamique ces pondérations ont étés proposées tels que des calculs basés sur l'entropie [3]. Cependant elles sont complexes à implémenter et lourdes à calculer.

Des solutions avec des calculs moins coûteux ont été proposées mais reposent sur des pondérations fixées [4] de manière empirique ou générique pour des cas d'applications bien particuliers, n'étant donc pas nécessairement optimales dans le cas de petits réseaux d'appareils de communication, tels que ceux déployés pendant des opérations de sauvetage.

### Problématique retenue
---
Comment l'intégration et l'optimisation paramétrique d'une métrique de routage prenant en compte la batterie dans le protocole AODV permet-elle d'optimiser l'utilisation de cette dernière ?

ou

Comment l'intégration et l'optimisation paramétrique d'une métrique de routage dans le protocole AODV permet-elle d'optimiser l'utilisation de la batterie ?

### Objectifs
---
1. Implémenter le protocole AODV en Python.

2. Déterminer les améliorations possibles à apporter à ce protocole afin de prendre en compte la batterie et les implémenter.

3. Déterminer l'algorithme d'optimisation paramétrique le plus adapté et déterminer les paramètres optimaux pour ce cas d'application dans différents cas.

4. Comparer les gains et/ou pertes sur plusieurs critères par rapport au protocole AODV classique.

### Références bibliographiques
---
\[1\] *Ad hoc On-Demand Distance Vector (AODV) Routing*, C. Perkins (Norme RFC 356), Juillet 2003 (http://www.rfc.fr/rfc/en/rfc3561.pdf)

\[2\] *Network Lifetime Analysis of AODV, DSR and ZRP at Different Network Parameters* Juillet 2012(https://arxiv.org/abs/1207.2584)

\[3\] *AODV-EOCW: An Energy-Optimized Combined Weighting AODV Protocol for Mobile Ad Hoc Networks*(https://www.mdpi.com/1424-8220/23/15/6759#B8-sensors-23-06759)

\[4\] *Implementation and Performance Evaluation of an Energy Constraint Routing Protocol for Mobile Ad Hoc Networks*, (https://ieeexplore.ieee.org/abstract/document/4215234)

\[5\] *New routing protocol “Dicho-AODV” for energy optimization in MANETS* (https://ieeexplore.ieee.org/abstract/document/6320126)

\[6\] Enhancing the Performance of AODV Using Node Remaining Energy and Aggregate Interface Queue Length (https://ieeexplore.ieee.org/abstract/document/6724327)