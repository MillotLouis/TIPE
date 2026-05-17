---
share: true
---
### Motivation
---
Étant personnellement intéressé par le domaine du réseau et des télécommunication et étant dérangé par la faible autonomie des drones grand public, j'ai concilié ces deux points et me suis demandé comment ce problème affectait il plus généralement les petits réseaux d'appareils mobiles.


### Ancrage au thème de l'année
---
Le protocole AODV à 

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

 
### Bibliographie commentée
---
La plupart des réseaux de communication aujourd'hui fonctionnent en se basant sur des points relais : une box pour le wifi, des antennes relai pour les téléphones, ...
Mais parfois les infrastructures nécessaires sont absentes. C'est le cas lors de situations d'urgence (des catastrophes naturelles par exemple). La solution prépondérante pour pallier ce manque est la mise en place d'un réseau dit *ad-hoc*. Ceci consiste à faire communiquer les appareils directement entre eux afin de se transmettre les messages de proche en proche (communication dite peer-to-peer) au lieu de passer par un ou plusieurs point relais centraux. Une "route" est donc définie comme une suite de noeuds qui vont transmettre le message d'un émetteur à un récepteur.
Il existe deux grand types de protocoles permettant de mettre en place de tels réseaux : ceux dits proactifs qui gardent en mémoire les différentes routes pour communiquer avec les autre noeuds et ceux dits *réactifs* ou *à la demande* qui établissent une route entre deux noeuds seulement quand cette dernière est requise.

Dans les années 2000, C. Perkins \[1\] décrit le protocole AODV (Ad-hoc On-demand Distance Vector), un protocole réactif qui établit des routes en propageant une requête à travers tout le réseau. Il est aujourd'hui l'un des plus utilisé dans cette catégorie et un des plus efficaces en termes de consommation énergétique \[2\]
Cependant, pour choisir une route, il ne prend en compte que le nombre de sauts c'est à dire de noeuds empruntés sur le chemin, qu'il cherche à minimiser. Cela tend à surcharger les noeuds centraux ce qui vide plus rapidement leur batterie, diminuant la durée de vie du réseau et/ou sa connectivité \[3\]. Ce problème se pose particulièrement dans le cas de petits réseaux d'appareils mobiles tels que des radios ou des caméras piétons. 

De nombreuses études ont eu pour objectif d'améliorer la durée de vie du réseau.
Une première approche est l'introduction de seuils de batterie en dessous desquels un nœud ne relaie pas les messages qu'il reçoit \[4\] ou la réduction du nombre de messages requis pour faire fonctionner le réseau. \[5\]
Une autre approche proposée est d'améliorer les métriques de routage : ne plus seulement choisir le chemin le plus court mais prendre en compte la batterie restante des noeuds, la congestion à certains points du réseau ou d'autres facteurs. \[6\]
Le défi majeur repose alors dans la pondération de ces différents critères dans le choix d'une route. Des méthodes permettant de calculer en temps réel ces pondérations ont étés proposées tels que des calculs basés sur l'entropie \[3\]. Cependant elles sont complexes à implémenter et lourdes à calculer, les rendant incompatibles avec les réseaux de systèmes embarqués ayant des ressources limitées \[3\].
Des solutions avec des calculs moins coûteux ont été proposées mais reposent sur des pondérations fixées \[4\] de manière empirique ou générique n'étant donc pas nécessairement optimales dans ce cas d'application. (476 mots)

>L'enjeu ici est donc de déterminer les pondérations optimales pour un protocole de routage modifié afin de garder une légèreté en calculs tout en améliorant l'efficacité, ceci grâce à une phase de simulation préalable ayant déterminé les coefficients optimaux.
### Problématique retenue
---
Comment l'intégration et l'optimisation paramétrique d'une métrique de routage dans le protocole AODV permet-elle d'optimiser l'utilisation de la batterie ? 

### Objectifs
---
1. Implémenter le protocole AODV en python ainsi que la consommation énergétique des noeuds.
2. Déterminer des améliorations possibles dans le protocole prenant en compte la batterie et les implémenter.
3. Déterminer à l'aide de simulations les paramètres optimaux dans ce contexte dans différents cas.
4. Comparer les gains et/ou pertes sur plusieurs critères par rapport au protocole AODV.

### Références bibliographiques
---
\[1\] *Ad hoc On-Demand Distance Vector (AODV) Routing*, C. Perkins (Norme RFC 356), Juillet 2003 (http://www.rfc.fr/rfc/en/rfc3561.pdf)

\[2\] *Network Lifetime Analysis of AODV, DSR and ZRP at Different Network Parameters* Juillet 2012(https://arxiv.org/abs/1207.2584)

\[3\] *AODV-EOCW: An Energy-Optimized Combined Weighting AODV Protocol for Mobile Ad Hoc Networks*(https://www.mdpi.com/1424-8220/23/15/6759#B8-sensors-23-06759)

\[4\] *Implementation and Performance Evaluation of an Energy Constraint Routing Protocol for Mobile Ad Hoc Networks*, (https://ieeexplore.ieee.org/abstract/document/4215234)

\[5\] *New routing protocol “Dicho-AODV” for energy optimization in MANETS* (https://ieeexplore.ieee.org/abstract/document/6320126)

\[6\] Enhancing the Performance of AODV Using Node Remaining Energy and Aggregate Interface Queue Length (https://ieeexplore.ieee.org/abstract/document/6724327)