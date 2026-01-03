---
share: true
---

Heuristique ? distance à vol d'oiseau ou bien si on se dirige "dans le bon sens"
	Stocker dans la table de routage last known location et si quand on recoit un message depuis un noeud on s'éloigne vraiment (ptet calcul angle) on défavorise la route
Adapter métriques en fonction de la situation pour éviter chute après 50%

---
---
Voici les points clés extraits du texte, classés par type d'approche pour améliorer AODV. Cela te permettra de structurer ta partie "État de l'art" ou "Travaux connexes".

### 1. Constat initial (Les défauts d'AODV)

- **Monocritère :** Se base uniquement sur le nombre de sauts (_hop count_).
    
- **Un chemin unique :** Une seule route est stockée, ce qui surcharge les nœuds de ce chemin.
    
- **Conséquence :** Épuisement rapide des batteries des nœuds sollicités et partitionnement du réseau.
    

### 2. Améliorations ciblées sur l'Énergie (Durée de vie)

L'objectif est d'éviter que les nœuds ne meurent trop vite.

- **Seuils de batterie (Frikha [14]) :** Un nœud refuse de relayer des paquets si son énergie descend sous un certain seuil.
    
- **Routes de secours (Ranjan [13]) :** Création préventive d'une route alternative (backup) basée sur l'estimation de l'énergie et du temps avant rupture du lien.
    
- **Réduction du "bavardage" (Boudhir [15], Darabkh [18]) :** Diminution des messages de contrôle (RREQ) via des méthodes dichotomiques ou directionnelles pour économiser l'énergie dépensée juste pour trouver la route.
    

### 3. Améliorations ciblées sur la Congestion (Charge réseau)

L'objectif est d'éviter les bouchons qui ralentissent tout et consomment de l'énergie inutilement.

- **Surveillance des files d'attente (Senthilkumaran [20], Wentao [12]) :** Utiliser la longueur de la file d'attente (buffer) ou la taille du cache comme indicateur de performance pour accepter ou refuser de router.
    
- **Métriques multiples (Baboo [19], Jabbar [22]) :** Ne plus regarder que les sauts, mais combiner : délai, débit, qualité du lien, et coût MAC. (Note : Souvent complexe et coûteux en calcul).
    
- **Système d'alerte :** Un nœud prévient ses voisins quand il est saturé pour qu'ils l'évitent.
    

### 4. Utilisation d'Algorithmes d'Optimisation

- **Colonies de fourmis (De Rango [16], Pu [24]) :** Utilisation d'algorithmes bio-inspirés pour trouver des chemins qui équilibrent plusieurs objectifs à la fois (délai, énergie, charge).
    
- **Ensembles de nœuds stables (Huang [23]) :** Sélectionner des groupes de nœuds basés sur leur stabilité et durée de vie prédictible.
    

### 5. Problème de la Pondération et Décision Multicritère (AHP)

Le texte souligne la difficulté de donner un "poids" correct à chaque métrique (ex: l'énergie est-elle deux fois plus importante que la vitesse ?).

- **Échec des poids fixes :** Les méthodes utilisant des poids constants (Kumbhar [26], Patsariya [27]) ne s'adaptent pas bien aux changements de topologie.
    
- **Solution AHP (Analytic Hierarchy Process) :** Utilisation de méthodes de décision hiérarchique (parfois floue/fuzzy) pour prendre des décisions intelligentes (Pabani [28], Tomar [29]). Cela est aussi utilisé pour la sécurité (détecter les nœuds malveillants/égoïstes).
    

---

**Pour ton TIPE :** Tu peux utiliser ce résumé pour justifier ton propre choix. Par exemple : _"Contrairement aux approches complexes de type AHP ou algorithmes génétiques, je vais me concentrer sur une modification simple de la métrique de base (type approche Frikha ou Barma) pour voir l'impact sur un petit réseau."_

---
---
