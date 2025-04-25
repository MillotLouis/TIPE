---
share: true
---
Le protocole **AODV (Ad-hoc On-Demand Distance Vector)** est un protocole de routage réactif conçu pour les réseaux mobiles ad hoc (MANETs). Il établit des routes uniquement lorsque nécessaire, réduisant ainsi la surcharge du réseau. Voici une explication détaillée de son fonctionnement :

---
#### **1. Concepts Clés**
- **Réactif (On-Demand)** : Aucune table de routage complète n'est maintenue. Les routes sont découvertes à la demande.
- **Distance Vector** : Utilise des compteurs de sauts pour mesurer la distance.
- **Séquence Numbers** : Garantissent la fraîcheur des routes et évitent les boucles.
- **Messages de Contrôle** :
    - **RREQ** (Route Request) : Pour découvrir une route.
    - **RREP** (Route Reply) : Pour confirmer une route.
    - **RERR** (Route Error) : Pour signaler une rupture de lien.
    - **Hello Messages** : Pour détecter la présence des voisins.
---
#### **2. Découverte de Route (Route Discovery)**
##### **Étape 1 : Diffusion du RREQ**
- **Lancement** : Un nœud source (ex: `S`) envoie un **RREQ** s'il n'a pas de route valide vers la destination (ex: `D`).
- **Contenu du RREQ** :
    - Adresses IP source et destination.
    - Numéros de séquence source (`S_seq`) et destination (`D_seq`).
    - Compteur de sauts (`hop_count`), initialisé à 0.
##### **Étape 2 : Traitement par les nœuds intermédiaires**
- Chaque nœud reçoit le RREQ :
    1. Vérifie s'il a une route **fraîche** vers `D` (via `D_seq` stocké ≥ celui du RREQ).
        - Si oui : Envoie un **RREP** directement au nœud source via le chemin inverse.
        - Si non : Incrémente `hop_count` et diffuse le RREQ à ses voisins.
    2. Stocke une **entrée de routage inverse** vers `S` pour acheminer le RREP.
##### **Étape 3 : Réponse de la destination (RREP)**
- La destination `D` reçoit le RREQ :
    1. Vérifie que son numéro de séquence est ≥ à celui du RREQ.
    2. Envoie un **RREP** via le chemin inverse enregistré.
    3. Incrémente son numéro de séquence pour garantir la fraîcheur.
##### **Étape 4 : Mise à jour des tables de routage**
- Chaque nœud sur le chemin du RREP :
    - Met à jour sa table de routage avec une **entrée de routage directe** vers `D`.
    - Stocke le prochain saut, le `hop_count`, et le `D_seq`.
---
#### **3. Maintenance de Route (Route Maintenance)**
##### **Détection des pannes**
- **Hello Messages** : Envoyés périodiquement pour confirmer la connectivité avec les voisins.
- **Surveillance des liens** : Si un nœud ne reçoit plus de messages d'un voisin, il considère le lien rompu.
##### **Gestion des erreurs (RERR)**
- Un nœud détectant une panne :
    1. Invalide toutes les routes utilisant le lien rompu.
    2. Envoie un **RERR** à tous les nœuds en amont (précurseurs) pour les informer.
    3. Les nœuds sources concernés relancent une découverte de route si nécessaire.
---
#### **4. Gestion des Numéros de Séquence**
- **Rôle** :
    - Éviter les boucles.
    - Identifier les routes obsolètes.
- **Incrémentation** :
    - Le nœud source incrémente `S_seq` à chaque nouveau RREQ.
    - Le nœud destination incrémente `D_seq` à chaque RREP.
---
#### **5. Structure des Tables de Routage**
Chaque entrée contient :
- Adresse destination.
- Prochain saut.
- Nombre de sauts.
- Numéro de séquence de la destination.
- Liste des précurseurs (nœuds utilisant cette route).
- Durée de vie (TTL) de l'entrée.
---
#### **6. Exemple de Scénario**
1. **Découverte** : `S` diffuse un RREQ pour joindre `D`.
2. **Réponse** : `D` envoie un RREP via le chemin inverse.
3. **Communication** : Les données circulent via `S → A → B → D`.
4. **Rupture** : Si le lien `A-B` tombe en panne :
    - `A` envoie un RERR à `S`.
    - `S` relance un RREQ pour trouver un nouveau chemin.
---
#### **7. Avantages et Limites**
- **Avantages** :
    - Faible surcharge (pas de mises à jour périodiques).
    - Adapté aux réseaux mobiles (routes dynamiques).
- **Limites** :
    - Latence lors de la découverte de route.
    - Risque de congestion avec les RREQ en cas de mobilité élevée.
---
#### **8. Comparaison avec d'autres Protocoles**
- **vs OLSR (proactif)** : AODV évite le maintien de routes inutilisées.
- **vs DSR** : AODV utilise des numéros de séquence et des tables de routage, contrairement à DSR qui stocke les routes dans les en-têtes des paquets.

---
En résumé, AODV optimise la gestion des routes dans les réseaux dynamiques en combinant la découverte à la demande, la maintenance réactive des liens, et des mécanismes anti-boucle via les numéros de séquence.
