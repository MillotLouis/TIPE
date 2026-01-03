---
share: true
---

==Comparer 50% noeuds et 75% noeuds car + pas réaliste==
Pour analyser les résultats je vais faire des scénarios et comparer des métriques pour les deux protocoles

**On définit :**
- Densité : $\lambda = \frac{\text{nb\_nodes}}{\text{area\_size}^2}$
- Portée relative $r = \frac{max\_dist}{area\_size}$ 

**Métrique(s) comparée(s)** : 
- Temps mort premier noeud
- Temps mort 10% des noeuds
- Temps partition du réseau
- Énergie consommée
- Std énergie finale
- Delivery ratio

#### Déjà fait : 
- en faisant varier nb_nodes à taille de simulation fixée pour analyser fnd, tpd, std final, total energy, dr

#### À faire :
- tracer delivery ratios en fonction du temps : 
	1. Tracer $D_r = \frac{len(\text{t\_send})}{len(\text{t\_received})}$  sur une fenêtre glissante de 100 par ex
		Permet de voir la qualité instantanée du réseau, uniquement les messages réellement envoyés
	2. Tracer $D_r = \frac{len(\text{t\_init})}{len(\text{t\_received})}$ sur une fenêtre glissante de 100 par ex
		Permet de voir le service rendu à l'application
Permet de distinguer «On n'arrive plus à trouver de routes» et «même en émettant on ne délivre plus»

##### Densité :
À $r$ fixé :
- Faire varier continuellement λ ie fixer `area_size` et faire varier `nb_nodes`
**Hypothèse** : Le protocole excelle plus en densité médiane

##### Portée radio
Fixer $\lambda$ et faire varier $r$ 
- seuil de connectivité $r$ tq $r = \sqrt{\frac{\log n}{n\pi}}$ 
**Hypothèse** : Mod excelle mieux avec $r$ faible car il y a plus de sauts à faire

##### TTL long ou court
Faire varier le TTL
**Hypothèse** : 
- TTL plus court -> Delivery rate augmente mais conso energie aussi (bcp plus pour modifié car transmet bcp plus de RREQ)
- TTL plus long -> Delivery rate baisse (plus de pertes car renouvellement éloigné des routes mais moins pour mod que pour reg car TTL dynamique)

