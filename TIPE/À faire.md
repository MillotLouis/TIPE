---
share: true
---
### Paramètres à optimiser
- `coeff_dist_weight` et on fixe `coeff_bat_weight = 1 - coeff_dist_weight` 
- `seuil_coeff`
- `penalite_seuil`
- `d_min` et `d_max`
- `max_duplicates`
- `weight_seuil`

---
- [ ] Trouver des paramètres fiables pour la simulation (module CC2420) (consommation, vitesse, temps de pause, fréquence d'envoi des messages, hello_intervall)
- [x] Ajouter la consommation à la réception
- [c] Ajouter conso idle
- [ ] Commencer plus tard pour le calcul du PDR, le temps que tout se mette en place (on mesure en régime permanent)
- [x] `yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))` -> Vitesse propagation et délai de traitement à ajuster
- [x] Déterminer taille paquets de type RR\* et data 
- [x] Déterminer BW pour avec ↑ avoir une conso réaliste
- [x] Justifier délai d'attente quand on reçoit premier RREQ
- [ ] Compter le poids d'une route que à l'aller ou au retour, à determiner ?
- [ ] Changer la méthode de récup de la distance dans calculate_weight pour se baser sur la dernière connue avec un message hello
- [ ] Compter end-to-end delai
- [ ] ![100](../Pasted/%C3%80%20faire-1779011301086.png)![100](../Pasted/%C3%80%20faire-1779011320606.png)![100](../Pasted/%C3%80%20faire-1779011334154.png)

# ==UTILISER PYPY==

- [ ] Harmoniser toutes les grandeurs
	- [x] Fixer init_bat
	- [x] Adapter les consommations pour avoir qqch de relatif
	- Faire tourner pour avoir une idée du temps
	- Adapter vitesse de déplacement
	- Changer jitter,... potentiellement
- [ ] Rendre seuil relatif
- [ ] Continuer à faire tourner la simulation jusqu'à un temps fixé et adapter la vitesse là dessus
- [ ] Vérifier qu'un nœud est bien à portée d'émission dans send_data avant d'envoyer 
- [ ] Modifier [Fonction de poids](../Fonction%20de%20poids.md)
- [ ] Modifier Network.\_\_init\_\_ pour prendre config : SimConfig en argument
- [ ] Utiliser random ou np.random pas les deux et donc changer les initialisation de seeds dans run_comp_sim 

---
AVANT de faire ça : faire tourner des simus pour avoir des résultats à comparer
- [ ] Ajouter la réponse des noeuds intermédiaires

---
- [ ] Changer de modèle de mobilité ?
- [ ] Ajouter une consommation de batterie à la reception ? Me paraît négligeable comparé à emission [Consommation radios tactiques](https://chatgpt.com/c/684e8756-d540-8011-a4ac-28612f8f2609) à voir
	- Serait potentiellement plus que émission ? [Optimisation AODV Réseau Ad-Hoc](https://chatgpt.com/c/68595b82-9e04-8011-81ab-587a8c4fb44a)
		- **Non** voir paramètres de simulation
- [ ] Ajouter réponse des noeuds intermédiaires
	https://chatgpt.com/g/g-p-68ae01179a9c819188df4800374ab039-louis/c/68fa54f8-4f28-832c-9d4f-c56ed61d3cfe
---
J'ai enlevé time_scale, le fait de déplacer les noeuds en continu (déplacements discrets mtn)
**Avant :**
Durée de simu BM : 5000
Time_scale = 0.01

**Donc mtn** : 
Durée de simu BM : 500000


- Enlever X et Y dans les simus BM, remplacer tout par size
- Faire correspondre les temps de simulation et de bonnmotion/de déplacement