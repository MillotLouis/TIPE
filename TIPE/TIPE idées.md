---
share: true
---
[TIPE Louis Millot](https://millotlouis.github.io/TIPE/)
___
Si sinistre ou catastrophe naturelle ou opération secours milieu isolé ou opération militaire $\Rightarrow$ réseaux traditionnels pas dispos donc faut un P2P pour faire communiquer secouristes, drones, équipements….
	$\Rightarrow$ ad hoc network, infrastructure-less network

> Un **réseau ad hoc** (généralement appelé [MANet](https://geekflare.com/fr/mobile-ad-hoc-network/) (*mobile ad hoc network*)) est un réseau décentralisé où les appareils (nœuds) communiquent directement entre eux, sans dépendre d'une infrastructure fixe (comme un routeur ou un point d'accès Wi-Fi). Il est souvent utilisé dans des situations où une configuration rapide et flexible est nécessaire, comme en milieu militaire, dans des zones sinistrées, ou pour des connexions temporaires entre appareils mobiles.

---
J'ai choisi pour ma part de modifier le critère de choix de chemin avec une pondération comprenant à la fois la batterie résiduelle et les distance séparant deux noeuds intermédiaires sur la route ainsi qu'une forte pénalisation appliquée aux routes empruntant un noeud sous un seuil de batterie défini. Finalement 
___
**[État actuel technologie](./Technos%20acutelles/%C3%89tat%20actuel%20technologie.md)**
___
> [!info]
> Aujourd'hui aucun programme ne combine vraiment tout cela : batterie restante, position, historique de fiabilité (trajectoire prédite : random waypoint mais probablement trop compliqué)

>[!question] Objectif
>Solution de routage se basant sur les technologies actuelles améliorant la durée de vie du réseau en réduisant la consommation énergétique : [Objectifs](Objectifs.md)

[Analyse des résultats](./Analyse.md) 
___
(Abandonné car trop compliqué pour l'instant : **[Idées Protocole](./OLD/Abandonn%C3%A9/Id%C3%A9es%20Protocole.md)**) 
___

**Déplacements :**
[panisson/pymobility: python implementation of mobility models](https://github.com/panisson/pymobility)
[seemoo-lab/natural-disaster-mobility: Natural Disaster Mobility Model and Scenarios in the ONE](https://github.com/seemoo-lab/natural-disaster-mobility)
	

**Cycles et boucles** :
- *count to infinity*
- [Rapport_Lohier, page 6](./Technos%20acutelles/Rapport_Lohier.pdf.md#page=6&selection=1,0,5,7) AODV

**Sources et liens** : 
https://chatgpt.com/c/67caefc2-2b58-8011-8077-41a73283d05c
[Routage ad hoc — Wikipédia](https://fr.wikipedia.org/wiki/Routage_ad_hoc)
https://chat.deepseek.com/a/chat/s/47e9b763-6d1d-443a-8c26-63b40a7f9914
[Chapitre 3 - Chapitre3.pdf](http://opera.inrialpes.fr/people/Tayeb.Lemlouma/Papers/MasterThesis/Chapitre3.pdf)
[INTRODUCTION - AdHoc_Presentation.pdf](http://opera.inrialpes.fr/people/Tayeb.Lemlouma/Papers/AdHoc_Presentation.pdf)
[BonnMotion - A mobility scenario generation and analysis tool](https://sys.cs.uos.de/bonnmotion/)

**Sources/thèses/papers sur le sujet :**
[A Composite Mobility Model for Ad Hoc Networks in Disaster Areas - CentraleSupélec](https://centralesupelec.hal.science/hal-00589846v1)
[AODV-EOCW: An Energy-Optimized Combined Weighting AODV Protocol for Mobile Ad Hoc Networks](https://www.mdpi.com/1424-8220/23/15/6759)
[Un protocole de routage ER-AODV à basse consommation energétique pour les rm ad hoc.pdf](https://dspace.univ-ouargla.dz/jspui/bitstream/123456789/11997/1/Un%20protocole%20de%20routage%20ER-AODV%20%C3%A0%20basse%20consomation%20energie%20pour%20les%20rm%20ad%20hoc.pdf)
[Energy Consumption Evaluation of AODV and AOMDV Routing Protocols](https://thesai.org/Downloads/Volume9No8/Paper_35-Energy_Consumption_Evaluation_of_AODV.pdf?utm_source=chatgpt.com)


--- 
#### Liens pour synchro github : 
[Résultats](./R%C3%A9sultats.md)
[Questions](./Questions.md)
[Protocole (batterie)](./Protocole%20(batterie).md)
[Changelog](./Changelog.md)
[Avancement](./Avancement.md)
[Analyse des résultats](./Analyse.md)
[À faire dans code](./%C3%80%20faire%20dans%20code.md)
[MCOT](./MCOT.md)
[Idées améliorations](./Id%C3%A9es%20am%C3%A9liorations.md)
