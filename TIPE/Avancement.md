---
share: true
---

1. Dans la réalité, AODV est utilisé dans des réseaux de noeuds mobiles donc des connexions se font et se défont donc le protocole AODV est utilisé uniquement afin de recréer une route ou en créer une si elle n'existe pas.
	Ici comme pas de déplacements pour l'instant je définis un TTL pour les routes afin de simuler ces créations de routes quand il y a effectivement des déplacements. cf : [AODV](./Technos%20acutelles/AODV.md)

2. Dans la norme **RFC 3561** décrivant le fonctionnement d'AODV il est écrit : "_The destination node... MUST send a RREP back to the source_" et ce RREP est le premier valide à être reçu par le noeud de destination. à ce moment là le nombre de saut de la route est calculé sur le chemin "retour".
	Cependant ici si on fait cela alors la pondération selon la batterie ne pourra pas être prise en compte : il faut donc [ajouter un petit délai d'attente](https://chat.deepseek.com/a/chat/s/101cc16b-010d-48c9-ba9e-eee1aaacbcbc) quand la destination reçoit le premier RREQ émanant de la source afin de tous les collecter et renvoyer uniquement le meilleur. Ici le poids de la route sera calculé à "l'aller" afin de pouvoir comparer à l'arrivée mais aussi au retour (indispensable afin de pouvoir ajouter le bon poids aux noeuds intermédiaires)

3. De même dans **RFC 3561** si un noeud intermédiaire a un chemin plus frais et valide il renvoie un RREP mais alors le noeud source peut recevoir plusieurs RREP (le noeud de destination renvoie **toujours** un RREP). Dans un premier temps je ne vais pas faire cela, seulement la destination répond

4. De plus dans AODV quand un noeud reçoit une RREQ il discard toutes les autres requêtes venant de ce noeud avec le même seq_num ie toutes celles lancées en même temps qui se sont séparées à des embranchements. Ici si je fait ca je vais perdre la comparaison avec la batterie car le noeud empruntant le chemin le plus rapide atteindra le noeud A en premier et tous les suivants seront discard même si ils ont un meilleur poids en pondérant par la batterie donc il faut que je regarde tous les RREQ venant d'un même noeud source avec un même seq_num : (source,seq_num,voisin) au lieu de juste (source,seq_num) dans le dico des RREQ vues
	1. **Update**: Si on fait ça on peut perdre des chemins meilleurs passant par le même prev_hop => pas bien. **ET** transmet des RREQs inutiles car ne seront pas choisis car poids plus grand qu'un qui est déjà passe
	2. Donc il faut comparer avec un dico `seen = (rreq.src_id, rreq.src_seq, rreq.prev_hop) : meilleur poids`
	3. Si je fais ça ça réduit de plus de 3x le nb de transmissions => noice

5. Normalement quand un noeud reçoit un RREQ de la source il ajoute un reverse path à sa table de routage pointant vers la source utilisant le chemin emprunté par le RREQ jusqu'ici pour pouvoir forward le RREP si ce chemin est choisi. Cependant cette entrée est marquée comme invalide, réservée à ce RREP, ici pour simplifier je n'implémente pas ça : les chemins vers la source sont mis à jour tout au long du chemin du RREQ si ils sont plus récent ou mieux et peuvent être utilisés pour n'importe quelle transmission

6. J'avais un problème : les délais n'était pas randomisés donc tous les RREQs était scheduled pour t = 0.0 et cela quasi à l'infini donc le temps de la simulation  n'avancait pas => RREPs pas collectés donc pas envoyés

7. Galère pour représenter les données sachant que je ne peux pas forcément faire varier un seul paramètre en fixant les autres cf [Analyse](./Analyse.md)

8. ![200](../Pasted/Avancement-1757969658658.png) les noeuds se déplaceaint trop vite pour des personnes, j'ai donc réduit la vitesse par 2 et j'ai augmenté les temps de pause

9. Au début je comprenais pas pk les résultats étaient mauvais en comparant la mort de 50% des noeuds mais c'est sûrement que comme il n'y a plus bcp de noeuds vivants avec reg_aodv ou que les noeuds "critiques"/"centraux" sont morts les messages ne peuvent pas être envoyés correctement. **à verifier**

10. à faire : tracer le delivery ratio et d'autres métriques de transfer de data afin de montrer ↑ vrai

11. J'ai fait cela : [cf commit](https://github.com/MillotLouis/TIPE/commit/380ce3f6358151acdf92ae10ee395908457bf434)
	![300](../Pasted/Avancement-1757967474400.png)![300](../Pasted/Avancement-1757967485914.png)
	Pas concluant, le delivery ratio ne chute pas assez pour que cela explique la baisse de performance
PEUT ETRE plot le nombre de messages envoyés pour confirmer que c'est pas à cause de ça / pour avoir des résultats relatifs aux nb de messages envoyés

12. ttl dynamique inefficace il me semble, **à vérifier**

13. trouver dans quel contexte (densité,...) il est le plus efficace et trouver meilleures pondérations après