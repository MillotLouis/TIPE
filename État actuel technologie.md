---
share: true
---
![600](./Pasted/Pasted%20image%2020250314135053.png)
Aussi *Link-State* : comme AODV basé sur chemin le plus court et ou le plus rapide en fonction du debit.
___
## Protocoles Naïfs :

##### Flooding :
- **Plus**
	- Rapide pas de temps de calcul ou quasiment pas 
	- Plus simple à implémenter
- **Moins**
	- Consommation batterie potentiellement élevée
	- Brutal et largement optimisable
##### Random walk
- *Cf photo*

##### Greedy forwarding
- *Cf photo*
- **Moins** 
	- Prend pas en compte obstacles physiques par exemple si juste calcule distance avec coordonées
## Protocoles existants/utilisés :
- [AODV](./AODV.md)
- 

> [!info] Mais
> Les algorithmes classiques (AODV, OLSR) négligent souvent les variations spatiales de couverture réseau.

Donc plus intéressant ici : 
#### Plus centré sur risk aware ou ce que je veux faire :
[Algorithmes de routage et de planification sensibles à la QoS : Guide](https://www.linkedin.com/advice/3/how-do-you-deal-uncertainty-dynamics-qos-aware) (sûrement utile (cycles et boucles))

https://chat.deepseek.com/a/chat/s/c110635a-240c-4359-b666-1e1a1beccd21
##### Trust-based routing
- Évalue la fiabilité des noeuds et évite les moins fiables

##### Power Aware Routing 
- à chercher, minimise consommation batterie

##### QoS-Aware Routing
- à chercher
