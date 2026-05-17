---
share: true
---
# Old
#### Au 26/08/2025 à 20h [voir commit](https://github.com/MillotLouis/TIPE/commit/e39759f7e84b879e819893f3d646e6e1de7844e5)
###### Résultats de `version_présentée.py` :
![500](../Pasted/R%C3%A9sultats-1756231387463.png)
###### Paramètres :
- Temps exécution = 3000
- Nb simulations = 3
- ![150](../Pasted/R%C3%A9sultats-1756231490963.png)

#### Au 28/08/2025 à 21h00 [voir commit](https://github.com/MillotLouis/TIPE/commit/16dc4614ec89d12ce8cd099b43bc67a2d9a00927)
```python
    params = {
        "nb_runs": 5,
        "size": 800,
        "conso": (1, 20),
        "seuil": 750,
        "coeff_dist": 0.6,
        "coeff_bat": 0.2,
        "coeff_conso": 0.005,
        "ttl": 100
    }
```
`nb_runs = 3`
`max_dist = 250`
`pas = 5`
`factor_min = 0.7`
`factor_max = 1.5`
**Arrêt de la simulation après 3000 unités de temps**, je l'ai modifié après



![Résultats-1756488906175.png](../Pasted/R%C3%A9sultats-1756488906175.png)
![Résultats-1756488913909.png](../Pasted/R%C3%A9sultats-1756488913909.png)
![Résultats-1756488920945.png](../Pasted/R%C3%A9sultats-1756488920945.png)


#### avec mobilité [voir commit](https://github.com/MillotLouis/TIPE/commit/51ca911cd0b2125a65bf357597229b3aedfb8abb): 
![400](../Pasted/R%C3%A9sultats-1756504577135.png)
![400](../Pasted/R%C3%A9sultats-1756504584967.png)
Voir dossier `résultats/simu1`



# Récent (16/05)

- **max_duplicates = 2, poids = $x\mapsto 1$, **
	![300](../Pasted/R%C3%A9sultats-1778929987839.png)
![677](../Pasted/R%C3%A9sultats-1778930010661.png)

---
``Seed_base = 12345``
`max_duplicates = 2`
![400](../Pasted/R%C3%A9sultats-1778930372091.png)
![400](../Pasted/R%C3%A9sultats-1778930421577.png)

**Résultat :**
![Résultats-1778930977869.png](../Pasted/R%C3%A9sultats-1778930977869.png)

---
``Seed_base = 12345``
`max_duplicates = 2`
poids : $x\mapsto 1$
![400](../Pasted/R%C3%A9sultats-1778930421577.png)

![Résultats-1778932560426.png](../Pasted/R%C3%A9sultats-1778932560426.png)

---
Commit `a53f1f5d63b7750fba8453e52efe3715d9cd9ba7`
`coeff_dist_weight = 0.6`
`coeff_bat_weight = 0.4`
![Résultats-1779008364335.png](../Pasted/R%C3%A9sultats-1779008364335.png)

---
Commit `0b852b221b0aa8495d03923d06d3bf6041b10396`
![Résultats-1779017211087.png](../Pasted/R%C3%A9sultats-1779017211087.png)
```
[(20, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 1022.3199119940231, 'energy_count': 5, 'msg_recv': 2145.6, 'msg_recv_count': 5, 'msg_sent': 2359.6, 'msg_sent_count': 5, 'rreq_sent': 3148.0, 'rreq_sent_count': 5, 'duration': 600.0020314482783, 'duration_count': 5, 'rrep_sent': 1333.4, 'rrep_sent_count': 5, 'messages_forwarded': 5235.4, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 159668.0, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 48.88400440004767, 'final_avg_bat_count': 5, 'final_std_bat': 9.651536753540045, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 1053.4349759940947, 'energy_count': 5, 'msg_recv': 2137.4, 'msg_recv_count': 5, 'msg_sent': 2356.4, 'msg_sent_count': 5, 'rreq_sent': 2777.6, 'rreq_sent_count': 5, 'duration': 600.0017247671207, 'duration_count': 5, 'rrep_sent': 1137.2, 'rrep_sent_count': 5, 'messages_forwarded': 5794.6, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 178934.2, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 47.32825120004209, 'final_avg_bat_count': 5, 'final_std_bat': 7.97216571303223, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}), (30, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 1671.0976799789878, 'energy_count': 5, 'msg_recv': 2208.2, 'msg_recv_count': 5, 'msg_sent': 2353.6, 'msg_sent_count': 5, 'rreq_sent': 2772.4, 'rreq_sent_count': 5, 'duration': 600.0020218049058, 'duration_count': 5, 'rrep_sent': 1740.0, 'rrep_sent_count': 5, 'messages_forwarded': 5369.6, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 296438.6, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 44.296744000063576, 'final_avg_bat_count': 5, 'final_std_bat': 10.164848933519767, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 1633.103799980901, 'energy_count': 5, 'msg_recv': 2180.2, 'msg_recv_count': 5, 'msg_sent': 2353.6, 'msg_sent_count': 5, 'rreq_sent': 2369.0, 'rreq_sent_count': 5, 'duration': 600.0020316106662, 'duration_count': 5, 'rrep_sent': 1456.0, 'rrep_sent_count': 5, 'messages_forwarded': 5951.8, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 334000.0, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 45.563206666724305, 'final_avg_bat_count': 5, 'final_std_bat': 8.120958003696202, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}), (40, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 2740.118231950076, 'energy_count': 5, 'msg_recv': 2215.0, 'msg_recv_count': 5, 'msg_sent': 2351.0, 'msg_sent_count': 5, 'rreq_sent': 2870.8, 'rreq_sent_count': 5, 'duration': 600.0015649275454, 'duration_count': 5, 'rrep_sent': 1978.4, 'rrep_sent_count': 5, 'messages_forwarded': 5420.4, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 530479.2, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 31.497044200069514, 'final_avg_bat_count': 5, 'final_std_bat': 10.746781147237098, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}, {'dead_nodes': 0.0, 'dead_nodes_count': 5, 'energy': 2618.2704959542707, 'energy_count': 5, 'msg_recv': 2209.8, 'msg_recv_count': 5, 'msg_sent': 2351.6, 'msg_sent_count': 5, 'rreq_sent': 2461.8, 'rreq_sent_count': 5, 'duration': 600.0011699265042, 'duration_count': 5, 'rrep_sent': 1670.8, 'rrep_sent_count': 5, 'messages_forwarded': 5943.2, 'messages_forwarded_count': 5, 'messages_initiated': 2401.0, 'messages_initiated_count': 5, 'rreq_forwarded': 609927.8, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': None, 'first_node_death_count': 0, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 34.54323760006544, 'final_avg_bat_count': 5, 'final_std_bat': 8.729831860544802, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}), (50, {'dead_nodes': 9.0, 'dead_nodes_count': 5, 'energy': 4283.75150395357, 'energy_count': 5, 'msg_recv': 2214.4, 'msg_recv_count': 5, 'msg_sent': 2292.4, 'msg_sent_count': 5, 'rreq_sent': 3088.6, 'rreq_sent_count': 5, 'duration': 600.0020121048256, 'duration_count': 5, 'rrep_sent': 2053.8, 'rrep_sent_count': 5, 'messages_forwarded': 5298.8, 'messages_forwarded_count': 5, 'messages_initiated': 2374.6, 'messages_initiated_count': 5, 'rreq_forwarded': 884749.6, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': 488.44336737060246, 'first_node_death_count': 5, 'ten_percent_death': 574.0054602383226, 'ten_percent_death_count': 4, 'final_avg_bat': 14.325570560066177, 'final_avg_bat_count': 5, 'final_std_bat': 11.779070249814973, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}, {'dead_nodes': 1.8, 'dead_nodes_count': 5, 'energy': 4025.0982479160784, 'energy_count': 5, 'msg_recv': 2230.0, 'msg_recv_count': 5, 'msg_sent': 2335.0, 'msg_sent_count': 5, 'rreq_sent': 2595.0, 'rreq_sent_count': 5, 'duration': 600.0020125558988, 'duration_count': 5, 'rrep_sent': 1813.0, 'rrep_sent_count': 5, 'messages_forwarded': 5760.6, 'messages_forwarded_count': 5, 'messages_initiated': 2397.8, 'messages_initiated_count': 5, 'rreq_forwarded': 1004176.6, 'rreq_forwarded_count': 5, 'seuiled': 8105.6, 'seuiled_count': 5, 'first_node_death': 560.3461632759966, 'first_node_death_count': 4, 'ten_percent_death': None, 'ten_percent_death_count': 0, 'final_avg_bat': 19.498058080074337, 'final_avg_bat_count': 5, 'final_std_bat': 11.430804481748462, 'final_std_bat_count': 5, 'fifty_percent_death': None, 'fifty_percent_death_count': 0}), (60, {'dead_nodes': 46.0, 'dead_nodes_count': 5, 'energy': 5881.768816322503, 'energy_count': 5, 'msg_recv': 1874.4, 'msg_recv_count': 5, 'msg_sent': 1961.0, 'msg_sent_count': 5, 'rreq_sent': 4078.6, 'rreq_sent_count': 5, 'duration': 600.0020080971007, 'duration_count': 5, 'rrep_sent': 1902.6, 'rrep_sent_count': 5, 'messages_forwarded': 4527.0, 'messages_forwarded_count': 5, 'messages_initiated': 2125.2, 'messages_initiated_count': 5, 'rreq_forwarded': 1290072.4, 'rreq_forwarded_count': 5, 'seuiled': 0.0, 'seuiled_count': 5, 'first_node_death': 407.24580176164585, 'first_node_death_count': 5, 'ten_percent_death': 463.09252940745, 'ten_percent_death_count': 5, 'final_avg_bat': 1.9724645333534288, 'final_avg_bat_count': 5, 'final_std_bat': 5.036235792218272, 'final_std_bat_count': 5, 'fifty_percent_death': 529.9274210844508, 'fifty_percent_death_count': 5}, {'dead_nodes': 43.4, 'dead_nodes_count': 5, 'energy': 5847.690928313141, 'energy_count': 5, 'msg_recv': 1970.6, 'msg_recv_count': 5, 'msg_sent': 2061.0, 'msg_sent_count': 5, 'rreq_sent': 3334.2, 'rreq_sent_count': 5, 'duration': 600.0014119744044, 'duration_count': 5, 'rrep_sent': 1770.2, 'rrep_sent_count': 5, 'messages_forwarded': 5001.6, 'messages_forwarded_count': 5, 'messages_initiated': 2206.4, 'messages_initiated_count': 5, 'rreq_forwarded': 1575511.8, 'rreq_forwarded_count': 5, 'seuiled': 125717.6, 'seuiled_count': 5, 'first_node_death': 440.2197810950717, 'first_node_death_count': 5, 'ten_percent_death': 492.5834175361051, 'ten_percent_death_count': 5, 'final_avg_bat': 2.5396964000232822, 'final_avg_bat_count': 5, 'final_std_bat': 5.569370310353588, 'final_std_bat_count': 5, 'fifty_percent_death': 552.9835901351, 'fifty_percent_death_count': 5})]
```

---
