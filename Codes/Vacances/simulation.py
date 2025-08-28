import matplotlib.pyplot as plt
import random
import numpy as np

from network import Network

class Simulation:
    def __init__(self,nb_nodes, area_size, max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl,reg_aodv,init_bat, node_positions = None):
        self.nb_nodes = nb_nodes
        self.area_size = area_size
        self.max_dist = max_dist
        self.reg_aodv = reg_aodv #true si on utilise AODV et false sinon
        self.energy_history = []
        self.dead_nodes_history = []
        self.time_points = []
        self.init_bat = init_bat
        self.avg_bat_history = []
        self.std_bat_history = []

        #création du réseau
        self.net = Network(
            conso=conso,
            seuil=seuil,
            coeff_dist=coeff_dist,
            coeff_bat=coeff_bat,
            coeff_conso=coeff_conso,
            nb_nodes=nb_nodes,
            ttl=ttl,
            reg_aodv = reg_aodv
        )

        #création des noeuds
        self.node_positions = node_positions or {} #si on a déjà une configuration on l'importe sinon on en crée une
        for i in range(nb_nodes):
            if i in self.node_positions:
                pos = self.node_positions[i]
            else:
                pos = (random.uniform(0, self.area_size), random.uniform(0, self.area_size))
                self.node_positions[i] = pos
            
            self.net.add_node(id=i, pos=pos, max_dist=max_dist, battery=self.init_bat,reg_aodv=self.reg_aodv)

                
        #création des noeuds
        self._create_links()
        
    def _create_links(self):
        for i in range(self.nb_nodes):
            for j in range(i + 1, self.nb_nodes):
                self.net.G.add_edge(i, j)
        # crée un réseau complet : toutes les connexions possibles sont crées mais pas utilisées car la distance est verifiée dans les fct de transmission

    def _random_communication(self):
        """Simule des communications tant que la simulation n'est pas terminée"""
        while not self.net.stop:
            src_id = random.randint(0, self.nb_nodes-1)
            dest_id = random.randint(0, self.nb_nodes-1)
            
            while dest_id == src_id:
                dest_id = random.randint(0, self.nb_nodes-1)
            #on choisit deux noeuds différents

            src_node = self.net.G.nodes[src_id]['obj']
            if src_node.alive:
                src_node.send_data(dest_id) # on lance le tranfert de données
            
            yield self.net.env.timeout(0.1) #petit délai pour pas flood

    def _monitor(self):
        while not self.net.stop:
            self.time_points.append(self.net.env.now) #points temporels pour ploter lels données
            self.energy_history.append(self.net.energy_consumed) #total d'énergie consommée
            self.dead_nodes_history.append(self.net.dead_nodes) #nb de noeuds morts
            
            avg_bat, std_bat = self.net.get_energy_stats()
            self.avg_bat_history.append(avg_bat) # moyenne de batterie des noeuds
            self.std_bat_history.append(std_bat) #écart type de batterie des noeuds
            
            yield self.net.env.timeout(0.2)  # ce qui donne tous les 2 messages envoyés

    def get_metrics(self):
        return {
            "dead_nodes": self.net.dead_nodes,
            "energy": self.net.energy_consumed,
            "msg_recv": self.net.messages_received,
            "msg_sent": self.net.messages_sent,
            "rreq_sent": self.net.rreq_sent,
            "duration": self.net.env.now,
            "rrep_sent" : self.net.rrep_sent,
            "messages_forwarded" : self.net.messages_forwarded,
            "messages_initiated":self.net.messages_initiated,
            "rreq_forwarded":self.net.rreq_forwarded,
            "seuiled":self.net.seuiled,
            "first_node_death": self.net.first_node_death_time,
            "ten_percent_death": self.net.ten_percent_death_time,
            "network_partition": self.net.network_partition_time,
            "final_avg_bat": self.avg_bat_history[-1],
            "final_std_bat": self.std_bat_history[-1]
        }
    
    def run(self):
        self.net.env.process(self._random_communication()) # on démarre les communications
        self.net.env.process(self._monitor()) # on démarre le monitoring pour récolter les données durant la simulation
        while not self.net.stop and self.net.env.now <= 3000:
            self.net.env.step()


    def print_results(self):
        print(f"Durée: {self.net.env.now:.2f} unités de temps")
        print(f"Noeuds morts: {self.net.dead_nodes}/{self.nb_nodes}")
        print(f"Énergie consommée: {self.net.energy_consumed:.2f}")
        print(f"Messages envoyés: {self.net.messages_sent}")
        print(f"Messages transmis: {self.net.messages_forwarded}")
        print(f"Messages reçus: {self.net.messages_received}")
        print(f"RREQ envoyés: {self.net.rreq_sent}")
        print(f"RREQ transmis: {self.net.rreq_forwarded}")
        print(f"RREP envoyés: {self.net.rrep_sent}")
        print(f"Seuiled: {self.net.seuiled}")
        print(f"Mort premier noeud: {self.net.first_node_death_time:.2f}" if self.net.first_node_death_time else "First Node Death: Not reached")
        print(f"Mort 10% noeuds: {self.net.ten_percent_death_time:.2f}" if self.net.ten_percent_death_time else "10% Node Death: Not reached")
        print(f"Partition du réseau: {self.net.network_partition_time:.2f}" if self.net.network_partition_time else "Network Partition: Not reached")
        print(f"Moyenne batterie finale: {self.avg_energy_history[-1]:.2f}")
        print(f"Écart type batterie finale: {self.std_energy_history[-1]:.2f}")

## Comparaison des protocoles ##

def run_comparison_simulations(nb_runs,nb_nodes,size,max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl):   
    reg_aodv_res = []
    mod_aodv_res = []

    params = {
        'nb_nodes': nb_nodes,
        'area_size': size,
        'max_dist': max_dist,
        'conso': conso,
        'seuil': seuil,
        'coeff_dist': coeff_dist,
        'coeff_bat': coeff_bat,
        'coeff_conso': coeff_conso,
        'ttl': ttl
    }

    for _ in range(nb_runs):
        positions = {} #sauvegarder la position des noeuds pour avoir la même pour les deux simulations sinon on peut pas comparer
        for i in range(params["nb_nodes"]):
            # génération aléatoire
            positions[i] = (
                random.uniform(0, params['area_size']), 
                random.uniform(0, params['area_size'])
            )
        
        #Simulation avec AODV classique
        print("\nstarting reg aodv sim")
        sim_reg = Simulation(
            node_positions=positions,
            reg_aodv=True,
            init_bat=100000,
            **params
        )
        sim_reg.run()
        reg_aodv_res.append(sim_reg.get_metrics())

        #Simulation avec AODV modifié
        print("\nstarting mod aodv sim")
        sim_mod = Simulation(
            node_positions=positions,
            reg_aodv=False,
            init_bat=100000, #pour ne pas avoir de valeurs trop petites dans les consommations
            **params
        )
        sim_mod.run()
        mod_aodv_res.append(sim_mod.get_metrics())

    # print_avg_results(reg_aodv_res,mod_aodv_res,nb_runs)
    return {"reg":reg_aodv_res,"mod":mod_aodv_res}

def calc_avg_metrics(res):
    metric_keys = res[0].keys() #on récupère les métriques sur lesquelles on doit calc la moyenne
    avg = {}

    for key in metric_keys:
        values = [r[key] for r in res if r[key] is not None]
        # On gère les None potentiels des situations pas atteintes
        if values:
            avg[key] = sum(values)/len(values)
            avg[f"{key}_count"] = len(values) #on garde le nb de simulations qui ont atteint cette métrique (ex : partition pas tjrs atteinte)
        else:
            avg[key] = None
            avg[f"{key}_count"] = 0
    return avg


def print_avg_results(reg_res,mod_res,nb_runs):
    print(f"\n\nMoyennes sur {nb_runs} simulations")
    print("="*60)

    print(f"\n{"Métrique":<25} {"Regular AODV":<15} {"Count Reg":<12} {"Modified AODV":<15} {"Count Mod":<12} {"Changement":<12}")
    print("-" * 94)

    reg_avg = calc_avg_metrics(reg_res)
    mod_avg = calc_avg_metrics(mod_res)
    metrics_to_compare = reg_res[0].keys()

    for key in metrics_to_compare:
        reg_val = reg_avg[key]
        count_reg = reg_avg[f"{key}_count"]
        mod_val = mod_avg.get(key)
        count_mod = mod_avg[f"{key}_count"]
        improvement = ((mod_val - reg_val) / reg_val * 100) if reg_val not in (None, 0) and mod_val is not None else None
        
        reg_val_str = f"{reg_val:.2f}" if reg_val is not None else "N/A"
        mod_val_str = f"{mod_val:.2f}" if mod_val is not None else "N/A"
        improvement_str = f"{improvement:.2f}%" if improvement is not None else "N/A"
        print(f"{key:<25} {reg_val_str:<15} {count_reg:<12} {mod_val_str:<15} {count_mod:<12} {improvement_str:<12}")

    reg_delivery_ratio = (reg_avg['msg_recv'] / reg_avg['messages_initiated']) * 100 if reg_avg['messages_initiated'] > 0 else 0
    mod_delivery_ratio = (mod_avg['msg_recv'] / mod_avg['messages_initiated']) * 100 if mod_avg['messages_initiated'] > 0 else 0
    
    print(f"\n{'Delivery Ratio':<25} {reg_delivery_ratio:<15.1f}% {"":<12} {mod_delivery_ratio:<15.1f}% {"":<12} {'':<12}")

    print("\n" + "="*60)


def densité(pas,max_dist,params):
    n_min = (params["size"]/max_dist)**2*np.pi #correspond à peu près à 1 noeud par cercle de max_dist de rayon
    print(n_min)
    nb_nodes_array = []
    res_reg_array = []
    res_mod_array = []
    for nb_nodes in (0.7*n_min,1.5*n_min,pas):
        nb_nodes = round(nb_nodes)
        nb_nodes_array.append(nb_nodes)
        result = run_comparison_simulations(nb_nodes=nb_nodes,max_dist=max_dist,**params)
        res_reg_array.append(calc_avg_metrics(result["reg"]))
        res_mod_array.append(calc_avg_metrics(result["mod"]))
    plt.figure()
    plt.plot(nb_nodes_array,[res["first_node_death"] for res in res_reg_array],label="Regular")
    plt.plot(nb_nodes_array,[res["first_node_death"] for res in res_mod_array],label="Modified")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # run_comparison_simulations(
    #     nb_runs=3,
    #     nb_nodes=25,
    #     size=800,
    #     max_dist= 400,
    #     conso=(1,20),
    #     seuil=750, #correspond à 0.75% avec 100% initialement
    #     coeff_dist=0.6,
    #     coeff_bat=0.2,
    #     coeff_conso=0.005,
    #     ttl=100
    # ) 
    params = {
        "nb_runs": 1,
        "size": 800,
        "conso": (1, 20),
        "seuil": 750, # correspond à 0.75% avec 100% initialement
        "coeff_dist": 0.6,
        "coeff_bat": 0.2,
        "coeff_conso": 0.005,
        "ttl": 100
    }
    densité(5,250,params)
    