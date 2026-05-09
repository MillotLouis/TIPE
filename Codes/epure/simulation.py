import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess
import gzip
import shutil

from network import Network


from dataclasses import dataclass,replace
from typing import Tuple, Optional, Dict

@dataclass
class SimConfig:
    nb_nodes: int
    """ Nombre de noeuds de la simulation """
    area_size: int
    """ Taille de la carte : area_size*area_size """
    max_dist: float
    """ Distance max à laquelle un noeud peut transmettre """
    init_bat: float
    """ batterie initiale pour tous les noeuds """
    conso: Tuple[float, float]
    """Consommation (d'un message de contrôle,d'un message de données)"""
    dt : float
    """Pas de temps pour la mise à jour de la position et de l'envoi aléatoire des messages de données"""
    #dt = 0.5 ?


    # Paramètres du Protocole (AODV / Energy Aware)
    ttl: int
    seuil: float # En pourcentage
    coeff_dist_weight: float
    coeff_bat_weight: float
    coeff_dist_bat: float
    
    # Paramètres de la Simulation
    duration: float
    window_size: float = 100.0


@dataclass
class BonnMotionConfig:
    bm_exe: str       # Chemin vers l'exécutable
    output_dir: str   # Dossier de sortie des traces
    scenario: str = "RandomWaypoint"
    
    # Paramètres du modèle de mobilité (Random Waypoint)
    vmin: float = 0.5
    vmax: float = 1.0
    pause: float = 50.0
    
    time_scale: float = 1.0  


class Simulation:
    def __init__(self, config, reg_aodv,node_positions,trace_file,traffic_seed):
        self.cfg = config
        
        self.time_points = []              
        """ Abscisse pour plot les résultats au cours du temps  """
        
        self.traffic_seed = traffic_seed   
        """ seed pour le générateur aléatoire de messages """

        self.MAX_DUPLICATES = 1 if reg_aodv else 3                     
        """ On s'autorise 3 RREQs max par (src_id,src_seq)  """
        
        self.WEIGHT_SEUIL = 1.0 if reg_aodv else 1.5                    
        """ Seuil à partir duquel on considère avoir vu une vraie amélioration """ 

        #création du réseau
        self.net = Network(
            config = self.cfg,
            reg_aodv = reg_aodv
        )


        if node_positions is None:
            raise NameError("Pas de position des noeuds lors de la simulation")
        else:
            self.node_positions = node_positions
        #création des noeuds
        for i in range(self.cfg.nb_nodes):
            self.net.add_node(id=i, pos=node_positions[i], max_dist=self.cfg.max_dist, battery=self.init_bat,reg_aodv=self.reg_aodv)

       
        if not os.path.exists(trace_file):
            raise FileNotFoundError(f"Fichier introuvabel : {trace_file}")
        # On lance le process SimPy
        self.net.env.process(self._bm_replay(trace_file))
        

    def _random_communication(self):
        """Simule des communications tant que la simulation n'est pas terminée"""
        rng = random.Random(self.traffic_seed)
        while self.net.env.now <= self.duration:
            src_id = rng.randint(0, self.nb_nodes-1)
            dest_id = rng.randint(0, self.nb_nodes-1)
            
            while dest_id == src_id:
                dest_id = rng.randint(0, self.nb_nodes-1)
            #on choisit deux noeuds différents

            src_node = self.net.G[src_id]
            if src_node.alive:
                src_node.send_data(dest_id) # on lance le tranfert de données
            
            yield self.net.env.timeout(self.cfg.dt) #petit délai pour pas flood

    # def _windowed_ratio(time_list, start_t, end_t, data_log):
    #     """Calcule le delivery ratio sur la fenêtre [start_t,end_t]"""
    #     if not time_list:
    #         return 0.0

    #     times_only = [ti for (ti, _) in time_list] # Tous les temps d'envoi/d'initialisation de messages
    #     lo = bisect.bisect_left(times_only, start_t) # Permet de déterminer l'indice dans la liste ↑ de start_t
    #     hi = bisect.bisect_right(times_only, end_t)  # ---------------------------------------------- end_t
        
    #     if lo >= hi:
    #         return 0.0

    #     keys_in_win = [time_list[i][1] for i in range(lo, hi)] #Toutes les keys des messages envoyés dans la fenêtre considérée
    #     delivered = sum(
    #         1
    #         for key in keys_in_win
    #         if (entry := data_log.get(key)) # Permet d'affecter entry et de tester en même temps si il n'est pas None
    #         and entry['t_recv'] is not None
    #         and entry['t_recv'] <= end_t
    #     )
    #     return 100.0 * delivered / len(keys_in_win)

    def _monitor(self):
        while self.net.env.now <= self.cfg.duration:
            current_time = self.net.env.now
            self.time_points.append(current_time) #points temporels pour ploter les données
            
            start_t = current_time - self.cfg.window_size
            # self.window_ratio_gen.append(
            #     self._windowed_ratio(self.net.data_init_times, start_t, current_time, self.net.data_log)
            # )
            # self.window_ratio_send.append(
            #     self._windowed_ratio(self.net.data_send_times, start_t, current_time, self.net.data_log)
            # )

            yield self.net.env.timeout(2*self.cfg.dt)  # ce qui donne tous les 2 messages envoyés

    def _bm_replay(self, file_path):
        """
        Rejoue les déplacements depuis un fichier .movements BonnMotion.
        """
        traces = {}
        with open(file_path, "r") as f:
            for nid, line in enumerate(f):
                vals = [float(v) for v in line.split()]
                # Format BonnMotion : t x y t x y ...
                if vals and nid < self.cfg.nb_nodes:
                     traces[nid] = list(zip(vals[0::3], vals[1::3], vals[2::3]))

        # Positionnement initial (t=0)
        indexes = {nid: 0 for nid in traces}
        for nid, seq in traces.items():
            if seq:
                self.net.G[nid].pos = (seq[0][1], seq[0][2])

        while self.net.env.now <= self.cfg.duration:
            sim_t = self.net.env.now
            
            for nid, seq in traces.items():
                if nid not in self.net.G or not self.net.G[nid].alive:
                    continue
                
                # On se place au bon endroit
                idx = indexes[nid]
                while idx + 1 < len(seq) and sim_t >= seq[idx + 1][0]:
                    idx += 1
                indexes[nid] = idx
                
                if idx + 1 < len(seq):
                    t0, x0, y0 = seq[idx]
                    t1, x1, y1 = seq[idx + 1]
                    
                    if t1 > t0 and sim_t >= t0:
                        u = (sim_t - t0) / (t1 - t0)
                        # Interpolation linéaire
                        new_x = x0 + u * (x1 - x0)
                        new_y = y0 + u * (y1 - y0)
                        
                        # Mise à jour
                        self.net.G[nid].pos = (new_x, new_y)
                else:
                    self.net.G[nid].pos = (seq[-1][1], seq[-1][2])

            yield self.net.env.timeout(self.cfg.dt)




    def get_metrics(self):
        final_avg_bat, final_std_bat = self.net.get_energy_stats()
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
            "final_avg_bat": final_avg_bat,
            "final_std_bat": final_std_bat,
            "fifty_percent_death": self.net.fifty_percent_death_time
        }
    
    def run(self):
        self.net.env.process(self._random_communication()) # on démarre les communications
        self.net.env.process(self._monitor()) # on démarre le monitoring pour récolter les données durant la simulation
        while not self.net.stop: 
            self.net.env.step()


def generate_bonnmotion_traces(sim_conf,bm_conf):
    os.makedirs(bm_conf.out_dir, exist_ok=True)
    movements_files = []

    for n_simu in range(sim_conf.nb_nodes):
        base = os.path.join(bm_conf.out_dir, f"{sim_conf.nb_nodes}rw{n_simu}")
        # Générer avec BM
        cmd = [
            bm_conf.bm_exe,
            "-f", base,
            "RandomWaypoint",
            "-n", str(sim_conf.nb_nodes),
            "-d", str(sim_conf.duration),
            "-x", str(sim_conf.area_size),
            "-y", str(sim_conf.area_size),
            "-l", str(bm_cfg.vmin),
            "-h", str(bm_cfg.vmax),
            "-p", str(bm_cfg.pause),
            "-o", str(bm_cfg.o),
        ]
        print("CMD>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        subprocess.run(cmd, check=True)

        # Décompresser .movements.gz
        gz_path = base + ".movements.gz"
        mov_path = base + ".movements"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rb") as f_in, open(mov_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
        
        # Supprimer le ".params"
        params_path = base + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)

        movements_files.append(mov_path)

    return movements_files


## Comparaison des protocoles ##

# Effectue nb_runs runs avec la configuration config
def run_comparison_simulations(config, nb_runs, seed_base, trace_files):
    reg_aodv_res = []
    mod_aodv_res = []

    for i in range(nb_runs):
        seed_i = seed_base + i
        
        random.seed(seed_i)
        np.random.seed(seed_i)
        
        positions = {
            i_node: (
                random.uniform(0, params["area_size"]),
                random.uniform(0, params["area_size"])
            )
            for i_node in range(config.nb_nodes)
        }
        
        for reg_aodv in [True, False]:
            random.seed(seed_i) #On réinitialise le rng pour avoir les mêmes transmissions et positions
            np.random.seed(seed_i)
            
            sim = Simulation(
                config=config,
                reg_aodv=reg_aodv,
                node_positions=positions,
                trace_file=trace_files[i],
                traffic_seed=seed_i
            )
            sim.run()
            
            if reg_aodv:
                reg_aodv_res.append(sim.get_metrics())
            else:
                mod_aodv_res.append(sim.get_metrics())
    
    return {"reg": reg_aodv_res, "mod": mod_aodv_res}

def calc_avg_metrics(res):
    if not res: return {}
    metric_keys = res[0].keys()
    avg = {}
    for key in metric_keys:
        values = [r[key] for r in res if r[key] is not None]
        if values:
            avg[key] = sum(values)/len(values)
            avg[f"{key}_count"] = len(values)
        else:
            avg[key] = None
            avg[f"{key}_count"] = 0
    return avg


# --- Parallélisation des runs + génération BM intégrée ---
from multiprocessing import Pool, cpu_count

def _one_point(args):
    (config,nb_runs,seed_base,trace_files) = args

    res = run_comparison_simulations(
        config= config,
        nb_runs=nb_runs,
        seed_base=seed_base,
        trace_files=trace_files
    )

    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return (reg_avg, mod_avg)



def densite_parallel(sim_conf,bm_conf,nb_runs,pas,factor_min=0.7, factor_max=1.5):
    """
    Fait la même chose qu'avant, mais :
      - génère d'abord, pour chaque N, les 'nb_runs' traces via BonnMotion
      - passe la liste des fichiers à run_comparison_simulations pour que les deux protocoles rejouent les mêmes chemins
    """
    n_min = (sim_conf.size / sim_conf.max_dist)**2 * np.pi
    # print(n_min)
    n_lo = max(2, int(round(factor_min * n_min)))
    n_hi = max(n_lo + 1, int(round(factor_max * n_min)))
    nb_nodes_list = list(range(n_lo, n_hi + 1, pas))

    tasks = []
    for N in nb_nodes_list:
        # --- Génère les nb_runs traces pour ce N ---
        sim_conf_for_this_N = replace(sim_conf,nb_runs=N) #Copie de la config 
        trace_files = generate_bonnmotion_traces(sim_conf_for_this_N,bm_conf)
        # Empile la tâche pour le pool
        tasks.append((sim_conf_for_this_N, nb_runs,12345,trace_files))

    # print(f"[densite] n_min={n_min:.2f}  n_lo={n_lo}  n_hi={n_hi}  pas={pas}")
    # print(f"[densite] nb_nodes_list={nb_nodes_list}")
    # print(f"[densite] nb_runs_par_N={params['nb_runs']}  => total_runs_par_protocole={len(nb_nodes_list)*params['nb_runs']}")
    
    # Lancement en parallèle
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(_one_point, tasks)

    # Trie les résultats par nb_nodes
    results.sort(key=lambda t: t[0])

    nb_nodes_array = []
    reg_first_death, mod_first_death = [], []
    reg_dr, mod_dr = [], []
    reg_ten_percent_death, mod_ten_percent_death = [], []
    reg_energy, mod_energy = [], []
    reg_std, mod_std = [], []
    reg_final_energy,mod_final_energy = [], []
    reg_fifty_percent_death, mod_fifty_percent_death = [], []

    for (N, reg_avg, mod_avg) in results:
        nb_nodes_array.append(N)

        reg_first_death.append(reg_avg.get("first_node_death", None))
        mod_first_death.append(mod_avg.get("first_node_death", None))

        reg_dr.append(reg_avg.get("msg_recv", 0) / max(1, reg_avg.get("messages_initiated",1)) * 100)
        mod_dr.append(mod_avg.get("msg_recv", 0) / max(1, mod_avg.get("messages_initiated",1)) * 100)

        reg_ten_percent_death.append(reg_avg.get("ten_percent_death", None))
        mod_ten_percent_death.append(mod_avg.get("ten_percent_death", None))

        reg_energy.append(reg_avg.get("energy", None))
        mod_energy.append(mod_avg.get("energy", None))

        reg_std.append(reg_avg.get("final_std_bat", None))
        mod_std.append(mod_avg.get("final_std_bat", None))

        reg_final_energy.append(reg_avg.get("final_avg_bat", None))
        mod_final_energy.append(mod_avg.get("final_avg_bat", None))

        reg_fifty_percent_death.append(reg_avg.get("fifty_percent_death", None))
        mod_fifty_percent_death.append(mod_avg.get("fifty_percent_death", None))



    # Affichage / ploting 
    plt.figure()
    plt.plot(nb_nodes_array, reg_first_death, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_first_death, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Temps (first node death)")
    plt.legend()
    plt.show()

    
    plt.figure()
    plt.plot(nb_nodes_array, reg_ten_percent_death, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_ten_percent_death, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Temps (10% death)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(nb_nodes_array, reg_fifty_percent_death, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_fifty_percent_death, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Temps (50% death)")
    plt.legend()
    plt.show()
    plt.figure()


    plt.plot(nb_nodes_array, reg_final_energy, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_final_energy, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Énergie résiduelle moyenne")
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(nb_nodes_array, reg_energy, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_energy, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Énergie totale consommée")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(nb_nodes_array, reg_std, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_std, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Écart type énergie finale")
    plt.legend()
    plt.show() 

    plt.figure()
    plt.plot(nb_nodes_array, reg_dr, marker='o', label="Regular")
    plt.plot(nb_nodes_array, mod_dr, marker='s', label="Modified")
    plt.xlabel("nb_nodes")
    plt.ylabel("Delivery ratio (%)")
    plt.legend()
    plt.show()

def plot_windowed_delivery_over_time(sim_reg, sim_mod ,W=None):
    import matplotlib.pyplot as plt
    if W is None: W = sim_reg.window_size

    plt.figure()
    plt.plot(sim_reg.time_points, sim_reg.window_ratio_gen, label="AODV classique (t_gen)")
    plt.plot(sim_mod.time_points, sim_mod.window_ratio_gen, label="AODV modifié (t_gen)")
    plt.xlabel("Temps")
    plt.ylabel("Taux de délivrance (%)")
    plt.title(f"Taux glissant basé sur t_gen (fenêtre={W})")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(sim_reg.time_points, sim_reg.window_ratio_send, label="AODV classique (t_send)")
    plt.plot(sim_mod.time_points, sim_mod.window_ratio_send, label="AODV modifié (t_send)")
    plt.xlabel("Temps")
    plt.ylabel("Taux de délivrance (%)")
    plt.title(f"Taux glissant basé sur t_send (fenêtre={W})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    bm_cfg = {
        "file": "",  # sera remplacé par chaque fichier généré OU par "files" injecté dans densite_parallel
        "time_scale": 0.01,       # 1 s BM = 100 s SimPy
    }

    params = {
        "nb_runs": 1,
        "size": 800,
        "conso": (0.0001,0.001),
        "seuil_coeff": 750,
        "coeff_dist_weight": 0.6,
        "coeff_bat_weight": 0.2,    
        "coeff_dist_bat": 0.005,
        "ttl": 100,
        "seed_base": 12345,
        "bm_cfg": bm_cfg,
        "init_bat" : 10000,
        "duration" : 100
    }
    out = densite_parallel(pas=2, max_dist=250, params=params, factor_min=0.5, factor_max=0.8,
                           bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité")


