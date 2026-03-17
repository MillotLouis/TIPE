import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess
import gzip
import shutil
import bisect

from network import Network


from dataclasses import dataclass
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
    
    # Paramètres du Protocole (AODV / Energy Aware)
    ttl: int
    seuil: float
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
    def __init__(self, config, reg_aodv,node_positions = None,trace_file = None,traffic_seed=None):
        self.cfg = config
        
        self.reg_aodv = reg_aodv           
        """  True si on utilise AODV et false sinon """
        
        self.time_points = []              
        """ Abscisse pour plot les résultats au cours du temps  """
        
        self.traffic_seed = traffic_seed   
        """ seed pour le générateur aléatoire de messages """
        
        self.window_size = 100.0           
        """ Taille de la fenêtre glissante pour représenter le delivery ratio """

        self.node_positions = node_positions or {} #si on a déjà une configuration on l'importe sinon on en crée une
        #création des noeuds
        for i in range(self.cfg.nb_nodes):
            if i in self.node_positions:
                pos = self.node_positions[i]
            else:
                pos = (random.uniform(0, self.area_size), random.uniform(0, self.area_size))
                self.node_positions[i] = pos
            
            self.net.add_node(id=i, pos=pos, max_dist=self.cfg.max_dist, battery=self.init_bat,reg_aodv=self.reg_aodv)        

        self.window_ratio_gen = []         # ?
        self.window_ratio_send = []


        #création du réseau
        self.net = Network(
            conso=self.cfg.conso,                   
            seuil=self.cfg.seuil,                
            coeff_dist_weight=self.cfg.coeff_dist_weight,
            coeff_bat_weight=self.cfg.coeff_bat_weight,
            coeff_dist_bat=self.cfg.coeff_dist_bat,
            nb_nodes=self.cfg.nb_nodes,
            ttl=self.cfg.ttl,
            reg_aodv = reg_aodv
        )

        if trace_file:
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
            
            yield self.net.env.timeout(0.1) #petit délai pour pas flood

    def _windowed_ratio(time_list, start_t, end_t, data_log):
        """Calcule le delivery ratio sur la fenêtre [start_t,end_t]"""
        if not time_list:
            return 0.0

        times_only = [ti for (ti, _) in time_list] # Tous les temps d'envoi/d'initialisation de messages
        lo = bisect.bisect_left(times_only, start_t) # Permet de déterminer l'indice dans la liste ↑ de start_t
        hi = bisect.bisect_right(times_only, end_t)  # ---------------------------------------------- end_t
        
        if lo >= hi:
            return 0.0

        keys_in_win = [time_list[i][1] for i in range(lo, hi)] #Toutes les keys des messages envoyés dans la fenêtre considérée
        delivered = sum(
            1
            for key in keys_in_win
            if (entry := data_log.get(key)) # Permet d'affecter entry et de tester en même temps si il n'est pas None
            and entry['t_recv'] is not None
            and entry['t_recv'] <= end_t
        )
        return 100.0 * delivered / len(keys_in_win)

    def _monitor(self):
        while self.net.env.now <= self.cfg.duration:
            current_time = self.net.env.now
            self.time_points.append(current_time) #points temporels pour ploter les données
            
            start_t = current_time - self.cfg.window_size
            self.window_ratio_gen.append(
                self._windowed_ratio(self.net.data_init_times, start_t, current_time, self.net.data_log)
            )
            self.window_ratio_send.append(
                self._windowed_ratio(self.net.data_send_times, start_t, current_time, self.net.data_log)
            )

            yield self.net.env.timeout(0.2)  # ce qui donne tous les 2 messages envoyés

    def _bm_replay(self, file_path):
        """
        Rejoue les déplacements depuis un fichier .movements standard BonnMotion.
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

        dt = 0.5 # Pas de temps de mise à jour de la position
        
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

            yield self.net.env.timeout(dt)




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


def generate_bonnmotion_traces(sim_conf,bm_conf,nb_runs):
    os.makedirs(bm_conf.out_dir, exist_ok=True)
    movements_files = []

    for n_simu in range(nb_runs):
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
        
        # Supprimer le .params
        params_path = base + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)

        movements_files.append(mov_path)

    return movements_files


## Comparaison des protocoles ##

def run_comparison_simulations(config, nb_runs, seed_base, trace_files):
    reg_aodv_res = []
    mod_aodv_res = []

    params = {
        'nb_nodes': config.nb_nodes,
        'area_size': config.size,
        'max_dist': config.max_dist,
        'conso': config.conso,
        'seuil': config.seuil,
        'coeff_dist_weight': config.coeff_dist_weight,
        'coeff_bat_weight': config.coeff_bat_weight,
        'coeff_dist_bat': config.coeff_dist_bat,
        'ttl': config.ttl
    }

    last_pair = None  # pour tracer à la fin si besoin

    for i in range(nb_runs):
        seed_i = seed_base + i

        # -- BM pour ce run
        bm_cfg_run = None
        if bm_cfg:
            bm_cfg_run = {k: v for k, v in bm_cfg.items()
                          if k not in ("files","out_dir","bm_exe","duration","X","Y","vmin","vmax","pause","o")}
            if bm_files:
                bm_cfg_run["file"] = bm_files[i % len(bm_files)]
            else:
                bm_cfg_run = None

        positions = {} # mêmes positions initiales pour les deux simulations
        for i_node in range(params["nb_nodes"]):
            positions[i_node] = (
                random.uniform(0, params['area_size']), 
                random.uniform(0, params['area_size'])
            )
        
        random.seed(seed_i)   
        np.random.seed(seed_i)

        #Simulation avec AODV classique
        print("\nstarting reg aodv sim")
        sim_reg = Simulation(
            node_positions=positions,
            reg_aodv=True,
            init_bat=init_bat,
            bonnmotion=bm_cfg_run,
            traffic_seed=seed_i,
            **params
        )
        sim_reg.run()
        reg_aodv_res.append(sim_reg.get_metrics())
        # sim_reg.plot_positions_before_after()
        # sim_reg.plot_paths()

        random.seed(seed_i)   
        np.random.seed(seed_i)

        #Simulation avec AODV modifié
        print("\nstarting mod aodv sim")
        sim_mod = Simulation(
            node_positions=positions,
            reg_aodv=False,
            init_bat=init_bat, 
            bonnmotion=bm_cfg_run,
            traffic_seed=seed_i,
            **params
        )
        sim_mod.run()
        mod_aodv_res.append(sim_mod.get_metrics())
        # sim_mod.plot_positions_before_after()
        # sim_mod.plot_paths()

        last_pair = (sim_reg, sim_mod)

        if plot_dr and plot_mode == 'each':
            plot_windowed_delivery_over_time(sim_reg, sim_mod, W=sim_reg.window_size)
    if plot_dr and plot_mode in ('last', 'final') and last_pair is not None:
        sim_reg, sim_mod = last_pair
        plot_windowed_delivery_over_time(sim_reg, sim_mod, W=sim_reg.window_size)
    


    return {"reg":reg_aodv_res,"mod":mod_aodv_res}

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

def print_avg_results(reg_res,mod_res,nb_runs):
    print(f"\n\nMoyennes sur {nb_runs} simulations")
    print("="*60)

    print(f"\n{'Métrique':<25} {'Regular AODV':<15} {'Count Reg':<12} {'Modified AODV':<15} {'Count Mod':<12} {'Changement':<12}")
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
    
    print(f"\n{'Delivery Ratio':<25} {reg_delivery_ratio:<15.1f}% {'':<12} {mod_delivery_ratio:<15.1f}% {'':<12} {'':<12}")

    print("\n" + "="*60)

# --- Parallélisation des runs + génération BM intégrée ---
from multiprocessing import Pool, cpu_count
import math

def _one_point(args):
    (nb_nodes, max_dist, params, bm_files_for_this_N) = args

    # prépare le bm_cfg en injectant la liste de fichiers pour ce N
    bm_cfg = params.get("bm_cfg", {}).copy() if params.get("bm_cfg") else {}
    bm_cfg["files"] = bm_files_for_this_N

    res = run_comparison_simulations(
        nb_nodes=nb_nodes,
        max_dist=max_dist,
        nb_runs=params["nb_runs"],
        size=params["size"],
        conso=params["conso"],
        seuil=params["seuil"],
        coeff_dist_weight=params["coeff_dist_weight"],
        coeff_bat_weight=params["coeff_bat_weight"],
        coeff_dist_bat=params["coeff_dist_bat"],
        ttl=params["ttl"],
        seed_base=params.get("seed_base", 12345),
        bm_cfg=bm_cfg,
        init_bat= params["init_bat"]
    )

    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return (nb_nodes, reg_avg, mod_avg)



def densite_parallel(pas, max_dist, params,factor_min=0.7, factor_max=1.5, procs=None,
                     bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité"):
    """
    Fait la même chose qu'avant, mais :
      - génère d'abord, pour chaque N, les 'nb_runs' traces via BonnMotion
      - passe la liste des fichiers à run_comparison_simulations pour que les deux protocoles rejouent les mêmes chemins
    """
    size = params["size"]
    n_min = (size / max_dist)**2 * math.pi
    # print(n_min)
    n_lo = max(2, int(round(factor_min * n_min)))
    n_hi = max(n_lo + 1, int(round(factor_max * n_min)))
    nb_nodes_list = list(range(n_lo, n_hi + 1, pas))

    if procs is None:
        procs = max(1, cpu_count() - 1)

    tasks = []
    for N in nb_nodes_list:
        # --- Génère les nb_runs traces pour ce N ---
        bm_files_for_this_N = _bm_generate_traces_for_N(
            nb_nodes=N,
            size = size,
            nb_runs=params["nb_runs"],
            out_dir=bm_out_dir,
            bm_exe=r"C:\Users\millo\Downloads\bonnmotion-3.0.1\bin\bm.bat",
            duration=params["duration"], X=size, Y=size, vmin=0.5, vmax=1, pause=10, o=2
        )
        # Empile la tâche pour le pool
        tasks.append((N, max_dist, params, bm_files_for_this_N))

    print(f"[densite] n_min={n_min:.2f}  n_lo={n_lo}  n_hi={n_hi}  pas={pas}")
    print(f"[densite] nb_nodes_list={nb_nodes_list}")
    print(f"[densite] nb_runs_par_N={params['nb_runs']}  => total_runs_par_protocole={len(nb_nodes_list)*params['nb_runs']}")
    
    # Lancement en parallèle
    with Pool(processes=procs) as pool:
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


