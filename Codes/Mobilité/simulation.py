import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess
import gzip
import shutil

from network import Network

class Simulation:
    def __init__(self,nb_nodes, area_size, max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl,reg_aodv,init_bat, node_positions = None,bonnmotion=None,traffic_seed=None):
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
        self.traffic_seed = traffic_seed

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

        #création des liens
        self._create_links()

        # Garde: ne jamais laisser "files" arriver dans _bm_replay
        if bonnmotion and "files" in bonnmotion:
            bonnmotion = dict(bonnmotion)
            bonnmotion.pop("files", None)

        if bonnmotion:
            self.net.env.process(self._bm_replay(**bonnmotion))
        
    def _create_links(self):
        for i in range(self.nb_nodes):
            for j in range(i + 1, self.nb_nodes):
                self.net.G.add_edge(i, j)
        # réseau complet : toutes les connexions possibles sont créées mais la distance est vérifiée au moment de la transmission

    def _random_communication(self):
        """Simule des communications tant que la simulation n'est pas terminée"""
        rng = random.Random(self.traffic_seed)
        while not self.net.stop:
            src_id = rng.randint(0, self.nb_nodes-1)
            dest_id = rng.randint(0, self.nb_nodes-1)
            
            while dest_id == src_id:
                dest_id = rng.randint(0, self.nb_nodes-1)
            #on choisit deux noeuds différents

            src_node = self.net.G.nodes[src_id]['obj']
            if src_node.alive:
                src_node.send_data(dest_id) # on lance le tranfert de données
            
            yield self.net.env.timeout(0.1) #petit délai pour pas flood

    def _monitor(self):
        while not self.net.stop:
            self.time_points.append(self.net.env.now) #points temporels pour ploter les données
            self.energy_history.append(self.net.energy_consumed) #total d'énergie consommée
            self.dead_nodes_history.append(self.net.dead_nodes) #nb de noeuds morts
            
            avg_bat, std_bat = self.net.get_energy_stats()
            self.avg_bat_history.append(avg_bat) # moyenne de batterie des noeuds
            self.std_bat_history.append(std_bat) #écart type de batterie des noeuds
            
            yield self.net.env.timeout(0.2)  # ce qui donne tous les 2 messages envoyés

    def _bm_parse_movements(self, path):
        """
        Lit un fichier BonnMotion 'par nœud' où chaque ligne = t0 x0 y0 t1 x1 y1 ...
        Retour: {node_id: [(t, x, y), ...]} trié par t croissant.
        """
        traces = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    traces[i] = []
                    continue
                if len(parts) % 3 != 0:
                    raise ValueError(f"Ligne {i}: le nombre d'éléments n'est pas multiple de 3")
                it = iter(parts)
                seq = []
                for t, x, y in zip(it, it, it):
                    seq.append((float(t), float(x), float(y)))
                seq.sort(key=lambda p: p[0])
                traces[i] = seq
        return traces

    def _bm_replay(self, file, time_scale=1.0, space_scale=1.0, offset=(0.0, 0.0),
               clamp_to_area=True, start_at=0.0, dt=0.05, node_map=None):
        """
        Rejoue une trace BM en mettant à jour node.pos en continu.
        - time_scale: BM_t = SimPy_t / time_scale
          * 1.0  => 1s SimPy = 1s BM
          * 0.01 => 1s SimPy = 0.01s BM (donc 100s SimPy = 1s BM) => mobilité très lente côté BM
          * 100  => 1s SimPy = 100s BM => mobilité accélérée
        - space_scale: échelle spatiale, offset: translation (dx,dy)
        - node_map: remap optionnel {id_BM -> id_graphe}
        """
        traces = self._bm_parse_movements(file)

        def map_id(bm_id):
            return node_map.get(bm_id, bm_id) if node_map else bm_id

        # Position initiale d'après le premier waypoint
        for bm_id, seq in traces.items():
            if not seq:
                continue
            nid = map_id(bm_id)
            if nid in self.net.G.nodes:
                t0, x0, y0 = seq[0]
                self.net.G.nodes[nid]['obj'].pos = (x0 * space_scale + offset[0],
                                                    y0 * space_scale + offset[1])

        # Pointeurs de segments
        seg_idx = {bm_id: 0 for bm_id in traces}

        # Démarrage différé
        yield self.net.env.timeout(start_at)

        # Boucle principale
        while not self.net.stop:
            sim_t = self.net.env.now
            bm_t = sim_t / max(1e-12, time_scale)

            for bm_id, seq in traces.items():
                nid = map_id(bm_id)
                if nid not in self.net.G.nodes or not seq:
                    continue

                node = self.net.G.nodes[nid]['obj']
                if not node.alive:
                    continue

                # avance l'index jusqu'au segment couvrant bm_t
                k = seg_idx[bm_id]
                while k + 1 < len(seq) and seq[k + 1][0] <= bm_t:
                    k += 1
                seg_idx[bm_id] = k

                if bm_t <= seq[0][0]:
                    _, x, y = seq[0]
                elif bm_t >= seq[-1][0]:
                    _, x, y = seq[-1]
                else:
                    t0, x0, y0 = seq[k]
                    t1, x1, y1 = seq[k + 1]
                    u = (bm_t - t0) / max(1e-9, (t1 - t0))
                    x = x0 + u * (x1 - x0)
                    y = y0 + u * (y1 - y0)

                nx = x * space_scale + offset[0]
                ny = y * space_scale + offset[1]

                if clamp_to_area:
                    nx = min(max(nx, 0.0), self.area_size)
                    ny = min(max(ny, 0.0), self.area_size)

                node.pos = (nx, ny)

            yield self.net.env.timeout(dt)


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
            "final_std_bat": self.std_bat_history[-1],
            "fifty_percent_death": self.net.fifty_percent_death_time
        }
    
    def run(self):
        self.net.env.process(self._random_communication()) # on démarre les communications
        self.net.env.process(self._monitor()) # on démarre le monitoring pour récolter les données durant la simulation
        while not self.net.stop: # and self.net.env.now <= 10000
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
        print(f"Moyenne batterie finale: {self.avg_bat_history[-1]:.2f}")
        print(f"Écart type batterie finale: {self.std_bat_history[-1]:.2f}")

## Comparaison des protocoles ##

def run_comparison_simulations(nb_runs,nb_nodes,size,max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl,seed_base,bm_cfg=None):
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

    for i in range(nb_runs):
        seed_i = seed_base + i

        # Choix du fichier BM pour ce run :
        bm_cfg_run = None
        if bm_cfg:
            bm_cfg_run = dict(bm_cfg)  # copie superficielle
            bm_files = bm_cfg_run.pop("files", None)  # RETIRE 'files' (clé non attendue par _bm_replay)
            if bm_files:
                bm_cfg_run["file"] = bm_files[i % len(bm_files)]
            else:
                if not bm_cfg_run.get("file"):
                    bm_cfg_run["file"] = f"rw{i}.movements"

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
            init_bat=100000,
            bonnmotion=bm_cfg_run,
            traffic_seed=seed_i,
            **params
        )
        sim_reg.run()
        reg_aodv_res.append(sim_reg.get_metrics())

        random.seed(seed_i)   
        np.random.seed(seed_i)

        #Simulation avec AODV modifié
        print("\nstarting mod aodv sim")
        sim_mod = Simulation(
            node_positions=positions,
            reg_aodv=False,
            init_bat=100000, #pour ne pas avoir de valeurs trop petites dans les consommations
            bonnmotion=bm_cfg_run,
            traffic_seed=seed_i,
            **params
        )
        sim_mod.run()
        mod_aodv_res.append(sim_mod.get_metrics())

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
        coeff_dist=params["coeff_dist"],
        coeff_bat=params["coeff_bat"],
        coeff_conso=params["coeff_conso"],
        ttl=params["ttl"],
        seed_base=params.get("seed_base", 12345),
        bm_cfg=bm_cfg
    )

    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return (nb_nodes, reg_avg, mod_avg)

def _bm_generate_traces_for_N(nb_nodes, nb_runs, out_dir,
                              bm_exe=r"C:\Users\millo\Downloads\bonnmotion-3.0.1\bin\bm.bat",
                              duration=5000, X=400, Y=400, vmin=1.0, vmax=2.0, pause=5, o=2):
    """
    Génère nb_runs traces pour une valeur de nb_nodes.
    Commande demandée :
      bm -f "<out_dir>\{nb_nodes}rw{n_simu}" RandomWaypoint -n {nb_nodes} -d 5000 -x 400 -y 400 -l 1.0 -h 2.0 -p 5 -o 2
    Retourne la liste complète des chemins .movements (décompressés).
    """
    os.makedirs(out_dir, exist_ok=True)
    movements_files = []

    for n_simu in range(nb_runs):
        base = os.path.join(out_dir, f"{nb_nodes}rw{n_simu}")
        # 1) Générer avec BM
        cmd = [
            bm_exe,
            "-f", base,
            "RandomWaypoint",
            "-n", str(nb_nodes),
            "-d", str(duration),
            "-x", str(X),
            "-y", str(Y),
            "-l", str(vmin),
            "-h", str(vmax),
            "-p", str(pause),
            "-o", str(o),
        ]
        print("CMD>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        subprocess.run(cmd, check=True)

        # 2) Décompresser .movements.gz vers .movements
        gz_path = base + ".movements.gz"
        mov_path = base + ".movements"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rb") as f_in, open(mov_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
        # 3) Optionnel : supprimer le .params pour ne garder que .movements
        params_path = base + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)

        movements_files.append(mov_path)

    return movements_files

def densite_parallel(pas, max_dist, params, factor_min=0.7, factor_max=1.5, procs=None,
                     bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité"):
    """
    Fait la même chose qu'avant, mais :
      - génère d'abord, pour chaque N, les 'nb_runs' traces via BonnMotion
      - passe la liste des fichiers à run_comparison_simulations pour que les deux protocoles rejouent les mêmes chemins
    """
    size = params["size"]
    n_min = (size / max_dist)**2 * math.pi
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
            nb_runs=params["nb_runs"],
            out_dir=bm_out_dir,
            bm_exe=r"C:\Users\millo\Downloads\bonnmotion-3.0.1\bin\bm.bat",
            duration=5000, X=800, Y=800, vmin=1.0, vmax=2.0, pause=5, o=2
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

if __name__ == "__main__":
    
    bm_cfg = {
        "file": "",  # sera remplacé par chaque fichier généré OU par "files" injecté dans densite_parallel
        "time_scale": 0.01,       # 1 s BM = 100 s SimPy
        "space_scale": 1.0,
        "offset": (0.0, 0.0),
        "clamp_to_area": True,
        "start_at": 0.0,
        "dt": 0.1
    }
    
    # Exemple simple : 10 runs sur N=20 sans densité parallèle
    # res = run_comparison_simulations(
    #     nb_runs=10, nb_nodes=20, size=800, max_dist=250,
    #     conso=(1,20), seuil=750, coeff_dist=0.6, coeff_bat=0.2,
    #     coeff_conso=0.005, ttl=100,seed_base=12345,
    #     bm_cfg=bm_cfg
    # )
    # print_avg_results(res["reg"],res["mod"],10)
    
    # Exemple d'appel de densite_parallel (avec génération BM intégrée) :
    params = {
        "nb_runs": 5,
        "size": 800,
        "conso": (1, 20),
        "seuil": 750,
        "coeff_dist": 0.6,
        "coeff_bat": 0.2,
        "coeff_conso": 0.005,
        "ttl": 100,
        "seed_base": 12345,
        "bm_cfg": bm_cfg
    }
    out = densite_parallel(pas=2, max_dist=250, params=params, factor_min=1.5, factor_max=1.5,
                           bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité")


