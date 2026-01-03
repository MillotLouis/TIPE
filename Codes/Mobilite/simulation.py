import matplotlib.pyplot as plt
import random
import numpy as np
import os
import subprocess
import gzip
import shutil
import bisect

from network import Network

class Simulation:
    def __init__(self,nb_nodes, area_size, max_dist,conso,seuil,coeff_dist_weight,coeff_bat_weight,coeff_dist_bat,ttl,reg_aodv,init_bat, node_positions = None,bonnmotion=None,traffic_seed=None):
        self.nb_nodes = nb_nodes           
        """ Nombre de noeuds de la simulation """
        
        self.area_size = area_size         
        """ Taille de la carte : area_size*area_size """
        
        self.max_dist = max_dist           
        """ Distance max à laquelle un noeud peut transmettre """
        
        self.reg_aodv = reg_aodv           
        """  True si on utilise AODV et false sinon """
        
        self.time_points = []              
        """ Abscisse pour plot les résultats au cours du temps  """
        
        self.init_bat = init_bat           
        """ batterie initiale pour tous les noeuds """
        
        self.traffic_seed = traffic_seed   
        """ seed pour le générateur aléatoire de messages """
        
        self.window_size = 100.0           
        """ Taille de la fenêtre glissante pour représenter le delivery ratio """
        
        self.window_ratio_gen = []         # ?
        self.window_ratio_send = []


        #création du réseau
        self.net = Network(
            conso=conso,                   
            seuil=seuil,                
            coeff_dist_weight=coeff_dist_weight,
            coeff_bat_weight=coeff_bat_weight,
            coeff_dist_bat=coeff_dist_bat,
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
        
        self.traj = {nid: [] for nid in self.track_ids}

        # Garde: ne jamais laisser "files" arriver dans _bm_replay
        if bonnmotion and "files" in bonnmotion:
            bonnmotion = {k: v for k, v in bonnmotion.items() if k != "files"}

        if bonnmotion:
            self.net.env.process(self._bm_replay(**bonnmotion))
        
    def _create_links(self):
        for i in range(self.nb_nodes):
            for j in range(i + 1, self.nb_nodes):
                self.net.G.add_edge(i, j)

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
        while not self.net.stop:
            current_time = self.net.env.now
            self.time_points.append(current_time) #points temporels pour ploter les données
            
            start_t = current_time - self.window_size
            self.window_ratio_gen.append(
                self._windowed_ratio(self.net.data_init_times, start_t, current_time, self.net.data_log)
            )
            self.window_ratio_send.append(
                self._windowed_ratio(self.net.data_send_times, start_t, current_time, self.net.data_log)
            )

            if self.track_ids:
                for nid in self.track_ids:
                    if nid not in self.net.G:
                        continue
                    node = self.net.G.nodes[nid]['obj']                 # Permet de visualiser les trajectoires
                    x, y = node.pos
                    self.traj[nid].append((current_time, x, y))
            

            yield self.net.env.timeout(0.2)  # ce qui donne tous les 2 messages envoyés

    """     def _bm_parse_movements(self, path):
        traces = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    traces[i] = []
                    continue
                it = iter(parts)
                seq = []
                for t, x, y in zip(it, it, it):
                    seq.append((float(t), float(x), float(y)))
                seq.sort(key=lambda p: p[0]) #Trie la liste par temps
                traces[i] = seq
        return traces """

    """     def _bm_replay(self, file, time_scale=1.0):
        all_traces = self._bm_parse_movements(file) # Dico de listes de tuples (t,x,y)
        traces = {}
        indexes = {}

        #Positions de départ
        for nid, seq in all_traces.items():
            if nid in self.net.G.nodes and seq:
                traces[nid] = seq
                indexes[nid] = 0
                self.net.G.nodes[nid]['obj'].pos = seq[0][1:]
        self.positions_start = {n: self.net.G.nodes[n]['obj'].pos for n in self.net.G.nodes}

        seg_idx = {bm_id: 0 for bm_id in traces}
        dt = 0.05  # pas de temps

        while not self.net.stop:
            sim_t = self.net.env.now
            bm_t = sim_t / max(1e-12, time_scale)

            for bm_id, seq in traces.items():
                nid = bm_id  # identité
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

                # Clamp par défaut aux bornes de la zone
                nx = min(max(x, 0.0), self.area_size)
                ny = min(max(y, 0.0), self.area_size)

                node.pos = (nx, ny)

            yield self.net.env.timeout(dt)
    """


    def _bm_replay(self, file, time_scale=1.0):
        """
        Rejoue les déplacements
        """
        traces = {}
        with open(file, "r") as f:
            for nid, line in enumerate(f):
                vals = [float(v) for v in line.split()]
                if not vals or nid not in self.net.G.nodes:
                    continue
                # Crée des triplets (t, x, y)
                traces[nid] = list(zip(vals[0::3], vals[1::3], vals[2::3]))

        # Place tout les noeuds au départ
        indexes = {nid: 0 for nid in traces} # Curseur de lecture pour chaque noeud
        for nid, seq in traces.items():
            x0, y0 = seq[0][1], seq[0][2]
            self.net.G.nodes[nid]['obj'].pos = (x0, y0)
        
        dt = 0.5

        # Boucle principale de mouvement
        while not self.net.stop:
            sim_t = self.net.env.now

            for nid, seq in traces.items():
                node = self.net.G.nodes[nid]['obj']
                if not node.alive:
                    continue

                # --- Logique du curseur (INDEXES) ---
                # On avance le curseur 'idx' quand le segment actuel est fini
                # seq[idx] est le point de départ du segment actuel
                # seq[idx+1] est le point d'arrivée (la prochaine destination)
                idx = indexes[nid]
                
                # Tant qu'il reste des points ET que le temps actuel dépasse l'arrivée du segment
                while idx + 1 < len(seq) and sim_t >= seq[idx + 1][0]:
                    idx += 1
                
                indexes[nid] = idx # Sauvegarde la nouvelle position du curseur

                # --- Interpolation ---
                if idx + 1 < len(seq):
                    # On est entre le point idx (A) et idx+1 (B)
                    t_start, x_start, y_start = seq[idx]
                    t_end, x_end, y_end = seq[idx + 1]

                    # Si on est bien dans l'intervalle de temps (et pas avant le début)
                    if sim_t >= t_start:
                        # Pourcentage du trajet effectué (u va de 0 à 1)
                        total_duration = t_end - t_start
                        if total_duration > 0:
                            u = (sim_t - t_start) / total_duration
                            
                            # Formule : Pos = Départ + u * (Arrivée - Départ)
                            new_x = x_start + u * (x_end - x_start)
                            new_y = y_start + u * (y_end - y_start)
                            node.pos = (new_x, new_y)
                else:
                    # Plus de points futurs, le noeud reste à sa dernière position connue
                    _, last_x, last_y = seq[-1]
                    node.pos = (last_x, last_y)

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
        while not self.net.stop: # and self.net.env.now <= 10000
            self.net.env.step()

    # def print_results(self):
    #     print(f"Durée: {self.net.env.now:.2f} unités de temps")
    #     print(f"Noeuds morts: {self.net.dead_nodes}/{self.nb_nodes}")
    #     print(f"Énergie consommée: {self.net.energy_consumed:.2f}")
    #     print(f"Messages envoyés: {self.net.messages_sent}")
    #     print(f"Messages transmis: {self.net.messages_forwarded}")
    #     print(f"Messages reçus: {self.net.messages_received}")
    #     print(f"RREQ envoyés: {self.net.rreq_sent}")
    #     print(f"RREQ transmis: {self.net.rreq_forwarded}")
    #     print(f"RREP envoyés: {self.net.rrep_sent}")
    #     print(f"Seuiled: {self.net.seuiled}")
    #     print(f"Mort premier noeud: {self.net.first_node_death_time:.2f}" if self.net.first_node_death_time else "First Node Death: Not reached")
    #     print(f"Mort 10% noeuds: {self.net.ten_percent_death_time:.2f}" if self.net.ten_percent_death_time else "10% Node Death: Not reached")
    #     print(f"Moyenne batterie finale: {self.:.2f}")
    #     print(f"Écart type batterie finale: {self.std_bat_history[-1]:.2f}")


    def plot_paths(self, ids=None, title="Trajectoires de quelques nœuds"):
        """Permet de visualiser la trajectoire des noeuds renseignés dans ids"""
        ids = list(ids) if ids is not None else self.track_ids

        # Nuage des positions finales de tous les nœuds (gris clair)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        all_pos = [self.net.G.nodes[n]['obj'].pos for n in self.net.G.nodes]
        ax.scatter([p[0] for p in all_pos], [p[1] for p in all_pos], s=18, alpha=0.25, label="Autres nœuds (final)")

        # Trajectoires pour les nœuds choisis
        for nid in ids:
            pts = self.traj.get(nid, [])
            if len(pts) < 2: 
                continue
            xs = [x for _, x, _ in pts]
            ys = [y for _, _, y in pts]
            ax.plot(xs, ys, linewidth=1.8, label=f"nœud {nid}")
            ax.scatter([xs[0]], [ys[0]], marker="^", s=40)   # départ
            ax.scatter([xs[-1]], [ys[-1]], marker="o", s=40) # arrivée
            ax.text(xs[-1], ys[-1], str(nid), fontsize=9, ha="left", va="bottom")

        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(0, self.area_size); ax.set_ylim(0, self.area_size)
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


def _bm_generate_traces_for_N(nb_nodes, nb_runs, out_dir,size,
                            bm_exe=r"C:\Users\millo\Downloads\bonnmotion-3.0.1\bin\bm.bat",
                            duration=5000, X=400, Y=400, vmin=0.5, vmax=1.0, pause=50, o=2):
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
        # Générer avec BM
        cmd = [
            bm_exe,
            "-f", base,
            "RandomWaypoint",
            "-n", str(nb_nodes),
            "-d", str(duration),
            "-x", str(),
            "-y", str(Y),
            "-l", str(vmin),
            "-h", str(vmax),
            "-p", str(pause),
            "-o", str(o),
        ]
        print("CMD>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        subprocess.run(cmd, check=True)

        # Décompresser .movements.gz vers .movements
        gz_path = base + ".movements.gz"
        mov_path = base + ".movements"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rb") as f_in, open(mov_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
        
        # Supprimer le .params pour ne garder que .movements
        params_path = base + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)

        movements_files.append(mov_path)

    return movements_files


## Comparaison des protocoles ##

def run_comparison_simulations(nb_runs,nb_nodes,size,max_dist,conso,seuil,coeff_dist_weight,coeff_bat_weight,coeff_dist_bat,ttl,seed_base,bm_cfg=None,plot_dr=False, plot_mode='last'):
    reg_aodv_res = []
    mod_aodv_res = []

    params = {
        'nb_nodes': nb_nodes,
        'area_size': size,
        'max_dist': max_dist,
        'conso': conso,
        'seuil': seuil,
        'coeff_dist_weight': coeff_dist_weight,
        'coeff_bat_weight': coeff_bat_weight,
        'coeff_dist_bat': coeff_dist_bat,
        'ttl': ttl
    }

        # --- Préparer les fichiers BM pour tous les runs (génération si nécessaire) ---
    out_dir = bm_cfg.get("out_dir", os.path.join(os.getcwd(), "BM"))
    bm_exe = bm_cfg.get("bm_exe", r"C:\Users\millo\Downloads\bonnmotion-3.0.1\bin\bm.bat")
    duration = bm_cfg.get("duration", 500000)
    X = bm_cfg.get("X", size)
    Y = bm_cfg.get("Y", size)
    vmin = bm_cfg.get("vmin", 1.0)
    vmax = bm_cfg.get("vmax", 2.0)
    pause = bm_cfg.get("pause", 5)
    o = bm_cfg.get("o", 2)
    bm_files = _bm_generate_traces_for_N(
        nb_nodes=nb_nodes,
        size = size,
        nb_runs=nb_runs,
        out_dir=out_dir,
        bm_exe=bm_exe,
        duration=duration, X=X, Y=Y,
        vmin=vmin, vmax=vmax, pause=pause, o=o
    )


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
            init_bat=100000,
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
            init_bat=100000, #pour ne pas avoir de valeurs trop petites dans les consommations
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
        coeff_dist_weight=params["coeff_dist_weight"],
        coeff_bat_weight=params["coeff_bat_weight"],
        coeff_dist_bat=params["coeff_dist_bat"],
        ttl=params["ttl"],
        seed_base=params.get("seed_base", 12345),
        bm_cfg=bm_cfg
    )

    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return (nb_nodes, reg_avg, mod_avg)



def densite_parallel(pas, max_dist, params, factor_min=0.7, factor_max=1.5, procs=None,
                     bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité"):
    """
    Fait la même chose qu'avant, mais :
      - génère d'abord, pour chaque N, les 'nb_runs' traces via BonnMotion
      - passe la liste des fichiers à run_comparison_simulations pour que les deux protocoles rejouent les mêmes chemins
    """
    size = params["size"]
    n_min = (size / max_dist)**2 * math.pi
    print(n_min)
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
            duration=500000, X=size, Y=size, vmin=1.0, vmax=2.0, pause=5, o=2
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
    
    # res = run_comparison_simulations(
    #     nb_runs=1, nb_nodes=20, size=800, max_dist=250,
    #     conso=(1,20), seuil=750, coeff_dist_weight=0.6, coeff_bat_weight=0.2,
    #     coeff_dist_bat=0.005, ttl=100, seed_base=12345, bm_cfg=bm_cfg,
    #     plot_dr=True,          # << active le plot
    #     plot_mode='last'       # 'each' pour tracer à chaque run
    # )

    # Exemple simple : 10 runs sur N=20 sans densité parallèle
    # res = run_comparison_simulations(
    #     nb_runs=10, nb_nodes=20, size=800, max_dist=250,
    #     conso=(1,20), seuil=750, coeff_dist_weight=0.6, coeff_bat_weight=0.2,
    #     coeff_dist_bat=0.005, ttl=100,seed_base=12345,
    #     bm_cfg=bm_cfg
    # )
    # print_avg_results(res["reg"],res["mod"],10)
    
    # Exemple d'appel de densite_parallel (avec génération BM intégrée) :
    params = {
        "nb_runs": 2,
        "size": 800,
        "conso": (1, 20),
        "seuil": 750,
        "coeff_dist_weight": 0.6,
        "coeff_bat_weight": 0.2,
        "coeff_dist_bat": 0.005,
        "ttl": 100,
        "seed_base": 12345,
        "bm_cfg": bm_cfg
    }
    out = densite_parallel(pas=2, max_dist=250, params=params, factor_min=0.5, factor_max=0.8,
                           bm_out_dir=r"C:\Users\millo\Documents\GitHub\TIPE\Codes\Mobilité")


