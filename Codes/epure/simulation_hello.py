from __future__ import annotations

import random
from dataclasses import dataclass,replace
from typing import Dict, Tuple

from time import time

import numpy as np
from matplotlib import pyplot as plt

from network_hello import Network
from node_hello import Node

from multiprocessing import Pool, cpu_count, current_process



@dataclass(frozen=True)
class SimConfig:
    """Paramètres globaux de simulation."""
    nb_nodes: int
    area_size: int
    max_dist: float
    init_bat: float #En Joules
    conso: Tuple[float, float, int] # RX, TX, TX(data)/TX(contrôle) = RX(data)/RX(contrôle)
    dt: float
    ttl: int
    seuil_coeff: float
    coeff_dist_weight: float
    coeff_bat_weight: float
    duration: float
    window_size: float = 100.0


@dataclass(frozen=True)
class ProtocolConfig:
    """Paramètres du protocole (AODV / Energy Aware)."""
    reg_aodv: bool
    max_duplicates: int
    weight_seuil: float

    @classmethod
    def from_mode(cls, reg_aodv: bool) -> "ProtocolConfig":
        return cls(reg_aodv=reg_aodv, max_duplicates=1 if reg_aodv else 2, weight_seuil=1.0 if reg_aodv else 1.5)


class Simulation:
    def __init__(self, config: SimConfig, protocol: ProtocolConfig, node_positions: Dict[int, Tuple[float, float]], trace_file: str, traffic_seed: int):
        self.cfg = config
        self.protocol = protocol
        self.traffic_seed = traffic_seed  # seed pour le générateur aléatoire de messages
        self.time_points = []  # Abscisse pour plot les résultats au cours du temps
        self.net = Network(config=config, reg_aodv=protocol.reg_aodv, protocol=protocol)

        if node_positions is None:
            raise ValueError("node_positions ne peut pas être None")

        for node_id in range(self.cfg.nb_nodes):
            node = Node(node_id=node_id, pos=node_positions[node_id], initial_battery=self.cfg.init_bat, max_dist=self.cfg.max_dist, network=self.net)
            self.net.add_node(node)

        self.net.env.process(self._bm_replay(trace_file))

    def _random_communication(self):
        rng = random.Random(self.traffic_seed)
        while self.net.env.now <= self.cfg.duration:
            src_id = rng.randint(0, self.cfg.nb_nodes - 1)
            dest_id = rng.randint(0, self.cfg.nb_nodes - 1)
            while dest_id == src_id:
                dest_id = rng.randint(0, self.cfg.nb_nodes - 1)
            # on choisit deux noeuds différents
            src_node = self.net.G[src_id]
            if src_node.alive:
                src_node.send_data(dest_id)  # on lance le tranfert de données
            yield self.net.env.timeout(self.cfg.dt)  # petit délai pour pas flood

    def _monitor(self):
        while self.net.env.now <= self.cfg.duration:
            self.time_points.append(self.net.env.now)  # points temporels pour ploter les données
            yield self.net.env.timeout(2 * self.cfg.dt)  # ce qui donne tous les 2 messages envoyés

    def _bm_replay(self, file_path):
        traces = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for nid, line in enumerate(f):
                vals = [float(v) for v in line.split()]
                # Format BonnMotion : t x y t x y ...
                if vals and nid < self.cfg.nb_nodes:
                    traces[nid] = list(zip(vals[0::3], vals[1::3], vals[2::3]))

        indexes = {nid: 0 for nid in traces}
        for nid, seq in traces.items():
            if seq:
                self.net.G[nid].pos = (seq[0][1], seq[0][2])

        while self.net.env.now <= self.cfg.duration:
            sim_t = self.net.env.now
            for nid, seq in traces.items():
                node = self.net.G.get(nid)
                if node is None or not node.alive:
                    continue
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
                        node.pos = (x0 + u * (x1 - x0), y0 + u * (y1 - y0))  # Mise à jour
                elif seq:
                    node.pos = (seq[-1][1], seq[-1][2])
            yield self.net.env.timeout(self.cfg.dt)

    def run(self):
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        while not self.net.stop and self.net.env.now <= self.cfg.duration:
            self.net.env.step()

    def get_metrics(self):
        final_avg_bat, final_std_bat = self.net.get_energy_stats()
        s = self.net.stats
        return {
            "dead_nodes": s.dead_nodes,
            "energy": s.energy_consumed,
            "msg_recv": s.messages_received,
            "msg_sent": s.messages_sent,
            "rreq_sent": s.rreq_sent,
            "duration": self.net.env.now,
            "rrep_sent": s.rrep_sent,
            "messages_forwarded": s.messages_forwarded,
            "messages_initiated": s.messages_initiated,
            "rreq_forwarded": s.rreq_forwarded,
            "seuiled": s.seuiled,
            "first_node_death": self.net.stats.first_node_death_time,
            "ten_percent_death": self.net.stats.ten_percent_death_time,
            "final_avg_bat": final_avg_bat,
            "final_std_bat": final_std_bat,
            "fifty_percent_death": self.net.stats.fifty_percent_death_time,
        }

@dataclass(frozen=True)
class BonnMotionConfig:
    bm_exe: str
    out_dir: str
    scenario: str = "RandomWaypoint"
    vmin: int = 0
    vmax: int = 1
    pause: int = 50
    o: int = 2


def generate_bonnmotion_traces(sim_conf: SimConfig, bm_conf: BonnMotionConfig, nb_runs : int):
    import gzip
    import os
    import shutil
    import subprocess

    os.makedirs(bm_conf.out_dir, exist_ok=True)
    movements_files = []

    for n_simu in range(nb_runs):
        base = os.path.join(bm_conf.out_dir, f"{sim_conf.nb_nodes}rw{n_simu}")
        cmd = " ".join([
            bm_conf.bm_exe,
            "-f", base,
            bm_conf.scenario,
            "-n", str(sim_conf.nb_nodes),
            "-d", str(sim_conf.duration),
            "-x", str(sim_conf.area_size),
            "-y", str(sim_conf.area_size),
            "-l", str(bm_conf.vmin),
            "-h", str(bm_conf.vmax),
            "-p", str(bm_conf.pause),
            "-o", str(bm_conf.o),
        ])
        print(cmd)
        subprocess.run(cmd, check=True)

        gz_path = base + ".movements.gz"
        mov_path = base + ".movements"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rb") as f_in, open(mov_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)

        params_path = base + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)
        movements_files.append(mov_path)

    return movements_files




def _run_single_mode(args):
    config, protocol, positions, trace_file, seed_i = args
    random.seed(seed_i)
    np.random.seed(seed_i)
    sim = Simulation(config=config, protocol=protocol, node_positions=positions, trace_file=trace_file, traffic_seed=seed_i)
    sim.run()
    return sim.get_metrics()


def _run_single_comparison(args):
    config, seed_i, trace_file = args
    random.seed(seed_i)
    np.random.seed(seed_i)

    positions = {
        i_node: (
            random.uniform(0, config.area_size),
            random.uniform(0, config.area_size),
        )
        for i_node in range(config.nb_nodes)
    }

    reg_protocol = ProtocolConfig.from_mode(True)
    mod_protocol = ProtocolConfig.from_mode(False)

    reg_metrics = _run_single_mode((config, reg_protocol, positions, trace_file, seed_i))
    mod_metrics = _run_single_mode((config, mod_protocol, positions, trace_file, seed_i))
    return reg_metrics, mod_metrics

def run_comparison_simulations(config: SimConfig, nb_runs: int, seed_base: int, trace_files):
    print(f"Simulations a {config.nb_nodes} noeuds débutées")

    tasks = [(config, seed_base + i, trace_files[i]) for i in range(nb_runs)]

    # Sous Windows (spawn), un worker de Pool est daemon et ne peut pas créer de sous-processus.
    # On garde donc la parallélisation des runs uniquement si on est dans le process principal.
    if current_process().daemon:
        run_results = [_run_single_comparison(task) for task in tasks]
    else:
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            run_results = pool.map(_run_single_comparison, tasks)

    reg_aodv_res = [reg_metrics for reg_metrics, _ in run_results]
    mod_aodv_res = [mod_metrics for _, mod_metrics in run_results]

    print(f"Simulations a {config.nb_nodes} noeuds terminées\n")
    return {"reg": reg_aodv_res, "mod": mod_aodv_res}


def calc_avg_metrics(res):
    if not res:
        return {}
    avg = {}
    for key in res[0].keys():
        values = [r[key] for r in res if r[key] is not None]
        avg[key] = (sum(values) / len(values)) if values else None
        avg[f"{key}_count"] = len(values)
    return avg

def _one_point(args):
    config, nb_runs, seed_base, trace_files = args
    res = run_comparison_simulations(config=config, nb_runs=nb_runs, seed_base=seed_base, trace_files=trace_files)
    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return config.nb_nodes, reg_avg, mod_avg


def densite_parallel(sim_conf: SimConfig, bm_conf: BonnMotionConfig, nb_runs: int, pas: int, deg_min: float = 0.7, deg_max: float = 1.5):
    n_crit_moins_1 = (sim_conf.area_size / sim_conf.max_dist) ** 2 / np.pi
    n_lo = max(2, int(round(deg_min * n_crit_moins_1))+1)
    n_hi = max(n_lo + 1, int(round(deg_max * n_crit_moins_1))+1)
    nb_nodes_list = list(range(n_lo, n_hi + 1, pas))

    tasks = []
    for n_nodes in nb_nodes_list:
        sim_conf_n = replace(sim_conf,nb_nodes=n_nodes) #Copie de la config 
        trace_files = generate_bonnmotion_traces(sim_conf_n, bm_conf,nb_runs)
        tasks.append((sim_conf_n, nb_runs, int(time()), trace_files))

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(_one_point, tasks)

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


    # Affichage dans une seule fenêtre : sous-graphiques côte à côte
    metrics_to_plot = [
        ("Temps (first node death)", reg_first_death, mod_first_death),
        ("Temps (10% death)", reg_ten_percent_death, mod_ten_percent_death),
        ("Temps (50% death)", reg_fifty_percent_death, mod_fifty_percent_death),
        ("Énergie résiduelle moyenne", reg_final_energy, mod_final_energy),
        ("Énergie totale consommée", reg_energy, mod_energy),
        ("Écart type énergie finale", reg_std, mod_std),
        ("Delivery ratio (%)", reg_dr, mod_dr),
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(metrics_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (ylabel, reg_vals, mod_vals) in zip(axes_flat, metrics_to_plot):
        ax.plot(nb_nodes_array, reg_vals, marker='o', label="Regular")
        ax.plot(nb_nodes_array, mod_vals, marker='s', label="Modified")
        ax.set_xlabel("nb_nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[len(metrics_to_plot):]:
        ax.axis("off")

    fig.suptitle("Comparaison AODV régulier vs modifié selon la densité", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()
    return results


def plot_windowed_delivery_over_time(sim_reg, sim_mod, W=None):

    if W is None:
        W = sim_reg.cfg.window_size

    if hasattr(sim_reg, "window_ratio_gen") and hasattr(sim_mod, "window_ratio_gen"):
        plt.figure()
        plt.plot(sim_reg.time_points, sim_reg.window_ratio_gen, label="AODV classique (t_gen)")
        plt.plot(sim_mod.time_points, sim_mod.window_ratio_gen, label="AODV modifié (t_gen)")
        plt.xlabel("Temps")
        plt.ylabel("Taux de délivrance (%)")
        plt.title(f"Taux glissant basé sur t_gen (fenêtre={W})")
        plt.legend()
        plt.show()

    if hasattr(sim_reg, "window_ratio_send") and hasattr(sim_mod, "window_ratio_send"):
        plt.figure()
        plt.plot(sim_reg.time_points, sim_reg.window_ratio_send, label="AODV classique (t_send)")
        plt.plot(sim_mod.time_points, sim_mod.window_ratio_send, label="AODV modifié (t_send)")
        plt.xlabel("Temps")
        plt.ylabel("Taux de délivrance (%)")
        plt.title(f"Taux glissant basé sur t_send (fenêtre={W})")
        plt.legend()
        plt.show()

if __name__ == "__main__" :
    sim_conf = SimConfig(
        nb_nodes=0,
        area_size=800,
        max_dist=200,
        init_bat=100,
        conso=(0.0082,0.00164,100),
        dt=0.25,
        ttl=100,
        seuil_coeff=0.075,  # 750 / 10000
        coeff_dist_weight=0.6,
        coeff_bat_weight=0.4,
        duration=100,
    )

    bm_conf = BonnMotionConfig(
        bm_exe="C:\\Users\\millo\\Documents\\bonnmotion-3.0.1\\bin\\bm.bat",
        out_dir="C:\\Users\\millo\\Documents\\GitHub\\TIPE\\bm_files\\",
        vmin=10,
        vmax=10,
        pause=0.5
    )
    res = densite_parallel(sim_conf,bm_conf,5,2,20,20)
    print(res)
