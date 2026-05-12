from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from network_gpt import Network
from node_gpt import Node


@dataclass(frozen=True)
class SimConfig:
    """Paramètres globaux de simulation."""
    nb_nodes: int
    area_size: int
    max_dist: float
    init_bat: float
    conso: Tuple[float, float]
    dt: float
    ttl: int
    seuil_coeff: float
    coeff_dist_weight: float
    coeff_bat_weight: float
    coeff_dist_bat: float
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
        return cls(reg_aodv=reg_aodv, max_duplicates=1 if reg_aodv else 3, weight_seuil=1.0 if reg_aodv else 1.5)


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
            "first_node_death": self.net.first_node_death_time,
            "ten_percent_death": self.net.ten_percent_death_time,
            "final_avg_bat": final_avg_bat,
            "final_std_bat": final_std_bat,
            "fifty_percent_death": self.net.fifty_percent_death_time,
        }

@dataclass(frozen=True)
class BonnMotionConfig:
    bm_exe: str
    out_dir: str
    scenario: str = "RandomWaypoint"
    vmin: float = 0.5
    vmax: float = 1.0
    pause: float = 50.0
    o: float = 0.0


def generate_bonnmotion_traces(sim_conf: SimConfig, bm_conf: BonnMotionConfig):
    import gzip
    import os
    import shutil
    import subprocess

    os.makedirs(bm_conf.out_dir, exist_ok=True)
    movements_files = []

    for n_simu in range(sim_conf.nb_nodes):
        base = os.path.join(bm_conf.out_dir, f"{sim_conf.nb_nodes}rw{n_simu}")
        cmd = [
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
        ]
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


def run_comparison_simulations(config: SimConfig, nb_runs: int, seed_base: int, trace_files):
    reg_aodv_res, mod_aodv_res = [], []
    for i in range(nb_runs):
        seed_i = seed_base + i
        random.seed(seed_i)
        np.random.seed(seed_i)

        positions = {
            i_node: (
                random.uniform(0, config.area_size),
                random.uniform(0, config.area_size),
            )
            for i_node in range(config.nb_nodes)
        }

        for reg_aodv in [True, False]:
            random.seed(seed_i)
            np.random.seed(seed_i)
            protocol = ProtocolConfig.from_mode(reg_aodv)
            sim = Simulation(config=config, protocol=protocol, node_positions=positions, trace_file=trace_files[i], traffic_seed=seed_i)
            sim.run()
            (reg_aodv_res if reg_aodv else mod_aodv_res).append(sim.get_metrics())
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

from multiprocessing import Pool, cpu_count


def _one_point(args):
    config, nb_runs, seed_base, trace_files = args
    res = run_comparison_simulations(config=config, nb_runs=nb_runs, seed_base=seed_base, trace_files=trace_files)
    reg_avg = calc_avg_metrics(res["reg"])
    mod_avg = calc_avg_metrics(res["mod"])
    return config.nb_nodes, reg_avg, mod_avg


def densite_parallel(sim_conf: SimConfig, bm_conf: BonnMotionConfig, nb_runs: int, pas: int, factor_min: float = 0.7, factor_max: float = 1.5):
    import numpy as _np

    n_min = (sim_conf.area_size / sim_conf.max_dist) ** 2 * _np.pi
    n_lo = max(2, int(round(factor_min * n_min)))
    n_hi = max(n_lo + 1, int(round(factor_max * n_min)))
    nb_nodes_list = list(range(n_lo, n_hi + 1, pas))

    tasks = []
    for n_nodes in nb_nodes_list:
        sim_conf_n = SimConfig(
            nb_nodes=n_nodes,
            area_size=sim_conf.area_size,
            max_dist=sim_conf.max_dist,
            init_bat=sim_conf.init_bat,
            conso=sim_conf.conso,
            dt=sim_conf.dt,
            ttl=sim_conf.ttl,
            seuil_coeff=sim_conf.seuil_coeff,
            coeff_dist_weight=sim_conf.coeff_dist_weight,
            coeff_bat_weight=sim_conf.coeff_bat_weight,
            coeff_dist_bat=sim_conf.coeff_dist_bat,
            duration=sim_conf.duration,
            window_size=sim_conf.window_size,
        )
        trace_files = generate_bonnmotion_traces(sim_conf_n, bm_conf)
        tasks.append((sim_conf_n, nb_runs, 12345, trace_files))

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(_one_point, tasks)

    results.sort(key=lambda t: t[0])
    return results


def plot_windowed_delivery_over_time(sim_reg, sim_mod, W=None):
    import matplotlib.pyplot as plt

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