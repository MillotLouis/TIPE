from __future__ import annotations

import random
from dataclasses import dataclass, replace
from time import time
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

from network_2 import Network
from node_2 import Node


@dataclass(frozen=True)
class SimConfig:
    """Paramètres globaux de simulation."""
    nb_nodes: int
    area_size: int
    max_dist: float
    init_bat: float
    conso: Tuple[float, float, int]
    dt: float
    ttl_max: int
    seuil_coeff: float
    coeff_dist_weight: float
    coeff_bat_weight: float
    duration: float

    d_min: float
    d_max: float
    penalite_seuil: float
    max_duplicates: int
    weight_seuil: float

    window_size: float = 100.0


@dataclass(frozen=True)
class BonnMotionConfig:
    bm_exe: str
    out_dir: str
    scenario: str = "RandomWaypoint"
    vmin: int = 0
    vmax: int = 1
    pause: int = 50
    o: int = 2


class Simulation:
    def __init__(
        self,
        config: SimConfig,
        reg_aodv: bool,
        node_positions: Dict[int, Tuple[float, float]],
        trace_file: str,
        traffic_seed: int,
    ):
        self.cfg = config
        self.reg_aodv = reg_aodv
        self.traffic_seed = traffic_seed
        self.time_points = []

        self.net = Network(config=config, reg_aodv=reg_aodv)

        if node_positions is None:
            raise ValueError("node_positions ne peut pas être None")

        for node_id in range(self.cfg.nb_nodes):
            node = Node(
                node_id=node_id,
                pos=node_positions[node_id],
                initial_battery=self.cfg.init_bat,
                max_dist=self.cfg.max_dist,
                network=self.net,
            )
            self.net.add_node(node)

        self.net.env.process(self._bm_replay(trace_file))

    def _random_communication(self):
        rng = random.Random(self.traffic_seed)

        while self.net.env.now <= self.cfg.duration and not self.net.stop:
            src_id = rng.randint(0, self.cfg.nb_nodes - 1)
            dest_id = rng.randint(0, self.cfg.nb_nodes - 1)

            while dest_id == src_id:
                dest_id = rng.randint(0, self.cfg.nb_nodes - 1)

            src_node = self.net.G[src_id]

            if src_node.alive:
                src_node.send_data(dest_id)

            yield self.net.env.timeout(self.cfg.dt)

    def _monitor(self):
        while self.net.env.now <= self.cfg.duration and not self.net.stop:
            self.time_points.append(self.net.env.now)
            yield self.net.env.timeout(2 * self.cfg.dt)

    def _bm_replay(self, file_path):
        traces = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for nid, line in enumerate(f):
                vals = [float(v) for v in line.split()]

                if vals and nid < self.cfg.nb_nodes:
                    traces[nid] = list(zip(vals[0::3], vals[1::3], vals[2::3]))

        indexes = {nid: 0 for nid in traces}

        for nid, seq in traces.items():
            if seq:
                self.net.G[nid].pos = (seq[0][1], seq[0][2])

        while self.net.env.now <= self.cfg.duration and not self.net.stop:
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
                        node.pos = (x0 + u * (x1 - x0), y0 + u * (y1 - y0))
                elif seq:
                    node.pos = (seq[-1][1], seq[-1][2])

            yield self.net.env.timeout(self.cfg.dt)

    def run(self):
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())

        while self.net.env.now <= self.cfg.duration and not self.net.stop:
            self.net.env.step()

    def get_metrics(self):
        final_avg_bat, final_std_bat = self.net.get_energy_stats()
        s = self.net.stats

        return {
            "dead_nodes": s.dead_nodes,
            "energy": s.energy_consumed,

            "msg_recv": s.messages_received,
            "msg_sent": s.messages_sent,
            "messages_forwarded": s.messages_forwarded,
            "messages_initiated": s.messages_initiated,

            "rreq_sent": s.rreq_sent,
            "rreq_forwarded": s.rreq_forwarded,
            "rrep_sent": s.rrep_sent,
            "hello_sent": s.hello_sent,
            "rerr_sent": s.rerr_sent,

            "duration": self.net.env.now,
            "seuiled": s.seuiled,

            "first_node_death": s.first_node_death_time,
            "ten_percent_death": s.ten_percent_death_time,
            "twenty_percent_death": s.twenty_percent_death_time,
            "fifty_percent_death": s.fifty_percent_death_time,

            "final_avg_bat": final_avg_bat,
            "final_std_bat": final_std_bat,

            "death_times": list(s.death_times),
        }


def generate_bonnmotion_traces(sim_conf: SimConfig, bm_conf: BonnMotionConfig, nb_runs: int):
    import gzip
    import os
    import shutil
    import subprocess

    os.makedirs(bm_conf.out_dir, exist_ok=True)

    movements_files = []

    for n_simu in range(nb_runs):
        base = os.path.join(bm_conf.out_dir, f"{sim_conf.nb_nodes}rw_2_{n_simu}")

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

        print(cmd, flush=True)
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
    print(f"Simulations à {config.nb_nodes} nœuds débutées", flush=True)

    reg_aodv_res = []
    mod_aodv_res = []

    for i in range(nb_runs):
        print(f"run {i + 1} débuté", flush=True)

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

            sim = Simulation(
                config=config,
                reg_aodv=reg_aodv,
                node_positions=positions,
                trace_file=trace_files[i],
                traffic_seed=seed_i,
            )

            sim.run()

            if reg_aodv:
                reg_aodv_res.append(sim.get_metrics())
            else:
                mod_aodv_res.append(sim.get_metrics())

    print(f"Simulations à {config.nb_nodes} nœuds terminées\n", flush=True)

    reg_avg = calc_avg_metrics(reg_aodv_res)
    mod_avg = calc_avg_metrics(mod_aodv_res)

    return {
        "reg": reg_aodv_res,
        "mod": mod_aodv_res,
        "reg_avg": [reg_avg],
        "mod_avg": [mod_avg],
    }


def calc_avg_metrics(res):
    if not res:
        return {}

    avg = {}

    for key in res[0].keys():
        if key == "death_times":
            avg[key] = [r[key] for r in res if r.get(key) is not None]
            avg[f"{key}_count"] = len(avg[key])
            continue

        values = [r[key] for r in res if r[key] is not None]
        avg[key] = (sum(values) / len(values)) if values else None
        avg[f"{key}_count"] = len(values)

    return avg


def _one_point(args):
    config, nb_runs, seed_base, trace_files = args

    res = run_comparison_simulations(
        config=config,
        nb_runs=nb_runs,
        seed_base=seed_base,
        trace_files=trace_files,
    )

    return config.nb_nodes, res["reg_avg"][0], res["mod_avg"][0]


def densite_parallel(
    sim_conf: SimConfig,
    bm_conf: BonnMotionConfig,
    nb_runs: int,
    pas: int,
    deg_min: float = 0.7,
    deg_max: float = 1.5,
):
    n_crit_moins_1 = (sim_conf.area_size / sim_conf.max_dist) ** 2 / np.pi
    n_lo = max(2, int(round(deg_min * n_crit_moins_1)) + 1)
    n_hi = max(n_lo + 1, int(round(deg_max * n_crit_moins_1)) + 1)

    nb_nodes_list = list(range(n_lo, n_hi + 1, pas))

    tasks = []

    for n_nodes in nb_nodes_list:
        sim_conf_n = replace(sim_conf, nb_nodes=n_nodes)
        trace_files = generate_bonnmotion_traces(sim_conf_n, bm_conf, nb_runs)
        tasks.append((sim_conf_n, nb_runs, int(time()), trace_files))

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(_one_point, tasks)

    results.sort(key=lambda t: t[0])

    nb_nodes_array = []
    reg_first_death, mod_first_death = [], []
    reg_ten_percent_death, mod_ten_percent_death = [], []
    reg_twenty_percent_death, mod_twenty_percent_death = [], []
    reg_fifty_percent_death, mod_fifty_percent_death = [], []
    reg_energy, mod_energy = [], []
    reg_std, mod_std = [], []
    reg_final_energy, mod_final_energy = [], []
    reg_dr, mod_dr = [], []

    for (N, reg_avg, mod_avg) in results:
        nb_nodes_array.append(N)

        reg_first_death.append(reg_avg.get("first_node_death", None))
        mod_first_death.append(mod_avg.get("first_node_death", None))

        reg_ten_percent_death.append(reg_avg.get("ten_percent_death", None))
        mod_ten_percent_death.append(mod_avg.get("ten_percent_death", None))

        reg_twenty_percent_death.append(reg_avg.get("twenty_percent_death", None))
        mod_twenty_percent_death.append(mod_avg.get("twenty_percent_death", None))

        reg_fifty_percent_death.append(reg_avg.get("fifty_percent_death", None))
        mod_fifty_percent_death.append(mod_avg.get("fifty_percent_death", None))

        reg_energy.append(reg_avg.get("energy", None))
        mod_energy.append(mod_avg.get("energy", None))

        reg_std.append(reg_avg.get("final_std_bat", None))
        mod_std.append(mod_avg.get("final_std_bat", None))

        reg_final_energy.append(reg_avg.get("final_avg_bat", None))
        mod_final_energy.append(mod_avg.get("final_avg_bat", None))

        reg_dr.append(reg_avg.get("msg_recv", 0) / max(1, reg_avg.get("messages_initiated", 1)) * 100)
        mod_dr.append(mod_avg.get("msg_recv", 0) / max(1, mod_avg.get("messages_initiated", 1)) * 100)

    metrics_to_plot = [
        ("Temps first node death", reg_first_death, mod_first_death),
        ("Temps 10% death", reg_ten_percent_death, mod_ten_percent_death),
        ("Temps 20% death", reg_twenty_percent_death, mod_twenty_percent_death),
        ("Temps 50% death", reg_fifty_percent_death, mod_fifty_percent_death),
        ("Énergie résiduelle moyenne", reg_final_energy, mod_final_energy),
        ("Énergie totale consommée", reg_energy, mod_energy),
        ("Écart-type énergie finale", reg_std, mod_std),
        ("Delivery ratio (%)", reg_dr, mod_dr),
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(metrics_to_plot) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, (ylabel, reg_vals, mod_vals) in zip(axes_flat, metrics_to_plot):
        ax.plot(nb_nodes_array, reg_vals, marker="o", label="Regular")
        ax.plot(nb_nodes_array, mod_vals, marker="s", label="Modified")
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