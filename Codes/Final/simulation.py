import random
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count
from time import time
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from network import Network
from node import Node


@dataclass(frozen=True)
class SimConfig:
    """Parametres communs a toutes les simulations."""

    nb_nodes: int
    area_size: int
    max_dist: float
    init_bat: float
    conso: Tuple[float, float, int]
    dt: float
    seuil_coeff: float
    coeff_dist_weight: float
    coeff_bat_weight: float
    duration: float
    d_min: float
    d_max: float
    penalite_seuil: float
    max_duplicates: int
    weight_seuil: float


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
        self.net = Network(config=config, reg_aodv=reg_aodv)

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
            src_id, dest_id = self._pick_distinct_nodes(rng)
            src_node = self.net.G[src_id]

            if src_node.alive:
                src_node.send_data(dest_id)

            yield self.net.env.timeout(self.cfg.dt)

    def _pick_distinct_nodes(self, rng):
        src_id = rng.randint(0, self.cfg.nb_nodes - 1)
        dest_id = rng.randint(0, self.cfg.nb_nodes - 1)
        while dest_id == src_id:
            dest_id = rng.randint(0, self.cfg.nb_nodes - 1)
        return src_id, dest_id

    def _bm_replay(self, file_path):
        traces = self._load_bonnmotion_trace(file_path)
        indexes = {node_id: 0 for node_id in traces}

        for node_id, positions in traces.items():
            if positions:
                self.net.G[node_id].pos = (positions[0][1], positions[0][2])

        while self.net.env.now <= self.cfg.duration and not self.net.stop:
            self._update_positions(traces, indexes)
            yield self.net.env.timeout(self.cfg.dt)

    def _load_bonnmotion_trace(self, file_path):
        traces = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for node_id, line in enumerate(file):
                values = [float(value) for value in line.split()]
                if values and node_id < self.cfg.nb_nodes:
                    traces[node_id] = list(zip(values[0::3], values[1::3], values[2::3]))
        return traces

    def _update_positions(self, traces, indexes):
        sim_time = self.net.env.now

        for node_id, positions in traces.items():
            node = self.net.G.get(node_id)
            if node is None or not node.alive:
                continue

            index = self._current_trace_index(positions, indexes[node_id], sim_time)
            indexes[node_id] = index
            node.pos = self._interpolate_position(positions, index, sim_time)

    @staticmethod
    def _current_trace_index(positions, index, sim_time):
        while index + 1 < len(positions) and sim_time >= positions[index + 1][0]:
            index += 1
        return index

    @staticmethod
    def _interpolate_position(positions, index, sim_time):
        if index + 1 >= len(positions):
            return positions[-1][1], positions[-1][2]

        t0, x0, y0 = positions[index]
        t1, x1, y1 = positions[index + 1]
        if t1 <= t0 or sim_time < t0:
            return x0, y0

        ratio = (sim_time - t0) / (t1 - t0)
        return x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0)

    def run(self):
        self.net.env.process(self._random_communication())
        while self.net.env.now <= self.cfg.duration and not self.net.stop:
            self.net.env.step()

    def get_metrics(self):
        final_avg_bat, final_std_bat = self.net.get_energy_stats()
        stats = self.net.stats

        return {
            "dead_nodes": stats.dead_nodes,
            "energy": stats.energy_consumed,
            "msg_recv": stats.messages_received,
            "msg_sent": stats.messages_sent,
            "rreq_sent": stats.rreq_sent,
            "duration": self.net.env.now,
            "rrep_sent": stats.rrep_sent,
            "messages_forwarded": stats.messages_forwarded,
            "messages_initiated": stats.messages_initiated,
            "rreq_forwarded": stats.rreq_forwarded,
            "seuiled": stats.seuiled,
            "first_node_death": stats.first_node_death_time,
            "ten_percent_death": stats.ten_percent_death_time,
            "final_avg_bat": final_avg_bat,
            "final_std_bat": final_std_bat,
            "twenty_percent_death": stats.twenty_percent_death_time,
        }


def generate_bonnmotion_traces(sim_conf: SimConfig, bm_conf: BonnMotionConfig, nb_runs: int):
    import gzip
    import os
    import shutil
    import subprocess

    os.makedirs(bm_conf.out_dir, exist_ok=True)
    movement_files = []

    for run_id in range(nb_runs):
        base_path = os.path.join(bm_conf.out_dir, f"{sim_conf.nb_nodes}rw{run_id}")
        command = " ".join(
            [
                bm_conf.bm_exe,
                "-f",
                base_path,
                bm_conf.scenario,
                "-n",
                str(sim_conf.nb_nodes),
                "-d",
                str(sim_conf.duration),
                "-x",
                str(sim_conf.area_size),
                "-y",
                str(sim_conf.area_size),
                "-l",
                str(bm_conf.vmin),
                "-h",
                str(bm_conf.vmax),
                "-p",
                str(bm_conf.pause),
                "-o",
                str(bm_conf.o),
            ]
        )
        print(command, flush=True)
        subprocess.run(command, check=True)

        movement_path = base_path + ".movements"
        gz_path = movement_path + ".gz"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rb") as source, open(movement_path, "wb") as target:
                shutil.copyfileobj(source, target)
            os.remove(gz_path)

        params_path = base_path + ".params"
        if os.path.exists(params_path):
            os.remove(params_path)

        movement_files.append(movement_path)

    return movement_files


def _run_one_sim(config, reg_aodv, trace_file, seed_i):
    random.seed(seed_i)
    positions = {
        node_id: (
            random.uniform(0, config.area_size),
            random.uniform(0, config.area_size),
        )
        for node_id in range(config.nb_nodes)
    }

    sim = Simulation(
        config=config,
        reg_aodv=reg_aodv,
        node_positions=positions,
        trace_file=trace_file,
        traffic_seed=seed_i,
    )
    sim.run()
    return sim.get_metrics()


def run_comparison_simulations(
    config: SimConfig,
    nb_runs: int,
    seed_base: int,
    trace_files: List[str],
):
    print(f"Simulations a {config.nb_nodes} noeuds debutees", flush=True)

    modified_results = []
    regular_results = []

    for run_id in range(nb_runs):
        print(f"run {run_id} commence", flush=True)
        seed_i = seed_base + run_id
        modified_results.append(_run_one_sim(config, False, trace_files[run_id], seed_i))
        regular_results.append(_run_one_sim(config, True, trace_files[run_id], seed_i))

    print(f"Simulations a {config.nb_nodes} noeuds terminees\n", flush=True)
    return {
        "mod": modified_results,
        "reg": regular_results,
        "mod_avg": [calc_avg_metrics(modified_results)],
    }


def calc_avg_metrics(results):
    if not results:
        return {}

    average = {}
    for key in results[0].keys():
        values = [result[key] for result in results if result[key] is not None]
        average[key] = sum(values) / len(values) if values else None
        average[f"{key}_count"] = len(values)
    return average


def _one_point(args):
    config, nb_runs, seed_base, trace_files = args
    results = run_comparison_simulations(
        config=config,
        nb_runs=nb_runs,
        seed_base=seed_base,
        trace_files=trace_files,
    )
    return (
        config.nb_nodes,
        calc_avg_metrics(results["reg"]),
        calc_avg_metrics(results["mod"]),
    )


def densite_parallel(
    sim_conf: SimConfig,
    bm_conf: BonnMotionConfig,
    nb_runs: int,
    pas: int,
    deg_min: float = 0.7,
    deg_max: float = 1.5,
):
    nb_nodes_list = [node_count for node_count in range(20, 70, 10)]
    print(nb_nodes_list, flush=True)

    tasks = []
    for node_count in nb_nodes_list:
        config = replace(sim_conf, nb_nodes=node_count)
        trace_files = generate_bonnmotion_traces(config, bm_conf, nb_runs)
        tasks.append((config, nb_runs, int(time()), trace_files))

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(_one_point, tasks)

    results.sort(key=lambda item: item[0])
    _plot_density_results(results)
    return results


def _plot_density_results(results):
    nb_nodes = []
    reg_first_death, mod_first_death = [], []
    reg_delivery, mod_delivery = [], []
    reg_ten_percent_death, mod_ten_percent_death = [], []
    reg_energy, mod_energy = [], []
    reg_std, mod_std = [], []
    reg_final_energy, mod_final_energy = [], []
    reg_twenty_percent_death, mod_twenty_percent_death = [], []

    for node_count, reg_avg, mod_avg in results:
        nb_nodes.append(node_count)

        reg_first_death.append(reg_avg.get("first_node_death"))
        mod_first_death.append(mod_avg.get("first_node_death"))

        reg_delivery.append(_delivery_ratio(reg_avg))
        mod_delivery.append(_delivery_ratio(mod_avg))

        reg_ten_percent_death.append(reg_avg.get("ten_percent_death"))
        mod_ten_percent_death.append(mod_avg.get("ten_percent_death"))

        reg_energy.append(reg_avg.get("energy"))
        mod_energy.append(mod_avg.get("energy"))

        reg_std.append(reg_avg.get("final_std_bat"))
        mod_std.append(mod_avg.get("final_std_bat"))

        reg_final_energy.append(reg_avg.get("final_avg_bat"))
        mod_final_energy.append(mod_avg.get("final_avg_bat"))

        reg_twenty_percent_death.append(reg_avg.get("twenty_percent_death"))
        mod_twenty_percent_death.append(mod_avg.get("twenty_percent_death"))

    metrics_to_plot = [
        ("Temps (first node death)", reg_first_death, mod_first_death),
        ("Temps (10% death)", reg_ten_percent_death, mod_ten_percent_death),
        ("Temps (20% death)", reg_twenty_percent_death, mod_twenty_percent_death),
        ("Energie residuelle moyenne", reg_final_energy, mod_final_energy),
        ("Energie totale consommee", reg_energy, mod_energy),
        ("Ecart type energie finale", reg_std, mod_std),
        ("Delivery ratio (%)", reg_delivery, mod_delivery),
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(metrics_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, (label, reg_values, mod_values) in zip(axes_flat, metrics_to_plot):
        axis.plot(nb_nodes, reg_values, marker="o", label="Regular")
        axis.plot(nb_nodes, mod_values, marker="s", label="Modified")
        axis.set_xlabel("nb_nodes")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes_flat[len(metrics_to_plot):]:
        axis.axis("off")

    fig.suptitle("Comparaison AODV regulier vs modifie", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


def _delivery_ratio(metrics):
    return metrics.get("msg_recv", 0) / max(1, metrics.get("messages_initiated", 1)) * 100


if __name__ == "__main__":
    sim_conf = SimConfig(
        nb_nodes=40,
        area_size=800,
        max_dist=250,
        init_bat=100,
        conso=(0.00164, 0.0082, 10),
        dt=0.25,
        seuil_coeff=0.03762018,
        coeff_dist_weight=0.23439377,
        coeff_bat_weight=0.4,
        duration=20000,
        d_min=0.12819995,
        d_max=0.94360313,
        penalite_seuil=2.29914294,
        max_duplicates=3,
        weight_seuil=1.41488483,
    )

    bm_conf = BonnMotionConfig(
        bm_exe="C:/Users/millo/Documents/bonnmotion-3.0.1/bin/bm.bat",
        out_dir="C:/Users/millo/Documents/GitHub/TIPE/bm_files/",
        vmin=5,
        vmax=5,
        pause=5,
    )

    results = densite_parallel(sim_conf, bm_conf, nb_runs=5, pas=2, deg_min=15, deg_max=15)
    print(results)
