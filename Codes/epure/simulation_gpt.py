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
