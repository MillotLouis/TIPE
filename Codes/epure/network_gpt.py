from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import simpy


@dataclass
class Message:
    """Message réseau : RREQ / RREP / DATA."""
    type: str
    src_id: int
    src_seq: int
    dest_id: int
    dest_seq: int = -1
    weight: float = 0.0
    prev_hop: int = -1


@dataclass
class NetworkStats:
    """Compteurs de métriques réseau."""
    messages_forwarded: int = 0
    messages_initiated: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    rreq_sent: int = 0
    rreq_forwarded: int = 0
    rrep_sent: int = 0
    energy_consumed: float = 0.0
    dead_nodes: int = 0
    seuiled: int = 0


class Network:
    def __init__(self, config, reg_aodv: bool, protocol):
        self.cfg = config
        self.protocol = protocol
        self.reg_aodv = reg_aodv  # True si on utilise AODV et false sinon
        self.env = simpy.Environment()
        self.G: Dict[int, "Node"] = {}
        self.stop = False  # passé à True quand on veut que la simulation s'arrête
        self.stats = NetworkStats()

        self.first_node_death_time: Optional[float] = None
        self.ten_percent_death_time: Optional[float] = None
        self.fifty_percent_death_time: Optional[float] = None
        self.death_times = []  # Liste des dates auxquelles des noeuds sont morts

        self.data_log = {}  # (src_id, data_seq) -> {'t_init','t_send','t_recv'}
        self.data_init_times = []
        self.data_send_times = []

    def add_node(self, node):
        self.G[node.id] = node

    def get_distance(self, n1, n2) -> float:
        return ((n2.pos[0] - n1.pos[0]) ** 2 + (n2.pos[1] - n1.pos[1]) ** 2) ** 0.5

    def update_battery(self, node, msg_type: str, dist: float) -> bool:
        base = node.initial_battery * (self.cfg.conso[0] if msg_type[:2] == "RR" else self.cfg.conso[1])
        energy_cost = self.cfg.coeff_dist_bat * dist + base
        node.battery = max(0.0, node.battery - energy_cost)
        self.stats.energy_consumed += energy_cost
        if node.battery == 0 and node.alive:
            self.env.process(self._kill_node(node))
        return node.battery > 0

    def _kill_node(self, node):
        yield self.env.timeout(0)
        node.alive = False
        self.stats.dead_nodes += 1
        self.death_times.append(self.env.now)

        if self.first_node_death_time is None:
            self.first_node_death_time = self.env.now
        if self.ten_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.1:
            self.ten_percent_death_time = self.env.now
        if self.fifty_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.5:
            self.fifty_percent_death_time = self.env.now
            self.stop = True

    def calculate_weight(self, n1, n2) -> float:
        if self.reg_aodv:
            return 1.0
        bat = max(n2.battery, 0.1)
        dist_norm = self.get_distance(n1, n2) / n1.max_dist
        bat_norm = 1 - (bat / n1.initial_battery)
        weight = (self.cfg.coeff_dist_weight * dist_norm) + (self.cfg.coeff_bat_weight * bat_norm)

        threshold = n2.initial_battery * self.cfg.seuil_coeff
        if bat < threshold:
            self.stats.seuiled += 1
            gap = (threshold - bat) / threshold
            weight += min(1.0, gap)
        return weight

    def get_energy_stats(self) -> Tuple[float, float]:
        alive_nodes = [node for node in self.G.values() if node.alive]
        if not alive_nodes:
            return 0.0, 0.0
        energies = [node.battery for node in alive_nodes]
        return float(np.mean(energies)), float(np.std(energies))

    def broadcast_rreq(self, node, rreq):
        for neighbor in self.G.values():
            if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                continue
            yield self.env.timeout(random.uniform(0.01, 0.05))  # jitter aléatoire avant transmission
            neighbor.pending.put(copy.deepcopy(rreq))
            self.stats.rreq_forwarded += 1

    def unicast_rrep(self, node, rrep):
        next_hop = node.routing_table.get(rrep.dest_id, (None, 0, 0, 0))[0]
        if next_hop is None:
            return
        next_node = self.G.get(next_hop)
        if next_node is None or not next_node.alive:
            return
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist and self.update_battery(node, "RREP", dist if not self.reg_aodv else node.max_dist):
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(rrep)

    def forward_data(self, node, data):
        next_hop = node.routing_table.get(data.dest_id, (None, 0, 0, 0))[0]
        data.prev_hop = node.id
        if next_hop is None:
            return
        next_node = self.G.get(next_hop)
        if next_node is None or not next_node.alive:
            return
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist and self.update_battery(node, "DATA", dist if not self.reg_aodv else node.max_dist):
            self.stats.messages_forwarded += 1
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(data)

    def log_data_init(self, src_id, data_seq, t_init):
        key = (src_id, data_seq)
        self.data_log[key] = {"t_init": t_init, "t_send": None, "t_recv": None}
        self.data_init_times.append((t_init, key))

    def log_data_send(self, src_id, data_seq, t_send):
        key = (src_id, data_seq)
        entry = self.data_log.get(key)
        if entry and entry["t_send"] is None:
            entry["t_send"] = t_send
            self.data_send_times.append((t_send, key))

    def log_data_recv(self, src_id, data_seq, t_recv):
        key = (src_id, data_seq)
        entry = self.data_log.get(key)
        if entry and entry["t_recv"] is None:
            entry["t_recv"] = t_recv