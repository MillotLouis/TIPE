from __future__ import annotations

import copy
import random
from dataclasses import dataclass,field
from typing import Dict, Optional, Tuple

import numpy as np
import simpy


@dataclass
class Message:
    """Message réseau : RREQ / RREP / DATA / HELLO / RERR."""
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
    hello_sent: int = 0
    rerr_sent: int = 0
    energy_consumed: float = 0.0
    dead_nodes: int = 0
    seuiled: int = 0
    first_node_death_time: float | None = None
    ten_percent_death_time: float | None = None
    fifty_percent_death_time: float | None = None
    death_times: list[float] = field(default_factory=list)




class Network:
    def __init__(self, config:"SimConfig", reg_aodv: bool, protocol):
        self.cfg = config
        self.protocol = protocol
        self.reg_aodv = reg_aodv  # True si on utilise AODV et false sinon
        self.env = simpy.Environment()
        self.G: Dict[int, "Node"] = {}
        self.stop = False  # passé à True quand on veut que la simulation s'arrête
        self.stats = NetworkStats()

        self.data_log = {}  # (src_id, data_seq) -> {'t_init','t_send','t_recv'}
        self.data_init_times = []
        self.data_send_times = []

        self.last_hello = {}  # (node_id, neighbor_id) -> dernière date HELLO reçue
        self.hello_interval = 1.0
        self.hello_timeout = 3.0
        self.env.process(self._hello_loop())
        self.env.process(self._hello_watchdog())

    def add_node(self, node):
        self.G[node.id] = node

    def get_distance(self, n1, n2) -> float:
        return ((n2.pos[0] - n1.pos[0]) ** 2 + (n2.pos[1] - n1.pos[1]) ** 2) ** 0.5

    def update_battery(self, node, msg_type: str) -> bool:
        # conso est exprimée en pourcentage de la batterie initiale par envoi
        coeff = self.cfg.conso[0] if msg_type[:2] == "RR" else self.cfg.conso[1]
        consommation = node.initial_battery * (coeff / 100.0)
        node.battery = max(0.0, node.battery - consommation)
        self.stats.energy_consumed += consommation
        if node.battery == 0 and node.alive:
            self.env.process(self._kill_node(node))
        return node.battery > 0

    def _kill_node(self, node):
        yield self.env.timeout(0)
        node.alive = False
        self.stats.dead_nodes += 1
        self.stats.death_times.append(self.env.now)

        if self.stats.first_node_death_time is None:
            self.stats.first_node_death_time = self.env.now
        if self.stats.ten_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.1:
            self.stats.ten_percent_death_time = self.env.now
        if self.stats.fifty_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.5:
            self.stats.fifty_percent_death_time = self.env.now
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
        if dist <= node.max_dist and self.update_battery(node, "RREP"):
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(rrep)

    def forward_data(self, node, data):
        next_hop = node.routing_table.get(data.dest_id, (None, 0, 0, 0))[0]
        data.prev_hop = node.id
        if next_hop is None:
            return
        next_node = self.G.get(next_hop)
        if next_node is None or not next_node.alive:
            invalidated = node.invalidate_route_via(next_hop)
            if invalidated:
                self.env.process(self.broadcast_rerr(node, invalidated))
            return
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist and self.update_battery(node, "DATA"):
            self.stats.messages_forwarded += 1
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(data)
        else:
            invalidated = node.invalidate_route_via(next_hop)
            if invalidated:
                self.env.process(self.broadcast_rerr(node, invalidated))

    def mark_neighbor_seen(self, node_id, neighbor_id):
        self.last_hello[(node_id, neighbor_id)] = self.env.now

    def _hello_loop(self):
        while not self.stop:
            for node in self.G.values():
                if not node.alive:
                    continue
                hello = Message(type="HELLO", src_id=node.id, src_seq=node.seq_num, dest_id=-1, prev_hop=node.id)
                for neighbor in self.G.values():
                    if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                        continue
                    neighbor.pending.put(copy.deepcopy(hello))
                    self.stats.hello_sent += 1
            yield self.env.timeout(self.hello_interval)

    def _hello_watchdog(self):
        while not self.stop:
            now = self.env.now
            for node in self.G.values():
                if not node.alive:
                    continue
                broken_neighbors = []
                for (next_hop, _, _, _) in set(node.routing_table.values()):
                    last = self.last_hello.get((node.id, next_hop), 0)
                    if (now - last) > self.hello_timeout:
                        broken_neighbors.append(next_hop)
                for broken in set(broken_neighbors): # Pour pas avoir de doublons
                    invalidated = node.invalidate_route_via(broken)
                    self.env.process(self.broadcast_rerr(node, invalidated))
            yield self.env.timeout(self.hello_interval)

    def broadcast_rerr(self, node, invalidated_destinations):
        if not invalidated_destinations:
            return
        for broken_dest in invalidated_destinations:
            rerr = Message(type="RERR", src_id=node.id, src_seq=node.seq_num, dest_id=broken_dest, prev_hop=node.id)
            for neighbor in self.G.values():
                if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                    continue
                yield self.env.timeout(random.uniform(0.001, 0.003))
                neighbor.pending.put(copy.deepcopy(rerr))
                self.stats.rerr_sent += 1

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