from __future__ import annotations

import copy
import random
from dataclasses import dataclass,field
from typing import Dict, Tuple

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
    ttl: int = 0


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


    def add_node(self, node):
        self.G[node.id] = node

    def get_distance(self, n1, n2) -> float:
        return ((n2.pos[0] - n1.pos[0]) ** 2 + (n2.pos[1] - n1.pos[1]) ** 2) ** 0.5

    def update_battery(self, node, msg_type: str, is_emission: bool = True) -> bool:
        """Met à jour la batterie pour une émission ou une réception de message."""
        control_msgs = {"RREQ", "RREP"}
        if is_emission:
            coeff = self.cfg.conso[1] if msg_type in control_msgs else self.cfg.conso[1]*self.cfg.conso[2]
        else:
            coeff = self.cfg.conso[0] if msg_type in control_msgs else self.cfg.conso[0]*self.cfg.conso[2]
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

    # def calculate_weight(self, n1, n2) -> float:
    #     if self.reg_aodv:
    #         return 1.0
    #     bat = max(n2.battery, 0.1)
    #     dist_norm = self.get_distance(n1, n2) / n1.max_dist
    #     bat_norm = 1 - (bat / n1.initial_battery)
    #     weight = (self.cfg.coeff_dist_weight * dist_norm) + (self.cfg.coeff_bat_weight * bat_norm)

    #     threshold = n2.initial_battery * self.cfg.seuil_coeff
    #     if bat < threshold:
    #         self.stats.seuiled += 1
    #         gap = (threshold - bat) / threshold
    #         weight += min(1.0, gap)
    #     return weight

    def calculate_weight(self, n1, n2, is_final_hop: bool = False) -> float:
        """
        Poids d'un lien n1 -> n2 sous la forme :

            poids = a * poids_distance + b * poids_batterie

        avec :
            a = self.cfg.coeff_dist_weight
            b = self.cfg.coeff_bat_weight

        Objectifs :
        - pénaliser les sauts trop courts ;
        - ne pas pénaliser le dernier saut uniquement parce qu'il est court ;
        - pénaliser les sauts proches de la limite d'émission ;
        - pénaliser les relais avec batterie faible.
        """

        if self.reg_aodv:
            return 1.0  

        d = self.get_distance(n1, n2)
        d_norm = d / n1.max_dist

        # Paramètres optimisables
        x_min = 0.15      # saut trop court si d < 30 % de la portée
        x_safe = 0.80     # saut risqué si d > 75 % de la portée

        poids_distance = 0.0

        # Pénalité des sauts trop courts.
        # Elle n'est pas appliquée au dernier saut vers la destination.
        if not is_final_hop and d_norm < x_min:
            poids_distance += ((x_min - d_norm) / x_min) ** 2

        # Pénalité des sauts proches de la limite radio.
        # Elle reste active même pour le dernier saut.
        if d_norm > x_safe:
            poids_distance += ((d_norm - x_safe) / (1.0 - x_safe)) ** 2

        poids_batterie = 0.0

        # La batterie du dernier nœud n'est pas critique pour le relais,
        # puisqu'il ne retransmet pas le paquet DATA ensuite.
        if not is_final_hop:
            bat = max(n2.battery, 0.0001)
            bat_norm = 1.0 - (bat / n2.initial_battery)

            # Pénalité progressive quand la batterie baisse
            poids_batterie = bat_norm ** 2

            # Sur-pénalité si le nœud est sous le seuil critique
            threshold = n2.initial_battery * self.cfg.seuil_coeff
            if bat < threshold:
                self.stats.seuiled += 1
                # gap = (threshold - bat) / max(threshold, eps)
                # poids_batterie += gap ** 2
                poids_batterie += 2

        return self.cfg.coeff_dist_weight * poids_distance + self.cfg.coeff_bat_weight * poids_batterie

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
        route = node.routing_table.get(rrep.dest_id)
        if route is None or route[3] <= self.env.now:
            node.routing_table.pop(rrep.dest_id, None)
            return
        next_hop = route[0]
        next_node = self.G.get(next_hop)
        if next_node is None or not next_node.alive:
            return
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist and self.update_battery(node, "RREP", is_emission=True):
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(rrep)

    def forward_data(self, node, data):
        route = node.routing_table.get(data.dest_id)
        data.prev_hop = node.id
        if route is None or route[3] <= self.env.now:
            node.routing_table.pop(data.dest_id, None)
            return
        next_hop = route[0]
        next_node = self.G.get(next_hop)
        if next_node is None or not next_node.alive:
            node.routing_table.pop(data.dest_id, None)
            return
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist and self.update_battery(node, "DATA", is_emission=True):
            self.stats.messages_forwarded += 1
            yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
            next_node.pending.put(data)
        else:
            node.routing_table.pop(data.dest_id, None)

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