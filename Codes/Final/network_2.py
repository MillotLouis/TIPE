from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

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
    hello_sent: int = 0
    rerr_sent: int = 0

    energy_consumed: float = 0.0
    dead_nodes: int = 0
    seuiled: int = 0

    first_node_death_time: float | None = None
    ten_percent_death_time: float | None = None
    twenty_percent_death_time: float | None = None
    fifty_percent_death_time: float | None = None
    death_times: list[float] = field(default_factory=list)


class Network:
    def __init__(self, config: "SimConfig", reg_aodv: bool):
        self.cfg = config
        self.reg_aodv = reg_aodv

        self.env = simpy.Environment()
        self.G: Dict[int, "Node"] = {}
        self.stop = False
        self.stats = NetworkStats()

        self.data_log = {}
        self.data_init_times = []
        self.data_send_times = []

        self.last_hello = {}
        self.hello_interval = 3.0
        self.hello_timeout = 6.0

        self.env.process(self._hello_loop())
        self.env.process(self._hello_watchdog())

    def add_node(self, node):
        self.G[node.id] = node

    def get_distance(self, n1, n2) -> float:
        return ((n2.pos[0] - n1.pos[0]) ** 2 + (n2.pos[1] - n1.pos[1]) ** 2) ** 0.5

    def update_battery(self, node, msg_type: str, is_emission: bool = True) -> bool:
        """
        Met à jour la batterie pour une émission ou une réception.

        conso = (RX_control, TX_control, ratio_data_vs_control)

        DATA coûte ratio_data_vs_control fois plus cher qu'un message de contrôle.
        """
        control_msgs = {"RREQ", "RREP", "RERR", "HELLO"}

        if is_emission:
            coeff = self.cfg.conso[1] if msg_type in control_msgs else self.cfg.conso[1] * self.cfg.conso[2]
        else:
            coeff = self.cfg.conso[0] if msg_type in control_msgs else self.cfg.conso[0] * self.cfg.conso[2]

        consommation = node.initial_battery * (coeff / 100.0)

        node.battery = max(0.0, node.battery - consommation)
        self.stats.energy_consumed += consommation

        if node.battery == 0.0 and node.alive:
            self.env.process(self._kill_node(node))

        return node.battery > 0.0

    def _kill_node(self, node):
        """
        Enregistre les seuils de mortalité sans arrêter la simulation.

        Point important pour NSGA-II :
        toutes les solutions doivent être évaluées sur la même durée.
        On ne stoppe donc pas à 10 %, 20 % ou 50 %.
        """
        yield self.env.timeout(0)

        if not node.alive:
            return

        node.alive = False
        self.stats.dead_nodes += 1
        self.stats.death_times.append(self.env.now)

        if self.stats.first_node_death_time is None:
            self.stats.first_node_death_time = self.env.now

        if self.stats.ten_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.10:
            self.stats.ten_percent_death_time = self.env.now

        if self.stats.twenty_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.20:
            self.stats.twenty_percent_death_time = self.env.now
            self.stop = True

        if self.stats.fifty_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.50:
            self.stats.fifty_percent_death_time = self.env.now

    def calculate_weight(self, n1, n2, is_final_hop: bool = False) -> float:
        """
        Fonction de poids pour AODV modifié.

        d_min et d_max sont maintenant de vraies bornes :
        - d_norm < d_min : saut trop court, pénalisé ;
        - d_min <= d_norm <= d_max : saut acceptable ;
        - d_norm > d_max : saut proche de la limite radio, pénalisé.

        La batterie du dernier saut vers la destination n'est pas pénalisée,
        car le destinataire final ne relaie pas le paquet DATA ensuite.
        """
        if self.reg_aodv:
            return 1.0

        poids_distance = 0.0
        poids_batterie = 0.0

        d = self.get_distance(n1, n2)
        d_norm = d / max(n1.max_dist, 1e-9)

        if not is_final_hop:
            if d_norm < self.cfg.d_min:
                poids_distance += ((self.cfg.d_min - d_norm) / max(self.cfg.d_min, 1e-9)) ** 2

            if d_norm > self.cfg.d_max:
                poids_distance += ((d_norm - self.cfg.d_max) / max(1.0 - self.cfg.d_max, 1e-9)) ** 2

            bat = max(n2.battery, 0.0)
            bat_norm = 1.0 - (bat / max(n2.initial_battery, 1e-9))
            poids_batterie = bat_norm ** 2

            threshold = n2.initial_battery * self.cfg.seuil_coeff
            if bat < threshold:
                self.stats.seuiled += 1
                poids_batterie += self.cfg.penalite_seuil

        return self.cfg.coeff_dist_weight * poids_distance + self.cfg.coeff_bat_weight * poids_batterie

    def get_energy_stats(self) -> Tuple[float, float]:
        alive_nodes = [node for node in self.G.values() if node.alive]

        if not alive_nodes:
            return 0.0, 0.0

        energies = [node.battery for node in alive_nodes]
        return float(np.mean(energies)), float(np.std(energies))

    def broadcast_rreq(self, node, rreq):
        """
        Broadcast RREQ.

        Correction : on compte maintenant l'énergie TX du broadcast côté émetteur.
        Le coût TX est payé une fois par broadcast, pas une fois par voisin.
        """
        if not node.alive:
            return

        if not self.update_battery(node, "RREQ", is_emission=True):
            return

        for neighbor in self.G.values():
            if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                continue

            yield self.env.timeout(random.uniform(0.002, 0.003))
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

        if dist <= node.max_dist and self.update_battery(node, "RREP", is_emission=True):
            yield self.env.timeout(dist * 0.001 + random.uniform(0.002, 0.003))
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

        if dist <= node.max_dist and self.update_battery(node, "DATA", is_emission=True):
            self.stats.messages_forwarded += 1
            yield self.env.timeout(dist * 0.001 + random.uniform(0.002, 0.003))
            next_node.pending.put(data)
        else:
            invalidated = node.invalidate_route_via(next_hop)
            if invalidated:
                self.env.process(self.broadcast_rerr(node, invalidated))

    def mark_neighbor_seen(self, node_id, neighbor_id):
        self.last_hello[(node_id, neighbor_id)] = self.env.now

    def _hello_loop(self):
        """
        Boucle HELLO.

        Correction : l'émission HELLO consomme maintenant de l'énergie côté émetteur.
        """
        while self.env.now <= self.cfg.duration and not self.stop:
            for node in self.G.values():
                if not node.alive:
                    continue

                if not self.update_battery(node, "HELLO", is_emission=True):
                    continue

                hello = Message(
                    type="HELLO",
                    src_id=node.id,
                    src_seq=node.seq_num,
                    dest_id=-1,
                    prev_hop=node.id,
                )

                for neighbor in self.G.values():
                    if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                        continue

                    neighbor.pending.put(copy.deepcopy(hello))
                    self.stats.hello_sent += 1

            yield self.env.timeout(self.hello_interval)

    def _hello_watchdog(self):
        while self.env.now <= self.cfg.duration and not self.stop:
            now = self.env.now

            for node in self.G.values():
                if not node.alive:
                    continue

                broken_neighbors = []

                for (next_hop, _, _, _) in set(node.routing_table.values()):
                    last = self.last_hello.get((node.id, next_hop), 0.0)

                    if (now - last) > self.hello_timeout:
                        broken_neighbors.append(next_hop)

                for broken in set(broken_neighbors):
                    invalidated = node.invalidate_route_via(broken)

                    if invalidated:
                        self.env.process(self.broadcast_rerr(node, invalidated))

            yield self.env.timeout(self.hello_interval)

    def broadcast_rerr(self, node, invalidated_destinations):
        """
        Broadcast RERR.

        Correction : on compte maintenant l'énergie TX du broadcast côté émetteur.
        """
        if not invalidated_destinations:
            return

        if not node.alive:
            return

        if not self.update_battery(node, "RERR", is_emission=True):
            return

        for broken_dest in invalidated_destinations:
            rerr = Message(
                type="RERR",
                src_id=node.id,
                src_seq=node.seq_num,
                dest_id=broken_dest,
                prev_hop=node.id,
            )

            for neighbor in self.G.values():
                if neighbor.id == node.id or (not neighbor.alive) or self.get_distance(node, neighbor) > node.max_dist:
                    continue

                yield self.env.timeout(random.uniform(0.002, 0.003))
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