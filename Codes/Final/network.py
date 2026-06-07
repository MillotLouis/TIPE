import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import simpy


@dataclass
class Message:
    """Message echange dans le reseau."""

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
    """Compteurs utiles pour comparer AODV classique et modifie."""

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
    death_times: list[float] = field(default_factory=list)


class Network:
    HELLO_INTERVAL = 3.0
    HELLO_TIMEOUT = 6.0
    TRANSMISSION_DELAY = (0.002, 0.003)
    CONTROL_MESSAGES = {"RREQ", "RREP", "RERR", "HELLO"}

    def __init__(self, config: "SimConfig", reg_aodv: bool):
        self.cfg = config
        self.reg_aodv = reg_aodv
        self.env = simpy.Environment()
        self.G: Dict[int, "Node"] = {}
        self.stop = False
        self.stats = NetworkStats()

        self.last_hello = {}
        self.env.process(self._hello_loop())
        self.env.process(self._hello_watchdog())

    def add_node(self, node):
        self.G[node.id] = node

    def get_distance(self, n1, n2) -> float:
        dx = n2.pos[0] - n1.pos[0]
        dy = n2.pos[1] - n1.pos[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def update_battery(self, node, msg_type: str, is_emission: bool = True) -> bool:
        base_cost = self.cfg.conso[1] if is_emission else self.cfg.conso[0]
        multiplier = 1 if msg_type in self.CONTROL_MESSAGES else self.cfg.conso[2]
        consumption = base_cost * multiplier

        node.battery = max(0.0, node.battery - consumption)
        self.stats.energy_consumed += consumption

        if node.battery == 0 and node.alive:
            self.env.process(self._kill_node(node))

        return node.battery > 0

    def _kill_node(self, node):
        yield self.env.timeout(0)
        if not node.alive:
            return

        node.alive = False
        self.stats.dead_nodes += 1
        self.stats.death_times.append(self.env.now)

        if self.stats.first_node_death_time is None:
            self.stats.first_node_death_time = self.env.now
        if self.stats.ten_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.1:
            self.stats.ten_percent_death_time = self.env.now
        if self.stats.twenty_percent_death_time is None and self.stats.dead_nodes >= self.cfg.nb_nodes * 0.5:
            self.stats.twenty_percent_death_time = self.env.now
            self.stop = True

    def calculate_weight(self, n1, n2, is_final_hop: bool = False) -> float:
        if self.reg_aodv:
            return 1.0
        if is_final_hop:
            return 0.0

        distance_weight = self._distance_weight(n1, n2)
        battery_weight = self._battery_weight(n2)

        return (
            self.cfg.coeff_dist_weight * distance_weight
            + self.cfg.coeff_bat_weight * battery_weight
        )

    def _distance_weight(self, n1, n2) -> float:
        normalized_distance = self.get_distance(n1, n2) / n1.max_dist
        target_middle = (self.cfg.d_min + self.cfg.d_max) / 2
        return ((normalized_distance - target_middle) / target_middle) ** 2

    def _battery_weight(self, node) -> float:
        battery_used = 1.0 - (node.battery / node.initial_battery)
        weight = battery_used ** 2

        if node.battery < node.initial_battery * self.cfg.seuil_coeff:
            self.stats.seuiled += 1
            weight += self.cfg.penalite_seuil

        return weight

    def get_energy_stats(self) -> Tuple[float, float]:
        energies = [node.battery for node in self.G.values()]
        return float(np.mean(energies)), float(np.std(energies))

    def broadcast_rreq(self, node, rreq):
        for neighbor in self._reachable_neighbors(node):
            yield self.env.timeout(self._random_delay())
            neighbor.pending.put(copy.deepcopy(rreq))
            self.stats.rreq_forwarded += 1

    def unicast_rrep(self, node, rrep):
        next_node = self._next_node(node, rrep.dest_id)
        if next_node is None:
            return

        if self.update_battery(node, "RREP", is_emission=True):
            yield self.env.timeout(self._random_delay())
            next_node.pending.put(rrep)

    def forward_data(self, node, data):
        next_node = self._next_node(node, data.dest_id)
        data.prev_hop = node.id
        if next_node is None:
            return

        if self.update_battery(node, "DATA", is_emission=True):
            self.stats.messages_forwarded += 1
            yield self.env.timeout(self._random_delay())
            next_node.pending.put(data)

    def _next_node(self, node, dest_id):
        next_hop = node.routing_table.get(dest_id, (None, 0, 0))[0]
        next_node = self.G.get(next_hop)

        if next_node is None or not next_node.alive:
            return None
        if self.get_distance(node, next_node) > node.max_dist:
            return None

        return next_node

    def _reachable_neighbors(self, node):
        for neighbor in self.G.values():
            if neighbor.id == node.id or not neighbor.alive:
                continue
            if self.get_distance(node, neighbor) <= node.max_dist:
                yield neighbor

    def _random_delay(self):
        return random.uniform(*self.TRANSMISSION_DELAY)

    def mark_neighbor_seen(self, node_id, neighbor_id):
        self.last_hello[(node_id, neighbor_id)] = self.env.now

    def _hello_loop(self):
        while self.env.now <= self.cfg.duration and not self.stop:
            for node in self.G.values():
                if not node.alive:
                    continue

                hello = Message(
                    type="HELLO",
                    src_id=node.id,
                    src_seq=node.seq_num,
                    dest_id=-1,
                    prev_hop=node.id,
                )
                for neighbor in self._reachable_neighbors(node):
                    neighbor.pending.put(copy.deepcopy(hello))
                    self.stats.hello_sent += 1

            yield self.env.timeout(self.HELLO_INTERVAL)

    def _hello_watchdog(self):
        while self.env.now <= self.cfg.duration and not self.stop:
            now = self.env.now
            for node in self.G.values():
                if not node.alive:
                    continue

                broken_neighbors = self._expired_neighbors(node, now)
                for broken_neighbor in broken_neighbors:
                    invalidated = node.invalidate_route_via(broken_neighbor)
                    self.env.process(self.broadcast_rerr(node, invalidated))

            yield self.env.timeout(self.HELLO_INTERVAL)

    def _expired_neighbors(self, node, now):
        if now <= self.HELLO_INTERVAL:
            return set()

        expired = set()
        for next_hop, _, _ in set(node.routing_table.values()):
            last_seen = self.last_hello.get((node.id, next_hop), 0)
            if now - last_seen > self.HELLO_TIMEOUT:
                expired.add(next_hop)
        return expired

    def broadcast_rerr(self, node, invalidated_destinations):
        if not invalidated_destinations:
            return

        for broken_dest in invalidated_destinations:
            rerr = Message(
                type="RERR",
                src_id=node.id,
                src_seq=node.seq_num,
                dest_id=broken_dest,
                prev_hop=node.id,
            )
            for neighbor in self._reachable_neighbors(node):
                yield self.env.timeout(self._random_delay())
                neighbor.pending.put(copy.deepcopy(rerr))
                self.stats.rerr_sent += 1
