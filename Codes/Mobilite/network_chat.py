import random
import simpy
import copy
import numpy as np


class Message:
    def __init__(self, typ, src_id, src_seq, dest_id, dest_seq=-1, weight=0.0, prev_hop=None):
        self.typ = typ
        self.src_id = src_id
        self.src_seq = src_seq
        self.dest_id = dest_id
        self.dest_seq = dest_seq
        self.weight = weight
        self.prev_hop = src_id if prev_hop is None else prev_hop


class NetworkConfig:
    """
    Un seul endroit pour TOUS les paramètres.
    Tu peux instancier cette config dans ton main et la passer à Network(cfg).
    """
    def __init__(
        self,
        nb_nodes,
        max_dist,
        ttl,

        # énergie
        conso_rreq_rrep=0.0001,         # coût "fixe" (relatif à battery initiale)
        conso_data=0.001,              # coût "fixe" (relatif à battery initiale)
        cost_per_distance=0.005,       # coût proportionnel à la distance
        
        # poids de route (mode modifié)
        dist_weight=0.6,
        bat_weight=0.2,
        battery_threshold_ratio=0.75,  # seuil = ratio * batterie_initiale
        penalty_cap=1.0,
        
        # contrôle des duplicats (mode modifié)
        max_duplicates=3,
        improve_factor=1.5,
        collect_timeout=0.2,
        
        # protocole
        reg_aodv=True
    ):
        self.nb_nodes = nb_nodes
        self.max_dist = max_dist
        self.ttl = ttl

        self.conso_rreq_rrep = conso_rreq_rrep
        self.conso_data = conso_data
        self.cost_per_distance = cost_per_distance

        self.dist_weight = dist_weight
        self.bat_weight = bat_weight
        self.battery_threshold_ratio = battery_threshold_ratio
        self.penalty_cap = penalty_cap

        self.max_duplicates = 1 if reg_aodv else max_duplicates
        self.improve_factor = 1.0 if reg_aodv else improve_factor
        self.collect_timeout = collect_timeout

        self.reg_aodv = reg_aodv


class Network:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.G = {}  # node_id -> Node

        # arrêt de la simu
        self.stop = False

        # métriques
        self.messages_forwarded = 0
        self.messages_initiated = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.rreq_sent = 0
        self.rreq_forwarded = 0
        self.rrep_sent = 0
        self.energy_consumed = 0.0
        self.dead_nodes = 0
        self.seuiled = 0

        self.first_node_death_time = None
        self.ten_percent_death_time = None
        self.fifty_percent_death_time = None

        # delivery ratio
        self.data_log = {}        # (src_id, data_seq) -> {'t_init','t_send','t_recv'}
        self.data_init_times = [] # [(t, key)]
        self.data_send_times = [] # [(t, key)]

    # ---------- Construction ----------
    def add_node(self, id, pos, battery):
        from node import Node
        n = Node(env=self.env, node_id=id, pos=pos, battery=battery, max_dist=self.cfg.max_dist, network=self)
        self.G[id] = n

    # ---------- Utils ----------
    def get_distance(self, n1, n2):
        dx = n2.pos[0] - n1.pos[0]
        dy = n2.pos[1] - n1.pos[1]
        return (dx * dx + dy * dy) ** 0.5

    def neighbors(self, node):
        for other in self.G.values():
            if other.id == node.id:
                continue
            if not other.alive:
                continue
            if self.get_distance(node, other) <= node.max_dist:
                yield other

    # ---------- Energie ----------
    def _energy_cost(self, node, msg_type, dist):
        if msg_type.startswith("RR"):
            fixed = node.initial_battery * self.cfg.conso_rreq_rrep
        else:
            fixed = node.initial_battery * self.cfg.conso_data
        return self.cfg.cost_per_distance * dist + fixed

    def update_battery(self, node, msg_type, dist):
        cost = self._energy_cost(node, msg_type, dist)
        node.battery = max(0.0, node.battery - cost)
        self.energy_consumed += cost

        if node.battery <= 0 and node.alive:
            self.env.process(self._kill_node(node))

        return node.battery > 0

    def _kill_node(self, node):
        yield self.env.timeout(0)
        node.alive = False
        self.dead_nodes += 1

        t = self.env.now

        if self.first_node_death_time is None:
            self.first_node_death_time = t

        if self.ten_percent_death_time is None and self.dead_nodes >= self.cfg.nb_nodes * 0.1:
            self.ten_percent_death_time = t

        if self.fifty_percent_death_time is None and self.dead_nodes >= self.cfg.nb_nodes * 0.5:
            self.fifty_percent_death_time = t
            self.stop = True

    # ---------- Poids / Routage ----------
    def calculate_weight(self, prev_node, cur_node):
        # AODV classique : tout vaut 1
        if self.cfg.reg_aodv:
            return 1.0

        dist = self.get_distance(prev_node, cur_node)
        dist_norm = dist / max(1e-12, prev_node.max_dist)

        bat = max(cur_node.battery, 0.1)
        bat_norm = 1.0 - (bat / max(1e-12, cur_node.initial_battery))

        w = self.cfg.dist_weight * dist_norm + self.cfg.bat_weight * bat_norm

        seuil = self.cfg.battery_threshold_ratio * cur_node.initial_battery
        if bat < seuil:
            self.seuiled += 1
            ecart = (seuil - bat) / max(1e-12, seuil)
            penalty = min(self.cfg.penalty_cap, ecart)
            w += penalty

        return w

    # ---------- Radio ----------
    def broadcast_rreq(self, node, rreq):
        neighs = [n for n in self.neighbors(node)]
        if not neighs:
            return

        if not self.update_battery(node, "RREQ", node.max_dist):
            return

        for neighbor in neighs:
            yield self.env.timeout(random.uniform(0.01, 0.05))
            msg = copy.deepcopy(rreq)
            neighbor.pending.put(msg)
            self.rreq_forwarded += 1

    def unicast_rrep(self, node, rrep):
        next_hop = node.routing_table.get(rrep.dest_id, (None, 0, 0.0, 0.0))[0]
        if next_hop is None:
            return

        nxt = self.G.get(next_hop)
        if nxt is None or not nxt.alive:
            return

        dist = self.get_distance(node, nxt)
        if dist > node.max_dist:
            return

        dist_for_cost = node.max_dist if self.cfg.reg_aodv else dist
        if not self.update_battery(node, "RREP", dist_for_cost):
            return

        yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
        nxt.pending.put(rrep)

    def forward_data(self, node, data):
        next_hop = node.routing_table.get(data.dest_id, (None, 0, 0.0, 0.0))[0]
        data.prev_hop = node.id

        if next_hop is None:
            return

        nxt = self.G.get(next_hop)
        if nxt is None or not nxt.alive:
            return

        dist = self.get_distance(node, nxt)
        if dist > node.max_dist:
            return

        dist_for_cost = node.max_dist if self.cfg.reg_aodv else dist
        if not self.update_battery(node, "DATA", dist_for_cost):
            return

        self.messages_forwarded += 1
        yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
        nxt.pending.put(data)

    # ---------- Logs delivery ----------
    def log_data_init(self, src_id, data_seq, t_init):
        key = (src_id, data_seq)
        self.data_log[key] = {"t_init": t_init, "t_send": None, "t_recv": None}
        self.data_init_times.append((t_init, key))

    def log_data_send(self, src_id, data_seq, t_send):
        key = (src_id, data_seq)
        e = self.data_log.get(key)
        if e and e["t_send"] is None:
            e["t_send"] = t_send
            self.data_send_times.append((t_send, key))

    def log_data_recv(self, src_id, data_seq, t_recv):
        key = (src_id, data_seq)
        e = self.data_log.get(key)
        if e and e["t_recv"] is None:
            e["t_recv"] = t_recv

    # ---------- Stats ----------
    def get_energy_stats(self):
        alive = [n for n in self.G.values() if n.alive]
        if not alive:
            return 0.0, 0.0
        arr = np.array([n.battery for n in alive], dtype=float)
        return float(arr.mean()), float(arr.std())
