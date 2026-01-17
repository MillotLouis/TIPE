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
        n = Node(env=self.env, node_id=id, pos=pos, battery=battery, max_dist=self.cfg.max_dist, network_
