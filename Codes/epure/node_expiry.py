from __future__ import annotations

import random
from collections import defaultdict
import math

import simpy

from network_expiry import Message


class Node:
    def __init__(self, node_id, pos, initial_battery, max_dist, network):
        self.id = node_id  # id du noeud
        self.pos = pos  # position (x,y) du noeud
        self.initial_battery = initial_battery
        self.battery = initial_battery  # batterie du noeud
        self.max_dist = max_dist  # distance max d'émission / portée du noeud
        self.network = network

        self.routing_table = {}  # dest : {next_hop,seq_num,weight,expiry}
        self.seq_num = 0
        self.data_seq = 0

        self.alive = True  # True si le noeud est vivant False si il est mort
        self.pending = simpy.Store(network.env)  # requêtes en attente
        self.seen = {}
        self.collected_rreqs = {}
        self.to_be_sent = defaultdict(list)
        self.rreq_state = {}  # dest_id -> {ttl, in_flight}

        self.network.env.process(self.process_messages())

    def process_messages(self):
        while self.alive:
            msg = yield self.pending.get()  # On bloque le process jusqu'à avoir un nouveau message
            if not self.network.update_battery(self, msg.type, is_emission=False):
                break
            yield self.network.env.timeout(random.uniform(0.001, 0.005))  # délai de processing
            if msg.type == "RREQ":
                self.handle_rreq(msg)
            elif msg.type == "RREP":
                self.handle_rrep(msg)
            elif msg.type == "DATA":
                self.handle_data(msg)

    def init_rreq(self, dest_id):
        state = self.rreq_state.get(dest_id)
        if state and state.get("in_flight"):
            # Si on est déjà en train d'envoyer des RREQ à ce destinataire
            return

        ttl_max = max(2, int(self.network.cfg.ttl_max))
        if state is None:
            ttl = 2
        else:
            ttl = min(ttl_max, max(state["ttl"] + 2, int(math.ceil(state["ttl"] * 1.5))))

        self.rreq_state[dest_id] = {"ttl": ttl, "in_flight": True}

        self.seq_num += 1
        self.network.stats.rreq_sent += 1
        rreq = Message(
            type="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0, 0))[1],
            prev_hop=self.id,
            weight=0.0,
            ttl=ttl,
        )
        self.network.env.process(self.network.broadcast_rreq(self, rreq))
        self.network.env.process(self._retry_rreq_if_needed(dest_id, rreq.src_seq))

    def handle_rreq(self, rreq):
        if rreq.ttl <= 0:
            return
        prev_node = self.network.G[rreq.prev_hop]

        is_final_hop = (self.id == rreq.dest_id)
        rreq.weight += self.network.calculate_weight(
            prev_node,
            self,
            is_final_hop=is_final_hop
        )
        
        if rreq.src_id == self.id:
            return  # éviter que les RREQs soient renvoyés à la source

        seen_key = (rreq.src_id, rreq.src_seq)
        count, min_weight = self.seen.get(seen_key, (0, float("inf")))
        if count >= self.network.protocol.max_duplicates or rreq.weight * self.network.protocol.weight_seuil >= min_weight:
            return
        self.seen[seen_key] = (count + 1, rreq.weight)

        if self.id == rreq.dest_id:  # Si on est la destination du RREQ
            key = (rreq.src_id, rreq.src_seq)
            if key not in self.collected_rreqs:
                self.collected_rreqs[key] = []
                self.network.env.process(self.collect_rreps(key))  # on commence la collecte des RREPs
            self.collected_rreqs[key].append(rreq)
            return

        self.update_route(rreq.src_id, rreq.prev_hop, rreq.src_seq, rreq.weight)
        if rreq.ttl <= 1:
            return

        rreq.prev_hop = self.id
        rreq.ttl -= 1
        self.network.env.process(self.network.broadcast_rreq(self, rreq))

    def handle_rrep(self, rrep):
        prev_node = self.network.G[rrep.prev_hop]

        # Le coût doit être évalué dans le sens futur DATA :
        # le nœud courant enverra vers prev_node pour atteindre rrep.src_id.
        is_final_hop = (prev_node.id == rrep.src_id)

        rrep.weight += self.network.calculate_weight(
            self,
            prev_node,
            is_final_hop=is_final_hop
        )

        self.update_route(rrep.src_id, rrep.prev_hop, rrep.src_seq, rrep.weight)

        if self.id == rrep.dest_id:
            self.rreq_state.pop(rrep.src_id, None)
            if rrep.src_id in self.to_be_sent:
                for msg in self.to_be_sent[rrep.src_id]:
                    self.network.env.process(self.network.forward_data(self, msg))
                    self.network.stats.messages_sent += 1
                    self.network.log_data_send(self.id, msg.src_seq, self.network.env.now)
                del self.to_be_sent[rrep.src_id]
            return

        rrep.prev_hop = self.id
        self.network.env.process(self.network.unicast_rrep(self, rrep))

    def send_rrep(self, rreq):
        self.seq_num += 1
        self.network.stats.rrep_sent += 1
        self.update_route(rreq.src_id, rreq.prev_hop, rreq.src_seq, rreq.weight)
        rrep = Message(
            type="RREP",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=rreq.src_id,
            prev_hop=self.id,
        )
        self.network.env.process(self.network.unicast_rrep(self, rrep))

    def update_route(self, dest, next_hop, seq_num, weight):
        current = self.routing_table.get(dest, (None, -1, float("inf"), 0))
        ttl = self.network.cfg.ttl_max
        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
            self.routing_table[dest] = (next_hop, seq_num, weight, self.network.env.now + ttl)

    def collect_rreps(self, key):
        yield self.network.env.timeout(0.2)
        # On attend pour que tous les RREQs arrivent à la dest
        if key in self.collected_rreqs:
            self.send_rrep(min(self.collected_rreqs[key], key=lambda r: r.weight))  # on envoie le meilleur

    def handle_data(self, data):
        if data.dest_id == self.id:
            self.network.stats.messages_received += 1
            self.network.log_data_recv(data.src_id, data.src_seq, self.network.env.now)
        else:
            self.network.env.process(self.network.forward_data(self, data))


    def send_data(self, dest_id):
        self.data_seq += 1
        self.network.stats.messages_initiated += 1
        msg = Message(type="DATA", src_id=self.id, src_seq=self.data_seq, dest_id=dest_id, prev_hop=self.id, weight=-1)
        self.network.log_data_init(self.id, self.data_seq, self.network.env.now)

        route = self.routing_table.get(dest_id)
        if route and route[3] > self.network.env.now:
            self.network.env.process(self.network.forward_data(self, msg))
            self.network.stats.messages_sent += 1
            self.network.log_data_send(self.id, self.data_seq, self.network.env.now)
        else:
            if route and route[3] <= self.network.env.now:
                del self.routing_table[dest_id]
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)
    def _retry_rreq_if_needed(self, dest_id, src_seq):
        yield self.network.env.timeout(2* 40*10**(-3)* 1.5*self.network.cfg.nb_nodes)

        state = self.rreq_state.get(dest_id)
        if not state or not state.get("in_flight"):
            # Si on a fini de chercher une route
            return

        if dest_id in self.routing_table:
            # Si on a fini de chercher une route et qu'elle est inscrite dans la table de routage
            self.rreq_state.pop(dest_id, None)
            return

        if self.seq_num != src_seq:
            state["in_flight"] = False
            return

        if state["ttl"] >= 7:
            state["in_flight"] = False
            return

        state["in_flight"] = False
        self.init_rreq(dest_id)
