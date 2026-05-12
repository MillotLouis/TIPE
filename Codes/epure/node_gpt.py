from __future__ import annotations

import random
from collections import defaultdict

import simpy

from network_gpt import Message


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

        self.network.env.process(self.process_messages())

    def process_messages(self):
        while self.alive:
            msg = yield self.pending.get()  # On bloque le process jusqu'à avoir un nouveau message
            yield self.network.env.timeout(random.uniform(0.001, 0.005))  # délai de processing
            if msg.type == "RREQ":
                self.handle_rreq(msg)
            elif msg.type == "RREP":
                self.handle_rrep(msg)
            elif msg.type == "DATA":
                self.handle_data(msg)

    def init_rreq(self, dest_id):
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
        )
        self.network.env.process(self.network.broadcast_rreq(self, rreq))

    def handle_rreq(self, rreq):
        prev_node = self.network.G[rreq.prev_hop]
        rreq.weight += self.network.calculate_weight(prev_node, self)
        if rreq.src_id == self.id:
            return  # éviter que les RREQs soient renvoyés à la source

        seen_key = (rreq.src_id, rreq.src_seq)
        count, min_weight = self.seen.get(seen_key, (0, float("inf")))
        if count > self.network.protocol.max_duplicates or rreq.weight * self.network.protocol.weight_seuil >= min_weight:
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
        rreq.prev_hop = self.id
        self.network.env.process(self.network.broadcast_rreq(self, rreq))

    def handle_rrep(self, rrep):
        prev_node = self.network.G[rrep.prev_hop]
        rrep.weight += self.network.calculate_weight(prev_node, self)
        self.update_route(rrep.src_id, rrep.prev_hop, rrep.src_seq, rrep.weight)

        if self.id == rrep.dest_id:
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
        ttl = self.network.cfg.ttl
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

        if dest_id in self.routing_table:
            self.network.env.process(self.network.forward_data(self, msg))
            self.network.stats.messages_sent += 1
            self.network.log_data_send(self.id, self.data_seq, self.network.env.now)
        else:
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)