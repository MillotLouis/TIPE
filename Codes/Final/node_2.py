from __future__ import annotations

import math
import random
from collections import defaultdict

import simpy

from network_2 import Message


class Node:
    def __init__(self, node_id, pos, initial_battery, max_dist, network):
        self.id = node_id
        self.pos = pos
        self.initial_battery = initial_battery
        self.battery = initial_battery
        self.max_dist = max_dist
        self.network = network

        self.routing_table = {}
        self.seq_num = 0
        self.data_seq = 0

        self.alive = True
        self.pending = simpy.Store(network.env)

        self.seen = {}
        self.collected_rreqs = {}
        self.to_be_sent = defaultdict(list)
        self.rreq_state = {}

        self.network.env.process(self.process_messages())

    def process_messages(self):
        while self.alive:
            msg = yield self.pending.get()

            if not self.network.update_battery(self, msg.type, is_emission=False):
                break

            yield self.network.env.timeout(random.uniform(0.001, 0.005))

            if msg.type == "RREQ":
                self.handle_rreq(msg)
            elif msg.type == "RREP":
                self.handle_rrep(msg)
            elif msg.type == "DATA":
                self.handle_data(msg)
            elif msg.type == "HELLO":
                self.handle_hello(msg)
            elif msg.type == "RERR":
                self.handle_rerr(msg)

    def init_rreq(self, dest_id):
        state = self.rreq_state.get(dest_id)

        if state and state.get("in_flight"):
            return

        ttl_max = max(2, int(self.network.cfg.ttl_max))

        if state is None:
            ttl = 2
        else:
            ttl = min(
                ttl_max,
                max(state["ttl"] + 2, int(math.ceil(state["ttl"] * 1.5))),
            )

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
        is_final_hop = self.id == rreq.dest_id

        rreq.weight += self.network.calculate_weight(
            prev_node,
            self,
            is_final_hop=is_final_hop,
        )

        if rreq.src_id == self.id:
            return

        seen_key = (rreq.src_id, rreq.src_seq)
        count, min_weight = self.seen.get(seen_key, (0, float("inf")))

        max_duplicates = 1 if self.network.reg_aodv else self.network.cfg.max_duplicates
        weight_seuil = 1.0 if self.network.reg_aodv else self.network.cfg.weight_seuil

        if count >= max_duplicates or rreq.weight * weight_seuil >= min_weight:
            return

        self.seen[seen_key] = (count + 1, rreq.weight)

        if self.id == rreq.dest_id:
            key = (rreq.src_id, rreq.src_seq)

            if key not in self.collected_rreqs:
                self.collected_rreqs[key] = []
                self.network.env.process(self.collect_rreps(key))

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

        is_final_hop = prev_node.id == rrep.src_id

        rrep.weight += self.network.calculate_weight(
            self,
            prev_node,
            is_final_hop=is_final_hop,
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
            self.routing_table[dest] = (
                next_hop,
                seq_num,
                weight,
                self.network.env.now + ttl,
            )

    def collect_rreps(self, key):
        yield self.network.env.timeout(0.2)

        if key in self.collected_rreqs:
            best_rreq = min(self.collected_rreqs[key], key=lambda r: r.weight)
            self.send_rrep(best_rreq)
            del self.collected_rreqs[key]

    def handle_data(self, data):
        if data.dest_id == self.id:
            self.network.stats.messages_received += 1
            self.network.log_data_recv(data.src_id, data.src_seq, self.network.env.now)
        else:
            self.network.env.process(self.network.forward_data(self, data))

    def handle_hello(self, hello):
        self.network.mark_neighbor_seen(self.id, hello.src_id)

    def invalidate_route_via(self, broken_neighbor):
        invalidated = []

        for dest, (next_hop, _, _, _) in list(self.routing_table.items()):
            if next_hop == broken_neighbor:
                del self.routing_table[dest]
                invalidated.append(dest)

        return invalidated

    def handle_rerr(self, rerr):
        if rerr.dest_id in self.routing_table:
            del self.routing_table[rerr.dest_id]
            self.network.env.process(self.network.broadcast_rerr(self, [rerr.dest_id]))

    def send_data(self, dest_id):
        self.data_seq += 1
        self.network.stats.messages_initiated += 1

        msg = Message(
            type="DATA",
            src_id=self.id,
            src_seq=self.data_seq,
            dest_id=dest_id,
            prev_hop=self.id,
            weight=-1,
        )

        self.network.log_data_init(self.id, self.data_seq, self.network.env.now)

        if dest_id in self.routing_table:
            self.network.env.process(self.network.forward_data(self, msg))
            self.network.stats.messages_sent += 1
            self.network.log_data_send(self.id, self.data_seq, self.network.env.now)
        else:
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)

    def _retry_rreq_if_needed(self, dest_id, src_seq):
        yield self.network.env.timeout(2 * 40 * 10 ** (-3) * 1.5 * self.network.cfg.nb_nodes)

        state = self.rreq_state.get(dest_id)

        if not state or not state.get("in_flight"):
            return

        if dest_id in self.routing_table:
            self.rreq_state.pop(dest_id, None)
            return

        if self.seq_num != src_seq:
            state["in_flight"] = False
            return

        if state["ttl"] >= self.network.cfg.ttl_max:
            state["in_flight"] = False
            return

        state["in_flight"] = False
        self.init_rreq(dest_id)