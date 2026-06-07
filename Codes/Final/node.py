import random
from collections import defaultdict

import simpy

from network import Message


class Node:
    """Noeud du reseau : routage AODV, batterie et messages en attente."""

    INITIAL_RREQ_TTL = 2
    RREQ_TTL_STEP = 2
    RREQ_TTL_LIMIT = 7

    def __init__(self, node_id, pos, initial_battery, max_dist, network):
        self.id = node_id
        self.pos = pos
        self.initial_battery = initial_battery
        self.battery = initial_battery
        self.max_dist = max_dist
        self.network = network

        # dest_id -> (next_hop, seq_num, weight)
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

            yield self.network.env.timeout(random.uniform(0.002, 0.003))

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
        if state and state["sent"]:
            return

        ttl = self._next_rreq_ttl(state)
        self.rreq_state[dest_id] = {"ttl": ttl, "sent": True}

        self.seq_num += 1
        self.network.stats.rreq_sent += 1

        rreq = Message(
            type="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0))[1],
            prev_hop=self.id,
            weight=0.0,
            ttl=ttl,
        )
        self.network.env.process(self.network.broadcast_rreq(self, rreq))
        self.network.env.process(self._retry_rreq_if_needed(dest_id))

    def _next_rreq_ttl(self, state):
        if state is None:
            return self.INITIAL_RREQ_TTL
        if state["ttl"] >= self.RREQ_TTL_LIMIT:
            return self.network.cfg.nb_nodes
        return state["ttl"] + self.RREQ_TTL_STEP

    def handle_rreq(self, rreq):
        if rreq.ttl <= 0 or rreq.src_id == self.id:
            return

        previous_node = self.network.G[rreq.prev_hop]
        is_final_hop = self.id == rreq.dest_id
        rreq.weight += self.network.calculate_weight(
            previous_node,
            self,
            is_final_hop=is_final_hop,
        )

        if not self._accept_rreq(rreq):
            return

        if is_final_hop:
            self._collect_candidate_rreq(rreq)
            return

        self.update_route(rreq.src_id, rreq.prev_hop, rreq.src_seq, rreq.weight)

        if rreq.ttl <= 1:
            return

        rreq.prev_hop = self.id
        rreq.ttl -= 1
        self.network.env.process(self.network.broadcast_rreq(self, rreq))

    def _accept_rreq(self, rreq):
        seen_key = (rreq.src_id, rreq.src_seq)
        count, min_weight = self.seen.get(seen_key, (0, float("inf")))
        max_duplicates = 1 if self.network.reg_aodv else self.network.cfg.max_duplicates

        too_many_copies = count >= max_duplicates
        clearly_worse = rreq.weight * self.network.cfg.weight_seuil >= min_weight
        if too_many_copies or clearly_worse:
            return False

        self.seen[seen_key] = (count + 1, rreq.weight)
        return True

    def _collect_candidate_rreq(self, rreq):
        key = (rreq.src_id, rreq.src_seq)
        if key not in self.collected_rreqs:
            self.collected_rreqs[key] = []
            self.network.env.process(self.collect_rreps(key))
        self.collected_rreqs[key].append(rreq)

    def handle_rrep(self, rrep):
        previous_node = self.network.G[rrep.prev_hop]
        is_final_hop = previous_node.id == rrep.src_id

        rrep.weight += self.network.calculate_weight(
            self,
            previous_node,
            is_final_hop=is_final_hop,
        )

        self.update_route(rrep.src_id, rrep.prev_hop, rrep.src_seq, rrep.weight)

        if self.id == rrep.dest_id:
            self.rreq_state.pop(rrep.src_id, None)
            self._send_pending_data(rrep.src_id)
            return

        rrep.prev_hop = self.id
        self.network.env.process(self.network.unicast_rrep(self, rrep))

    def _send_pending_data(self, dest_id):
        for msg in self.to_be_sent.pop(dest_id, []):
            self.network.env.process(self.network.forward_data(self, msg))
            self.network.stats.messages_sent += 1

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

    def update_route(self, dest_id, next_hop, seq_num, weight):
        current = self.routing_table.get(dest_id, (None, -1, float("inf")))
        newer_route = seq_num > current[1]
        lighter_route = seq_num == current[1] and weight < current[2]

        if newer_route or lighter_route:
            self.routing_table[dest_id] = (next_hop, seq_num, weight)

    def collect_rreps(self, key):
        yield self.network.env.timeout(0.006 * self.network.cfg.nb_nodes)
        candidates = self.collected_rreqs.get(key)
        if candidates:
            self.send_rrep(min(candidates, key=lambda rreq: rreq.weight))

    def handle_data(self, data):
        if data.dest_id == self.id:
            self.network.stats.messages_received += 1
        else:
            self.network.env.process(self.network.forward_data(self, data))

    def handle_hello(self, hello):
        self.network.mark_neighbor_seen(self.id, hello.src_id)

    def invalidate_route_via(self, broken_neighbor):
        invalidated = []
        for dest_id, (next_hop, _, _) in list(self.routing_table.items()):
            if next_hop == broken_neighbor:
                del self.routing_table[dest_id]
                invalidated.append(dest_id)
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

        if dest_id in self.routing_table:
            self.network.env.process(self.network.forward_data(self, msg))
            self.network.stats.messages_sent += 1
        else:
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)

    def _retry_rreq_if_needed(self, dest_id):
        retry_delay = 2 * 40 * 10 ** (-3) * self.network.cfg.nb_nodes
        yield self.network.env.timeout(retry_delay)

        state = self.rreq_state.get(dest_id)
        if not state or not state["sent"]:
            return

        if dest_id in self.routing_table:
            self.rreq_state.pop(dest_id, None)
            return

        state["sent"] = False
        self.init_rreq(dest_id)
