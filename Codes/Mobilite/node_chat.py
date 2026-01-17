import random
import simpy


class Node:
    def __init__(self, env, node_id, pos, battery, max_dist, network):
        self.env = env
        self.id = node_id
        self.pos = pos
        self.battery = battery
        self.initial_battery = battery
        self.max_dist = max_dist
        self.network = network

        self.alive = True

        # dest -> (next_hop, seq_num, weight, expiry)
        self.routing_table = {}

        self.seq_num = 0
        self.data_seq = 0

        self.pending = simpy.Store(env)

        # anti-duplicates / collecte
        self.seen = {}            # (src_id, src_seq) -> (count, best_weight)
        self.collected_rreqs = {} # (src_id, src_seq) -> [rreq...]
        self.to_be_sent = {}      # dest_id -> [data...]

        self.env.process(self._process())

    def _process(self):
        while self.alive:
            msg = yield self.pending.get()
            yield self.env.timeout(random.uniform(0.001, 0.005))

            if msg.typ == "RREQ":
                self._handle_rreq(msg)
            elif msg.typ == "RREP":
                self._handle_rrep(msg)
            elif msg.typ == "DATA":
                self._handle_data(msg)

    def send_data(self, dest_id):
        from network import Message

        self.data_seq += 1
        msg = Message("DATA", self.id, self.data_seq, dest_id, dest_seq=-1, weight=0.0, prev_hop=self.id)

        self.network.messages_initiated += 1
        self.network.log_data_init(self.id, self.data_seq, self.env.now)

        route = self.routing_table.get(dest_id)
        if route and route[3] >= self.env.now:
            self.network.messages_sent += 1
            self.env.process(self.network.forward_data(self, msg))
            self.network.log_data_send(self.id, self.data_seq, self.env.now)
        else:
            self.to_be_sent.setdefault(dest_id, []).append(msg)
            self._init_rreq(dest_id)

    # --------- Routage ----------
    def _init_rreq(self, dest_id):
        from network import Message

        self.seq_num += 1
        self.network.rreq_sent += 1

        known_dest_seq = self.routing_table.get(dest_id, (None, 0, 0.0, 0.0))[1]

        rreq = Message(
            "RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=known_dest_seq,
            weight=0.0,
            prev_hop=self.id
        )

        self.env.process(self.network.broadcast_rreq(self, rreq))

    def _handle_rreq(self, rreq):
        # éviter boucle : ne jamais ré-accepter un RREQ qu'on a émis
        if rreq.src_id == self.id:
            return

        prev = self.network.G.get(rreq.prev_hop)
        if prev is None:
            return

        rreq.weight += self.network.calculate_weight(prev, self)

        key = (rreq.src_id, rreq.src_seq)
        count, best = self.seen.get(key, (0, float("inf")))

        if count >= self.network.cfg.max_duplicates:
            return

        if rreq.weight * self.network.cfg.improve_factor >= best:
            return

        self.seen[key] = (count + 1, rreq.weight)

        # si on est la destination
        if self.id == rreq.dest_id:
            if self.network.cfg.reg_aodv:
                self._send_rrep(rreq)
            else:
                self.collected_rreqs.setdefault(key, []).append(rreq)
                if len(self.collected_rreqs[key]) == 1:
                    self.env.process(self._collect_and_reply(key))
            return

        # route inverse vers la source
        self._update_route(dest=rreq.src_id, next_hop=rreq.prev_hop, seq_num=rreq.src_seq, weight=rreq.weight)

        # forward
        rreq.prev_hop = self.id
        self.env.process(self.network.broadcast_rreq(self, rreq))

    def _collect_and_reply(self, key):
        yield self.env.timeout(self.network.cfg.collect_timeout)
        rreqs = self.collected_rreqs.get(key, [])
        if not rreqs:
            return
        best = min(rreqs, key=lambda r: r.weight)
        self._send_rrep(best)

    def _send_rrep(self, rreq):
        from network import Message

        self.seq_num += 1
        self.network.rrep_sent += 1

        # route inverse (dest -> source)
        self._update_route(dest=rreq.src_id, next_hop=rreq.prev_hop, seq_num=rreq.src_seq, weight=rreq.weight)

        rrep = Message(
            "RREP",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=rreq.src_id,
            dest_seq=-1,
            weight=0.0,
            prev_hop=self.id
        )

        self.env.process(self.network.unicast_rrep(self, rrep))

    def _handle_rrep(self, rrep):
        prev = self.network.G.get(rrep.prev_hop)
        if prev is None:
            return

        rrep.weight += self.network.calculate_weight(prev, self)

        # route vers la destination (celui qui a envoyé ce RREP)
        self._update_route(dest=rrep.src_id, next_hop=rrep.prev_hop, seq_num=rrep.src_seq, weight=rrep.weight)

        # si on est la source du RREQ initial
        if self.id == rrep.dest_id:
            pending = self.to_be_sent.pop(rrep.src_id, [])
            for msg in pending:
                self.env.process(self.network.forward_data(self, msg))
                self.network.messages_sent += 1
                self.network.log_data_send(self.id, msg.src_seq, self.env.now)
        else:
            rrep.prev_hop = self.id
            self.env.process(self.network.unicast_rrep(self, rrep))

    def _update_route(self, dest, next_hop, seq_num, weight):
        cur = self.routing_table.get(dest, (None, -1, float("inf"), 0.0))
        ttl = self.network.cfg.ttl

        if (seq_num > cur[1]) or (seq_num == cur[1] and weight < cur[2]):
            self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + ttl)

    # --------- DATA ----------
    def _handle_data(self, data):
        if data.dest_id == self.id:
            self.network.messages_received += 1
            self.network.log_data_recv(data.src_id, data.src_seq, self.env.now)
        else:
            self.env.process(self.network.forward_data(self, data))
